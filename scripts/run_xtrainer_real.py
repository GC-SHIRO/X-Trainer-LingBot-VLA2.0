import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.websocket_client_policy import WebsocketClientPolicy
from deploy.xtrainer_real import XTrainerRealEnvironment


@dataclass
class PrefetchResult:
    actions: np.ndarray
    request_step: int


def _servo_range(value: str) -> tuple[int, int]:
    try:
        minimum, maximum = (int(part) for part in value.split(",", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected MIN,MAX") from exc
    if minimum >= maximum:
        raise argparse.ArgumentTypeError("MIN must be less than MAX")
    return minimum, maximum


def _extract_action_chunk(response: dict, action_horizon: int) -> np.ndarray:
    if "action" not in response:
        raise KeyError(f"Missing 'action' in policy response: {tuple(response.keys())}")
    actions = np.asarray(response["action"], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected action shape (H, 14), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("Policy returned an empty action chunk")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned non-finite actions")
    return actions[:action_horizon]


def _rate_limit_action(action: np.ndarray, last_action: np.ndarray | None, max_delta_per_step: float) -> np.ndarray:
    target = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if last_action is None or max_delta_per_step <= 0:
        return target
    previous = np.asarray(last_action, dtype=np.float64).reshape(-1)
    return previous + np.clip(target - previous, -max_delta_per_step, max_delta_per_step)


def _smooth_chunk_boundary(
    chunk: np.ndarray,
    last_action: np.ndarray | None,
    *,
    max_switch_delta: float,
    blend_steps: int,
) -> np.ndarray:
    smoothed = np.asarray(chunk, dtype=np.float64).copy()
    if last_action is None or len(smoothed) == 0 or max_switch_delta <= 0:
        return smoothed

    previous = np.asarray(last_action, dtype=np.float64).reshape(-1)
    delta = float(np.max(np.abs(smoothed[0] - previous)))
    if delta <= max_switch_delta:
        return smoothed

    steps = min(max(blend_steps, 1), len(smoothed))
    for index in range(steps):
        weight = float(index + 1) / float(steps + 1)
        smoothed[index] = previous + weight * (smoothed[index] - previous)
    logging.warning(
        "Blended action chunk boundary: max delta %.4f exceeded threshold %.4f over %d steps",
        delta,
        max_switch_delta,
        steps,
    )
    return smoothed


def _infer_action_chunk(policy: WebsocketClientPolicy, observation: dict, action_horizon: int) -> np.ndarray:
    response = policy.infer(observation)
    return _extract_action_chunk(response, action_horizon)


def _infer_prefetch_chunk(
    policy: WebsocketClientPolicy,
    observation: dict,
    action_horizon: int,
    request_step: int,
) -> PrefetchResult:
    return PrefetchResult(
        actions=_infer_action_chunk(policy, observation, action_horizon),
        request_step=request_step,
    )


def _with_projected_state(observation: dict, projected_state: np.ndarray | None) -> dict:
    if projected_state is None:
        return observation
    projected = dict(observation)
    projected["observation.state"] = np.asarray(projected_state, dtype=np.float32).copy()
    return projected


def _project_chunk_end_state(
    action_chunk: np.ndarray,
    action_index: int,
    last_sent_action: np.ndarray | None,
    max_delta_per_step: float,
) -> np.ndarray | None:
    if action_index >= len(action_chunk):
        return last_sent_action.copy() if last_sent_action is not None else None

    if max_delta_per_step <= 0 or last_sent_action is None:
        return np.asarray(action_chunk[-1], dtype=np.float64).copy()

    projected = np.asarray(last_sent_action, dtype=np.float64).copy()
    for action in action_chunk[action_index:]:
        projected = _rate_limit_action(action, projected, max_delta_per_step)
    return projected


def _apply_prefetched_chunk(
    action_chunk: np.ndarray,
    action_index: int,
    next_chunk: np.ndarray | None,
    prefetched_chunk: np.ndarray,
    last_sent_action: np.ndarray | None,
    *,
    apply_mode: str,
    max_switch_delta: float,
    blend_steps: int,
) -> tuple[np.ndarray, int, np.ndarray | None, int]:
    if apply_mode == "replace":
        discarded_actions = max(len(action_chunk) - action_index, 0)
        return (
            _smooth_chunk_boundary(
                prefetched_chunk,
                last_sent_action,
                max_switch_delta=max_switch_delta,
                blend_steps=blend_steps,
            ),
            0,
            None,
            discarded_actions,
        )
    if apply_mode == "boundary":
        return action_chunk, action_index, prefetched_chunk, 0
    raise ValueError(f"Unsupported prefetch apply mode: {apply_mode}")


def _align_prefetched_chunk(
    prefetched_chunk: np.ndarray,
    request_step: int,
    current_step: int,
    last_sent_action: np.ndarray | None = None,
    *,
    mode: str = "nearest",
    search_window: int = 12,
) -> tuple[np.ndarray, int]:
    elapsed_steps = max(int(current_step - request_step), 0)
    if elapsed_steps >= len(prefetched_chunk):
        return np.empty((0, prefetched_chunk.shape[1]), dtype=np.float64), elapsed_steps

    start_index = elapsed_steps
    if mode == "elapsed":
        pass
    elif mode == "nearest":
        if last_sent_action is not None and len(prefetched_chunk) > 0:
            search_start = elapsed_steps
            search_end = min(len(prefetched_chunk), elapsed_steps + max(search_window, 1) + 1)
            candidates = np.asarray(prefetched_chunk[search_start:search_end], dtype=np.float64)
            previous = np.asarray(last_sent_action, dtype=np.float64).reshape(1, -1)
            distances = np.max(np.abs(candidates - previous), axis=1)
            start_index = search_start + int(np.argmin(distances))
    else:
        raise ValueError(f"Unsupported prefetch alignment mode: {mode}")

    return np.asarray(prefetched_chunk[start_index:], dtype=np.float64).copy(), start_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LingBot-VLA 2.0 on an X-Trainer robot")
    parser.add_argument("--host", required=True, help="LingBot policy server address")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--left-robot-ip", default="192.168.5.1")
    parser.add_argument("--right-robot-ip", default="192.168.5.2")
    parser.add_argument("--left-gripper-port", default="/dev/ttyUSB1")
    parser.add_argument("--right-gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--left-gripper-id", type=int, default=21)
    parser.add_argument("--right-gripper-id", type=int, default=22)
    parser.add_argument("--left-gripper-servo-pos", type=_servo_range, default=(2048, 3052), metavar="MIN,MAX")
    parser.add_argument("--right-gripper-servo-pos", type=_servo_range, default=(2048, 3052), metavar="MIN,MAX")
    parser.add_argument("--camera-top-serial", required=True)
    parser.add_argument("--camera-left-wrist-serial", required=True)
    parser.add_argument("--camera-right-wrist-serial", required=True)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=float("inf"),
        help="Optional environment joint delta limit in radians; default disables it",
    )
    parser.add_argument("--ramp-step", type=float, default=0.01)
    parser.add_argument("--ramp-max-steps", type=int, default=100)
    parser.add_argument(
        "--gripper-update-threshold",
        type=float,
        default=0.0,
        help="Minimum gripper target change to transmit; default sends every change",
    )
    parser.add_argument(
        "--servo-step-limit",
        type=float,
        default=float("inf"),
        help="Optional follower joint jump limit in radians; default disables it",
    )
    parser.add_argument(
        "--prefetch-remaining",
        type=int,
        default=0,
        help="Start background inference when this many actions remain in the current chunk; default disables prefetch",
    )
    parser.add_argument(
        "--prefetch-state-mode",
        choices=("chunk-end", "current"),
        default="current",
        help=(
            "State used in the observation sent by async prefetch. "
            "'chunk-end' replaces observation.state with the planned end state of the current chunk; "
            "'current' keeps the sampled state unchanged."
        ),
    )
    parser.add_argument(
        "--prefetch-apply-mode",
        choices=("replace", "boundary"),
        default="replace",
        help=(
            "How to use a completed prefetch. 'replace' immediately discards the remaining current chunk "
            "and starts the new chunk; 'boundary' waits until the current chunk is exhausted."
        ),
    )
    parser.add_argument(
        "--disable-prefetch-alignment",
        action="store_true",
        help="Do not drop elapsed actions from a completed prefetch before applying it",
    )
    parser.add_argument(
        "--prefetch-alignment-mode",
        choices=("nearest", "elapsed"),
        default="nearest",
        help=(
            "How to align a returned prefetch. 'elapsed' drops actions by elapsed control steps; "
            "'nearest' additionally starts from the action closest to the last sent action."
        ),
    )
    parser.add_argument(
        "--prefetch-alignment-search",
        type=int,
        default=12,
        help="Number of post-latency actions searched when --prefetch-alignment-mode=nearest",
    )
    parser.add_argument(
        "--min-prefetch-actions",
        type=int,
        default=12,
        help="Discard an aligned prefetched chunk if fewer than this many actions remain",
    )
    parser.add_argument(
        "--switch-blend-steps",
        type=int,
        default=5,
        help="Number of actions used to blend across a large chunk-boundary jump",
    )
    parser.add_argument(
        "--max-switch-delta",
        type=float,
        default=0.12,
        help="Blend the next chunk if its first action differs from the last sent action by more than this",
    )
    parser.add_argument(
        "--max-delta-per-step",
        type=float,
        default=0.0,
        help="Optional final per-control-step action delta limit; <=0 disables this client-side limiter",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "port": args.port,
        "action_horizon": args.action_horizon,
        "control_hz": args.control_hz,
        "max_steps": args.max_steps,
        "camera_fps": args.camera_fps,
        "ramp_step": args.ramp_step,
        "ramp_max_steps": args.ramp_max_steps,
        "servo_step_limit": args.servo_step_limit,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"Expected positive values for: {', '.join(invalid)}")
    if args.max_joint_delta < 0 or args.gripper_update_threshold < 0:
        raise ValueError("Action thresholds must be non-negative")
    non_negative_values = {
        "prefetch_remaining": args.prefetch_remaining,
        "switch_blend_steps": args.switch_blend_steps,
        "max_switch_delta": args.max_switch_delta,
        "max_delta_per_step": args.max_delta_per_step,
        "prefetch_alignment_search": args.prefetch_alignment_search,
        "min_prefetch_actions": args.min_prefetch_actions,
    }
    invalid = [name for name, value in non_negative_values.items() if value < 0]
    if invalid:
        raise ValueError(f"Expected non-negative values for: {', '.join(invalid)}")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    if metadata.get("model_type") not in (None, "lingbot-vla-2.0"):
        raise RuntimeError(f"Unexpected model type: {metadata.get('model_type')}")
    if metadata.get("robot") not in (None, "xtrainer"):
        raise RuntimeError(f"Unexpected robot config: {metadata.get('robot')}")
    if metadata.get("mock_policy"):
        logging.warning("Connected to a mock hold-current policy; no learned actions will be executed")

    environment = XTrainerRealEnvironment(
        left_robot_ip=args.left_robot_ip,
        right_robot_ip=args.right_robot_ip,
        left_gripper_port=args.left_gripper_port,
        right_gripper_port=args.right_gripper_port,
        left_gripper_id=args.left_gripper_id,
        right_gripper_id=args.right_gripper_id,
        left_gripper_servo_pos=args.left_gripper_servo_pos,
        right_gripper_servo_pos=args.right_gripper_servo_pos,
        camera_top_serial=args.camera_top_serial,
        camera_left_wrist_serial=args.camera_left_wrist_serial,
        camera_right_wrist_serial=args.camera_right_wrist_serial,
        camera_fps=args.camera_fps,
        task=args.task,
        reset_pose=metadata.get("reset_pose"),
        max_joint_delta=args.max_joint_delta,
        ramp_step=args.ramp_step,
        ramp_max_steps=args.ramp_max_steps,
        gripper_update_threshold=args.gripper_update_threshold,
        servo_step_limit=args.servo_step_limit,
    )

    action_chunk = np.empty((0, 14), dtype=np.float64)
    action_index = 0
    next_chunk: np.ndarray | None = None
    prefetch_future: Future | None = None
    last_sent_action: np.ndarray | None = None
    period = 1.0 / args.control_hz
    deadline = time.monotonic()
    try:
        environment.reset()
        with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
            for step in range(args.max_steps):
                if prefetch_future is not None and prefetch_future.done():
                    prefetch_result = prefetch_future.result()
                    prefetch_future = None
                    prefetched_chunk = prefetch_result.actions
                    skipped_actions = 0
                    if not args.disable_prefetch_alignment:
                        prefetched_chunk, skipped_actions = _align_prefetched_chunk(
                            prefetched_chunk,
                            prefetch_result.request_step,
                            step,
                            last_sent_action,
                            mode=args.prefetch_alignment_mode,
                            search_window=args.prefetch_alignment_search,
                        )
                    logging.info(
                        "Prefetched %d actions requested at step %d, applying at step %d after skipping %d elapsed actions",
                        len(prefetched_chunk),
                        prefetch_result.request_step,
                        step,
                        skipped_actions,
                    )
                    if len(prefetched_chunk) == 0:
                        logging.warning(
                            "Discarded stale prefetched chunk requested at step %d; no aligned actions remain",
                            prefetch_result.request_step,
                        )
                        continue
                    if len(prefetched_chunk) < args.min_prefetch_actions:
                        logging.warning(
                            "Discarded short prefetched chunk requested at step %d; only %d aligned actions remain (< %d)",
                            prefetch_result.request_step,
                            len(prefetched_chunk),
                            args.min_prefetch_actions,
                        )
                        continue
                    action_chunk, action_index, next_chunk, discarded_actions = _apply_prefetched_chunk(
                        action_chunk,
                        action_index,
                        next_chunk,
                        prefetched_chunk,
                        last_sent_action,
                        apply_mode=args.prefetch_apply_mode,
                        max_switch_delta=args.max_switch_delta,
                        blend_steps=args.switch_blend_steps,
                    )
                    if args.prefetch_apply_mode == "replace":
                        logging.info(
                            "Replaced current action chunk at step %d; discarded %d remaining actions",
                            step,
                            discarded_actions,
                        )

                if action_index >= len(action_chunk):
                    if next_chunk is not None:
                        action_chunk = _smooth_chunk_boundary(
                            next_chunk,
                            last_sent_action,
                            max_switch_delta=args.max_switch_delta,
                            blend_steps=args.switch_blend_steps,
                        )
                        next_chunk = None
                    elif prefetch_future is not None:
                        if last_sent_action is None:
                            logging.warning("Action chunk exhausted while prefetch is running and no last action is available")
                            continue
                        logging.warning("Action chunk exhausted before prefetch finished; holding last action")
                        action = last_sent_action.copy()
                        environment.apply_action(action)
                        deadline += period
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            time.sleep(remaining)
                        else:
                            deadline = time.monotonic()
                        continue
                    else:
                        action_chunk = _infer_action_chunk(policy, environment.get_observation(), args.action_horizon)
                    action_index = 0
                    logging.info("Received %d actions at step %d", len(action_chunk), step)

                if action_index < len(action_chunk):
                    remaining_actions = len(action_chunk) - action_index
                    if (
                        args.prefetch_remaining > 0
                        and remaining_actions <= args.prefetch_remaining
                        and prefetch_future is None
                        and next_chunk is None
                    ):
                        observation = environment.get_observation()
                        if args.prefetch_state_mode == "chunk-end":
                            sampled_state = np.asarray(observation.get("observation.state"), dtype=np.float64)
                            projected_state = _project_chunk_end_state(
                                action_chunk,
                                action_index,
                                last_sent_action,
                                args.max_delta_per_step,
                            )
                            observation = _with_projected_state(observation, projected_state)
                            if projected_state is not None:
                                state_delta = float(np.max(np.abs(projected_state - sampled_state)))
                                logging.debug(
                                    "Using projected chunk-end state for prefetch at step %d; max state delta %.4f",
                                    step,
                                    state_delta,
                                )
                        prefetch_future = prefetch_executor.submit(
                            _infer_prefetch_chunk,
                            policy,
                            observation,
                            args.action_horizon,
                            step,
                        )
                        logging.info(
                            "Started action prefetch at step %d with %d actions remaining",
                            step,
                            remaining_actions,
                        )

                    action = _rate_limit_action(action_chunk[action_index], last_sent_action, args.max_delta_per_step)
                    action_index += 1
                elif last_sent_action is not None and prefetch_future is not None:
                    logging.warning("Action chunk exhausted before prefetch finished; holding last action")
                    action = last_sent_action.copy()
                else:
                    action = _infer_action_chunk(policy, environment.get_observation(), args.action_horizon)[0]

                environment.apply_action(action)
                last_sent_action = action.copy()
                deadline += period
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.monotonic()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        environment.close()
        policy.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
