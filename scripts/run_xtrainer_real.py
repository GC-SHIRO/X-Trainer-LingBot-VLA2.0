import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.websocket_client_policy import WebsocketClientPolicy
from deploy.inference_logging import InferenceRecorder
from deploy.xtrainer_real import XTrainerRealEnvironment


JOINT_INDICES = np.r_[0:6, 7:13]


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


def _chunk_blend_offset(last_sent_action: np.ndarray | None, first_action: np.ndarray) -> np.ndarray | None:
    """Offset between where the arm is and where the new chunk starts.

    Captured once per chunk from the last *applied* action, before any blending
    is applied to it.
    """
    if last_sent_action is None:
        return None
    return np.asarray(last_sent_action, dtype=np.float64).reshape(-1) - np.asarray(
        first_action, dtype=np.float64
    ).reshape(-1)


def _blend_chunk_action(
    action: np.ndarray,
    blend_offset: np.ndarray | None,
    index: int,
    blend_steps: int,
) -> np.ndarray:
    """Decay the chunk-splice offset over the first actions of a new chunk.

    Only a constant offset is faded out, so the new trajectory's own motion is
    preserved while the splice discontinuity disappears. Grippers are never
    blended, so an open/close command is not smeared by the transition.
    """
    target = np.asarray(action, dtype=np.float64).copy()
    if blend_offset is None or blend_steps <= 1 or index >= blend_steps:
        return target
    progress = float(index + 1) / float(blend_steps)
    weight = progress * progress * (3.0 - 2.0 * progress)
    target[JOINT_INDICES] += (1.0 - weight) * blend_offset[JOINT_INDICES]
    return target


def _hold_action_from_observation(observation: dict, last_sent_action: np.ndarray) -> np.ndarray:
    """Hold measured arm joints while retaining the previous gripper targets."""
    if "observation.state" not in observation:
        raise KeyError("Missing 'observation.state' in client observation")
    state = np.asarray(observation["observation.state"], dtype=np.float64).reshape(-1).copy()
    if state.shape[0] != 14:
        raise ValueError(f"Expected observation.state length 14, got {state.shape[0]}")
    if not np.all(np.isfinite(state)):
        raise ValueError("observation.state contains non-finite values")
    previous = np.asarray(last_sent_action, dtype=np.float64).reshape(-1)
    if previous.shape[0] != 14 or not np.all(np.isfinite(previous)):
        raise ValueError("last_sent_action must contain 14 finite values")
    state[[6, 13]] = previous[[6, 13]]
    return state


def _log_server_timing(response: dict) -> None:
    timing = response.get("server_timing")
    if not isinstance(timing, dict):
        return
    infer_ms = timing.get("infer_ms")
    prev_total_ms = timing.get("prev_total_ms")
    logging.info(
        "Server timing: infer=%.1f ms, previous round trip=%.1f ms",
        infer_ms if infer_ms is not None else float("nan"),
        prev_total_ms if prev_total_ms is not None else float("nan"),
    )


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
        "--chunk-blend-steps",
        type=int,
        default=6,
        help="Actions over which the chunk-splice offset decays at a chunk boundary; <=1 disables blending",
    )
    parser.add_argument(
        "--max-delta-per-step",
        type=float,
        default=0.0,
        help="Optional final per-control-step action delta limit; <=0 disables this client-side limiter",
    )
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for camera observations sent to the server; 0 sends raw ndarrays",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Write raw policy requests/responses, input PNGs, and applied actions under ./log",
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
        "chunk_blend_steps": args.chunk_blend_steps,
        "max_delta_per_step": args.max_delta_per_step,
    }
    invalid = [name for name, value in non_negative_values.items() if value < 0]
    if invalid:
        raise ValueError(f"Expected non-negative values for: {', '.join(invalid)}")
    if not 0 <= args.image_jpeg_quality <= 100:
        raise ValueError("--image-jpeg-quality must be in [0, 100]")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    recorder = InferenceRecorder("real") if args.log else None
    policy = WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        inference_callback=recorder.record_inference if recorder is not None else None,
        image_jpeg_quality=args.image_jpeg_quality,
    )
    metadata = policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    if metadata.get("model_type") not in (None, "lingbot-vla-2.0"):
        raise RuntimeError(f"Unexpected model type: {metadata.get('model_type')}")
    if metadata.get("robot") not in (None, "xtrainer"):
        raise RuntimeError(f"Unexpected robot config: {metadata.get('robot')}")
    if metadata.get("mock_policy"):
        logging.warning("Connected to a mock hold-current policy; no learned actions will be executed")
    robot_name = metadata.get("robot") or "xtrainer"

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
    blend_offset: np.ndarray | None = None
    blend_index = 0
    last_sent_action: np.ndarray | None = None
    period = 1.0 / args.control_hz
    deadline = time.monotonic()
    try:
        environment.reset()
        # Clear server-side episode state once the arm is at the reset pose, so
        # a long-lived server cannot carry an action cache into this run.
        policy.reset(robot_name)
        logging.info("Server policy reset for robot config %s", robot_name)

        for step in range(args.max_steps):
            if action_index >= len(action_chunk):
                observation = environment.get_observation()
                if last_sent_action is not None:
                    # Command the measured pose as a stationary hold. Holding the
                    # last *commanded* target instead would keep pushing toward a
                    # stale target if the arm was pushed or sagged during the
                    # chunk; holding the measured pose stops it where it is.
                    # Keep gripper targets so contact with an object does not
                    # turn a closing command into a hold at the blocked opening.
                    hold_action = _hold_action_from_observation(observation, last_sent_action)
                    environment.apply_action(hold_action)
                    last_sent_action = hold_action.copy()
                    if recorder is not None:
                        recorder.record_applied_action(hold_action, step=step, source="hold")
                    logging.info(
                        "Action chunk exhausted at step %d; holding measured pose while requesting the next chunk",
                        step,
                    )
                response = policy.infer(observation)
                _log_server_timing(response)
                action_chunk = _extract_action_chunk(response, args.action_horizon)
                action_index = 0
                blend_index = 0
                blend_offset = _chunk_blend_offset(last_sent_action, action_chunk[0])
                logging.info("Received %d actions at step %d", len(action_chunk), step)

            target = _blend_chunk_action(
                action_chunk[action_index],
                blend_offset,
                blend_index,
                args.chunk_blend_steps,
            )
            blend_index += 1
            action = _rate_limit_action(target, last_sent_action, args.max_delta_per_step)
            action_index += 1

            environment.apply_action(action)
            if recorder is not None:
                recorder.record_applied_action(action, step=step, source="chunk")
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
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
