"""Sequential X-Trainer joint and gripper test without a policy server.

This is intentionally not named ``test_*.py`` so automated test discovery
cannot move real hardware. It uses the same policy response contract as the
WebSocket server: ``{"action": np.ndarray}`` with shape ``(H, 14)``.
"""

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.xtrainer_real import XTrainerRealEnvironment


ACTION_DIM = 14
JOINT_DELTA_RAD = float(np.deg2rad(5.0))
JOINTS = (
    *(("left", joint + 1, joint) for joint in range(6)),
    *(("right", joint + 1, 7 + joint) for joint in range(6)),
)
GRIPPERS = (("left", 6), ("right", 13))


def _servo_range(value: str) -> tuple[int, int]:
    try:
        minimum, maximum = (int(part) for part in value.split(",", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected MIN,MAX") from exc
    if minimum >= maximum:
        raise argparse.ArgumentTypeError("MIN must be less than MAX")
    return minimum, maximum


def _policy_response(start: np.ndarray, target: np.ndarray, duration: float, control_hz: float) -> dict:
    """Build the same ``action`` chunk shape returned by the policy server."""
    steps = max(1, int(round(duration * control_hz)))
    weights = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float64)[:, None]
    action_chunk = start[None, :] + weights * (target - start)[None, :]
    return {"action": action_chunk.astype(np.float32)}


def _hold_response(target: np.ndarray, duration: float, control_hz: float) -> dict:
    steps = max(1, int(round(duration * control_hz)))
    return {"action": np.repeat(target[None, :], steps, axis=0).astype(np.float32)}


def _extract_action_chunk(response: dict) -> np.ndarray:
    if "action" not in response:
        raise KeyError(f"Missing 'action' in policy response: {tuple(response)}")
    actions = np.asarray(response["action"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] == 0:
        raise ValueError(f"Expected action shape (H, {ACTION_DIM}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Action chunk contains NaN or Inf")
    return actions


def _execute_response(environment: XTrainerRealEnvironment, response: dict, control_hz: float) -> None:
    actions = _extract_action_chunk(response)
    period = 1.0 / control_hz
    deadline = time.monotonic()
    for action in actions:
        environment.apply_action(action)
        deadline += period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            deadline = time.monotonic()


def _move(
    environment: XTrainerRealEnvironment,
    start: np.ndarray,
    target: np.ndarray,
    move_seconds: float,
    hold_seconds: float,
    control_hz: float,
) -> np.ndarray:
    _execute_response(environment, _policy_response(start, target, move_seconds, control_hz), control_hz)
    _execute_response(environment, _hold_response(target, hold_seconds, control_hz), control_hz)
    return target.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move each X-Trainer joint by +/-5 degrees sequentially, then test each gripper"
    )
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
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--move-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-close", type=float, default=0.0)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the workspace is clear and every joint may move by five degrees",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit(
            "Refusing to move hardware without confirmation. Clear the workspace, "
            "stand by the emergency stop, then pass --yes."
        )
    if args.camera_fps <= 0 or args.control_hz <= 0 or args.move_seconds <= 0 or args.hold_seconds <= 0:
        raise ValueError("FPS, control rate, move time, and hold time must be positive")
    if not 0.0 <= args.gripper_open <= 1.0 or not 0.0 <= args.gripper_close <= 1.0:
        raise ValueError("Gripper positions must be within [0, 1]")
    if args.gripper_open == args.gripper_close:
        raise ValueError("Open and close gripper positions must differ")


def _preflight_cameras(serials: tuple[str, str, str]) -> None:
    """Fail before enabling any robot when a required USB camera is unavailable."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required on the control computer") from exc

    connected = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        connected[serial] = name

    missing = [serial for serial in serials if serial not in connected]
    if missing:
        discovered = ", ".join(f"{name} ({serial})" for serial, name in connected.items()) or "none"
        raise RuntimeError(
            "RealSense preflight failed before enabling robots. "
            f"Missing serials: {', '.join(missing)}. USB cameras visible on this computer: {discovered}"
        )
    logging.info("RealSense preflight passed: %s", ", ".join(serials))


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _preflight_cameras(
        (
            args.camera_top_serial,
            args.camera_left_wrist_serial,
            args.camera_right_wrist_serial,
        )
    )
    logging.warning(
        "The existing hardware interface initializes each gripper during connection before the explicit test sequence"
    )
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
        task="sequential basic control test",
    )

    baseline = None
    try:
        environment.reset()
        observation = environment.get_observation()
        baseline = np.asarray(observation["observation.state"], dtype=np.float64)
        if baseline.shape != (ACTION_DIM,) or not np.all(np.isfinite(baseline)):
            raise ValueError(f"Expected finite observation.state shape ({ACTION_DIM},), got {baseline.shape}")

        current = baseline.copy()
        logging.info("Baseline state: %s", np.array2string(baseline, precision=4))

        for side, joint_number, action_index in JOINTS:
            for direction, signed_delta in (("positive", JOINT_DELTA_RAD), ("negative", -JOINT_DELTA_RAD)):
                target = baseline.copy()
                target[action_index] += signed_delta
                logging.info("Testing %s joint %d: %s 5 degrees", side, joint_number, direction)
                current = _move(
                    environment,
                    current,
                    target,
                    args.move_seconds,
                    args.hold_seconds,
                    args.control_hz,
                )
                current = _move(
                    environment,
                    current,
                    baseline,
                    args.move_seconds,
                    args.hold_seconds,
                    args.control_hz,
                )

        for side, action_index in GRIPPERS:
            for state_name, value in (("open", args.gripper_open), ("close", args.gripper_close)):
                target = baseline.copy()
                target[action_index] = value
                logging.info("Testing %s gripper: %s (%.3f)", side, state_name, value)
                current = _move(
                    environment,
                    current,
                    target,
                    args.move_seconds,
                    args.hold_seconds,
                    args.control_hz,
                )
            current = _move(
                environment,
                current,
                baseline,
                args.move_seconds,
                args.hold_seconds,
                args.control_hz,
            )

        logging.info("Sequential X-Trainer basic control test passed")
    except KeyboardInterrupt:
        logging.warning("Interrupted; attempting to restore the baseline pose")
    finally:
        if baseline is not None:
            with contextlib.suppress(Exception):
                environment.apply_action(baseline)
                time.sleep(args.hold_seconds)
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
