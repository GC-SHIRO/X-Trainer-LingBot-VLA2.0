import contextlib
import logging
import time
from typing import Optional

import numpy as np

from .hardware import DobotXTrainer
from .image_transforms import prepare_xtrainer_camera_image

logger = logging.getLogger(__name__)


def _obs_dict_to_arm_array(obs: dict[str, float]) -> np.ndarray:
    arm = np.zeros(7, dtype=np.float64)
    for joint_index in range(6):
        arm[joint_index] = float(obs[f"joint{joint_index + 1}.pos"])
    arm[6] = float(obs.get("gripper.pos", 1.0))
    return arm


class XTrainerRealEnvironment:
    """Minimal real-robot environment for the LingBot policy client."""

    def __init__(
        self,
        *,
        left_robot_ip: str = "192.168.5.1",
        right_robot_ip: str = "192.168.5.2",
        left_gripper_port: str = "/dev/ttyUSB1",
        right_gripper_port: str = "/dev/ttyUSB0",
        left_gripper_id: int = 21,
        right_gripper_id: int = 22,
        left_gripper_servo_pos: tuple[int, int] = (2048, 3052),
        right_gripper_servo_pos: tuple[int, int] = (2048, 3052),
        camera_top_serial: str,
        camera_left_wrist_serial: str,
        camera_right_wrist_serial: str,
        camera_fps: float = 30.0,
        task: str,
        reset_pose: Optional[list[float]] = None,
        max_joint_delta: float = float("inf"),
        ramp_step: float = 0.01,
        ramp_max_steps: int = 100,
        gripper_update_threshold: float = 0.0,
        servo_step_limit: float = float("inf"),
    ) -> None:
        self._follower_left = DobotXTrainer(
            robot_ip=left_robot_ip,
            gripper_port=left_gripper_port,
            gripper_id=left_gripper_id,
            gripper_servo_pos=left_gripper_servo_pos,
            read_gripper_position=False,
            max_delta_per_step=servo_step_limit,
            camera_serials={
                "cam_top": camera_top_serial,
                "cam_left_wrist": camera_left_wrist_serial,
            },
            camera_fps=camera_fps,
        )
        self._follower_right = DobotXTrainer(
            robot_ip=right_robot_ip,
            gripper_port=right_gripper_port,
            gripper_id=right_gripper_id,
            gripper_servo_pos=right_gripper_servo_pos,
            read_gripper_position=False,
            max_delta_per_step=servo_step_limit,
            camera_serials={"cam_right_wrist": camera_right_wrist_serial},
            camera_fps=camera_fps,
        )

        self._task = task
        self._max_joint_delta = max_joint_delta
        self._ramp_step = ramp_step
        self._ramp_max_steps = max(ramp_max_steps, 1)
        self._gripper_update_threshold = max(gripper_update_threshold, 0.0)
        self._connected = False
        self._last_action: np.ndarray | None = None
        self._last_gripper_sent = np.array([1.0, 1.0], dtype=np.float64)

        self._reset_pose = None
        if reset_pose is not None:
            reset = np.asarray(reset_pose, dtype=np.float64).reshape(-1)
            if reset.shape[0] != 14:
                raise ValueError(f"Expected reset_pose length 14, got {reset.shape[0]}")
            if not np.all(np.isfinite(reset)):
                raise ValueError("reset_pose contains non-finite values")
            self._reset_pose = reset

    def reset(self) -> None:
        self._ensure_connected()
        if self._reset_pose is not None:
            self._move_smooth(self._get_bimanual_qpos(), self._reset_pose)
            time.sleep(0.2)
        self._last_action = self._get_bimanual_qpos()
        self._last_gripper_sent = self._last_action[[6, 13]].copy()

    def get_observation(self) -> dict:
        self._ensure_connected()
        observation = {
            "observation.state": self._get_bimanual_qpos().astype(np.float32),
            "task": self._task,
        }
        for camera_name in ("top", "left_wrist", "right_wrist"):
            frame = self._read_camera_frame(camera_name)
            frame = prepare_xtrainer_camera_image(camera_name, frame)
            observation[f"observation.images.{camera_name}"] = frame
        return observation

    def apply_action(self, action: np.ndarray) -> None:
        self._ensure_connected()
        target = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        if target.shape[0] != 14:
            raise ValueError(f"Expected action length 14, got {target.shape[0]}")
        if not np.all(np.isfinite(target)):
            raise ValueError("Action contains non-finite values")

        target[6] = float(np.clip(target[6], 0.0, 1.0))
        target[13] = float(np.clip(target[13], 0.0, 1.0))
        if self._last_action is None:
            self._last_action = self._get_bimanual_qpos()

        max_joint_delta = max(
            float(np.max(np.abs(target[:6] - self._last_action[:6]))),
            float(np.max(np.abs(target[7:13] - self._last_action[7:13]))),
        )
        if max_joint_delta > self._max_joint_delta:
            self._move_smooth(self._last_action, target)
        else:
            self._send_bimanual_action(target)
        self._last_action = target

    def close(self) -> None:
        if not self._connected:
            return
        for follower in (self._follower_left, self._follower_right):
            try:
                follower.disconnect()
            except Exception:
                logger.exception("Failed to disconnect follower cleanly")
        self._connected = False

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        self._follower_left.connect()
        try:
            self._follower_right.connect()
        except Exception:
            self._follower_left.disconnect()
            raise
        self._connected = True
        self._last_action = self._get_bimanual_qpos()
        self._last_gripper_sent = self._last_action[[6, 13]].copy()

    def _read_camera_frame(self, camera_name: str, retries: int = 5) -> np.ndarray:
        camera = self._get_camera(camera_name)
        for _ in range(max(retries, 1)):
            try:
                frame = camera.async_read(timeout_ms=50)
            except Exception:
                frame = None
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                return frame
            time.sleep(0.005)
        raise RuntimeError(f"Failed to read camera frame: {camera_name}")

    def _get_camera(self, camera_name: str):
        camera_locations = {
            "top": (self._follower_left, "cam_top"),
            "left_wrist": (self._follower_left, "cam_left_wrist"),
            "right_wrist": (self._follower_right, "cam_right_wrist"),
        }
        follower, key = camera_locations[camera_name]
        if key not in follower.cameras:
            raise RuntimeError(f"Missing camera {key}")
        return follower.cameras[key]

    def _get_bimanual_qpos(self) -> np.ndarray:
        left = _obs_dict_to_arm_array(self._follower_left.get_low_latency_observation())
        right = _obs_dict_to_arm_array(self._follower_right.get_low_latency_observation())
        return np.concatenate([left, right])

    def _send_bimanual_action(self, action: np.ndarray) -> None:
        left = {f"joint{index + 1}.pos": float(action[index]) for index in range(6)}
        left["gripper.pos"] = float(action[6])
        right = {f"joint{index + 1}.pos": float(action[7 + index]) for index in range(6)}
        right["gripper.pos"] = float(action[13])

        for index, arm_action in enumerate((left, right)):
            gripper = float(arm_action["gripper.pos"])
            if abs(gripper - float(self._last_gripper_sent[index])) < self._gripper_update_threshold:
                arm_action.pop("gripper.pos")
            else:
                self._last_gripper_sent[index] = gripper

        self._follower_left.send_action(left)
        self._follower_right.send_action(right)

    def _move_smooth(self, start_action: np.ndarray, goal_action: np.ndarray) -> None:
        max_delta = float(np.max(np.abs(goal_action - start_action)))
        steps = min(int(np.ceil(max_delta / max(self._ramp_step, 1e-6))), self._ramp_max_steps)
        if steps <= 1:
            self._send_bimanual_action(goal_action)
            return
        for action in np.linspace(start_action, goal_action, steps):
            self._send_bimanual_action(action)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
