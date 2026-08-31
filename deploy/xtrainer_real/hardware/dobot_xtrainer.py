import logging
import os
import socket
import threading
import time

import numpy as np

from ..scservo_sdk import COMM_SUCCESS
from ..scservo_sdk import PortHandler
from ..scservo_sdk import protocol_packet_handler
from ..scservo_sdk import sms_sts
from .realsense_camera import RealSenseCamera

logger = logging.getLogger(__name__)

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


class DobotConnection:
    """TCP connection wrapper for Dobot controller."""

    def __init__(self, ip: str, port: int, timeout: float = 3.0):
        self.ip = ip
        self.port = int(port)
        self.timeout = float(timeout)
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self._socket = socket.socket()
        self._socket.settimeout(self.timeout)
        self._socket.connect((self.ip, self.port))

    def send_recv(self, cmd: str) -> str:
        if self._socket is None:
            raise RuntimeError("Dobot socket is not connected")
        with self._lock:
            self._socket.send(cmd.encode("utf-8"))
            try:
                data = self._socket.recv(1024)
                return data.decode("utf-8") if data else ""
            except Exception as exc:
                logger.warning("Dobot recv error: %s", exc)
                return ""

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None


class DobotGripperWrapper:
    """Feetech SMS/STS servo wrapper."""

    def __init__(self, port: str, servo_id: int, servo_pos: tuple[int, int], torque_limit: int = 300):
        self.port = port
        self.servo_id = int(servo_id)
        self.servo_pos_min = int(servo_pos[0])
        self.servo_pos_max = int(servo_pos[1])
        self.torque_limit = int(torque_limit)
        self._port_handler: PortHandler | None = None
        self._servo: sms_sts | None = None
        self._pack_handler = None
        self._io_lock = threading.Lock()

    def connect(self) -> None:
        self._port_handler = PortHandler(self.port)
        self._pack_handler = protocol_packet_handler(portHandler=self._port_handler, protocol_end=0)
        self._servo = sms_sts(self._port_handler)

        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open gripper port {self.port}")
        self._port_handler.setBaudRate(1000000)

        self._set_latency_timer()

        _, result, error = self._servo.ping(self.servo_id)
        if error != 0:
            raise ConnectionError(
                f"Failed to ping gripper servo {self.servo_id} on {self.port}, result={result}, error={error}"
            )

        comm_result, _ = self._pack_handler.write2ByteTxRx(self.servo_id, 48, self.torque_limit)
        if comm_result != COMM_SUCCESS:
            logger.warning("Failed to set gripper torque limit: %s", comm_result)

    def _set_latency_timer(self) -> None:
        port_name = self.port.split("/")[-1]
        latency_path = f"/sys/bus/usb-serial/devices/{port_name}/latency_timer"
        try:
            with open(latency_path, "w") as file_obj:
                file_obj.write("1")
        except PermissionError:
            try:
                os.system(f"echo 1 | sudo tee {latency_path} > /dev/null")
            except Exception:
                logger.debug("Could not set latency timer for %s", port_name)
        except FileNotFoundError:
            logger.debug("Latency timer path not found: %s", latency_path)

    def get_position(self, retries: int = 2) -> float | None:
        if self._servo is None:
            raise RuntimeError("Gripper servo is not connected")
        last_result = 0
        last_error = 0
        for _ in range(max(1, retries)):
            with self._io_lock:
                position, result, error = self._servo.ReadPos(self.servo_id)
            if result == 0:
                normalized = (position - self.servo_pos_min) / (self.servo_pos_max - self.servo_pos_min)
                return float(np.clip(normalized, 0.0, 1.0))
            last_result = result
            last_error = error
            time.sleep(0.001)

        logger.debug("Failed to read gripper position: result=%s, error=%s", last_result, last_error)
        return None

    def set_position(self, pos: float, speed: int = 100) -> None:
        if self._servo is None:
            raise RuntimeError("Gripper servo is not connected")
        pos = float(np.clip(pos, 0.0, 1.0))
        servo_pos = int(pos * (self.servo_pos_max - self.servo_pos_min) + self.servo_pos_min)
        servo_speed = int(speed / 100.0 * 4096)
        with self._io_lock:
            self._servo.WritePosEx(self.servo_id, servo_pos, servo_speed, 0)

    def disconnect(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None


class DobotXTrainer:
    """Minimal xtrainer follower arm for inference runtime."""

    def __init__(
        self,
        *,
        robot_ip: str,
        gripper_port: str,
        gripper_id: int,
        gripper_servo_pos: tuple[int, int],
        use_gripper: bool = True,
        read_gripper_position: bool = False,
        tool_offset: tuple[float, float, float, float, float, float] = (0, 0, 197, 0, 0, 0),
        speed_factor: int = 20,
        acc_j: int = 20,
        speed_j: int = 20,
        servo_j_time: float = 0.03,
        disable_torque_on_disconnect: bool = True,
        max_delta_per_step: float = float("inf"),
        camera_serials: dict[str, str] | None = None,
        camera_fps: float = 30.0,
    ):
        self.robot_ip = robot_ip
        self.read_gripper_position = read_gripper_position
        self.tool_offset = tool_offset
        self.speed_factor = speed_factor
        self.acc_j = acc_j
        self.speed_j = speed_j
        self.servo_j_time = servo_j_time
        self.disable_torque_on_disconnect = disable_torque_on_disconnect
        self.max_delta_per_step = max_delta_per_step

        self._dashboard: DobotConnection | None = None
        self._move: DobotConnection | None = None
        self._connected = False
        self._last_action_rad: np.ndarray | None = None
        self._last_gripper_pos = 1.0

        self._gripper = None
        if use_gripper:
            self._gripper = DobotGripperWrapper(
                port=gripper_port,
                servo_id=gripper_id,
                servo_pos=gripper_servo_pos,
            )

        camera_serials = camera_serials or {}
        self.cameras: dict[str, RealSenseCamera] = {
            name: RealSenseCamera(serial, fps=round(camera_fps)) for name, serial in camera_serials.items() if serial
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and all(camera.is_connected for camera in self.cameras.values())

    def connect(self) -> None:
        if self._connected:
            return

        self._dashboard = DobotConnection(self.robot_ip, 29999)
        self._dashboard.connect()

        self._move = DobotConnection(self.robot_ip, 30003, timeout=1.0)
        self._move.connect()

        self._dashboard.send_recv("EnableRobot()")
        time.sleep(0.5)
        self._dashboard.send_recv(f"SpeedFactor({self.speed_factor})")
        self._dashboard.send_recv(f"AccJ({self.acc_j})")
        self._dashboard.send_recv(f"SpeedJ({self.speed_j})")

        tx, ty, tz, trx, try_, trz = self.tool_offset
        self._dashboard.send_recv(f"SetTool(1,{{{tx:.3f},{ty:.3f},{tz:.3f},{trx:.3f},{try_:.3f},{trz:.3f}}})")
        self._dashboard.send_recv("Tool(1)")
        self._dashboard.send_recv("StopDrag()")

        if self._gripper is not None:
            self._gripper.connect()
            current_gripper = self._gripper.get_position(retries=3)
            if current_gripper is not None:
                self._last_gripper_pos = current_gripper
            self._gripper.set_position(0.0)
            time.sleep(0.3)
            self._gripper.set_position(1.0)

        for camera in self.cameras.values():
            camera.connect()

        self._connected = True

    def set_do_status(self, channel: int, status: int) -> None:
        if self._dashboard is None:
            raise RuntimeError("Dashboard connection is not available")
        self._dashboard.send_recv(f"DO({int(channel)},{int(status)})")

    def _read_joint_positions(self) -> list[float]:
        if self._dashboard is None:
            raise RuntimeError("Dashboard connection is not available")
        angle_str = self._dashboard.send_recv("GetAngle()")
        try:
            angles_deg = list(map(float, angle_str.split("{")[1].split("}")[0].split(",")))
            return [float(np.deg2rad(angle)) for angle in angles_deg]
        except (IndexError, ValueError) as exc:
            logger.error("Failed to parse GetAngle response: %s, error: %s", angle_str, exc)
            return [0.0] * 6

    def get_low_latency_observation(self) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        observation: dict[str, float] = {}
        angles_rad = self._read_joint_positions()
        for index, name in enumerate(JOINT_NAMES):
            observation[f"{name}.pos"] = float(angles_rad[index])
        if self._gripper is not None:
            observation["gripper.pos"] = float(self._last_gripper_pos)
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("Robot not connected")
        if self._move is None:
            raise RuntimeError("Move connection is not available")

        goal_rad = np.array([action.get(f"{name}.pos", 0.0) for name in JOINT_NAMES], dtype=np.float64)

        max_delta = self.max_delta_per_step
        if max_delta is not None and max_delta > 0 and self._last_action_rad is not None:
            delta = goal_rad - self._last_action_rad
            abs_delta = np.abs(delta)
            if np.any(abs_delta > max_delta):
                for joint_index in range(6):
                    if abs_delta[joint_index] <= max_delta:
                        continue
                    pi_ratio = delta[joint_index] / np.pi
                    if 1.85 < abs(pi_ratio) < 2.15:
                        goal_rad[joint_index] = goal_rad[joint_index] - 2 * np.pi * np.sign(pi_ratio)
                    else:
                        raise RuntimeError(
                            f"Action jump too large on joint {joint_index}: delta={abs_delta[joint_index]:.3f}rad"
                        )
        self._last_action_rad = goal_rad.copy()

        goal_deg = [float(np.rad2deg(value)) for value in goal_rad]
        command = (
            f"ServoJ({goal_deg[0]:f},{goal_deg[1]:f},{goal_deg[2]:f},"
            f"{goal_deg[3]:f},{goal_deg[4]:f},{goal_deg[5]:f},{self.servo_j_time:f},gain=500)"
        )
        self._move.send_recv(command)

        if self._gripper is not None and "gripper.pos" in action:
            self._last_gripper_pos = float(np.clip(action["gripper.pos"], 0.0, 1.0))
            self._gripper.set_position(self._last_gripper_pos)

        sent = {f"{name}.pos": float(goal_rad[index]) for index, name in enumerate(JOINT_NAMES)}
        if self._gripper is not None:
            sent["gripper.pos"] = float(self._last_gripper_pos)
        return sent

    def disconnect(self) -> None:
        if not self._connected:
            return

        for camera in self.cameras.values():
            camera.disconnect()

        if self._gripper is not None:
            self._gripper.disconnect()

        if self._dashboard is not None:
            if self.disable_torque_on_disconnect:
                self._dashboard.send_recv("DisableRobot()")
            self._dashboard.close()
            self._dashboard = None

        if self._move is not None:
            self._move.close()
            self._move = None

        self._connected = False
