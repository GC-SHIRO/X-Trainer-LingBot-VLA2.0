import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_client_module():
    websocket_module = types.ModuleType("deploy.websocket_client_policy")
    websocket_module.WebsocketClientPolicy = object
    environment_module = types.ModuleType("deploy.xtrainer_real")
    environment_module.XTrainerRealEnvironment = object
    sys.modules[websocket_module.__name__] = websocket_module
    sys.modules[environment_module.__name__] = environment_module

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_xtrainer_real.py"
    spec = importlib.util.spec_from_file_location("run_xtrainer_real", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_default_args(client):
    import unittest.mock

    with unittest.mock.patch.object(
        sys,
        "argv",
        [
            "run_xtrainer_real.py",
            "--host",
            "127.0.0.1",
            "--camera-top-serial",
            "top",
            "--camera-left-wrist-serial",
            "left",
            "--camera-right-wrist-serial",
            "right",
        ],
    ):
        return client.parse_args()


class ActionChunkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _load_client_module()

    def test_truncates_chunk_to_horizon(self) -> None:
        actions = np.zeros((50, 14), dtype=np.float32)
        self.assertEqual(self.client._extract_action_chunk({"action": actions}, 25).shape, (25, 14))

    def test_promotes_single_action(self) -> None:
        action = np.zeros(14, dtype=np.float32)
        self.assertEqual(self.client._extract_action_chunk({"action": action}, 25).shape, (1, 14))

    def test_rejects_unsafe_responses(self) -> None:
        invalid_actions = (
            np.zeros((2, 13)),
            np.empty((0, 14)),
            np.full((1, 14), np.nan),
        )
        for actions in invalid_actions:
            with self.subTest(shape=actions.shape):
                with self.assertRaises(ValueError):
                    self.client._extract_action_chunk({"action": actions}, 25)

        with self.assertRaises(KeyError):
            self.client._extract_action_chunk({}, 25)

    def test_rate_limits_action_delta(self) -> None:
        last_action = np.zeros(14)
        action = np.ones(14)

        limited = self.client._rate_limit_action(action, last_action, 0.25)

        np.testing.assert_allclose(limited, np.full(14, 0.25))


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _load_client_module()

    def test_cli_defaults_execute_full_chunk_without_execution_limits(self) -> None:
        args = _parse_default_args(self.client)

        self.assertEqual(args.action_horizon, 50)
        self.assertEqual(args.control_hz, 30.0)
        self.assertEqual(args.chunk_blend_steps, 6)
        self.assertEqual(args.image_jpeg_quality, 85)
        self.assertFalse(args.log)
        self.assertTrue(math.isinf(args.max_joint_delta))
        self.assertEqual(args.gripper_update_threshold, 0.0)
        self.assertTrue(math.isinf(args.servo_step_limit))

    def test_no_prefetch_options_remain(self) -> None:
        args = _parse_default_args(self.client)

        stale = [name for name in vars(args) if "prefetch" in name or "switch" in name]
        self.assertEqual(stale, [])
        self.assertFalse(hasattr(self.client, "_align_prefetched_chunk"))
        self.assertFalse(hasattr(self.client, "_apply_prefetched_chunk"))
        self.assertFalse(hasattr(self.client, "_infer_prefetch_chunk"))


class ChunkBlendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _load_client_module()

    def test_blend_offset_is_last_sent_minus_first_action(self) -> None:
        offset = self.client._chunk_blend_offset(np.full(14, 0.5), np.full(14, 0.1))

        np.testing.assert_allclose(offset, np.full(14, 0.4))

    def test_blend_offset_is_none_on_first_chunk(self) -> None:
        self.assertIsNone(self.client._chunk_blend_offset(None, np.zeros(14)))

    def test_blend_starts_near_the_previous_action_and_lands_on_target(self) -> None:
        action = np.zeros(14)
        offset = np.ones(14)

        first = self.client._blend_chunk_action(action, offset, 0, 6)
        last = self.client._blend_chunk_action(action, offset, 5, 6)

        # smoothstep(1/6) == 0.074074..., so the first step keeps ~92.6% of the splice offset
        np.testing.assert_allclose(first[:6], np.full(6, 1.0 - 0.07407407407407407), rtol=1e-9)
        np.testing.assert_allclose(first[7:13], np.full(6, 1.0 - 0.07407407407407407), rtol=1e-9)
        # the last blended step is exactly the model target
        np.testing.assert_allclose(last, action)

    def test_blend_stops_after_blend_steps(self) -> None:
        action = np.zeros(14)
        offset = np.ones(14)

        beyond = self.client._blend_chunk_action(action, offset, 6, 6)

        np.testing.assert_allclose(beyond, action)

    def test_blend_is_monotonic_toward_the_target(self) -> None:
        action = np.zeros(14)
        offset = np.ones(14)

        residuals = [self.client._blend_chunk_action(action, offset, index, 6)[0] for index in range(6)]

        self.assertEqual(residuals, sorted(residuals, reverse=True))

    def test_blend_never_touches_grippers(self) -> None:
        action = np.zeros(14)
        action[6] = action[13] = 0.25
        offset = np.ones(14)

        blended = self.client._blend_chunk_action(action, offset, 0, 6)

        self.assertEqual(blended[6], 0.25)
        self.assertEqual(blended[13], 0.25)

    def test_blend_disabled_when_steps_below_two(self) -> None:
        action = np.arange(14, dtype=np.float64)
        offset = np.ones(14)

        for blend_steps in (0, 1):
            with self.subTest(blend_steps=blend_steps):
                np.testing.assert_allclose(
                    self.client._blend_chunk_action(action, offset, 0, blend_steps), action
                )

    def test_blend_without_offset_returns_action(self) -> None:
        action = np.arange(14, dtype=np.float64)

        np.testing.assert_allclose(self.client._blend_chunk_action(action, None, 0, 6), action)


class HoldActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _load_client_module()

    def test_returns_the_measured_pose_as_float64(self) -> None:
        state = np.arange(14, dtype=np.float32)

        hold = self.client._hold_action_from_observation({"observation.state": state})

        self.assertEqual(hold.dtype, np.float64)
        np.testing.assert_allclose(hold, np.arange(14))

    def test_returns_a_copy_so_the_observation_is_not_aliased(self) -> None:
        # float64 so that the dtype conversion cannot be what makes the copy.
        state = np.zeros(14, dtype=np.float64)
        observation = {"observation.state": state}

        hold = self.client._hold_action_from_observation(observation)
        hold[0] = 5.0

        self.assertEqual(observation["observation.state"][0], 0.0)

    def test_rejects_wrong_length_and_non_finite_state(self) -> None:
        for state in (np.zeros(13), np.zeros(15), np.full(14, np.nan), np.full(14, np.inf)):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    self.client._hold_action_from_observation({"observation.state": state})

    def test_rejects_missing_state(self) -> None:
        with self.assertRaises(KeyError):
            self.client._hold_action_from_observation({"task": "test"})


class ServerTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _load_client_module()

    def test_logs_server_timing_without_raising(self) -> None:
        with self.assertLogs(level="INFO") as captured:
            self.client._log_server_timing({"server_timing": {"infer_ms": 12.5, "prev_total_ms": 40.0}})

        self.assertIn("infer=12.5 ms", captured.output[0])

    def test_tolerates_first_response_without_previous_total(self) -> None:
        with self.assertLogs(level="INFO"):
            self.client._log_server_timing({"server_timing": {"infer_ms": 12.5}})

    def test_ignores_responses_without_timing(self) -> None:
        self.client._log_server_timing({"action": np.zeros((2, 14))})


if __name__ == "__main__":
    unittest.main()
