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


class ActionChunkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _load_client_module()

    def test_truncates_chunk_to_horizon(self) -> None:
        actions = np.zeros((50, 14), dtype=np.float32)
        self.assertEqual(self.client._extract_action_chunk({"action": actions}, 25).shape, (25, 14))

    def test_cli_defaults_execute_full_chunk_without_prefetch_or_execution_limits(self) -> None:
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
            args = self.client.parse_args()

        self.assertEqual(args.action_horizon, 50)
        self.assertEqual(args.control_hz, 30.0)
        self.assertEqual(args.prefetch_remaining, 0)
        self.assertTrue(math.isinf(args.max_joint_delta))
        self.assertEqual(args.gripper_update_threshold, 0.0)
        self.assertTrue(math.isinf(args.servo_step_limit))

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

    def test_boundary_smoothing_keeps_small_delta(self) -> None:
        chunk = np.full((3, 14), 0.05)
        smoothed = self.client._smooth_chunk_boundary(
            chunk,
            np.zeros(14),
            max_switch_delta=0.1,
            blend_steps=2,
        )

        np.testing.assert_allclose(smoothed, chunk)

    def test_boundary_smoothing_blends_large_delta(self) -> None:
        chunk = np.ones((4, 14))
        smoothed = self.client._smooth_chunk_boundary(
            chunk,
            np.zeros(14),
            max_switch_delta=0.1,
            blend_steps=3,
        )

        np.testing.assert_allclose(smoothed[0], np.full(14, 0.25))
        np.testing.assert_allclose(smoothed[1], np.full(14, 0.5))
        np.testing.assert_allclose(smoothed[2], np.full(14, 0.75))
        np.testing.assert_allclose(smoothed[3], np.ones(14))

    def test_projected_state_uses_chunk_end_without_rate_limit(self) -> None:
        chunk = np.arange(4 * 14, dtype=np.float64).reshape(4, 14)

        projected = self.client._project_chunk_end_state(
            chunk,
            action_index=1,
            last_sent_action=np.zeros(14),
            max_delta_per_step=0.0,
        )

        np.testing.assert_allclose(projected, chunk[-1])

    def test_projected_state_respects_rate_limit(self) -> None:
        chunk = np.vstack([np.full(14, 1.0), np.full(14, 2.0), np.full(14, 3.0)])

        projected = self.client._project_chunk_end_state(
            chunk,
            action_index=0,
            last_sent_action=np.zeros(14),
            max_delta_per_step=0.5,
        )

        np.testing.assert_allclose(projected, np.full(14, 1.5))

    def test_projected_state_replaces_observation_state_copy(self) -> None:
        observation = {
            "observation.state": np.zeros(14, dtype=np.float32),
            "task": "test",
        }
        projected = np.ones(14, dtype=np.float64)

        updated = self.client._with_projected_state(observation, projected)

        self.assertIsNot(updated, observation)
        np.testing.assert_allclose(updated["observation.state"], np.ones(14, dtype=np.float32))
        np.testing.assert_allclose(observation["observation.state"], np.zeros(14, dtype=np.float32))

    def test_prefetched_chunk_replace_discards_current_remainder(self) -> None:
        current = np.zeros((5, 14), dtype=np.float64)
        prefetched = np.ones((3, 14), dtype=np.float64)

        action_chunk, action_index, next_chunk, discarded = self.client._apply_prefetched_chunk(
            current,
            action_index=2,
            next_chunk=None,
            prefetched_chunk=prefetched,
            last_sent_action=np.ones(14),
            apply_mode="replace",
            max_switch_delta=0.0,
            blend_steps=0,
        )

        np.testing.assert_allclose(action_chunk, prefetched)
        self.assertEqual(action_index, 0)
        self.assertIsNone(next_chunk)
        self.assertEqual(discarded, 3)

    def test_prefetched_chunk_boundary_keeps_current_until_exhausted(self) -> None:
        current = np.zeros((5, 14), dtype=np.float64)
        prefetched = np.ones((3, 14), dtype=np.float64)

        action_chunk, action_index, next_chunk, discarded = self.client._apply_prefetched_chunk(
            current,
            action_index=2,
            next_chunk=None,
            prefetched_chunk=prefetched,
            last_sent_action=np.ones(14),
            apply_mode="boundary",
            max_switch_delta=0.0,
            blend_steps=0,
        )

        np.testing.assert_allclose(action_chunk, current)
        self.assertEqual(action_index, 2)
        np.testing.assert_allclose(next_chunk, prefetched)
        self.assertEqual(discarded, 0)

    def test_align_prefetched_chunk_skips_elapsed_actions(self) -> None:
        prefetched = np.arange(5 * 14, dtype=np.float64).reshape(5, 14)

        aligned, skipped = self.client._align_prefetched_chunk(
            prefetched,
            request_step=10,
            current_step=12,
            mode="elapsed",
        )

        np.testing.assert_allclose(aligned, prefetched[2:])
        self.assertEqual(skipped, 2)

    def test_align_prefetched_chunk_keeps_future_or_same_step_actions(self) -> None:
        prefetched = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)

        aligned, skipped = self.client._align_prefetched_chunk(
            prefetched,
            request_step=10,
            current_step=10,
            mode="elapsed",
        )

        np.testing.assert_allclose(aligned, prefetched)
        self.assertEqual(skipped, 0)

    def test_align_prefetched_chunk_returns_empty_when_stale(self) -> None:
        prefetched = np.arange(3 * 14, dtype=np.float64).reshape(3, 14)

        aligned, skipped = self.client._align_prefetched_chunk(
            prefetched,
            request_step=10,
            current_step=15,
            mode="elapsed",
        )

        self.assertEqual(aligned.shape, (0, 14))
        self.assertEqual(skipped, 5)

    def test_align_prefetched_chunk_nearest_searches_after_elapsed_step(self) -> None:
        prefetched = np.vstack(
            [
                np.full(14, 0.0),
                np.full(14, 0.5),
                np.full(14, 1.0),
                np.full(14, 0.2),
                np.full(14, 0.9),
            ]
        )

        aligned, skipped = self.client._align_prefetched_chunk(
            prefetched,
            request_step=10,
            current_step=12,
            last_sent_action=np.full(14, 0.2),
            mode="nearest",
            search_window=2,
        )

        np.testing.assert_allclose(aligned, prefetched[3:])
        self.assertEqual(skipped, 3)


if __name__ == "__main__":
    unittest.main()
