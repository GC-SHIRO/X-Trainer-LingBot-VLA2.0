"""Serve a model-free X-Trainer policy that holds the measured position."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.image_codec import JPEG_RGB_ENCODING
from deploy.websocket_policy_server import WebsocketPolicyServer


class HoldCurrentPolicy:
    """Return an action chunk equal to the state supplied by the client."""

    def __init__(self, horizon: int = 50) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self._horizon = horizon

    def infer(self, observation: dict) -> dict:
        if observation.get("reset"):
            return {"action": None}
        if "observation.state" not in observation:
            raise KeyError("Missing 'observation.state' in client observation")

        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected observation.state shape (14,), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("observation.state contains NaN or Inf")

        return {"action": np.repeat(state[None, :], self._horizon, axis=0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a model-free hold-current policy for X-Trainer integration tests"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen address; use 0.0.0.0 for a remote client")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=50, help="Number of repeated hold actions per response")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    policy = HoldCurrentPolicy(args.horizon)
    metadata = {
        "model_type": "lingbot-vla-2.0",
        "robot": "xtrainer",
        "mock_policy": True,
        "action_mode": "hold-current",
        # Advertised so the model-free hardware test exercises the same JPEG
        # transport the real server uses.
        "image_encodings": ["raw_ndarray", JPEG_RGB_ENCODING],
    }
    logging.warning("Starting model-free policy: actions will hold the client-reported position")
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
