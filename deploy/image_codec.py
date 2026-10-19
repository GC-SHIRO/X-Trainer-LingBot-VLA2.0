"""JPEG wire encoding for X-trainer RGB camera observations.

The X-trainer client ships three 640x480 RGB frames per inference request. Sent
as raw msgpack ndarrays that is roughly 2.7 MB per request, which is a
meaningful share of the round-trip latency at a 30 Hz control rate. Encoding
the frames as JPEG shrinks that by roughly an order of magnitude for real camera
frames, and never below ~3.5x even for uncorrelated noise (JPEG's worst case).
The server decodes them back to the original resolution before the model sees
them, so the model input is unchanged.

The encoding is negotiated through the server metadata so that a client can talk
to a server that does not implement it: `raw_ndarray` is always acceptable.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

JPEG_RGB_ENCODING = "jpeg_rgb"
JPEG_IMAGE_MARKER = "__lingbot_jpeg_rgb__"
IMAGE_KEY_PREFIX = "observation.images."


def encode_policy_images(payload: dict[str, Any], quality: int) -> dict[str, Any]:
    """Return a payload copy whose camera images are JPEG byte strings."""
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")

    encoded_payload = payload.copy()
    changed = False
    for key, value in payload.items():
        if not key.startswith(IMAGE_KEY_PREFIX) or not isinstance(value, np.ndarray):
            continue
        encoded_payload[key] = _encode_rgb_image(value, quality)
        changed = True
    return encoded_payload if changed else payload


def decode_policy_images(payload: dict[str, Any]) -> dict[str, Any]:
    """Decode JPEG-marked camera values while leaving raw ndarray payloads unchanged."""
    decoded_payload = payload.copy()
    changed = False
    for key, value in payload.items():
        if not key.startswith(IMAGE_KEY_PREFIX):
            continue
        if isinstance(value, dict) and value.get(JPEG_IMAGE_MARKER) is True:
            decoded_payload[key] = _decode_rgb_image(value)
            changed = True
    return decoded_payload if changed else payload


def _encode_rgb_image(image: Any, quality: int) -> dict[str, Any]:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"JPEG input must be uint8 HxWx3 RGB, got dtype={array.dtype}, shape={array.shape}")
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode RGB image as JPEG")
    return {
        JPEG_IMAGE_MARKER: True,
        "data": encoded.tobytes(),
    }


def _decode_rgb_image(payload: dict[str, Any]) -> np.ndarray:
    data = payload.get("data")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("JPEG image payload must contain non-empty bytes")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode JPEG image payload")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
