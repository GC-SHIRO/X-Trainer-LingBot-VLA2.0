"""Opt-in, lossless protocol logging for policy inference."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .msgpack_numpy import Packer


class InferenceRecorder:
    """Persist inference payloads under ``./log`` without affecting control flow.

    The msgpack files preserve every protocol field. PNG copies make the image
    inputs directly inspectable without requiring a msgpack decoder.
    """

    def __init__(self, component: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.root = Path.cwd() / "log" / f"{timestamp}-{component}"
        self._payload_dir = self.root / "payloads"
        self._image_dir = self.root / "images"
        self._action_dir = self.root / "applied_actions"
        self._lock = threading.Lock()
        self._sequence = 0
        self._enabled = True
        self._packer = Packer()
        try:
            self._payload_dir.mkdir(parents=True, exist_ok=False)
            self._image_dir.mkdir()
            self._action_dir.mkdir()
            self._events = (self.root / "events.jsonl").open("a", encoding="utf-8")
            self._write_json(
                {
                    "schema_version": 1,
                    "component": component,
                    "created_at": datetime.now().astimezone().isoformat(),
                }
            )
            logging.info("Inference logging enabled: %s", self.root)
        except Exception as exc:
            self._enabled = False
            self._events = None
            logging.warning("Unable to enable inference logging: %s", exc)

    def record_inference(self, request: Mapping[str, Any], response: Mapping[str, Any]) -> None:
        """Record one protocol-level model request and response."""
        with self._lock:
            if not self._enabled:
                return
            try:
                sequence = self._next_sequence()
                request_path, request_images = self._write_payload(sequence, "request", request)
                response_path, response_images = self._write_payload(sequence, "response", response)
                self._write_json(
                    {
                        "event": "inference",
                        "sequence": sequence,
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "request": request_path,
                        "response": response_path,
                        "request_images": request_images,
                        "response_images": response_images,
                    }
                )
            except Exception as exc:
                self._disable(exc)

    def record_applied_action(self, action: np.ndarray, *, step: int, source: str) -> None:
        """Record the final action sent to the real-robot environment."""
        with self._lock:
            if not self._enabled:
                return
            try:
                sequence = self._next_sequence()
                path = self._action_dir / f"{sequence:06d}.npy"
                np.save(path, np.asarray(action))
                self._write_json(
                    {
                        "event": "applied_action",
                        "sequence": sequence,
                        "step": step,
                        "source": source,
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "action": str(path.relative_to(self.root)),
                    }
                )
            except Exception as exc:
                self._disable(exc)

    def close(self) -> None:
        with self._lock:
            if self._events is not None:
                self._events.close()
                self._events = None

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _write_payload(
        self,
        sequence: int,
        direction: str,
        payload: Mapping[str, Any],
    ) -> tuple[str, list[str]]:
        payload_path = self._payload_dir / f"{sequence:06d}-{direction}.msgpack"
        payload_path.write_bytes(self._packer.pack(dict(payload)))
        images = self._write_images(sequence, direction, payload)
        return str(payload_path.relative_to(self.root)), images

    def _write_images(self, sequence: int, direction: str, payload: Mapping[str, Any]) -> list[str]:
        paths: list[str] = []
        for key, value in payload.items():
            if not key.startswith("observation.images."):
                continue
            image = np.asarray(value)
            if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
                continue
            filename = f"{sequence:06d}-{direction}-{_safe_filename(key)}.png"
            image_path = self._image_dir / filename
            Image.fromarray(image).save(image_path, format="PNG")
            paths.append(str(image_path.relative_to(self.root)))
        return paths

    def _write_json(self, event: dict[str, Any]) -> None:
        if self._events is None:
            return
        self._events.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self._events.flush()

    def _disable(self, exc: Exception) -> None:
        logging.warning("Inference logging disabled after write failure: %s", exc)
        self._enabled = False
        if self._events is not None:
            self._events.close()
            self._events = None


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)
