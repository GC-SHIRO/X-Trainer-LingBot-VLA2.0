"""Camera transforms shared by X-trainer LingBot inference and tests."""

from __future__ import annotations

import numpy as np
from PIL import Image


def crop_and_rotate_top_image(image: np.ndarray) -> np.ndarray:
    """Crop 20% from every side, restore the original size, then rotate 180 degrees."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"top camera image must have shape HxWx3, got {image.shape}")

    height, width, _ = image.shape
    top_px = int(0.2 * height)
    bottom_px = int(0.2 * height)
    left_px = int(0.2 * width)
    right_px = int(0.2 * width)

    top_px = max(0, min(top_px, height - 1))
    bottom_px = max(0, min(bottom_px, height - 1 - top_px))
    left_px = max(0, min(left_px, width - 1))
    right_px = max(0, min(right_px, width - 1 - left_px))

    cropped = image[top_px : height - bottom_px, left_px : width - right_px]
    resized = np.asarray(Image.fromarray(cropped).resize((width, height), resample=Image.BILINEAR))
    return np.ascontiguousarray(resized[::-1, ::-1])


def prepare_xtrainer_camera_image(camera_name: str, image: np.ndarray) -> np.ndarray:
    """Return the camera image in the orientation used by LingBot inference."""
    if camera_name == "top":
        return crop_and_rotate_top_image(image)
    if camera_name in {"left_wrist", "right_wrist"}:
        return np.ascontiguousarray(image)
    raise KeyError(f"Unsupported X-trainer camera: {camera_name}")
