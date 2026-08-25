import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def _load_image_transforms_module():
    path = Path(__file__).resolve().parents[1] / "deploy" / "xtrainer_real" / "image_transforms.py"
    spec = importlib.util.spec_from_file_location("xtrainer_image_transforms", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


image_transforms = _load_image_transforms_module()


class XTrainerImageTransformsTest(unittest.TestCase):
    def test_top_camera_crops_resizes_and_rotates(self) -> None:
        image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)

        transformed = image_transforms.crop_and_rotate_top_image(image)

        expected = np.asarray(Image.fromarray(image[2:8, 4:16]).resize((20, 10), resample=Image.BILINEAR))[::-1, ::-1]
        np.testing.assert_array_equal(transformed, expected)
        self.assertEqual(transformed.shape, image.shape)
        self.assertTrue(transformed.flags.c_contiguous)

    def test_wrist_cameras_keep_the_original_orientation(self) -> None:
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

        for camera_name in ("left_wrist", "right_wrist"):
            with self.subTest(camera_name=camera_name):
                np.testing.assert_array_equal(image_transforms.prepare_xtrainer_camera_image(camera_name, image), image)

    def test_unknown_camera_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            image_transforms.prepare_xtrainer_camera_image("unknown", np.zeros((2, 2, 3), dtype=np.uint8))
