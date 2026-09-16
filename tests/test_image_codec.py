import importlib.util
import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(CV2_AVAILABLE, "opencv-python-headless is required for the JPEG image codec")
class ImageCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from deploy.image_codec import JPEG_IMAGE_MARKER, decode_policy_images, encode_policy_images

        cls.encode = staticmethod(encode_policy_images)
        cls.decode = staticmethod(decode_policy_images)
        cls.marker = JPEG_IMAGE_MARKER

    def _payload(self) -> dict:
        rng = np.random.default_rng(0)
        return {
            "observation.state": np.zeros(14, dtype=np.float32),
            "task": "pick up the object",
            "observation.images.top": rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8),
            "observation.images.left_wrist": rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8),
            "observation.images.right_wrist": rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8),
        }

    def test_round_trip_preserves_shape_and_approximate_colors(self) -> None:
        payload = self._payload()

        decoded = self.decode(self.encode(payload, 95))

        for key in ("observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"):
            with self.subTest(key=key):
                self.assertEqual(decoded[key].shape, payload[key].shape)
                self.assertEqual(decoded[key].dtype, np.uint8)
                self.assertTrue(
                    np.allclose(decoded[key].mean(axis=(0, 1)), payload[key].mean(axis=(0, 1)), atol=3.0)
                )

    def test_encoding_shrinks_the_payload(self) -> None:
        # The fixture is uniform random noise, which is JPEG's worst case (~3.5x
        # at quality 85). Real camera frames compress far better; this only pins
        # down that encoding happens and that it never inflates the payload.
        payload = self._payload()

        encoded = self.encode(payload, 85)
        raw_bytes = sum(payload[key].nbytes for key in payload if key.startswith("observation.images."))
        encoded_bytes = sum(len(encoded[key]["data"]) for key in encoded if key.startswith("observation.images."))

        self.assertLess(encoded_bytes, raw_bytes // 3)

    def test_encode_does_not_mutate_the_input(self) -> None:
        payload = self._payload()

        self.encode(payload, 85)

        self.assertIsInstance(payload["observation.images.top"], np.ndarray)

    def test_non_image_keys_are_untouched(self) -> None:
        encoded = self.encode(self._payload(), 85)

        self.assertEqual(encoded["task"], "pick up the object")
        np.testing.assert_allclose(encoded["observation.state"], np.zeros(14, dtype=np.float32))

    def test_decode_leaves_raw_payloads_unchanged(self) -> None:
        payload = self._payload()

        decoded = self.decode(payload)

        self.assertIs(decoded, payload)

    def test_decode_marks_and_decodes_only_marked_images(self) -> None:
        payload = self._payload()
        encoded = self.encode(payload, 85)

        decoded = self.decode(encoded)

        self.assertNotIn(self.marker, str(decoded["observation.images.top"].dtype))
        self.assertIsInstance(decoded["observation.images.top"], np.ndarray)

    def test_rejects_out_of_range_quality(self) -> None:
        payload = self._payload()

        for quality in (0, 101, -1):
            with self.subTest(quality=quality):
                with self.assertRaises(ValueError):
                    self.encode(payload, quality)

    def test_rejects_non_uint8_images(self) -> None:
        payload = self._payload()
        payload["observation.images.top"] = payload["observation.images.top"].astype(np.float32)

        with self.assertRaises(ValueError):
            self.encode(payload, 85)

    def test_rejects_corrupt_jpeg_bytes(self) -> None:
        payload = self._payload()
        payload["observation.images.top"] = {self.marker: True, "data": b"not a jpeg"}

        with self.assertRaises(ValueError):
            self.decode(payload)


if __name__ == "__main__":
    unittest.main()
