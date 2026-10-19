import unittest
from pathlib import Path

import numpy as np

from tools.transform_xtrainer_dataset_images import CAMERA_KEYS, _video_paths, transform_frame


class TransformXTrainerDatasetImagesTest(unittest.TestCase):
    def test_camera_orientation_rules(self) -> None:
        image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)[..., None]

        np.testing.assert_array_equal(transform_frame(CAMERA_KEYS[0], image), image)
        np.testing.assert_array_equal(transform_frame(CAMERA_KEYS[1], image), image)
        np.testing.assert_array_equal(transform_frame(CAMERA_KEYS[2], image)[..., 0], [[6, 5, 4], [3, 2, 1]])

    def test_video_paths_use_chunk_and_episode_as_the_alignment_key(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for camera_key in CAMERA_KEYS:
                camera_dir = root / "videos" / "chunk-000" / camera_key
                camera_dir.mkdir(parents=True)
                (camera_dir / "episode_000000.mp4").touch()

            expected = {Path("chunk-000/episode_000000.mp4")}
            paths = {camera_key: _video_paths(root, camera_key) for camera_key in CAMERA_KEYS}
            self.assertTrue(all(set(camera_paths) == expected for camera_paths in paths.values()))
