import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from deploy.inference_logging import InferenceRecorder
from deploy.msgpack_numpy import unpackb


class InferenceRecorderTest(unittest.TestCase):
    def test_records_lossless_protocol_payloads_and_input_images(self) -> None:
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        request = {
            "observation.state": np.arange(14, dtype=np.float32),
            "observation.images.top": image,
            "task": "pick up the object",
        }
        response = {"action": np.arange(28, dtype=np.float32).reshape(2, 14)}

        with tempfile.TemporaryDirectory() as temporary_directory:
            original_directory = Path.cwd()
            os.chdir(temporary_directory)
            try:
                recorder = InferenceRecorder("server")
                recorder.record_inference(request, response)
                recorder.record_applied_action(response["action"][0], step=7, source="synchronous")
                recorder.close()
            finally:
                os.chdir(original_directory)

            run_directories = list((Path(temporary_directory) / "log").iterdir())
            self.assertEqual(len(run_directories), 1)
            run_directory = run_directories[0]

            saved_request = unpackb((run_directory / "payloads" / "000001-request.msgpack").read_bytes())
            saved_response = unpackb((run_directory / "payloads" / "000001-response.msgpack").read_bytes())
            np.testing.assert_array_equal(saved_request["observation.state"], request["observation.state"])
            np.testing.assert_array_equal(saved_request["observation.images.top"], image)
            self.assertEqual(saved_request["task"], request["task"])
            np.testing.assert_array_equal(saved_response["action"], response["action"])
            np.testing.assert_array_equal(
                np.load(run_directory / "applied_actions" / "000002.npy"),
                response["action"][0],
            )

            saved_image = np.asarray(Image.open(run_directory / "images" / "000001-request-observation_images_top.png"))
            np.testing.assert_array_equal(saved_image, image)

            events = [json.loads(line) for line in (run_directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[1]["event"], "inference")
            self.assertEqual(events[2]["source"], "synchronous")


if __name__ == "__main__":
    unittest.main()
