"""Calibration reuse checks do not require model dependencies."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

path = Path(__file__).resolve().parents[1] / "hybrid_sft/calibration.py"
spec = importlib.util.spec_from_file_location("training_calibration", path)
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class CalibrationTests(unittest.TestCase):
    def test_reuse_preserves_scale_and_files_but_rejects_mismatched_data(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data, config = root / "train.jsonl", root / "calibration.json"
            data.write_text('{"question":"q"}\n')
            record = {"output_scale": 0.012689582847333439, "pairs": 1628,
                      "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
            config.write_text(json.dumps(record))
            original = config.read_bytes()
            reused = calibration.read_training_calibration(config, data)
            self.assertEqual(reused["output_scale"], record["output_scale"])
            self.assertEqual(reused["pairs"], 1628)
            self.assertTrue(reused["reused"])
            self.assertEqual(reused["source_calibration_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(config.read_bytes(), original)
            for invalid in (0, -1, float("nan"), float("inf")):
                config.write_text(json.dumps({**record, "output_scale": invalid}))
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    calibration.read_training_calibration(config, data)
            config.write_bytes(original)
            data.write_text('{"question":"different"}\n')
            with self.assertRaisesRegex(ValueError, "training data"):
                calibration.read_training_calibration(config, data)


if __name__ == "__main__":
    unittest.main()
