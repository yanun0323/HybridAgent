"""Read an existing training-only magnitude calibration without changing it."""
import hashlib
import json
import math
from pathlib import Path


def read_training_calibration(path, data_file):
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    data_hash = hashlib.sha256(Path(data_file).read_bytes()).hexdigest()
    if record.get("data_sha256") != data_hash:
        raise ValueError("calibration does not match the training data file")
    scale = float(record["output_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("calibration output_scale must be finite and positive")
    return {**record, "output_scale": scale, "reused": True,
            "source_calibration_sha256": hashlib.sha256(raw).hexdigest()}
