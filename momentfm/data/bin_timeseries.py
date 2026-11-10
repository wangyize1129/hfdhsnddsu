"""Read PADS movement binaries without changing the files on disk.

Each file contains little-endian float32 values in channel-first order:
132 channels (11 task segments x 12 bilateral IMU axes), 976 points each.
"""

from pathlib import Path

import numpy as np


RAW_N_CHANNELS = 132
RAW_SEQ_LEN = 976
SEQUENCE_LENGTHS = (512, 976)


def load_movement_bin(path: str | Path, seq_len: int = RAW_SEQ_LEN) -> np.ndarray:
    """Return (132, seq_len) float32 data for one subject.

    seq_len=976 preserves every original point without interpolation.
    seq_len=512 linearly resamples the full segment, preserving its endpoints.
    Channel order and segment duration are unchanged. Resampling does not mean
    the resulting sequence still has the original 100 Hz sampling rate.
    """
    if seq_len not in SEQUENCE_LENGTHS:
        raise ValueError(f"seq_len must be one of {SEQUENCE_LENGTHS}, got {seq_len}")

    path = Path(path)
    dtype = np.dtype("<f4")
    expected_bytes = RAW_N_CHANNELS * RAW_SEQ_LEN * dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes for a (132, 976) movement "
            "binary. Questionnaire binaries are not supported here."
        )

    data = np.fromfile(path, dtype=dtype)
    if data.size != RAW_N_CHANNELS * RAW_SEQ_LEN:
        raise ValueError(f"{path}: incomplete movement binary")
    data = data.reshape(RAW_N_CHANNELS, RAW_SEQ_LEN)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: movement binary contains NaN or infinity")

    if seq_len == RAW_SEQ_LEN:
        return data

    old_time = np.linspace(0.0, 1.0, RAW_SEQ_LEN)
    new_time = np.linspace(0.0, 1.0, seq_len)
    return np.stack([
        np.interp(new_time, old_time, channel) for channel in data
    ]).astype(np.float32)
