"""Subject-level PADS classification using preprocessed movement binaries."""

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from momentfm.data.bin_timeseries import load_movement_bin


class PADS_dataset(Dataset):
    """One subject per sample, returning (132, seq_len) and an integer label.

    Train/validation/test sets are stratified by label and split by subject.
    All phases use the same seed and metadata order to reproduce the partition.
    No extra scaling is applied: MOMENT already normalizes each input channel.
    """

    n_channels = 132
    n_classes = 3

    def __init__(self, config, phase):
        if phase not in {"train", "val", "test"}:
            raise ValueError("phase must be train, val, or test")
        self.phase = phase
        self.seq_len = config.seq_len
        self.base_path = Path(config.base_path).expanduser().resolve()
        self.movement_path = self.base_path / "movement"

        with (self.base_path / "file_list.csv").open(
            encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)
            if not {"id", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("file_list.csv must contain id and label columns")
            records = list(reader)
        if not records:
            raise ValueError("file_list.csv is empty")

        subject_ids = [f"{int(record['id']):03d}" for record in records]
        labels = np.array([int(record["label"]) for record in records], dtype=np.int64)
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("Duplicate subject IDs in file_list.csv")
        if set(labels.tolist()) != {0, 1, 2}:
            raise ValueError("Expected three PADS classes with labels 0, 1, 2")
        paths = [self.movement_path / f"{sid}_ml.bin" for sid in subject_ids]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing movement binary: {path}")
            if path.stat().st_size != 132 * 976 * 4:
                raise ValueError(f"Invalid movement binary size: {path}")

        val_ratio = config.val_ratio
        test_ratio = config.test_ratio
        if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
            raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
        indices = np.arange(len(records))
        train_val, test = train_test_split(
            indices, test_size=test_ratio, random_state=config.seed, stratify=labels
        )
        train, val = train_test_split(
            train_val,
            test_size=val_ratio / (1 - test_ratio),
            random_state=config.seed,
            stratify=labels[train_val],
        )
        selected = {"train": train, "val": val, "test": test}[phase]
        self.subject_ids = [subject_ids[i] for i in selected]
        self.paths = [paths[i] for i in selected]
        self.labels = labels[selected]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_movement_bin(self.paths[index], self.seq_len), self.labels[index]
