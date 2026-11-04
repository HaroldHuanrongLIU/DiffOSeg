from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from datasets import lidc


def patient_folds(series_uids, seed=1337, folds=4):
    """Match D-Persona's patient ordering, seeded shuffle and four-fold split."""
    groups = {}
    for index, uid in enumerate(series_uids):
        groups.setdefault(str(uid).split("_")[0], []).append(index)
    patients = list(groups)
    np.random.RandomState(seed).shuffle(patients)
    result = []
    for test_positions in np.array_split(np.arange(len(patients)), folds):
        test_positions = set(test_positions.tolist())
        train_patients = [p for i, p in enumerate(patients) if i not in test_positions]
        test_patients = [p for i, p in enumerate(patients) if i in test_positions]
        result.append({
            "train_patients": train_patients,
            "test_patients": test_patients,
            "train_indices": [i for p in train_patients for i in groups[p]],
            "test_indices": [i for p in test_patients for i in groups[p]],
        })
    return result


def datasets_for_run(data_root, name, stage, fold, seed):
    root = Path(data_root)
    if name == "lidc":
        path = root / "lidc_attributes.pkl"
        train = lidc.LIDC_IDRI(stage, str(path), transform=lidc.transform)
        test = lidc.Test_LIDC(stage, str(path), transform=lidc.batch_transform)
        split = patient_folds(train.series_uid, seed)[fold]
        assert not set(split["train_patients"]) & set(split["test_patients"])
        metadata = {"dataset": name, "fold": fold, "evaluation_split": "held_out_patients", **split}
        return Subset(train, split["train_indices"]), Subset(test, split["test_indices"]), metadata
    raise ValueError(f"Unknown dataset: {name}")
