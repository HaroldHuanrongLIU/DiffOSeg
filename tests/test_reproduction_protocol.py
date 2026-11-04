import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from reproduction.data import patient_folds
from reproduction.metrics import consensus_scores, pairwise_scores, preference_scores
from reproduction.run import paper_comparison, save_masks, load_masks, load_config
from metrics_set import generalized_energy_distance


class ReproductionProtocolTests(unittest.TestCase):
    def test_config_expands_user_home_in_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.yml"
            config_path.write_text("data_root: ${HOME}/dataset\n")
            with patch.dict("os.environ", {"HOME": temporary}):
                config = load_config(config_path)
            self.assertEqual(config["data_root"], str(Path(temporary) / "dataset"))

    def test_patient_folds_never_share_a_patient(self):
        uids = [f"{patient}_slice{image}" for patient in range(214) for image in range(2)]
        folds = patient_folds(uids)
        self.assertEqual([len(f["test_patients"]) for f in folds], [54, 54, 53, 53])
        seen = []
        for fold in folds:
            self.assertFalse(set(fold["train_patients"]) & set(fold["test_patients"]))
            self.assertFalse(set(fold["train_indices"]) & set(fold["test_indices"]))
            self.assertEqual(len(fold["train_indices"]) + len(fold["test_indices"]), len(uids))
            seen.extend(fold["test_indices"])
        self.assertEqual(sorted(seen), list(range(len(uids))))
        self.assertEqual(folds, patient_folds(uids))

    def test_empty_masks_match(self):
        empty = np.zeros((4, 8, 8), dtype=bool)
        dice, iou = pairwise_scores(empty, empty)
        np.testing.assert_array_equal(dice, 1)
        np.testing.assert_array_equal(iou, 1)
        score = consensus_scores(empty, empty)
        self.assertEqual(score["GED"], 0)
        self.assertEqual(score["Dice_soft"], 1)

    def test_reported_ged_matches_released_reference_with_empty_masks(self):
        labels = np.zeros((4, 8, 8), dtype=bool)
        labels[2:] = True
        predictions = np.zeros((4, 8, 8), dtype=bool)
        reference = generalized_energy_distance(
            torch.from_numpy(labels[None]), torch.from_numpy(predictions[None])
        )
        self.assertAlmostEqual(consensus_scores(labels, predictions)["GED"], reference, places=6)
        self.assertEqual(consensus_scores(labels, predictions)["GED_empty_match"], 0.5)

    def test_saved_masks_round_trip_with_padding_bits(self):
        generator = np.random.default_rng(7)
        truth = generator.integers(0, 2, (4, 2, 3, 3)).astype(bool)
        predictions = generator.integers(0, 2, (5, 2, 3, 3)).astype(bool)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "masks.npz"
            save_masks(path, truth, predictions)
            restored_truth, restored_predictions = load_masks(path)
        np.testing.assert_array_equal(restored_truth, truth)
        np.testing.assert_array_equal(restored_predictions, predictions)

    def test_consensus_matching_must_not_replace_expert_identity(self):
        labels = np.eye(4, dtype=bool).reshape(4, 2, 2)
        shuffled = np.roll(labels, 1, axis=0)
        self.assertEqual(consensus_scores(labels, shuffled)["Dice_match"], 1)
        self.assertEqual(preference_scores(labels, shuffled)["Dice_mean"], 0)
        self.assertEqual(preference_scores(labels, labels)["Dice_each"], [1, 1, 1, 1])
        with self.assertRaises(ValueError):
            preference_scores(labels, labels[:3])

    def test_partial_folds_and_smoke_do_not_become_paper_comparisons(self):
        rows = []
        for fold in range(4):
            rows.append({"results": {
                "dataset": "lidc", "stage": 2, "fold": fold,
                "classification": "paper_budget_with_documented_assumptions",
                "metrics": {"Dice_each": [0.5] * 4},
            }})
        self.assertEqual(paper_comparison(rows[:3])["lidc_stage_2"]["state"], "incomplete")
        self.assertEqual(paper_comparison(rows)["lidc_stage_2"]["metrics"][-1]["observed"], 0.5)
        for row in rows:
            row["results"]["classification"] = "engineering_smoke"
        self.assertEqual(paper_comparison(rows)["lidc_stage_2"]["completed_folds"], 0)


if __name__ == "__main__":
    unittest.main()
