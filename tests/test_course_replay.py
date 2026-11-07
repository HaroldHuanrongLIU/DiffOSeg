import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_course_results import score_masks


class CourseReplayTests(unittest.TestCase):
    def test_identical_deterministic_masks(self):
        mask = np.eye(4, dtype=bool)
        released, high_precision = score_masks(
            np.repeat(mask[None], 4, 0), np.repeat(mask[None], 30, 0)
        )
        np.testing.assert_allclose(released, [0, 1], atol=1e-12)
        np.testing.assert_allclose(high_precision, [0, 1], atol=1e-12)

    def test_disjoint_deterministic_masks(self):
        truth = np.zeros((4, 4, 4), dtype=bool)
        predictions = np.zeros((30, 4, 4), dtype=bool)
        truth[:, 0, 0] = True
        predictions[:, 3, 3] = True
        released, high_precision = score_masks(truth, predictions)
        np.testing.assert_allclose(released, [2, 0], atol=1e-12)
        np.testing.assert_allclose(high_precision, [2, 0], atol=1e-12)

    def test_same_empirical_distribution_with_different_bank_sizes(self):
        first = np.eye(4, dtype=bool)
        second = np.fliplr(first)
        truth = np.stack([first, second, first, second])
        predictions = np.stack([first, second] * 15)
        released, high_precision = score_masks(truth, predictions)
        np.testing.assert_allclose(released, [0, 1], atol=1e-12)
        np.testing.assert_allclose(high_precision, [0, 1], atol=1e-12)

    def test_released_empty_mask_convention(self):
        truth = np.ones((4, 4, 4), dtype=bool)
        predictions = np.zeros((30, 4, 4), dtype=bool)
        released, high_precision = score_masks(truth, predictions)
        np.testing.assert_allclose(released, [1, 0], atol=1e-12)
        np.testing.assert_allclose(high_precision, [1, 0], atol=1e-12)
        empty, empty64 = score_masks(np.zeros_like(truth), predictions)
        np.testing.assert_allclose(empty, [0, 1], atol=1e-12)
        np.testing.assert_allclose(empty64, [0, 1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
