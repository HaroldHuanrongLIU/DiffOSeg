import unittest
import numpy as np
from reproduction.centroid_exchange import BUDGETS, exchange_path, mean_centroid, distance_squared


class CentroidExchangeTests(unittest.TestCase):
    def test_area_and_consensus_moment_are_preserved_or_improved(self):
        scores = np.full((4, 18, 18), -4.0, np.float32)
        scores[:, 6:11, 8:13] = 2.0
        target = np.array([8, 8])
        masks, counts = exchange_path(scores, target, True)
        self.assertEqual(counts[0], 0)
        self.assertGreater(counts[-1], 0)
        for budget, number, mask in zip(BUDGETS, counts, masks):
            np.testing.assert_array_equal(mask.sum(axis=(1, 2)), np.full(4, 25))
            self.assertLessEqual(number, budget)
        distances = [distance_squared(mean_centroid(mask), target) for mask in masks]
        self.assertTrue(np.all(np.diff(distances) <= 1e-9))
        self.assertLess(distances[-1], distances[0])
        np.testing.assert_array_equal(masks[0], scores > 0)

    def test_guards_for_empty_centered_and_offcenter_masks(self):
        scores = np.full((4, 18, 18), -4.0)
        scores[:, 6:11, 6:11] = 4.0
        for target, empty in (([8, 8], False), ([2, 8], False), ([8, 9], True)):
            logits = scores.copy()
            if empty:
                logits[0] = -2
            masks, counts = exchange_path(logits, np.asarray(target), True)
            self.assertFalse(counts.any())
            np.testing.assert_array_equal(masks[-1], logits > 0)

    def test_first_swap_is_minimum_confidence_on_tied_geometry(self):
        scores = np.full((4, 14, 14), -10.0)
        scores[:, 5:8, 7:10] = 10.0
        scores[3, 6, 9] = 0.01
        scores[3, 6, 6] = -0.01
        masks, counts = exchange_path(scores, np.array([6., 6.]), True)
        self.assertEqual(counts[1], 1)
        self.assertFalse(masks[1, 3, 6, 9])
        self.assertTrue(masks[1, 3, 6, 6])
