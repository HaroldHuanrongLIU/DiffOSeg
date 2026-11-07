import unittest
import numpy as np
from reproduction.topology_control import clean_mask


class TopologyControlTests(unittest.TestCase):
    def test_small_components_removed_and_equal_size_ties_retained(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[1:4, 1:4] = True
        mask[8:11, 8:11] = True
        mask[16, 16] = True
        expected = mask.copy()
        expected[16, 16] = False
        np.testing.assert_array_equal(clean_mask(mask, 0, 0), mask)
        np.testing.assert_array_equal(clean_mask(mask, 1, 0), expected)

    def test_hole_fill_respects_border_and_size(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:17, 3:17] = True
        mask[6, 6] = False
        mask[10:13, 10:13] = False
        expected = mask.copy()
        expected[6, 6] = True
        np.testing.assert_array_equal(clean_mask(mask, 0, 4), expected)
        expected[10:13, 10:13] = True
        np.testing.assert_array_equal(clean_mask(mask, 0, 16), expected)

    def test_empty_and_full_masks(self):
        for mask in (np.zeros((5, 5), bool), np.ones((5, 5), bool)):
            np.testing.assert_array_equal(clean_mask(mask, 1, 16384), mask)
