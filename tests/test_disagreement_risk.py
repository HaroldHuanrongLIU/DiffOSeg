import unittest
import numpy as np
from reproduction.disagreement_risk import features, select_mask


class DisagreementTests(unittest.TestCase):
    def test_both_disagreement_directions_and_uncertain_fallback(self):
        old = np.array([[[False, True, True, False, True, False]]])
        new = np.array([[[True, False, False, True, True, False]]])
        scalar = np.array([[[False, True, True, True, False, True]]])
        points = np.nonzero(old != new)
        result = select_mask(old, new, scalar, points, np.array([.9, .8, .2, .5]))
        np.testing.assert_array_equal(result, [[[True, False, True, True, True, False]]])

    def test_rows_follow_the_disagreement_pixel_order(self):
        old = np.full((4, 16, 16), -2, dtype=np.float32)
        new = old.copy()
        new[0, 8, 8] = 3
        new[3, 4, 9] = 4
        matrix, points = features(np.zeros((16, 16), np.uint8), old, new, np.zeros((16, 16)))
        np.testing.assert_array_equal(points, [[0, 3], [8, 4], [8, 9]])
        np.testing.assert_array_equal(matrix[:, :3], [[0, -2, 3], [3, -2, 4]])
