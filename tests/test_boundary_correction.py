import unittest

import numpy as np

from reproduction.boundary_eval import COUNTS, correct
from reproduction.rank_contours import signed


class BoundaryCorrectionTest(unittest.TestCase):
    def test_empty_and_full_have_finite_distances(self):
        a = np.zeros((12, 12), dtype=bool)
        np.testing.assert_array_equal(signed(a), np.full(a.shape, -8))
        np.testing.assert_array_equal(signed(~a), np.full(a.shape, 8))
        a[3:9, 3:9] = True
        np.testing.assert_array_equal(signed(a) > 0, a)

    def test_agreeing_contours_recover_identity(self):
        mask = np.zeros((24, 24), dtype=bool)
        mask[6:17, 7:18] = True
        consensus = np.repeat(mask[None], 50, axis=0)
        expert = consensus[:4].copy()
        banks, pref = correct(consensus, expert, expert)
        np.testing.assert_array_equal(pref, expert)
        for k in COUNTS:
            np.testing.assert_array_equal(banks[k], consensus[:k])

    def test_prefix_and_fixed_expert_alignment(self):
        rng = np.random.default_rng(57)
        bank = rng.random((50, 16, 16)) > .8
        expert = np.zeros((4, 16, 16), dtype=bool)
        for rank in range(4):
            expert[rank, rank:rank + 6, rank:rank + 4] = True
        original, pref = correct(bank, expert, expert)
        np.testing.assert_array_equal(expert, pref)
        bank[10:] = ~bank[10:]
        modified, _ = correct(bank, expert, expert)
        np.testing.assert_array_equal(original[10], modified[10])


if __name__ == "__main__":
    unittest.main()
