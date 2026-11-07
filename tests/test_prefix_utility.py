import unittest
import numpy as np
from reproduction.prefix_utility import OFFSETS, ZERO, apply, case_curves, prefix_dice


class PrefixUtilityTests(unittest.TestCase):
    def test_prefix_scores_equal_independent_small_sets(self):
        y=np.array([1,0,1,0],bool)
        order=np.array([2,1,0,3])
        np.testing.assert_allclose(prefix_dice(y,order,np.arange(5)),[0,2/3,1/2,4/5,2/3])
        np.testing.assert_allclose(prefix_dice(np.zeros(4,bool),order,[0,1]),[1,0])

    def test_zero_shift_replays_original_mask_even_with_ties(self):
        z=np.full((4,128,128),-1.,np.float32)
        for rank in range(4):
            z[rank,55:65+rank,61:69]=1
        curves=case_curves(np.zeros((128,128),np.uint8),z,np.zeros(12,np.float32))
        np.testing.assert_array_equal(apply(curves,[3,-2,1,4],0),z>0)
        for rank,curve in enumerate(curves):
            self.assertEqual(curve.sizes[ZERO],int((z[rank]>0).sum()))
            self.assertEqual(curve.features.shape[0],len(OFFSETS))
            self.assertTrue(np.isfinite(curve.features).all())
