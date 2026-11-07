import unittest
import numpy as np
import torch
from reproduction.reader_error_transfer import anatomy, candidate_donors, choose_guide, transplant_grid


class ReaderTransferTests(unittest.TestCase):
    def test_donors_are_other_distinct_patients_with_nearest_scale(self):
        names=np.array(["a","a","b","b","c","d","e","f"])
        radii=np.array([8.,9.,20.,8.,10.,11.,7.,12.])
        candidates=candidate_donors(0,names,radii)
        self.assertNotIn(0,candidates);self.assertNotIn(1,candidates);self.assertNotIn(2,candidates)
        self.assertIn(3,candidates)
        self.assertEqual(len(set(names[candidates])),len(candidates))

    def test_transplant_grid_preserves_native_center_and_relative_scale(self):
        target=(np.array([60.,62.]),8.,0.)
        donor=(np.array([70.,61.]),12.,0.)
        grid=transplant_grid(target,donor,[1.,1.])
        np.testing.assert_allclose(grid[62,60],donor[0])
        np.testing.assert_allclose(grid[62,64],[76.,61.])
        mask=np.zeros((4,128,128),bool);mask[:,50:54,60:68]=1
        center,radius,angle=anatomy(mask)
        np.testing.assert_allclose(center,[63.5,51.5])
        self.assertAlmostEqual(radius,np.sqrt(32/np.pi))
        self.assertAlmostEqual(angle,0.)

    def test_cyclic_shuffling_keeps_every_rank_marginal_exact(self):
        bank=torch.arange(2*4*4*3*5).reshape(2,4,4,3,5)
        indices=torch.tensor([1,1,1,1]);draws=torch.arange(4)
        full=choose_guide(bank,indices,draws,"joint")
        shuffled=choose_guide(bank,indices,draws,"shuffled")
        for r in range(4):
            torch.testing.assert_close(shuffled[:,r],torch.roll(full[:,r],-r,0))
        self.assertFalse(torch.equal(full,shuffled))
