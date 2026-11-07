import unittest

import numpy as np
import torch

from reproduction import crossfit_contours as cf
from reproduction import crossfit_residual as residual


class CrossFitTests(unittest.TestCase):
    def test_every_training_patient_is_held_exactly_once(self):
        uids=[f"P{p}_slice{s}" for p in range(12) for s in range(4)]
        indices=list(range(len(uids)))
        parts=cf.partitions(indices,uids,1);seen=[]
        for train,held in parts:
            self.assertTrue({uids[i].split("_")[0] for i in train}.isdisjoint({uids[i].split("_")[0] for i in held}))
            self.assertEqual(set(train+held),set(indices))
            seen+=held
        self.assertEqual(sorted(seen),indices)

    def test_identity_initialization_and_image_independence(self):
        torch.manual_seed(57);model=residual.Correction().eval()
        x=torch.rand(2,1,32,32);g=torch.randn(2,4,32,32)
        with torch.no_grad():
            z,r=model(x,g)
            torch.testing.assert_close(z,g)
            torch.testing.assert_close(r,((g>0).sum((-2,-1)).clamp_min(1)/np.pi).sqrt())
