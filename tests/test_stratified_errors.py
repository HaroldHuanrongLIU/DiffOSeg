import unittest
import numpy as np
import torch
from reproduction import stratified_errors as study


class StratifiedErrorsTest(unittest.TestCase):
    def test_ties_and_weighted_strata(self):
        np.testing.assert_allclose(study.percentile(np.array([0,0,2,3])),[.25,.25,.625,.875])
        rng=np.random.default_rng(723)
        y=rng.random((85,4,12,12))<rng.uniform(.1,.9,size=(85,1,1,1))
        p=y^(rng.random(y.shape)<.2)
        _,_,group,raw,balanced=study.difficulty(y,p)
        self.assertGreater(np.max(abs(raw-balanced)),.05)
        for rank in range(4):
            self.assertAlmostEqual(float(raw[:,rank].mean()),1.,places=6)
            for g in np.unique(group[:,rank]):
                self.assertAlmostEqual(float(balanced[group[:,rank]==g,rank].mean()),1.,places=6)

    def test_uniform_reproduces_original_loss_and_gradients(self):
        z=torch.randn(3,4,7,9,requires_grad=True)
        y=torch.rand_like(z)
        w=torch.ones(3,4)
        ordinary=study.vote.mask_loss(z,y)
        weighted=study.mask_objective(z,y,w)
        torch.testing.assert_close(weighted,ordinary)
        a=torch.autograd.grad(ordinary,z)[0]
        b=torch.autograd.grad(weighted,z)[0]
        torch.testing.assert_close(a,b)

    def test_weighting_changes_only_targeted_rank_contributions(self):
        z=torch.zeros(1,4,2,2,requires_grad=True)
        y=torch.ones_like(z)
        weight=torch.tensor([[.5,1.,1.,1.5]])
        study.mask_objective(z,y,weight).backward()
        self.assertAlmostEqual(float(z.grad[0,3,0,0]/z.grad[0,0,0,0]),3.,places=6)
        self.assertEqual([study.ramp(s) for s in (1,500,4500,5250,6000)],[0.,1.,1.,.5,0.])
