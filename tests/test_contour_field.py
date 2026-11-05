import unittest

import numpy as np
import torch

from reproduction import contour_field as field
from reproduction import field_study as study
from reproduction.metrics import consensus_scores


class FieldTests(unittest.TestCase):
    def test_identical_basis_shares_random_rank_and_antithetic_symmetry(self):
        b=torch.zeros(1,field.DIM,7,8); b[:,0]=1
        z=torch.randn(1,6,field.DIM)
        u=field.sample_percentiles(b,torch.cat((z,-z),1))
        torch.testing.assert_close(u[:,:6]+u[:,6:],torch.ones_like(u[:,:6]))
        torch.testing.assert_close(u,u[:,:,:1,:1].expand_as(u))

    def test_field_order_depends_only_on_prefix_and_preserves_pixel_counts(self):
        rng=np.random.default_rng(25)
        b=rng.normal(size=(field.DIM,7,9)).astype(np.float32)
        q=rng.uniform(size=(7,9))
        z=study.field_ranks(b,10,43)
        v=rng.random((10,7,9)).argsort(0).astype(np.uint8)
        for w in (0.,0.5,1.):
            mixed=study.mix_order(v,z,w)
            np.testing.assert_array_equal(np.sort(mixed,axis=0),np.broadcast_to(np.arange(10)[:,None,None],mixed.shape))
            np.testing.assert_array_equal(study.bank_at(q,mixed).sum(0),np.floor(10*q+0.5))
        np.testing.assert_equal(z,study.field_ranks(b,10,43))

    def test_energy_term_matches_released_metric_and_has_gradient(self):
        torch.manual_seed(61)
        b=torch.nn.functional.normalize(torch.randn(1,field.DIM,7,8),dim=1).requires_grad_()
        y=(torch.rand(1,4,7,8)>0.5).float()
        f=torch.rand(1,7,8).requires_grad_()
        z=torch.randn(1,12,field.DIM)
        samples=(f[:,None]>field.sample_percentiles(b,z)).detach().numpy()[0]
        ged=consensus_scores(y[0].numpy(),samples)["GED"]
        soft=[]
        for t in (.1,.3,.5,.7,.9):
            a=f.detach().numpy()[0]>t
            c=y.numpy()[0].mean(0)>t
            den=a.sum()+c.sum()
            soft.append(2*(a&c).sum()/den if den else 1.)
        loss=field.population_objective(f,b,y,z)
        self.assertAlmostEqual(loss.item(),ged+1.6*(1-np.mean(soft)),places=5)
        loss.backward()
        self.assertGreater(f.grad.abs().sum().item(),0)
        self.assertGreater(b.grad.abs().sum().item(),0)
