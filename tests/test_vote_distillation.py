import unittest
import numpy as np
import torch
from scipy.special import logit
from reproduction.vote_distillation import projected_targets, vote_kl


class VoteDistillationTests(unittest.TestCase):
    def test_vote_mass_and_relative_log_odds_preserved(self):
        logits=np.array([[-2.,-1.,.5,1.,2.],[-1.,3.,1.,-1.,1.],
                         [1.,0.,-.5,2.,-2.],[2.,1.,1.5,3.,0.]],dtype=float)[:,None,:]
        y=np.stack([np.arange(5)>i for i in range(4)])[:,None,:]
        q=projected_targets(logits,y)
        np.testing.assert_allclose(q.sum(0),y.sum(0),atol=2e-6)
        np.testing.assert_array_equal(q[:,:,0],np.zeros((4,1)))
        np.testing.assert_array_equal(q[:,:,4],np.ones((4,1)))
        shifts=logit(q[:,:,1:4].astype(float))-logits[:,:,1:4]/2
        np.testing.assert_allclose(shifts,np.broadcast_to(shifts[:1],shifts.shape),atol=2e-6)

    def test_kl_has_only_ambiguous_label_gradients(self):
        y=torch.zeros(1,4,4,5)
        y[0,:2,1,2]=1;y[0,:,3,4]=1
        q=y.mean(1,keepdim=True).expand_as(y)
        logits=torch.ones_like(y,requires_grad=True)
        loss=vote_kl(logits,q,y);loss.backward()
        self.assertGreater(loss.item(),0)
        grad=logits.grad.detach().clone()
        self.assertTrue((grad[:,:,1,2]!=0).all())
        grad[:,:,1,2]=0
        self.assertFalse(grad.any())
        self.assertEqual(vote_kl(logits,torch.zeros_like(y),torch.zeros_like(y)).item(),0.)
