import unittest

import numpy as np
import torch

from reproduction import joint_rater as joint
from reproduction import vote_eval as v3


class JointRaterTests(unittest.TestCase):
    def test_exact_marginals_and_rank_only_limit(self):
        gen=torch.Generator().manual_seed(702)
        rank=torch.randn(3,4,7,9,generator=gen)
        vote=torch.randn(3,4,7,9,generator=gen)
        net=joint.JointDecoder([3,4])
        e,c=joint.count_condition(rank,vote,0.27,net.bits)
        torch.testing.assert_close(e.sum(1),c.sum(1),rtol=2e-6,atol=1e-6)
        self.assertTrue(bool((c[:,:-1]>=c[:,1:]).all()))
        e,_=joint.count_condition(rank,vote,1.,net.bits)
        torch.testing.assert_close(e,rank.sigmoid(),atol=2e-6,rtol=2e-6)

    def test_symmetric_experts_share_one_vote(self):
        logits=torch.zeros(1,4,1,1)
        votes=torch.tensor([20.,-20.,-20.,-20.]).reshape(1,4,1,1)
        e,s=joint.count_condition(logits,votes,0.,joint.JointDecoder([0,0]).bits)
        torch.testing.assert_close(e,torch.full_like(e,0.25))
        self.assertAlmostEqual(float(s.sum()),1.,places=6)

    def test_shift_direction_and_far_landmark(self):
        x=torch.zeros(1,1,13,15)
        x[0,0,4,7]=1
        torch.testing.assert_close(joint.centroid(joint.translate(x,torch.tensor([[2.,-1.]])).squeeze(1)),
                                   torch.tensor([[9.,3.]]),atol=1e-5,rtol=1e-5)
        net=joint.JointDecoder([1000,1000])
        fields=torch.randn(1,8,13,15)
        a=net(fields,location=False,joint=False)
        b=net(fields,location=True,joint=False)
        for s,t in zip(a,b):
            torch.testing.assert_close(s,t,atol=1e-5,rtol=1e-5)

    def test_draw_order_preserves_exact_target_and_prefix(self):
        rng=np.random.default_rng(771)
        bank=rng.random((50,9,11))>0.45
        e=rng.uniform(0.01,0.99,(4,9,11))
        support=rng.uniform(size=(4,9,11))
        lock={"hardness":0.5,"strength":0.9}
        outputs=[]
        for weight in (0,0.2):
            b=bank[:10]
            outputs.append(joint.consensus(b,v3.bank_ranks(b),e,support,lock,weight))
        np.testing.assert_equal(outputs[0].sum(0),outputs[1].sum(0))
        bank[10:]=~bank[10:]
        b=bank[:10]
        np.testing.assert_equal(outputs[1],joint.consensus(b,v3.bank_ranks(b),e,support,lock,0.2))
