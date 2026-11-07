import unittest
import torch
from reproduction import rank_detail as study


class RankDetailTest(unittest.TestCase):
    def test_shared_or_block_constant_error_has_zero_fine_penalty(self):
        target=torch.zeros(1,4,4,4)
        shared=torch.arange(16).float().reshape(1,1,4,4).expand_as(target)/16
        self.assertEqual(study.detail_loss(shared,target,"all").item(),0.)
        block=torch.zeros_like(target)
        block[:,0,:2,:2]=1
        self.assertGreater(study.detail_loss(block,target,"all").item(),0)
        self.assertEqual(study.detail_loss(block,target,"fine").item(),0.)

    def test_checkered_rank_error_energy_and_rotation(self):
        p=torch.zeros(1,4,2,2)
        p[:,0]=torch.tensor([[1.,0.],[0.,1.]])
        p[:,1]=1-p[:,0]
        target=torch.zeros_like(p)
        self.assertAlmostEqual(study.detail_loss(p,target,"fine").item(),2.4*2/16,places=6)
        torch.testing.assert_close(study.detail_loss(p,target,"fine"),study.detail_loss(torch.rot90(p,1,(-2,-1)),target,"fine"))
        p.requires_grad_()
        study.detail_loss(p,target,"fine").backward()
        torch.testing.assert_close(p.grad.sum(1),torch.zeros_like(p.grad[:,0]),atol=1e-6,rtol=0)
