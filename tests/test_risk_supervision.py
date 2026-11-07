import unittest
import torch
from reproduction import risk_supervision as risk


class RiskSupervisionTest(unittest.TestCase):
    def test_equal_case_weight_and_zero_gradient_outside_budget(self):
        z=torch.tensor([0.,1.]).reshape(2,1,1,1).expand(2,4,3,3).clone().requires_grad_()
        y=torch.zeros_like(z);mask=torch.zeros(2,3,3);mask[0,0,1]=1;mask[1,1:,:]=1
        value=risk.region_loss(z,y,mask)
        expected=(torch.nn.functional.softplus(torch.tensor(0.))+torch.nn.functional.softplus(torch.tensor(1.)))/2
        torch.testing.assert_close(value,expected)
        value.backward();self.assertTrue((z.grad[~mask[:,None].expand_as(z).bool()]==0).all())
        self.assertTrue((z.grad[mask[:,None].expand_as(z).bool()]>0).all())

    def test_empty_and_rotated_regions(self):
        rng=torch.Generator().manual_seed(952)
        z=torch.randn(3,4,8,8,generator=rng);y=torch.rand(z.shape,generator=rng);mask=y[:,0]>.6
        self.assertEqual(risk.region_loss(z,y,torch.zeros_like(mask)).item(),0.)
        rot=lambda a:torch.rot90(a,1,(-2,-1))
        torch.testing.assert_close(risk.region_loss(z,y,mask),risk.region_loss(rot(z),rot(y),rot(mask)))
