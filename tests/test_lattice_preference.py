import unittest
import torch
from reproduction.lattice_preference import MassProjection, fixed_area_mask


class LatticeTests(unittest.TestCase):
    def test_hard_area_is_exact_even_for_ties(self):
        x=torch.zeros(2,4,7,9); a=torch.tensor([[0,1,7,12],[63,32,31,3]])
        torch.testing.assert_close(fixed_area_mask(x,a).sum((-2,-1)),a.float())
    def test_constant_logit_shift_is_null_direction(self):
        torch.manual_seed(13);x=torch.randn(2,4,7,9,requires_grad=True)
        a=torch.tensor([[2,8,25,32],[3,9,24,31]])
        p=MassProjection.apply(x,a,.65)
        torch.testing.assert_close(p.sum((-2,-1)),a.float(),atol=.006,rtol=0)
        (p*torch.randn_like(p)).sum().backward()
        torch.testing.assert_close(x.grad.sum((-2,-1)),torch.zeros_like(a.float()),atol=2e-6,rtol=0)
