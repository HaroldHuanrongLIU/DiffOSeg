import unittest
import numpy as np
import torch
from torch import nn
from reproduction import registered_matte as matte
from reproduction import aligned_preference as aligned


class RegisteredMatteTest(unittest.TestCase):
    def test_registration_recovers_shift_in_xy_convention(self):
        target=np.zeros((3,30,30),dtype=np.float32)
        target[:,8:18,10:17]=.75
        target[:,12:15,17:20]=.25
        source=np.roll(target,(2,-1),axis=(1,2))
        shift,before,after=matte.estimate_matte_shift(source,target)
        self.assertEqual(shift,(1,-2))
        self.assertGreater(before,0)
        self.assertEqual(after,0)
        corrected=aligned.registered_ct(torch.from_numpy(source[:,None]),*shift)
        np.testing.assert_array_equal(corrected[:,0,3:-3,3:-3].numpy(),target[:,3:-3,3:-3])

    def test_linear_basis_conserves_mass_and_interpolates(self):
        values=torch.tensor([0.,.125,.25,.625,1.]).reshape(1,1,1,-1)
        expected=torch.tensor([[1,.5,0,0,0],[0,.5,1,0,0],[0,0,0,.5,0],
                               [0,0,0,.5,0],[0,0,0,0,1]]).reshape(1,5,1,-1)
        torch.testing.assert_close(matte.spline(values),expected)

    def test_warm_readout_and_gradient_into_both_new_heads(self):
        torch.manual_seed(14)
        warm=nn.Conv2d(7,12,1)
        layer=matte.MatteHead(warm)
        x=torch.randn(2,7,13,13)
        old=warm(x)
        new=layer(x)
        torch.testing.assert_close(new[:,:12],old)
        torch.testing.assert_close(new[:,12:13],old[:,:4].mean(1,keepdim=True))
        y=torch.randint(0,2,(2,4,13,13)).float()
        soft=torch.rand(2,1,13,13)
        loss=matte.vote.mask_loss(new[:,:4],y)+.3*matte.matte_loss(new[:,12:],soft,y)
        loss.backward()
        self.assertGreater(layer.coefficients.weight.grad.abs().sum().item(),0)
        self.assertGreater(layer.continuous.weight.grad.abs().sum().item(),0)
