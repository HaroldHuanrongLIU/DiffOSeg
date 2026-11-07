import unittest
import numpy as np
import torch
from torch.nn import functional as F
from reproduction import native_views as study
from reproduction import aligned_preference as aligned
from reproduction import vote_contours as vote


class NativeViewTest(unittest.TestCase):
    def test_translation_is_reversed_on_physical_pixels(self):
        center=torch.tensor([[65.5,64.5],[63.5,64.5]])
        forward,back=aligned.transform_grids(center,torch.ones(2),torch.zeros(2),torch.ones(2))
        target=torch.zeros(1,4,128,128);target[:,:,50:55,70:75]=1
        view=F.grid_sample(target.repeat(2,1,1,1),forward,align_corners=True)
        self.assertGreater(view[0,0,50,69].item(),.99)
        pulled=F.grid_sample(view,back,align_corners=True)
        torch.testing.assert_close(pulled[:,:,:,5:-5],target.repeat(2,1,1,1)[:,:,:,5:-5],atol=3e-5,rtol=0)

    def test_common_support_ignores_outside_and_matches_vote_on_full_support(self):
        g=torch.Generator().manual_seed(71)
        z=torch.randn(2,4,8,8,generator=g)
        y=torch.rand(2,4,8,8,generator=g)
        roi=torch.ones(2,1,8,8,dtype=torch.bool)
        torch.testing.assert_close(study.masked_preference(z,y,roi),vote.mask_loss(z,y))
        roi[:,:,:,0]=False
        z.requires_grad_()
        study.masked_preference(z,y,roi).backward()
        self.assertTrue((z.grad[:,:,:,0]==0).all())
        self.assertGreater(z.grad[:,:,:,1:].abs().sum().item(),0)

    def test_fusion_precedes_nonlinear_loss_and_both_views_receive_gradient(self):
        center=torch.full((2,2),64.5)
        _,back=aligned.transform_grids(center,torch.ones(2),torch.zeros(2),torch.ones(2))
        z=torch.cat((torch.full((1,4,128,128),-4.),torch.full((1,4,128,128),4.))).requires_grad_()
        y=torch.ones(1,4,128,128)
        value=study.native_objective(z,back,y,"joint")
        expected=1-(128-2)**2/((128-2)**2*1.5+1)-1/((128-2)**2*1.5+1)+5*np.log(2)
        self.assertAlmostEqual(value.item(),expected,places=5)
        self.assertGreater(study.native_objective(z,back,y,"single").item(),value.item())
        value.backward()
        self.assertTrue(all(v.abs().sum().item()>0 for v in z.grad.chunk(2)))

    def test_zero_delta_preserves_anchor_and_selects_zero(self):
        z=np.array([[[[2.,-1.]]]],np.float32)
        truth=z>0
        pred,selected=study.insert(z,z,z,truth)
        np.testing.assert_array_equal(pred,truth)
        self.assertEqual(selected["weight"],0.)
