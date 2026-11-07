import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from reproduction import hierarchical_sam as hierarchy
from reproduction import masked_ct as study


class MaskedCTTest(unittest.TestCase):
    def test_exact_hidden_blocks_and_no_hidden_input(self):
        rng=np.random.default_rng(341)
        order=rng.random((2,256)).argsort(1)
        mask=study.block_mask(order,"cpu")
        self.assertEqual(mask.shape,(2,1,128,128))
        self.assertTrue(torch.all(mask.sum((-2,-1))==8192))
        blocks=mask.reshape(2,1,16,8,16,8)
        self.assertTrue((blocks==blocks[:,:,:,0:1,:,0:1]).all())
        encoder=study.NativeEncoder().eval()
        image=torch.rand(2,1,128,128)
        changed=torch.where(mask,image+10,image)
        for a,b in zip(encoder(image,mask),encoder(changed,mask)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_region_loss_scores_only_requested_pixels_and_gradients(self):
        x=torch.tensor([[[[.1,.2],[.4,.7]]]])
        y=study.reconstruction_target(x)
        torch.testing.assert_close(y[0,1],torch.tensor([[.4,0],[1.2,0]]))
        torch.testing.assert_close(y[0,2],torch.tensor([[1.2,2.],[0,0]]))
        selected=torch.tensor([[[[True,False],[False,True]]]])
        pred=(y+1).requires_grad_()
        study.region_loss(pred,y,selected,torch.ones(2,2)).backward()
        self.assertTrue((pred.grad[~selected.expand_as(pred)]==0).all())
        self.assertTrue((pred.grad[selected.expand_as(pred)]>0).all())

    def test_transfer_preserves_e_and_has_gradients_at_both_scales(self):
        class Heads(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_s0=nn.Identity()
                self.conv_s1=nn.Identity()
        with patch.object(hierarchy,"open_public",return_value={}),patch.object(hierarchy,"decoder_heads",
                side_effect=lambda _:(nn.Identity(),Heads())):
            torch.manual_seed(411)
            original=hierarchy.HierarchicalRanks(None)
            rng=torch.rand(4)
            torch.manual_seed(411)
            model=study.MaskedRanks(None)
            torch.testing.assert_close(torch.rand(4),rng,rtol=0,atol=0)
        packed=torch.randn(2,1024,4,4)
        image=torch.rand(2,1,16,16)
        a,skips=original.spatial_features(packed,None,image)
        b,with_ct=model.spatial_features(packed,None,image)
        for left,right in zip([a,*skips],[b,*with_ct]):
            torch.testing.assert_close(left,right,atol=0,rtol=0)
        sum(x.square().mean() for x in with_ct).backward()
        for projection in model.projections:
            self.assertGreater(projection.weight.grad.abs().sum().item(),0)
