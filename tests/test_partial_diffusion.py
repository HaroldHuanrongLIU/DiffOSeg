import unittest
import numpy as np
import torch
from torch import nn
from reproduction.prior_diffusion import predict, schedule


class EmptyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter=nn.Parameter(torch.tensor(0.))
        self.levels=[]

    def forward(self,image,guide,observed,prior,retention):
        self.levels.append(float(retention[0,0,0,0]))
        output=torch.full_like(guide,-25)
        return output,output,output


class PartialDiffusionTests(unittest.TestCase):
    def test_partial_band_preserves_known_native_regions(self):
        model=EmptyDenoiser()
        image=np.zeros((1,128,128),np.uint8)
        guide=np.full((1,4,128,128),-4.,np.float32)
        guide[:,:,48:80,48:80]=4
        field,mean=predict(model,image,guide,[-1,-1],"guided",1,start_step=16,band_width=3.)
        np.testing.assert_allclose(model.levels,schedule("cpu")[[16,12,8,4]].numpy())
        self.assertTrue((field[:,:,60:68,60:68]>0).all())
        self.assertFalse((field[:,:,:16,:16]>0).any())
        np.testing.assert_array_equal(mean[:,:,60:68,60:68],np.ones((1,4,8,8)))
        np.testing.assert_array_equal(mean[:,:,:16,:16],np.zeros((1,4,16,16)))

    def test_full_empty_denoiser_can_remove_the_same_core(self):
        guide=np.full((1,4,128,128),-4.,np.float32)
        guide[:,:,48:80,48:80]=4
        field,_=predict(EmptyDenoiser(),np.zeros((1,128,128),np.uint8),guide,[-1,-1],"guided",1)
        self.assertFalse((field>0).any())
