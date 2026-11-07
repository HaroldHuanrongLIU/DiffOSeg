import unittest
import numpy as np
import torch
from reproduction.roi_prompt import geometry, View


class EchoImage(torch.nn.Module):
    def forward(self,embedding,guide,image):
        value=image.expand(-1,4,-1,-1)
        return value,value


class ROIPromptTests(unittest.TestCase):
    def test_empty_and_off_center_regions(self):
        logits=np.full((2,4,128,128),-1.,np.float32)
        logits[1,:,40:60,72:92]=1
        logits[1,:,4,4]=1
        center,side=geometry(logits,True)
        np.testing.assert_allclose(center,[[63.5,63.5],[81.5,49.5]])
        np.testing.assert_allclose(side,[128,58])

    def test_native_interior_and_exterior_across_rotations(self):
        view=View([[81.5,49.5]],[58],True,"cpu")
        image=torch.arange(128).float()[None,None,None,:].expand(1,1,128,128)/127
        guide=torch.full((1,4,128,128),-3.)
        for k in range(4):
            z,_=view.decode(EchoImage(),None,torch.rot90(guide,k,(-2,-1)),
                torch.rot90(image,k,(-2,-1)),torch.tensor([0]),k)
            z=torch.rot90(z,-k,(-2,-1))
            torch.testing.assert_close(z[0,0,48:52,78:82],image[0,0,48:52,78:82],atol=2e-6,rtol=0)
            self.assertEqual(z[0,0,0,0].item(),-3.)

    def test_global_control_is_exact_identity(self):
        view=View([[63.5,63.5]],[128],False,"cpu")
        image=torch.rand(1,1,128,128);guide=torch.zeros(1,4,128,128)
        z,_=view.decode(EchoImage(),None,guide,image,torch.tensor([0]),0)
        torch.testing.assert_close(z,image.expand_as(guide),rtol=0,atol=0)
