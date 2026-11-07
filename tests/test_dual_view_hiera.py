import unittest
import torch
from reproduction.dual_view_hiera import fuse_registered, DualView
from reproduction.native_local import grids


class DualViewTests(unittest.TestCase):
    def test_native_interior_exterior_and_rotation(self):
        grid,back=grids(torch.tensor([[80.,64.]]),torch.eye(2)[None]*0.5,128)
        window=((1-back.abs().amax(-1))*8).clamp(0,1)[:,None]
        image=torch.arange(128).float()[None,None,None,:].expand(1,1,128,128)/127
        local=torch.nn.functional.grid_sample(image,grid,align_corners=True,padding_mode="border")
        glob=torch.full_like(image,-2)
        for k in range(4):
            mixed=fuse_registered(torch.rot90(glob,k,(-2,-1)),torch.rot90(local,k,(-2,-1)),back,window,k)
            native=torch.rot90(mixed,-k,(-2,-1))
            torch.testing.assert_close(native[:,:,:3,:],glob[:,:,:3,:],rtol=0,atol=0)
            torch.testing.assert_close(native[:,:,60:68,77:84],(glob+image)[:,:,60:68,77:84]/2,rtol=0,atol=2e-6)

    def test_full_control_uses_two_outputs_without_view_or_label_information(self):
        view=DualView.__new__(DualView);view.local=False
        values=[]
        def decoder(embedding,guide,image):
            values.append(image)
            return image+len(values),guide+len(values)
        image=torch.zeros(1,1,8,8);guide=torch.ones(1,4,8,8)
        a,b=view.decode(decoder,None,guide,image,torch.tensor([0]),0)
        self.assertEqual(len(values),2)
        torch.testing.assert_close(a,image+1.5)
        torch.testing.assert_close(b,guide+1.5)
