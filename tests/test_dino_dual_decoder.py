import unittest
import torch
from reproduction import dino_dual_decoder as dual


class DualTokenTest(unittest.TestCase):
    def test_native_position_and_roi_fallback(self):
        x=torch.arange(32.).expand(32,32)
        maps=torch.stack((x,x+100)).reshape(1,2,1,32,32).expand(1,2,64,32,32)
        ctr=torch.tensor([[63.5,63.5]])
        ys,xs=torch.meshgrid(torch.arange(128.),torch.arange(128.),indexing="ij")
        grid=torch.stack((xs,ys),-1)[None]/63.5-1
        values=dual.sample_tokens(maps,ctr,grid,128,True)
        self.assertAlmostEqual(values[0,0,64,64].item(),15.625,places=4)
        self.assertAlmostEqual(values[0,64,64,64].item(),115.75,places=4)
        self.assertEqual(values[0,0,64,5].item(),values[0,64,64,5].item())
        whole=dual.sample_tokens(maps,ctr,grid,64,False)
        torch.testing.assert_close(whole[:,:64],whole[:,64:])
