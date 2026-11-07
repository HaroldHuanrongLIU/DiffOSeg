import unittest
import numpy as np
import torch
from reproduction import dino_local_probe as probe


class DinoLocalProbeTest(unittest.TestCase):
    def test_centers_only_depend_on_current_two_teachers(self):
        u=np.full((2,4,128,128),-1.,np.float32)
        u[0,:2,38:43,20:25]=1.
        expected=np.array([[22,40],[63.5,63.5]])
        np.testing.assert_array_equal(probe.centers_from_guides(u,u),expected)

    def test_projected_tokens_follow_native_coordinates(self):
        y,x=np.indices((32,32));tokens=(x+2*y)[None].astype(np.float16)
        center=np.array([49.5,72.5])
        for side in (64,128):
            xx=np.array([5.,10.5,22.75]);yy=np.array([2.,18.25,14.5])
            native_x=center[0]+(xx-15.5)*side/32
            native_y=center[1]+(yy-15.5)*side/32
            result=probe.tokens_at(tokens,native_x,native_y,center,side)
            np.testing.assert_allclose(result[:,0],xx+2*yy,rtol=0,atol=1e-6)

    def test_crops_share_per_slice_normalization_and_center(self):
        image=torch.arange(128.).repeat(128,1)[None,None]
        ctr=torch.tensor([[63.5,63.5]])
        g=probe.view_image(image,ctr,128);l=probe.view_image(image,ctr,64)
        self.assertEqual(g.shape,(1,3,448,448))
        torch.testing.assert_close(g[:,:,223:225,223:225].mean((2,3)),l[:,:,223:225,223:225].mean((2,3)))
        self.assertGreater(g[0,0,224,400].item(),l[0,0,224,400].item())
