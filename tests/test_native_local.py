import unittest

import numpy as np
import torch
from torch.nn import functional as F

from reproduction import native_local as native
from reproduction.native_study import moment_project


class NativeGridTests(unittest.TestCase):
    def test_affine_native_queries_invert_rotation_and_reflection(self):
        c=torch.tensor([[51.,62.],[65.,68.]])
        a=native.matrix_for(torch.tensor([.47,.7]),torch.tensor([.52,-.7]),torch.tensor([-1.,1.]))
        src,back=native.grids(c,a,144)
        v=(src+1)*63.5
        # Read native coordinate ramps on the local grid, then pull them into native space.
        out=F.grid_sample(v.permute(0,3,1,2),back,align_corners=True)
        native_xy=native.coordinates(128,"cpu").permute(2,0,1)[None].expand_as(out)
        inside=(back.abs().amax(-1)<.97)[:,None].expand_as(out)
        torch.testing.assert_close(out[inside],native_xy[inside],rtol=0,atol=5e-5)

    def test_locator_image_only_and_eccentric_peak(self):
        z=torch.zeros(1,2,128,128)
        z[0,0,64,30]=18
        c=native.detect(z)
        self.assertLess(torch.norm(c[0]-torch.tensor([30.,64.])).item(),0.1)

    def test_four_noninterchangeable_area_masks_share_average_landmark(self):
        m=torch.zeros(1,4,128,128)
        m[0,0,60,60]=1;m[0,1,60,64]=1;m[0,2,64,62]=1;m[0,3,64,66]=1
        torch.testing.assert_close(native.rater_center(m),torch.tensor([[63.,62.]]))

    def test_moment_control_is_identity_when_disabled(self):
        z=np.random.RandomState(9).normal(size=(2,4,128,128)).astype("float32")
        np.testing.assert_array_equal(moment_project(z,np.full((2,2),64.5),0),z)
