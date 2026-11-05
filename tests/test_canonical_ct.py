import unittest

import numpy as np
import torch

from reproduction import canonical_ct as ct
from reproduction import canonical_study as study


class CanonicalTests(unittest.TestCase):
    def test_empty_fallback_and_forward_inverse_coordinate_pair(self):
        w=torch.zeros(2,128,128)
        w[1,48:67,35:74]=1
        center,matrix=ct.geometry(w,22.)
        torch.testing.assert_close(center[0],torch.tensor([64.5,64.5]))
        torch.testing.assert_close(matrix[0],torch.eye(2))
        torch.testing.assert_close(center[1],torch.tensor([54.,57.]))
        eig=torch.linalg.eigvalsh(matrix)
        self.assertTrue(bool((eig>=0.45-1e-5).all() and (eig<=1.8+1e-5).all()))
        out=ct.image_grid(center,matrix,128,128,False)
        original=(out.reshape(2,-1,2)-center[:,None])@torch.linalg.inv(matrix).transpose(1,2)+64.5
        torch.testing.assert_close(original.reshape_as(out),ct.pixel_grid(128,128,"cpu").expand_as(out),atol=3e-5,rtol=0)

    def test_bilinear_restore_and_native_fallback(self):
        yy,xx=torch.meshgrid(torch.arange(128),torch.arange(128),indexing="ij")
        source=(xx+2*yy).float()[None,None]
        center=torch.tensor([[54.,67.]])
        matrix=torch.tensor([[[0.8,0.1],[0.1,1.1]]])
        grid=ct.image_grid(center,matrix,128,128,False)
        aligned=(grid[...,0]+2*grid[...,1])[:,None]
        back=ct.pull_back(aligned,source,center,matrix)
        torch.testing.assert_close(back,source,atol=0.00015,rtol=0)

    def test_panel_is_convex_and_disabled_limit_is_exact(self):
        rng=np.random.default_rng(701)
        s=rng.random((4,11,9))
        h,n=study.panel(s)
        q=rng.random((11,9))
        np.testing.assert_equal(study.shrink(q,h,n,None),q)
        v=study.shrink(q,h,n,4.)
        self.assertTrue(np.all(v>=np.minimum(h,q)-1e-7))
        self.assertTrue(np.all(v<=np.maximum(h,q)+1e-7))
