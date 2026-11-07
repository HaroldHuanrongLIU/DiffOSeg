import unittest
import numpy as np
import torch
from reproduction.registered_seed_field import cpu_refinement, cpu_signed


class StableRefinementTests(unittest.TestCase):
    def test_exact_euclidean_geometry_and_empty_region(self):
        mask = np.zeros((9,9), dtype=bool)
        mask[2,2] = True
        signed = cpu_signed(mask)
        self.assertEqual(signed[2,2], .5)
        self.assertEqual(signed[5,6], -4.5)
        self.assertTrue((cpu_signed(np.zeros_like(mask))==-8).all())
        self.assertTrue((cpu_signed(np.ones_like(mask))==8).all())

    def test_eight_view_cpu_prediction_preserves_image_orientation(self):
        class Add(torch.nn.Module):
            def forward(self,image,guide):
                return image + guide
        image=np.arange(2*8*8,dtype=np.uint8).reshape(2,8,8)
        guide=np.ones((2,1,8,8),np.float32)
        out=cpu_refinement(Add().cpu(),image,guide)
        expected=(image[:,None].astype(np.float32)/255+guide)/.65
        np.testing.assert_allclose(out,expected,atol=5e-7)
        np.testing.assert_array_equal(out,cpu_refinement(Add().cpu(),image,guide))
