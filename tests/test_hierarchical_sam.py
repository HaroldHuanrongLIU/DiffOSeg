import unittest

import torch

from reproduction.hierarchical_sam import pack, unpack


class HierarchicalSAMTests(unittest.TestCase):
    def test_spatial_pyramid_roundtrip_and_zero_skip_control(self):
        high = torch.arange(32*16*16).reshape(1,32,16,16).float()
        middle = torch.arange(64*8*8).reshape(1,64,8,8).float()
        coarse = torch.arange(256*4*4).reshape(1,256,4,4).float()
        packed = pack(high, middle, coarse)
        self.assertEqual(packed.shape, (1,1024,4,4))
        deep, restored = unpack(packed, True)
        torch.testing.assert_close(deep, coarse, rtol=0, atol=0)
        torch.testing.assert_close(restored[0], high, rtol=0, atol=0)
        torch.testing.assert_close(restored[1], middle, rtol=0, atol=0)
        deep_control, zero = unpack(packed, False)
        torch.testing.assert_close(deep_control, deep, rtol=0, atol=0)
        self.assertFalse(zero[0].any() or zero[1].any())

    def test_packed_samples_have_independent_gradient(self):
        high = torch.randn(2,32,16,16,requires_grad=True)
        middle = torch.randn(2,64,8,8,requires_grad=True)
        coarse = torch.randn(2,256,4,4,requires_grad=True)
        deep, maps = unpack(pack(high,middle,coarse), True)
        (deep[0].sum() + sum(x[0].sum() for x in maps)).backward()
        for x in (high,middle,coarse):
            self.assertTrue((x.grad[0] == 1).all())
            self.assertFalse(x.grad[1].any())
