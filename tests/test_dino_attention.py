import unittest
import torch
from torch import nn
from reproduction import dino_attention as tail


class DinoTailTest(unittest.TestCase):
    def test_token_layout_roundtrip_preserves_class_and_patch_order(self):
        x=torch.arange(257*1024,dtype=torch.float32).reshape(1,257,1024)
        packed=tail.pack_tokens(x)
        self.assertEqual(tuple(packed.shape),(1,257,32,32))
        torch.testing.assert_close(packed[:,-1].reshape(1,-1),x[:,0])
        torch.testing.assert_close(tail.unpack_tokens(packed),x,rtol=0,atol=0)

    def test_zero_low_rank_adapter_preserves_public_projection(self):
        original=nn.Linear(12,12).requires_grad_(False)
        update=tail.LowRank(12,3)
        model=tail.AdaptedLinear(original,update);x=torch.randn(2,5,12)
        torch.testing.assert_close(model(x),original(x),rtol=0,atol=0)
        model(x).square().mean().backward()
        self.assertGreater(update.up.weight.grad.norm().item(),0)
        self.assertIsNone(original.weight.grad)
        self.assertEqual(sum(p.numel() for p in update.parameters()),2*12*3)
