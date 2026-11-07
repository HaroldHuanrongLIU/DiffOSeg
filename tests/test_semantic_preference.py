import unittest
import torch
from reproduction.semantic_preference import DinoSmall, TokenBlock


class SemanticTests(unittest.TestCase):
    def test_token_attention_is_independent_per_image(self):
        torch.manual_seed(4);m=TokenBlock().eval()
        x=torch.randn(1,7,384)
        with torch.no_grad():
            first=m(torch.cat((x,torch.ones_like(x)),0))[0]
            second=m(torch.cat((x,torch.zeros_like(x)),0))[0]
        torch.testing.assert_close(first,second)
    def test_partial_finetune_scope(self):
        model=DinoSmall()
        self.assertFalse(model.patch.weight.requires_grad)
        self.assertFalse(model.blocks[9].qkv.weight.requires_grad)
        self.assertTrue(model.blocks[10].qkv.weight.requires_grad)
        self.assertTrue(model.blocks[11].qkv.weight.requires_grad)
        self.assertFalse(model.norm.weight.requires_grad)
