import unittest
import torch
from torch import nn
from reproduction.early_attention import LowRankQv


class EarlyAttentionTests(unittest.TestCase):
    def test_zero_initialization_matches_all_original_components(self):
        torch.manual_seed(741)
        projection=nn.Linear(16,48)
        adapter=LowRankQv(projection,rank=4)
        x=torch.randn(2,3,4,16)
        torch.testing.assert_close(adapter(x),projection(x),rtol=0,atol=0)
        self.assertFalse(projection.weight.requires_grad)
        adapter(x).sum().backward()
        self.assertGreater(adapter.query_up.weight.grad.abs().sum().item(),0)
        self.assertGreater(adapter.value_up.weight.grad.abs().sum().item(),0)
        self.assertIsNone(projection.weight.grad)

    def test_key_projection_stays_identical_after_updates(self):
        torch.manual_seed(743)
        projection=nn.Linear(12,36)
        model=LowRankQv(projection,rank=3)
        with torch.no_grad():
            model.query_up.weight.normal_(std=.1)
            model.value_up.weight.normal_(std=.1)
        x=torch.randn(3,2,2,12)
        q,k,v=model(x).chunk(3,-1)
        original_q,original_k,original_v=projection(x).chunk(3,-1)
        torch.testing.assert_close(k,original_k,rtol=0,atol=0)
        self.assertGreater((q-original_q).abs().max().item(),0)
        self.assertGreater((v-original_v).abs().max().item(),0)
