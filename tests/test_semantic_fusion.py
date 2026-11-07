import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from reproduction import hierarchical_sam as hierarchy
from reproduction import semantic_fusion as fusion


class SemanticFusionTest(unittest.TestCase):
    def test_whitening_is_fitted_on_specified_patients_only(self):
        rng = np.random.default_rng(9128)
        data = rng.normal(size=(4,1,6,32,32)).astype(np.float32)
        mean, weight, values = fusion.fit_pca(data, [1,3], 6, 4)
        data[[0,2]] += 100
        again = fusion.fit_pca(data, [1,3], 6, 4)
        for before, after in zip((mean,weight,values), again):
            np.testing.assert_array_equal(before, after)
        rows = []
        for i in (1,3):
            for coords in (range(2,32,4), range(9,25,2)):
                rows.extend(data[i,0,:,y,x] for y in coords for x in coords)
        projected = (np.stack(rows)-mean) @ weight.T
        np.testing.assert_allclose(projected.mean(0), 0., atol=1e-6)
        np.testing.assert_allclose(np.cov(projected.T, bias=True), np.eye(4), atol=1e-6)

    def test_joint_cache_preserves_case_view_and_channels(self):
        a = np.arange(3*4*2).reshape(3,4,2,1,1)
        b = a[:,:,:1]+100
        joined = fusion.Joined(a,b)[[2,0]]
        np.testing.assert_array_equal(joined[:,:,:2], a[[2,0]])
        np.testing.assert_array_equal(joined[:,:,2:], b[[2,0]])

    def test_adapter_starts_at_e_and_has_a_learning_signal(self):
        class TestHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_s0 = nn.Identity()
                self.conv_s1 = nn.Identity()
        with patch.object(hierarchy, "open_public", return_value={}), patch.object(
                hierarchy, "decoder_heads", side_effect=lambda _: (nn.Identity(), TestHeads())):
            torch.manual_seed(131)
            base = hierarchy.HierarchicalRanks(None)
            after_base = torch.rand(8)
            torch.manual_seed(131)
            model = fusion.FusionRanks(None)
            torch.testing.assert_close(torch.rand(8), after_base, rtol=0, atol=0)
        packed = torch.randn(2,1024,3,3)
        extra = torch.randn(2,64,3,3)
        original, skips = base.spatial_features(packed,None,None)
        out, updated_skips = model.spatial_features(torch.cat((packed,extra),1),None,None)
        torch.testing.assert_close(out, original, atol=0, rtol=0)
        for a,b in zip(skips,updated_skips):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        out.square().mean().backward()
        self.assertGreater(model.adapter[-1].weight.grad.abs().sum().item(), 0.)
