import unittest

import numpy as np
import torch
from torch.nn import functional as F

from reproduction import vote_contours as model
from reproduction import vote_eval as study


class VoteTransportTests(unittest.TestCase):
    def test_non_nested_experts_produce_empirical_vote_targets(self):
        masks=torch.tensor([[[[1,0,1]],[[0,1,1]],[[1,0,1]],[[0,0,1]]]])
        expected=torch.tensor([[[[1,1,1]],[[1,0,1]],[[0,0,1]],[[0,0,1]]]],dtype=torch.float32)
        torch.testing.assert_close(model.votes(masks),expected)

    def test_view_geometry_matches_original_pixel_coordinates(self):
        image=torch.arange(128,dtype=torch.float32).expand(1,1,128,-1)
        raw,_=model.sampling_grid(1,"fullfield","cpu",False)
        zoom,_=model.sampling_grid(1,"zoom","cpu",False)
        full=F.grid_sample(image,raw,align_corners=True)
        local=F.grid_sample(image,zoom,align_corners=True)
        torch.testing.assert_close(full,image,atol=2e-5,rtol=0)
        torch.testing.assert_close(model.combine_views(full,local),image,atol=2e-5,rtol=0)

    def test_original_bank_returns_exactly_at_zero_strength(self):
        rng=np.random.default_rng(42)
        bank=rng.random((10,24,24))>.75
        bank[0]=False
        ranks=study.bank_ranks(bank)
        freq=bank.mean(0)
        np.testing.assert_array_equal(study.transport(ranks,freq,1-freq,freq,.2,0),bank)

    def test_requested_counts_hold_even_when_distances_tie(self):
        bank=np.zeros((10,9,9),bool)
        bank[:5,2:6,2:6]=True
        levels=np.tile(np.array([0,.25,.5,.75,1,0,.25,.5,1]),(9,1))
        seg=study.transport(study.bank_ranks(bank),bank.mean(0),levels,levels,1,1)
        expected=np.tile(np.array([0,3,5,8,10,0,3,5,10]),(9,1))
        np.testing.assert_array_equal(seg.sum(0),expected)


if __name__=="__main__":
    unittest.main()
