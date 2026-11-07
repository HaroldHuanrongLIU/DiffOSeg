import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from reproduction import encoder_adaptation as adapter
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc


class AddIndex(nn.Module):
    def __init__(self,index):
        super().__init__()
        self.index=index

    def forward(self,x):
        return x+self.index


class SAMCapacityTests(unittest.TestCase):
    def test_split_keeps_exactly_four_terminal_blocks_at_both_depths(self):
        for depth in (12,32):
            toy=nn.Module()
            toy.patch_embed=nn.Identity()
            toy.pos_embed=None
            toy.blocks=nn.ModuleList(AddIndex(i+1) for i in range(depth))
            toy.neck=nn.Identity()
            x=torch.zeros(1,2,2,1)
            prefix=adapter.prefix_forward(toy,x)
            self.assertEqual(float(prefix[0,0,0,0]),sum(range(1,depth-3)))
            tail=adapter.EncoderTail(toy)
            self.assertEqual(len(tail.blocks),4)
            torch.testing.assert_close(tail(prefix),torch.full((1,1,2,2),float(sum(range(1,depth+1)))))

    def test_equal_shaped_embedding_cache_still_requires_the_matching_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"image_features.json").write_text(json.dumps({"source":rc.hash_file(prompt.__file__),
                "data":rc.DATA_HASH,"public_weights":prompt.WEIGHT,"image_size":512}))
            args=SimpleNamespace(run=root,feature_cache=None,backbone="vit_h",image_size=512)
            with self.assertRaises(ValueError):
                prompt.features_file(args)
