import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from reproduction import hierarchical_sam as hierarchy
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc


class MaskedEcho(torch.nn.Module):
    def forward(self, embedding, guide, image):
        result = embedding * (guide + image)
        return result, result


class MirrorViewTests(unittest.TestCase):
    def test_restore_asymmetric_native_pixels_for_all_rotations(self):
        images = np.zeros((2,128,128),np.uint8)
        images[1,31:48,69:83] = 153
        images[0,90:110,13:28] = 87
        ids = [1,0]
        guide = np.zeros((2,4,128,128),np.float32)
        guide[0,:,17:37,68:78] = 2
        guide[1,:,84:94,14:25] = 3
        # Translate the CT by (-1,-1) on the original lattice, with border replication.
        coord = np.minimum(np.arange(128)+1,127)
        shifted = images[ids][:,coord][:,:,coord].astype(np.float32)/255
        patterns = np.zeros((2,1,128,128),np.float32)
        patterns[1,:,22:43,63:76] = 1.5
        patterns[0,:,79:99,10:23] = 0.5
        expected = patterns[ids] * (guide+shifted[:,None])
        for reflected in (False,True):
            oriented = patterns[:,:,:,::-1] if reflected else patterns
            cache = np.stack([np.rot90(oriented,k,axes=(-2,-1)) for k in range(4)],axis=1).copy()
            with patch.object(torch.Tensor,"cuda",lambda x:x), patch.object(torch.cuda,"is_bf16_supported",return_value=False):
                got = prompt.predict(MaskedEcho(),cache,ids,images,guide,reflected=reflected)
            np.testing.assert_allclose(got,expected,atol=1e-6,rtol=0)

    def test_cache_rejects_opposite_orientation_including_older_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.save(root/"features.npy",np.zeros(2,np.float16))
            (root/"runtime.sha256").write_text("synthetic")
            args=SimpleNamespace(run=root,public=root)
            meta={"source":rc.hash_file(Path(hierarchy.__file__)),"payload":rc.hash_file(root/"features.npy"),
                  "dataset":rc.DATA_HASH,"weights":hierarchy.WEIGHT,"runtime":rc.hash_file(root/"runtime.sha256")}
            (root/"features.json").write_text(json.dumps(meta))
            hierarchy.feature_cache(args)
            with self.assertRaises(ValueError):
                hierarchy.feature_cache(args,reflected=True)
            (root/"features.json").write_text(json.dumps({**meta,"horizontal_reflection":True}))
            hierarchy.feature_cache(args,reflected=True)
            with self.assertRaises(ValueError):
                hierarchy.feature_cache(args)
