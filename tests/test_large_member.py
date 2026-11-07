import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from reproduction import hierarchical_sam as image
from reproduction import rank_contours as rc


class LargeMemberTests(unittest.TestCase):
    def test_cache_binds_to_frozen_image_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=SimpleNamespace(run=root,public=root)
            np.save(root/"features.npy",np.zeros(2,np.float16))
            (root/"runtime.sha256").write_text("synthetic")
            record={"source":rc.hash_file(Path(image.__file__)),"payload":rc.hash_file(root/"features.npy"),
                    "dataset":rc.DATA_HASH,"weights":image.BACKBONES["sam2.1-large"].digest,
                    "runtime":rc.hash_file(root/"runtime.sha256")}
            (root/"features.json").write_text(json.dumps(record))
            image.feature_cache(args,backbone="sam2.1-large")
            with self.assertRaises(ValueError):image.feature_cache(args)

    def test_original_default_keeps_its_public_identity(self):
        tiny=image.BACKBONES["sam2.1-tiny"]
        self.assertEqual(tiny.digest,image.WEIGHT)
        self.assertEqual(tiny.filename,"sam2.1_hiera_tiny.pt")
        self.assertEqual((tiny.width,tiny.stages,tiny.windows),(96,(1,2,7,2),(8,4,14,7)))
