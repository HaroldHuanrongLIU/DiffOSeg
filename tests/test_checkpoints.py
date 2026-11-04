import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from reproduction.run import read_checkpoint, sample_experts


class CheckpointAndExpertTests(unittest.TestCase):
    def test_checkpoint_rejects_different_source_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.pt"
            torch.save({"source_id": "old", "config_id": "config", "step": 2}, path)
            with self.assertRaises(ValueError):
                read_checkpoint(path, "new", "config")
            with self.assertRaises(ValueError):
                read_checkpoint(path, "old", "different")
            self.assertEqual(read_checkpoint(path, "old", "config")["step"], 2)

    def test_batched_experts_keep_identity_across_chunk_boundaries(self):
        class EchoExpert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))

            def forward(self, noise, condition):
                foreground = condition["txt"].float()[:, None, None].expand(-1, 2, 2) / 3
                return {"diffusion_out": torch.stack((1 - foreground, foreground), dim=1)}

        predictions = sample_experts(EchoExpert(), torch.zeros(1, 2, 2), 5, 7)
        self.assertEqual(predictions.shape, (4, 5, 2, 2))
        for expert in range(4):
            np.testing.assert_allclose(predictions[expert], expert / 3, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
