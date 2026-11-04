import copy
import random
import pickle
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from datasets import lidc
from ddpm.models.diffusion_denoising import DiffusionModel, DenoisingModel
from ddpm.polyak import PolyakAverager
from ddpm.trainer import Trainer


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1)
        self.timesteps = []

    def forward(self, x, condition, feature_condition=None, timesteps=None):
        self.timesteps.extend(timesteps.tolist())
        return {"diffusion_out": self.conv(x).softmax(1)}


class ReproductionRegressions(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_ranked_masks_match_training_prompts_and_evaluation(self):
        empty = np.zeros((2, 2), dtype=np.uint8)
        one_a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        one_b = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        two = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        raw = [one_b, two, empty, one_a]
        expected = [empty, one_b, one_a, two]
        sample = {"sample": {"image": np.zeros((2, 2), dtype=np.uint8),
                              "masks": raw, "series_uid": "patient_0"}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.pkl"
            with path.open("wb") as stream:
                pickle.dump(sample, stream)
            train = lidc.LIDC_IDRI(2, str(path), input_size=2)
            evaluation = lidc.Test_LIDC(2, str(path), input_size=2)
            for expert, wanted in enumerate(expected):
                with patch("datasets.lidc.random.randint", return_value=expert):
                    _, selected, mask = train[0]
                self.assertEqual(selected, expert)
                np.testing.assert_array_equal(mask, wanted)
                np.testing.assert_array_equal(evaluation[0][1][str(expert)], wanted)

    def test_consensus_probability_survives_training_transforms(self):
        for module, channels in ((lidc, 1),):
            with self.subTest(dataset=module.__name__):
                image = np.zeros((channels, 128, 128), dtype=np.uint8)
                consensus = np.full((128, 128), 0.5, dtype=np.float32)
                _, encoded = module.transform(image, consensus)
                self.assertTrue(torch.any(encoded[1] == 0.5).item())
                torch.testing.assert_close(encoded.sum(0), torch.ones((128, 128)))

    def test_cosine_marginal_matches_forward_markov_chain(self):
        diffusion = DiffusionModel("cosine", 250, 2)
        torch.testing.assert_close(
            diffusion.cumalphas, torch.cumprod(diffusion.alphas, dim=0)
        )
        x0 = torch.zeros(1, 2, 4, 4)
        x0[:, 0] = 1
        t = torch.ones(1, dtype=torch.long)
        torch.testing.assert_close(
            diffusion.q_xt_given_x0(x0, t).probs,
            diffusion.q_xt_given_xtm1(x0, t).probs,
        )

    def test_training_images_follow_paper_unit_range(self):
        for module, channels in ((lidc, 1),):
            with self.subTest(dataset=module.__name__):
                image = np.full((channels, 128, 128), 255, dtype=np.uint8)
                image, _ = module.transform(image, np.zeros((128, 128), dtype=np.float32))
                self.assertGreaterEqual(image.min().item(), 0.0)
                self.assertLessEqual(image.max().item(), 1.0)

    def test_train_step_samples_multiple_diffusion_timesteps(self):
        model = DenoisingModel(DiffusionModel("cosine", 250, 2), TinyUNet(), "datasets.lidc")
        polyak = PolyakAverager(model, copy.deepcopy(model))
        trainer = Trainer(
            polyak, torch.optim.Adam(model.parameters(), lr=1e-4), None,
            torch.ones(2), None, None,
            {"stage": 1, "feature_cond_encoder": {"train": False}},
        )
        target = torch.nn.functional.one_hot(torch.randint(2, (16, 4, 4)), 2)
        batch = (torch.zeros(16, 1, 4, 4), target.permute(0, 3, 1, 2).float())
        with patch("ddpm.trainer.idist.device", return_value=torch.device("cpu")), patch("ddpm.trainer.np.save"):
            trainer.train_step(None, batch)
            trainer.train_step(None, batch)
        self.assertGreater(len(set(model.unet.timesteps)), 1)
        self.assertTrue(all(1 <= t <= 250 for t in model.unet.timesteps))


if __name__ == "__main__":
    unittest.main()
