import unittest

import numpy as np
import torch

from reproduction import joint_energy as energy
from reproduction.metrics import consensus_scores


class EnergyTrainingTests(unittest.TestCase):
    def test_training_forward_is_the_discrete_published_objective(self):
        rng=np.random.default_rng(9)
        ranks=np.argsort(rng.random((10,9,9)),axis=0).astype(np.uint8)
        freq=torch.tensor(rng.uniform(size=(1,9,9)),dtype=torch.float32,requires_grad=True)
        truth=torch.tensor(rng.random((1,4,9,9))>0.6)
        bank=ranks<np.floor(10*freq.detach().numpy()[0]+0.5)[None]
        expected=consensus_scores(truth[0].numpy(),bank)
        actual=energy.distribution_loss(freq,torch.tensor(ranks)[None],truth.float())
        self.assertAlmostEqual(actual.item(),expected["GED"]+3*(1-expected["Dice_soft"]),places=5)
        actual.backward()
        self.assertTrue(bool(freq.grad.isfinite().all()))
        self.assertGreater(freq.grad.abs().sum().item(),0)

    def test_empty_foreground_convention_and_reader_counts(self):
        freq=torch.zeros(1,4,4)
        ranks=torch.arange(10)[None,:,None,None].expand(1,-1,4,4)
        truth=torch.zeros(1,4,4,4)
        self.assertAlmostEqual(energy.distribution_loss(freq,ranks,truth).item(),0.,places=6)
