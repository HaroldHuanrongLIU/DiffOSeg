import unittest

import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F

from reproduction.backprojection import sampling_operator, solve_occupancy


class BackprojectionTests(unittest.TestCase):
    def test_forward_matches_torch_with_zero_padding(self):
        rng = np.random.default_rng(713)
        image = rng.normal(size=(5, 7))
        grid = rng.uniform(-1.4, 1.4, size=(3, 6, 2))
        operator = sampling_operator(grid, image.shape)
        expected = F.grid_sample(
            torch.from_numpy(image)[None, None], torch.from_numpy(grid)[None],
            align_corners=True, padding_mode="zeros"
        )[0, 0].numpy()
        np.testing.assert_allclose(
            (operator @ image.ravel()).reshape(3, 6), expected, atol=1e-12
        )
        probe = rng.normal(size=18)
        self.assertAlmostEqual(float(probe @ (operator @ image.ravel())),
                               float((operator.T @ probe) @ image.ravel()), places=12)

    def test_identity_recovers_native_binary_mask(self):
        mask = np.zeros((5, 7))
        mask[1:4, 2:5] = 1
        operator = sparse.eye(35, format="csr")
        rhs = np.stack((mask.ravel(), 1-mask.ravel()), axis=1)
        soft, hard = solve_occupancy(operator, rhs, np.full_like(rhs, 0.5), 0, mask.shape)
        np.testing.assert_allclose(soft, rhs)
        np.testing.assert_array_equal(hard, rhs.astype(bool))

    def test_binary_updates_decrease_the_actual_forward_energy(self):
        rng = np.random.default_rng(421)
        grid = rng.uniform(-1, 1, size=(20, 20, 2))
        operator = sampling_operator(grid, (5, 7))
        target = (rng.random((35, 2)) > 0.6).astype(float)
        measurements = operator @ target
        normal = operator.T @ operator
        rhs = operator.T @ measurements
        prior = np.clip(target * 0.65 + rng.uniform(0, 0.35, target.shape), 0, 1)
        strength = 0.1
        soft, hard = solve_occupancy(normal, rhs, prior, strength, (5, 7))
        anchor = strength * np.asarray(normal.sum(axis=1)).ravel()
        def energy(value):
            return np.square(operator @ value - measurements).sum() + (
                anchor[:, None] * np.square(value-prior)).sum()
        self.assertLessEqual(energy(hard), energy(soft >= 0.5) + 1e-9)
