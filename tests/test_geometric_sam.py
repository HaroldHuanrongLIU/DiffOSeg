import unittest

import torch

from reproduction import geometric_sam, native_local


class Echo(torch.nn.Module):
    def forward(self, prefix, guide, image, rotation):
        return image.expand_as(guide), image.expand_as(guide) * 2


class GeometricSamTests(unittest.TestCase):
    def test_known_translation_and_native_loss_gradient(self):
        image = torch.arange(128.0)[None, None, None, :].expand(1, 1, 128, 128).clone().requires_grad_()
        guide = torch.full((1, 4, 128, 128), -7.0)
        matrix = torch.eye(2)[None]
        grid, back = native_local.grids(torch.tensor([[67.5, 63.5]]), matrix, 128)
        logits, raw = geometric_sam.native_predictions(Echo(), image, guide, 0, grid, back)
        torch.testing.assert_close(logits[:, :, :, :4], guide[:, :, :, :4])
        torch.testing.assert_close(logits[0, 0, 32:96, 4:126], image[0, 0, 32:96, 4:126], atol=3e-5, rtol=0)
        torch.testing.assert_close(raw[0, 0, 64, 54], torch.tensor(108.0), atol=3e-5, rtol=0)
        logits[0, 0, 64, 54].backward()
        self.assertAlmostEqual(image.grad[0, 0, 64, 54].item(), 1.0, places=4)
        self.assertAlmostEqual(image.grad.sum().item(), 1.0, places=5)

    def test_separate_rng_and_oblique_gradient(self):
        before = torch.random.get_rng_state()
        generator = torch.Generator().manual_seed(716)
        grid, back = geometric_sam.random_grids(3, generator)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        second = geometric_sam.random_grids(3, torch.Generator().manual_seed(716))
        torch.testing.assert_close(grid, second[0])
        self.assertGreater((grid[0] - grid[1]).abs().max().item(), 0.1)
        image = torch.ones(3, 1, 128, 128, requires_grad=True)
        guide = torch.zeros(3, 4, 128, 128)
        logits, _ = geometric_sam.native_predictions(Echo(), image, guide, 1, grid, back)
        torch.testing.assert_close(logits[:, :, 64, 64], torch.ones(3, 4))
        logits.mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.sum().item(), 0.5)
