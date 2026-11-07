import unittest
import torch
from reproduction.encoder_adaptation import CTInputAdapter


class CTInputAdapterTests(unittest.TestCase):
    def test_initial_residual_is_zero_but_projection_has_gradient(self):
        torch.manual_seed(81)
        model = CTInputAdapter()
        image = torch.rand(2, 1, 128, 128)
        residual = model(image)
        self.assertEqual(tuple(residual.shape), (2, 32, 32, 768))
        self.assertEqual(torch.count_nonzero(residual).item(), 0)
        residual.sum().backward()
        self.assertGreater(model.network[-1].weight.grad.abs().sum().item(), 0)

    def test_native_features_are_independent_of_other_cases(self):
        torch.manual_seed(83)
        model = CTInputAdapter().eval()
        with torch.no_grad():
            model.network[-1].weight.normal_(std=0.01)
            image = torch.rand(2, 1, 128, 128)
            alone = model(image[:1])
            batch = model(torch.cat((image[:1], torch.zeros_like(image[1:])), dim=0))
        torch.testing.assert_close(alone, batch[:1], atol=2e-5, rtol=2e-5)
