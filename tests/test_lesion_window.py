import unittest
import torch
from reproduction.lesion_window import LesionWindow, anatomy_windows


class LesionWindowTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.full((2,1,32,32), 0.1)
        self.image[0,:,10:22,10:22] = 0.72
        self.guide = torch.full((2,4,32,32), -2.)
        self.guide[0,:,10:22,10:22] = 2.

    def test_quantiles_fallback_and_case_independence(self):
        channels, valid = anatomy_windows(self.image,self.guide)
        self.assertEqual(valid.tolist(), [True,False])
        self.assertTrue(torch.isfinite(channels).all())
        single, keep = anatomy_windows(self.image[:1],self.guide[:1])
        torch.testing.assert_close(single,channels[:1],rtol=0,atol=0)
        self.assertTrue(keep.item())
        torch.testing.assert_close(channels[0,:2,15,15], torch.ones(2))
        torch.testing.assert_close(channels[0,:2,2,2], torch.zeros(2), atol=1e-6,rtol=0)
        torch.testing.assert_close(channels[0,2,15,15], torch.tensor(.5), atol=2e-5,rtol=0)

    def test_zero_initialization_gradient_and_empty_support(self):
        adapter = LesionWindow()
        baseline = torch.randn(2,3,64,64)
        actual = adapter(baseline,self.image,self.guide)
        torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
        actual[0,:,:,20:28].mean().backward()
        self.assertTrue(torch.isfinite(adapter.gates.grad).all())
        self.assertGreater(adapter.gates.grad.abs().sum().item(),0)
        with torch.no_grad(): adapter.gates.fill_(2)
        changed = adapter(baseline,self.image,self.guide)
        torch.testing.assert_close(changed[1],baseline[1],rtol=0,atol=0)
        self.assertGreater((changed[0]-baseline[0]).abs().max().item(),0.1)
