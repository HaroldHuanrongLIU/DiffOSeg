import unittest
import torch
from reproduction.gated_hierarchy import SpatialSkipGate


class SpatialSkipGateTests(unittest.TestCase):
    def test_zero_path_initial_gradient_and_independent_samples(self):
        fine=torch.randn(2,32,16,16)
        mid=torch.randn(2,64,8,8)
        image=torch.randn(2,1,16,16)
        guide=torch.randn(2,4,16,16)
        rng=torch.random.get_rng_state()
        gate=SpatialSkipGate()
        torch.testing.assert_close(torch.random.get_rng_state(),rng)
        outputs=gate([fine,mid],guide,image)
        self.assertTrue(all(torch.count_nonzero(t)==0 for t in outputs))
        sum(t.square().sum()+t.sum() for t in outputs).backward()
        self.assertGreater(gate.projection.weight.grad.abs().sum().item(),0)
        with torch.no_grad():
            gate.projection.weight.fill_(.08)
        batch=gate([fine,mid],guide,image)
        single=gate([fine[:1],mid[:1]],guide[:1],image[:1])
        for a,b in zip(batch,single):
            torch.testing.assert_close(a[:1],b,rtol=1e-5,atol=1e-6)

    def test_uniform_bias_is_exactly_a_bounded_skip_weight(self):
        gate=SpatialSkipGate()
        with torch.no_grad():
            gate.projection.bias.copy_(torch.tensor([.4,-.7]))
        high=torch.ones(1,32,12,12)
        middle=2*torch.ones(1,64,6,6)
        out=gate([high,middle],torch.zeros(1,4,12,12),torch.zeros(1,1,12,12))
        torch.testing.assert_close(out[0],high*torch.tanh(torch.tensor(.4)))
        torch.testing.assert_close(out[1],middle*torch.tanh(torch.tensor(-.7)))
