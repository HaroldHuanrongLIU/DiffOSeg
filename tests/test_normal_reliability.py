import unittest

import torch

from reproduction import joint_energy as energy
from reproduction import normal_reliability as method


class ReliabilityTests(unittest.TestCase):
    def test_neutral_preference_gate_preserves_matching_logits(self):
        net=method.PreferenceGate()
        logits=torch.linspace(-4,4,100).reshape(1,4,5,5)
        old=logits.sigmoid()
        for contrast in (-1.,1.):
            q=net(old,logits,torch.ones_like(old[:,:1]),torch.full_like(old[:,:1],contrast))
            torch.testing.assert_close(q,old)

    def test_frequency_gate_reduces_to_frozen_readout_for_agreeing_inputs(self):
        ref=energy.FrequencyReadout()
        model=method.PopulationGate(ref)
        logits=torch.linspace(-3,3,100).reshape(1,4,5,5)
        s=logits.sigmoid()
        prior=s.mean(1)
        q=model(s,s,logits,torch.zeros_like(s[:,:1]),prior)
        torch.testing.assert_close(q,ref(s,prior))
        q.sum().backward()
        self.assertTrue(all(p.grad is None for p in ref.parameters()))
        self.assertIsNotNone(model.temp.grad)
        clone=method.restore(method.PopulationGate(energy.FrequencyReadout()),method.export(model),"cpu")
        torch.testing.assert_close(clone(s,s,logits,torch.zeros_like(s[:,:1]),prior),q.detach())
