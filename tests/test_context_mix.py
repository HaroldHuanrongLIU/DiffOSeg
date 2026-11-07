import unittest
import numpy as np
import torch

from reproduction.context_mix import ContextMix, background_layers, matched_partners


class ContextMixTests(unittest.TestCase):
    def test_rim_and_matte_are_protected(self):
        image = np.arange(32*32,dtype=np.uint16).reshape(32,32).astype(np.uint8)
        mask = np.zeros((4,32,32),np.uint8)
        mask[:,12:18,12:18] = 1
        alpha = np.zeros((32,32),np.uint8)
        alpha[2,3] = 255
        _, editable, profile = background_layers(image,mask,alpha)
        self.assertEqual(float(editable[12:18,9:21].max()),0)
        self.assertEqual(editable[2,3],0)
        self.assertEqual(editable[30,30],1)
        self.assertTrue(np.isfinite(profile).all())

    def test_sampling_changes_only_eligible_pixels_using_other_patients(self):
        rng = np.random.default_rng(37)
        raw = torch.from_numpy(rng.integers(0,200,size=(8,1,32,32),dtype=np.uint8))
        masks = np.zeros((8,4,32,32),np.uint8)
        masks[:,:,12:18,12:18]=1
        alpha = np.zeros((8,32,32),np.uint8)
        names = np.repeat(["A","B","C","D"],2)
        mixer = ContextMix(raw,masks,alpha,names,912)
        self.assertTrue(np.all(names[:,None] != names[mixer.partners.numpy()]))
        indices = torch.arange(8).repeat(4)
        torch.testing.assert_close(mixer(indices,1),raw[indices])
        before = torch.random.get_rng_state()
        mixed = mixer(indices,2)
        torch.testing.assert_close(before,torch.random.get_rng_state())
        protected = mixer.editable[indices] == 0
        torch.testing.assert_close(mixed[protected],raw[indices].float()[protected])
        self.assertTrue((mixed-raw[indices]).abs().max().item()>1)

    def test_single_patient_cannot_supply_a_partner(self):
        with self.assertRaises(ValueError):
            matched_partners(np.zeros((2,5)),["A","A"])
