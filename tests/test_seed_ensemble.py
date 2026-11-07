import unittest
import numpy as np
from reproduction.seed_ensemble import mean_saved_logits


class SeedEnsembleTests(unittest.TestCase):
    def test_average_uses_float32_and_preserves_each_case(self):
        stored=[np.array([64000.,-1.],dtype=np.float16),np.array([64000.,2.],dtype=np.float16),
                np.array([-64000.,5.],dtype=np.float16)]
        expected=np.array([64000./3,2.],dtype=np.float32)
        actual=mean_saved_logits(stored)
        self.assertEqual(actual.dtype,np.float32)
        np.testing.assert_array_equal(actual,expected)

    def test_rejects_incomplete_or_asymmetric_member_precision(self):
        sample=np.array([1.,2.],dtype=np.float16)
        with self.assertRaises(ValueError):mean_saved_logits([sample,sample])
        with self.assertRaises(ValueError):mean_saved_logits([sample,sample,sample.astype(np.float32)])
