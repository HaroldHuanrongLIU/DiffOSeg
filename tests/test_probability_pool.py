import unittest
import numpy as np
from reproduction import probability_pool as pool


class ProbabilityPoolTest(unittest.TestCase):
    def test_pooling_distinguishes_one_large_opposing_score(self):
        g=np.array([1.2]);z=np.array([[-1.],[-1.],[-10.]])
        self.assertTrue(pool.pooling(g,z,1,1,.32,"probability")[0])
        self.assertFalse(pool.pooling(g,z,1,1,.32,"logits")[0])

    def test_agreement_and_tiny_logits_have_original_sign(self):
        g=np.array([-1e-12,0,1e-12])
        members=np.tile(g,(3,1))
        for weight in pool.WEIGHTS:
            np.testing.assert_array_equal(pool.pooling(g,members,3,.25,weight,"probability"),g>0)

    def test_nll_penalizes_overconfidence_at_wrong_labels(self):
        scores=np.array([10.,-10.]);labels=np.array([0.,1.]);w=np.array([2.,1.])
        self.assertEqual(pool.temperatures(scores,labels,w)[0],4.)
        self.assertEqual(pool.temperatures(scores,1-labels,w)[0],.25)
