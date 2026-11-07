import unittest
import numpy as np
from reproduction import spread_gain as gain


class GainTest(unittest.TestCase):
    def test_optimum_at_prescribed_scale_and_low_benefit_abstention(self):
        t=np.log(1.5)
        self.assertEqual(gain.panel.SCALES[int(gain.action(np.array([4*t,2.])))],1.5)
        self.assertEqual(gain.action(np.array([.01,-.5])),0)

    def test_no_held_coefficients_used_by_predictor(self):
        rng=np.random.default_rng(933);x=rng.normal(size=(24,7));c=rng.normal(size=(24,2))
        ti=np.arange(18);hi=np.arange(18,24);w=np.ones(18)
        expected=gain.regress(x,c,ti,hi,w)
        c[hi]=80
        np.testing.assert_array_equal(expected,gain.regress(x,c,ti,hi,w))
