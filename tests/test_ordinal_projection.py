import unittest

import numpy as np
import torch

from reproduction import ordinal_projection as op


class ProjectionTests(unittest.TestCase):
    def test_all_panel_classes_minimally_project_without_threshold_leak(self):
        for k in (10,30,50):
            ranks=np.arange(k)[:,None,None]
            for c in range(5):
                for target in (0.,0.13,0.33,0.51,0.69,0.87,1.):
                    seg=op.project(np.array([[target]]),np.array([[c]]),ranks)
                    mask=seg.mean(0).item()
                    bits=mask>np.array(op.PUBLISHED)
                    np.testing.assert_array_equal(bits,c/4>np.array(op.PUBLISHED))
                    allowed=[n for n in range(k+1) if np.array_equal(n/k>np.array(op.PUBLISHED),bits)]
                    self.assertAlmostEqual(abs(seg.sum()-k*target),min(abs(n-k*target) for n in allowed))
    def test_missing_class_is_an_error(self):
        with self.assertRaises(ValueError): op.intervals(2)
    def test_ordered_supports_and_restoration(self):
        model=op.OrdinalStack()
        x=torch.randn(2,4,5,7,8)
        s=model(x)
        self.assertTrue(bool((s[:,:-1]>=s[:,1:]).all()))
        torch.testing.assert_close(s,op.OrdinalStack.restore(model.coefficients())(x))
