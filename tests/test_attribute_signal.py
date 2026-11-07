import unittest
import numpy as np
from reproduction import attribute_signal as study


class AttributeSignalTest(unittest.TestCase):
    def test_no_assumed_reader_order_or_missing_value(self):
        entries=[{"texture":1,"malignancy":3},{"texture":5},{"texture":99}]
        x=study.attributes(entries)
        np.testing.assert_array_equal(x,study.attributes(entries[::-1]))
        np.testing.assert_allclose(x[:3],[.5,.5,2/3])
        np.testing.assert_allclose(x[3:6],[.5,0,0])
        np.testing.assert_allclose(x[-3:],[.5,0,1/3])

    def test_zero_rank_correction_keeps_every_pixel(self):
        rng=np.random.default_rng(7)
        q=rng.integers(-2,3,size=(4,16,16)).astype(np.float32)
        q[0]=0
        np.testing.assert_array_equal(study.adjust(q,np.zeros(4)),q>0)

    def test_regression_is_independent_of_held_patient_targets(self):
        rng=np.random.default_rng(104)
        x=rng.normal(size=(22,6))
        y=rng.normal(size=(22,4))
        fit=np.arange(18);held=np.arange(18,22);names=np.arange(22)//2
        a=study.regress(x,y,fit,held,names)
        y[held]=999.
        np.testing.assert_array_equal(a,study.regress(x,y,fit,held,names))
