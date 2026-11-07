import unittest
from types import SimpleNamespace
import numpy as np
from reproduction import subset_witness as study


class WitnessTest(unittest.TestCase):
    def test_exchange_symmetry_and_unanimous_block(self):
        panel=SimpleNamespace(points=np.array([0,1,3]),group=np.array([0,0,1]),count=np.array([2,1]),initial=np.array([1.,0.]))
        a=np.array([[-2.,-3.],[0.,2.]])
        b=np.array([[-1.,1.],[0.,4.]])
        left=study.witness_features(a,b,panel);right=study.witness_features(b,a,panel)
        for x,y in zip(left,right):np.testing.assert_allclose(x,y)
        np.testing.assert_array_equal(left[2],[False,True])
        b[0,1]=-1.
        np.testing.assert_array_equal(study.witness_features(a,b,panel)[2],[True,True])
