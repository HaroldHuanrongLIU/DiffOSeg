import unittest
import numpy as np
from reproduction import adaptive_pool as routing


class RoutingTest(unittest.TestCase):
    def test_case_aggregation_is_distinct_from_per_rank(self):
        x=np.zeros((3,4,2));gain=np.zeros((2,4,9))
        gain[:,0,2]=1.;gain[:,1:,2]=-.5
        result=routing.fit_predict(x,gain,["a","b"])
        np.testing.assert_array_equal(result["fixed_rank"][2],[2,0,0,0])
        np.testing.assert_array_equal(result["case"][2],[0,0,0,0])

    def test_threshold_and_tie_priority(self):
        p=np.zeros((2,9));p[0,[2,3]]=1.;p[1,4]=.05
        np.testing.assert_array_equal(routing.select(p),[2,0])
