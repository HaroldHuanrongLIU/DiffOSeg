import itertools
import unittest
import numpy as np
from reproduction.boundary_graph import CAPACITY_SCALE, cut, distance, neighbors


class BoundaryGraphTests(unittest.TestCase):
    def test_every_small_graph_cut_matches_exhaustive_minimum(self):
        rng=np.random.default_rng(327)
        left,right=neighbors((2,3))
        for _ in range(20):
            prior=rng.integers(0,2,(2,3)).astype(bool)
            editable=np.array([0,1,4,5])
            contact=np.isin(left,editable)|np.isin(right,editable)
            a,b=left[contact],right[contact]
            unary=rng.integers(-10,11,len(editable))/4
            weight=rng.integers(0,12,len(a))/4
            got=cut(prior,editable,unary,a,b,weight)
            def energy(arr):
                v=arr.ravel()[editable]
                u=np.where(v,np.maximum(-unary,0),np.maximum(unary,0))
                return np.rint(CAPACITY_SCALE*u).sum()+np.rint(CAPACITY_SCALE*weight[arr.ravel()[a]!=arr.ravel()[b]]).sum()
            alternatives=[]
            for assignment in itertools.product((False,True),repeat=len(editable)):
                m=prior.copy();m.ravel()[editable]=assignment
                alternatives.append(energy(m))
            self.assertEqual(energy(got),min(alternatives))
            np.testing.assert_array_equal(got.ravel()[[2,3]],prior.ravel()[[2,3]])

    def test_distance_empty_full_and_native_edge(self):
        m=np.zeros((16,16),bool)
        np.testing.assert_array_equal(distance(m),np.full(m.shape,-8))
        np.testing.assert_array_equal(distance(~m),np.full(m.shape,8))
        m[5:11,5:11]=1
        d=distance(m)
        self.assertEqual(d[5,7],.5)
        self.assertEqual(d[4,7],-.5)
        self.assertGreater(d[7,7],1.5)
        self.assertLess(d[0,0],-3)
