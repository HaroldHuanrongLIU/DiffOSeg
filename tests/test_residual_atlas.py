import copy
import unittest
import numpy as np
from reproduction import residual_atlas as atlas


class AtlasTest(unittest.TestCase):
    def test_canonical_origin_and_axis_roundtrip(self):
        m=np.zeros((4,128,128),bool);m[:,52:63,41:58]=True
        forward,inverse=atlas.atlas_maps(m)
        np.testing.assert_allclose(forward[16,16],[49,57],atol=1e-5)
        np.testing.assert_allclose(inverse[57,49],[16,16],atol=1e-5)

    def test_basis_recovers_low_rank_signal(self):
        x=np.arange(19.)
        data=np.column_stack((2*x,-x,3*x+1))
        mean,modes=atlas.fit_basis(data,np.ones(len(data)))
        self.assertEqual(len(modes),1)
        np.testing.assert_allclose(mean+(data-mean)@modes.T@modes,data,atol=1e-8)

    def test_held_truth_cannot_change_deployable_outputs(self):
        rng=np.random.default_rng(811)
        def case():
            return atlas.Example(rng.normal(size=(8,8,8)),rng.normal(size=(2,8,8)),rng.normal(size=12),
                rng.normal(size=(4,33,33)),rng.normal(size=(4,12,12)),np.ones((4,12,12),bool),
                rng.uniform(0,32,size=(12,12,2)).astype(np.float32))
        train=[case() for _ in range(10)];held=case()
        a,_=atlas.fit_predict(train,[held],list(range(10)))
        other=copy.copy(held);other.target=np.zeros_like(held.target)+5
        b,_=atlas.fit_predict(train,[other],list(range(10)))
        for method in ("mean","shape","ct"):
            np.testing.assert_array_equal(a[method],b[method])
