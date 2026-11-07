import unittest
from types import SimpleNamespace
import numpy as np
from reproduction import slice_prototypes as study


class SlicePrototypesTest(unittest.TestCase):
    def test_seed_excludes_disagreement_and_boundary(self):
        mask=np.zeros((4,16,16),bool);mask[:,4:12,4:12]=True
        u=np.where(mask,2.,-2.)
        s=u.copy();s[:,7,7]=-2.
        descriptor=np.stack([np.ones((16,16)),np.zeros((16,16))])
        descriptor[1,7,7]=100.
        counts,moments=study.seed_stats(descriptor,u,s,mask)
        self.assertTrue((counts>0).all())
        np.testing.assert_allclose(moments[:,:,0,0],1)
        np.testing.assert_allclose(moments[:,:,0,1],0)

    def test_population_patient_weighting_and_missing_local_tissue(self):
        counts=np.ones((4,2),int)
        def make(value):
            moment=np.full((4,2,2,6),value)
            return SimpleNamespace(seed_moments=moment,seed_count=counts)
        pop,w=study.pooled([make(0),make(0),make(1)],["a","a","b"])
        np.testing.assert_allclose(w,[7/24,7/24,5/12])
        np.testing.assert_allclose(pop,5/12)
        local=np.ones_like(pop)
        np.testing.assert_allclose(study.adaptive_moments(pop,local,np.zeros_like(counts)),pop)
        np.testing.assert_allclose(study.adaptive_moments(pop,local,16*counts),(pop+local)/2)

    def test_two_tissue_likelihood_prefers_matching_descriptor(self):
        model=np.zeros((2,2,8));model[:,0]=[[0]*8,[2]*8]
        model[:,1]=model[:,0]**2+1.
        x=np.array([[0]*8,[2]*8])
        out=study.distances(x,x**2,model)
        np.testing.assert_allclose(out[:,2],[4,-4])
        np.testing.assert_allclose(out[:,5],[4,-4])
