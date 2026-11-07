import unittest
import numpy as np
from reproduction.component_utility import Components, apply, flip_utility, partition


class ComponentUtilityTests(unittest.TestCase):
    def test_four_neighbors_and_flip_direction_define_groups(self):
        u=-np.ones((5,6));s=u.copy()
        u[1,1:4]=1;u[2,4]=1
        reference=u>0;reference[1,3]=False
        pts,g=partition(u,s,reference,False)
        table=dict(zip(pts,g))
        self.assertEqual(len(np.unique(g)),3)
        self.assertEqual(table[7],table[8])
        self.assertNotEqual(table[8],table[9])
        self.assertNotEqual(table[9],table[16])

    def test_closed_form_gain_equals_flipping_each_mask_directly(self):
        ref=np.zeros((4,5),bool);ref[1,1:4]=1;ref[2,1:3]=1
        truth=np.zeros_like(ref);truth[1,2:5]=1;truth[2,2:4]=1
        points=np.array([6,7,9,13]);group=np.array([0,0,1,1])
        comp=Components(points,group,np.array([2,2]),np.array([1.,0.]),None,None)
        def dice(mask):
            size=mask.sum()+truth.sum()
            return 2*np.count_nonzero(mask&truth)/size if size else 1.
        for j,value in enumerate(flip_utility(comp,ref,truth)):
            new=ref.copy();new.ravel()[points[group==j]]^=True
            self.assertAlmostEqual(value,(dice(new)-dice(ref))*(ref.sum()+1)/2,places=6)

    def test_only_registered_disagreements_can_change(self):
        ref=np.ones((2,3),bool)
        gains=np.full(ref.shape,-np.inf);gains[0,0]=.2
        np.testing.assert_array_equal(apply(ref,gains,None),ref)
        expected=ref.copy();expected[0,0]=False
        np.testing.assert_array_equal(apply(ref,gains,.075),expected)
