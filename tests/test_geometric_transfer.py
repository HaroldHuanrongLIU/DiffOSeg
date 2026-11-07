import unittest
import numpy as np
from reproduction import geometric_transfer as motion
from reproduction.rater_area import sdf


class GeometricTransferTest(unittest.TestCase):
    def test_constant_and_missing_boundary(self):
        mask=np.zeros((32,32),bool);mask[8:24,8:24]=True
        surface=sdf(mask)
        expected=motion.boundary_motion(surface,np.full(surface.shape,1.25,np.float32))
        np.testing.assert_allclose(expected[10:22,8],1.25,atol=1e-5)
        np.testing.assert_array_equal(motion.boundary_motion(sdf(np.zeros_like(mask)),np.ones_like(surface)),0.)

    def test_unchanged_members_preserve_e_even_with_different_margins(self):
        mask=np.zeros((4,32,32),bool);mask[:,8:24,8:24]=True
        anchor=np.where(mask,.07,-.3).astype(np.float32)
        initial=np.roll(mask,1,axis=2)
        signed,direct,boundary=motion.geometric_fields(anchor,initial,np.stack([initial]*3))
        np.testing.assert_array_equal(signed>0,mask)
        np.testing.assert_array_equal(direct,0.)
        np.testing.assert_array_equal(boundary,0.)

    def test_two_of_three_expansions_and_band_protection(self):
        mask=np.zeros((4,40,40),bool);mask[:,12:28,12:28]=True
        larger=mask.copy();larger[:,11:29,11:29]=True
        anchor=np.where(mask,1.,-1.).astype(np.float32)
        signed,direct,boundary=motion.geometric_fields(anchor,mask,np.stack([larger,mask,larger]))
        self.assertGreater(boundary[0,11,20],.8)
        self.assertTrue((signed+boundary>0)[0,11,20])
        self.assertEqual(boundary[0,0,0],0.)
        self.assertEqual(direct[0,0,0],0.)
        _,_,one=motion.geometric_fields(anchor,mask,np.stack([mask,larger,mask]))
        np.testing.assert_array_equal(one,0.)
