import unittest
import numpy as np
from reproduction.canonical_guide import calibrate, codes, reconstruct


class CanonicalGuideTests(unittest.TestCase):
    def test_distance_codes_and_exact_archived_sign(self):
        image=np.arange(256,dtype=np.uint8).reshape(16,16)
        masks=np.zeros((4,16,16),bool)
        masks[1,4:11,4:11]=True
        masks[2]=True
        d,c=codes(image,masks)
        self.assertTrue((d[0]==8).all())
        self.assertTrue((d[2]==17).all())
        self.assertEqual(d[1,4,4],9)
        self.assertEqual(d[1,3,4],0)
        self.assertGreater(d[1,7,7],d[1,4,4])
        self.assertTrue(((c>=0)&(c<6)).all())
        base,fine=calibrate([image],[masks],[np.where(masks,3.,-2.)],["a"])
        for table in (base,fine):
            guide=reconstruct(image,masks,table)
            np.testing.assert_array_equal(guide>0,masks)
            np.testing.assert_allclose(guide[masks],3.,rtol=1e-5)
            np.testing.assert_allclose(guide[~masks],-2.,rtol=1e-5)

    def test_patient_equal_profile_and_repeatable_calibration(self):
        image=np.zeros((12,12),np.uint8)
        mask=np.zeros((4,12,12),bool);mask[:,3:8,3:8]=True
        a=np.where(mask,1.,-1.);b=np.where(mask,7.,-7.)
        once=calibrate([image]*2,[mask]*2,[a,b],["patientA","patientB"])
        repeated=calibrate([image]*3,[mask]*3,[a,a,b],["patientA","patientA","patientB"])
        for x,y in zip(once,repeated):
            np.testing.assert_array_equal(x,y)
            np.testing.assert_allclose(np.abs(reconstruct(image,mask,x)),3.,atol=1e-6)
