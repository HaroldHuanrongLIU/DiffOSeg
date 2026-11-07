import unittest
import numpy as np
from reproduction import ct_response as response


class CTResponseTest(unittest.TestCase):
    def test_opposite_intensity_perturbations_preserve_constant_regions(self):
        flat=np.full((24,24),93,np.uint8)
        for v in response.perturb(flat):np.testing.assert_array_equal(v,flat)
        image=flat.copy();image[12,12]=146
        blur,sharp=response.perturb(image)
        self.assertLess(blur[12,12],image[12,12])
        self.assertGreater(sharp[12,12],image[12,12])
        np.testing.assert_array_equal(blur.astype(int)+sharp,2*image.astype(int))

    def test_response_retains_sign_and_separates_rank(self):
        base=np.full((4,8,8),-2.,np.float32);base[:,3:5,3:5]=2.
        bigger=base.copy();bigger[0,2:6,2:6]=2.
        small=base.copy();small[0,3,3]=-2.
        vals=response.response(base,bigger,small).reshape(2,3,4)
        self.assertGreater(vals[0,0,0],0)
        self.assertLess(vals[1,0,0],0)
        np.testing.assert_array_equal(vals[:,:,1:],0)
