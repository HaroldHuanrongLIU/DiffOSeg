import unittest
import numpy as np
from reproduction import ridge_profile as ridge


class RidgeProfileTest(unittest.TestCase):
    def test_longitudinal_bright_strip_is_distinct_from_tangent(self):
        image=np.zeros((32,32),np.float32)
        image[15:18,12:28]=1
        coord=np.array([[16],[12]])
        axial=ridge.directional_values(image,coord,np.array([[0],[1]]))
        transverse=ridge.directional_values(image,coord,np.array([[1],[0]]))
        np.testing.assert_array_equal(axial[:,0::3],[[1,1,1]])
        np.testing.assert_array_equal(axial[:,1::3],[[1,1,1]])
        np.testing.assert_array_equal(transverse[:,0::3],[[0,0,0]])

    def test_band_coordinates_and_zero_residual_reproduce_source(self):
        image=np.zeros((128,128),np.uint8)
        score=np.full((4,128,128),-4,np.float32)
        score[:,57:70,60:71]=1
        mask=score>0
        case=ridge.example(image,score,score,mask)
        self.assertEqual(case.features["normal"].shape,case.features["tangent"].shape)
        restored=np.unpackbits(case.reference,axis=-1)
        restored.ravel()[case.index]=case.margin>0
        np.testing.assert_array_equal(restored,mask)
        np.testing.assert_allclose(case.residual,mask.ravel()[case.index]-(case.margin+.5))
