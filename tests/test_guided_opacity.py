import unittest
import numpy as np
from reproduction import guided_opacity as guided


class OpacityTest(unittest.TestCase):
    def test_constant_ct_removes_only_the_intensity_contribution(self):
        image=np.full((32,32),.4)
        prob=np.zeros_like(image);prob[:,:16]=.9;prob[:,16:]=.1
        h,s=guided.guided(image,prob)
        np.testing.assert_allclose(h,s,atol=1e-12)
        self.assertGreater(s[16,16],prob[16,16])

    def test_aligned_contrast_recovers_the_direction_of_the_edge(self):
        image=np.zeros((32,32));image[:,16:]=1
        prob=.1+.8*image
        h,s=guided.guided(image,prob)
        self.assertGreater(h[16,16],s[16,16])
        self.assertLess(h[16,15],s[16,15])
        self.assertLess(abs(h[16,16]-prob[16,16]),abs(s[16,16]-prob[16,16]))

    def test_zero_strength_is_an_exact_replay(self):
        z=np.full((4,128,128),-2.,np.float32);z[:,55:75,54:71]=1.e-8
        x=np.zeros((128,128),np.uint8)
        for bank in guided.proposals(x,z).values():np.testing.assert_array_equal(bank[0],np.packbits(z>0,axis=-1))
