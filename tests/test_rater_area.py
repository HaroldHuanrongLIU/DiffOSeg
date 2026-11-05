import unittest

import numpy as np

from reproduction import rater_area as area


class AreaDecisionTests(unittest.TestCase):
    def test_target_area_adds_only_best_remaining_pixels(self):
        rng=np.random.default_rng(9)
        value=rng.normal(size=(4,12,12))
        old=value>0.3
        radius=np.sqrt((old.sum((1,2))+4)/np.pi)
        new=area.apply_radii(value,old,radius,1,np.zeros(4))
        for j in range(4):
            self.assertEqual(int(new[j].sum()),int(old[j].sum())+4)
            self.assertTrue(np.all(new[j][old[j]]))
            self.assertGreater(value[j][new[j]&~old[j]].min(),value[j][~new[j]].max())
        np.testing.assert_equal(area.apply_radii(value,old,radius,0,np.zeros(4)),old)

    def test_radiomics_are_finite_for_empty_and_constant_images(self):
        x=np.ones((128,128),np.uint8)*10
        m=np.zeros_like(x,dtype=bool)
        a=area.describe(x,m)
        m[56:72,56:72]=True
        b=area.describe(x,m)
        self.assertEqual(a.shape,b.shape)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
