import unittest
import numpy as np
from reproduction import ribbon_preference as ribbon
from reproduction import rater_area


class RibbonTests(unittest.TestCase):
    def test_circle_reconstruction_and_outward_normal(self):
        y,x=np.indices((128,128)); m=(x-61)**2+(y-64)**2<=14**2
        d=rater_area.sdf(m)
        c,n=ribbon.contour(d)
        self.assertGreater(np.mean(np.sum((c-[61,64])*n,axis=1)),13)
        reconstructed=ribbon.polygon_sdf(c)>0
        self.assertGreater((m&reconstructed).sum()/(m|reconstructed).sum(),.97)

    def test_outside_target_has_positive_offset(self):
        y,x=np.indices((128,128))
        d=rater_area.sdf((x-65)**2+(y-64)**2<=12**2)
        t=rater_area.sdf((x-65)**2+(y-64)**2<=14**2)
        _,_,_,delta=ribbon.sample_ribbon(np.zeros_like(d),d,np.stack([t]*4))
        self.assertGreater(delta.mean(),1.7)
        self.assertLess(delta.mean(),2.3)
