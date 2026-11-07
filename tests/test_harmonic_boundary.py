import unittest
import numpy as np
from reproduction import harmonic_boundary as method


class HarmonicBoundaryTest(unittest.TestCase):
    def test_low_mode_reconstruction(self):
        theta = np.linspace(0, 2*np.pi, 43, endpoint=False)
        series = .2 + np.cos(method.ANGLES) - .35*np.sin(2*method.ANGLES) + .5*np.cos(3*method.ANGLES)
        np.testing.assert_allclose(method.smooth(series,theta),.6*(.2+np.cos(theta)-.35*np.sin(2*theta)), atol=1e-6)
        np.testing.assert_allclose(method.smooth(series,theta,0),.12,atol=1e-6)

    def test_zero_displacement_and_sector_area(self):
        rng=np.random.default_rng(3371)
        logit=rng.normal(size=(4,22,22))
        logit[0]=0.
        mask=logit>0
        sector,_=method.sectors(mask)
        radii=method.sector_radii(mask,sector)
        np.testing.assert_allclose(np.square(radii).sum(1)*np.pi/method.SECTORS,mask.sum((1,2)))
        np.testing.assert_array_equal(method.continuous_field(logit)>0,mask)

    def test_meta_holdouts_exclude_entire_patients(self):
        uids=[f"P{i//3}_image{i}" for i in range(30)]
        groups=method.case_partition(list(range(30)),uids,243)
        self.assertEqual(sorted(i for g in groups for i in g),list(range(30)))
        names=[{uids[i].split("_")[0] for i in g} for g in groups]
        self.assertFalse(names[0]&names[1] or names[0]&names[2] or names[1]&names[2])
