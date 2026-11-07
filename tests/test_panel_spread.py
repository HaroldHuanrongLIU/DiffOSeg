import unittest
import numpy as np
from reproduction import panel_spread as panel


class PanelSpreadTest(unittest.TestCase):
    def test_width_counts_pairwise_disagreements_and_grid_perimeter(self):
        m=np.zeros((4,12,12),bool)
        for r in range(4):m[r,3:6,3:6+r]=1
        perimeter=np.mean([2*(3+3+r) for r in range(4)])
        self.assertEqual(panel.mean_perimeter(m),perimeter)
        expected=sum((m[a]^m[b]).sum() for a in range(4) for b in range(a+1,4))/(6*perimeter)
        self.assertEqual(panel.spread(m,perimeter),expected)

    def test_identical_and_crossing_panels(self):
        m=np.zeros((4,18,18),bool);m[:,5:12,5:12]=True
        q=np.where(m,2.,-2.).astype(np.float32)
        candidates=panel.actions(q)
        for c in candidates:np.testing.assert_array_equal(c,m)
        q[0,10:14,11:14]=1.;q[3,4:6,4:6]=2.
        np.testing.assert_array_equal(panel.actions(q)[0],q>0)
