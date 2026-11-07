import unittest
import numpy as np
from reproduction import event_routing as routing


class RoutingTest(unittest.TestCase):
    def test_budget_ceil_ties_and_empty(self):
        points=np.array([3,4,5,7,8,9])
        scores=np.array([0,1,1,0,1,0.])
        self.assertEqual(np.flatnonzero(routing.budget_mask(points,scores,(4,4))).tolist(),[4,5])
        self.assertFalse(routing.budget_mask(points[:0],scores[:0],(4,4)).any())

    def test_opposition_requires_both_teachers_inside_budget(self):
        ref=np.zeros((4,2,2),bool);first=np.ones_like(ref);second=first.copy()
        second[0,0,0]=False
        budget=np.array([[1,0],[0,0]],bool)
        out=routing.apply(ref,first,second,budget)
        self.assertEqual(out.sum(),3);self.assertFalse(out[0,0,0])
