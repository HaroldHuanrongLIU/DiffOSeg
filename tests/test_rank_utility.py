from itertools import combinations
import unittest

import numpy as np

from reproduction import rank_utility as method


class OrderDecisionTests(unittest.TestCase):
    def test_order_weights_equal_explicit_four_reader_panels(self):
        areas=np.array([13,4,6,9,40,20])
        weights=method.ordered_weights(areas)
        counted=np.zeros_like(weights)
        panels=list(combinations(range(len(areas)),4))
        for panel in panels:
            for rank,idx in enumerate(sorted(panel,key=lambda j:areas[j])):
                counted[rank,idx]+=1/len(panels)
        np.testing.assert_allclose(weights,counted,atol=1e-12)
        np.testing.assert_allclose(weights.mean(0),np.full(len(areas),1/len(areas)))

    def test_ties_do_not_favor_bank_indices(self):
        w=method.ordered_weights(np.array([4,7,7,8,9]))
        np.testing.assert_allclose(w[:,1],w[:,2])
        np.testing.assert_allclose(w.sum(1),np.ones(4))

    def test_prefix_risk_matches_direct_enumeration(self):
        rng=np.random.default_rng(831)
        bank=rng.random((6,3,4))>.62
        image=rng.uniform(size=(4,3,4))
        anchor=image>.45
        eta=.65
        chosen=method.decisions(bank,image,anchor,[0,eta])
        np.testing.assert_equal(chosen[0],anchor)
        w=method.ordered_weights(bank.sum((1,2)))
        def dice(a,b):
            n=a.sum()+b.sum()
            return 2*(a&b).sum()/n if n else 1.
        for r in range(4):
            union=bank.any(0)|anchor[r]
            s=.8*image[r,union]+.2*(w[r]@bank[:,union])
            order=np.argsort(-s,kind="stable")
            def score(m):
                return (1-eta)*dice(m,anchor[r])+eta*sum(z*dice(m,v) for z,v in zip(w[r],bank))
            values=[score(anchor[r])]
            for k in range(len(order)+1):
                p=np.zeros(len(order),bool);p[order[:k]]=True
                m=np.zeros_like(union);m[union]=p;values.append(score(m))
            self.assertAlmostEqual(score(chosen[eta][r]),max(values),places=10)
