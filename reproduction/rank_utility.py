"""Finite four-reader order statistics and expected-Dice decisions."""
from math import comb

import numpy as np


def ordered_weights(areas):
    """Probability of each bank member being rank r in a uniform four-subset."""
    k=len(areas)
    if k<4:
        raise ValueError("Four distinct candidate draws are needed")
    order=np.argsort(areas,kind="stable")
    denom=comb(k,4)
    w=np.array([[comb(i,r)*comb(k-1-i,3-r)/denom for i in range(k)] for r in range(4)])
    # Rank is ambiguous between equal-area masks. Split the probability symmetrically.
    for a in np.unique(areas):
        ix=np.flatnonzero(areas[order]==a)
        w[:,ix]=w[:,ix].mean(1,keepdims=True)
    result=np.empty_like(w); result[:,order]=w
    return result


def ratio(intersection,denominator):
    return np.divide(2*intersection,denominator,out=np.ones(np.broadcast_shapes(intersection.shape,denominator.shape)),where=denominator>0)


def decisions(bank,probability,anchor,strengths):
    """Optimize over a nested image/shape-ranked family plus the anchor mask itself."""
    sizes=bank.sum((1,2))
    weights=ordered_weights(sizes)
    result={eta:anchor.copy() for eta in strengths}
    for r in range(4):
        support=bank.any(0)|anchor[r]
        if not support.any():
            continue
        draws=bank[:,support]
        base=anchor[r,support]
        score=0.8*probability[r,support]+0.2*(weights[r]@draws)
        order=np.argsort(-score,kind="stable")
        m=np.arange(len(order)+1)
        overlap=np.pad(draws[:,order].cumsum(1),((0,0),(1,0)))
        u=weights[r]@ratio(overlap,sizes[:,None]+m)
        a=np.pad(base[order].cumsum(),(1,0))
        u_img=ratio(a,base.sum()+m)
        anchor_reward=float(weights[r]@ratio((draws&base).sum(1),sizes+base.sum()))
        for eta in strengths:
            if eta==0: continue
            utility=eta*u+(1-eta)*u_img
            best=int(np.argmax(utility))
            if utility[best]>eta*anchor_reward+(1-eta)+1e-12:
                flat=np.zeros(len(order),bool); flat[order[:best]]=True
                result[eta][r,support]=flat
    return result
