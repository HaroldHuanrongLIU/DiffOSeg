"""Expected preference loss over small perturbations of annotation-area ranks."""
from itertools import permutations
import math

import torch
import torch.nn.functional as F


def assignment_weights(masks):
    radius=(masks.float().sum((-2,-1))/math.pi).sqrt()
    jitter=radius.new_tensor(list(permutations((-0.75,-0.25,0.25,0.75))))
    order=torch.argsort(radius[:,None]+jitter[None],dim=2,stable=True)
    frequencies=F.one_hot(order,num_classes=4).float().mean(1)
    identity=torch.eye(4,device=masks.device,dtype=frequencies.dtype)
    return 0.5*identity[None]+0.5*frequencies


def expected_mask_loss(logits, target, assignment):
    z=logits.float().flatten(2)
    y=target.float().flatten(2)
    p=z.sigmoid()
    transpose=y.transpose(1,2)
    intersection=torch.bmm(p,transpose)
    dice=(2*intersection+1)/(p.sum(2)[:,:,None]+y.sum(2)[:,None,:]+1)
    softplus=F.softplus(z)
    weighted_bce=(softplus.mean(2)[:,:,None]
                  +torch.bmm(4*softplus-z,transpose)/z.shape[2]
                  -4*torch.bmm(z,y.square().transpose(1,2))/z.shape[2])
    return ((1-dice+weighted_bce)*assignment).sum((1,2)).mean()/4
