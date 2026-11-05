"""Predict nested reader-vote sets, then minimally modify a continuous sample count."""
import numpy as np
import torch
from torch import nn

PUBLISHED=(0.1,0.3,0.5,0.7,0.9)
HALF_VOTE=(0.125,0.375,0.625,0.875)
DIAGNOSTIC=tuple((np.arange(20)+0.5)/20)


def intervals(k,thresholds=PUBLISHED):
    levels=np.asarray(thresholds)
    bank=np.arange(k+1)/k
    decision=bank[:,None]>levels
    lo,hi=[],[]
    for c in range(5):
        ok=np.flatnonzero((decision==(c/4>levels)[None]).all(1))
        if not len(ok):
            raise ValueError(f"Unrepresentable vote class {c} with K={k}")
        lo.append(ok[0]);hi.append(ok[-1])
    return np.array(lo),np.array(hi)


def project(prior,counts,ranks,mode="full"):
    k=len(ranks)
    nominal=np.floor(k*prior+0.5)
    if mode=="direct_count":
        target=np.floor(k*counts/4+0.5)
    elif mode=="unconstrained":
        target=nominal
    else:
        bounds=intervals(k,HALF_VOTE if mode=="half_vote" else PUBLISHED)
        target=np.clip(nominal,bounds[0][counts],bounds[1][counts])
    return ranks<target[None]


class OrdinalStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([[0.,-0.5,0.4,0.,0.4]]*4))
        self.bias=nn.Parameter(torch.zeros(4))

    def forward(self,features):
        # A separate convex combination for each cumulative vote level.
        weights=self.weight.softmax(1)[None,:,:,None,None]
        logits=(weights*features).sum(2)+2.5*self.bias.tanh()[None,:,None,None]
        return logits.sort(1,descending=True).values

    def coefficients(self):
        return {"weight":self.weight.detach().cpu().tolist(),"bias":self.bias.detach().cpu().tolist()}

    @classmethod
    def restore(cls,record):
        net=cls()
        with torch.no_grad():
            net.weight.copy_(torch.tensor(record["weight"]))
            net.bias.copy_(torch.tensor(record["bias"]))
        return net.eval()


def fit(features,truth,patients,seed):
    model=OrdinalStack().cuda()
    x=torch.from_numpy(features).cuda()
    t=torch.from_numpy(truth).cuda().sum(1,keepdim=True)>=torch.arange(1,5,device="cuda")[None,:,None,None]
    y=t.float()
    _,inv,n=np.unique(patients,return_inverse=True,return_counts=True)
    w=torch.from_numpy(1/n[inv]).float().cuda()
    rng=torch.Generator(device="cuda").manual_seed(seed)
    optim=torch.optim.Adam(model.parameters(),lr=0.025)
    importance=x.new_tensor([1.,1.,2.,1.])[None,:]/5
    starting=model.weight.detach().clone()
    for step in range(480):
        ix=torch.multinomial(w,12,replacement=True,generator=rng)
        logits=model(x[ix])
        soft=(1.8*logits).sigmoid()
        # Hard forward evaluates the intended support decision. Soft probabilities provide gradients.
        p=(logits>0).float()+(soft-soft.detach())
        target=y[ix]
        den=(p+target).sum((-2,-1))
        d=torch.where(den>0,2*(p*target).sum((-2,-1))/den.clamp_min(1),1.)
        loss=1-(d*importance).sum(1).mean()+0.00035*((model.weight-starting).square().mean()+model.bias.square().mean())
        optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True)
        optim.param_groups[0]["lr"]=0.025*(1-0.92*step/479)
        optim.step()
    return model.cpu().eval()


@torch.inference_mode()
def counts(model,features):
    return np.concatenate([(model(block)>0).sum(1).numpy().astype(np.int32)
                           for block in torch.from_numpy(features).split(24)])
