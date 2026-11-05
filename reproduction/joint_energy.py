"""Supervise the predictive frequency through discrete diffusion-bank samples."""
import numpy as np
import torch
from torch import nn


class FrequencyReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset=nn.Parameter(torch.zeros(4))
        self.temperature=nn.Parameter(torch.zeros(4))
        self.hard_logit=nn.Parameter(torch.tensor(-0.7))
        self.strength_logit=nn.Parameter(torch.tensor(1.0))

    def forward(self,support,prior):
        logit=torch.logit(support.clamp(1e-6,1-1e-6))
        logit=logit*(1.3*self.temperature.tanh()).exp()[None,:,None,None]+1.5*self.offset.tanh()[None,:,None,None]
        soft=logit.sigmoid()
        smooth=(3*logit).sigmoid()
        hard=(logit>0).to(soft.dtype)+(smooth-smooth.detach())
        h=self.hard_logit.sigmoid()
        strength=0.5+0.5*self.strength_logit.sigmoid()
        return (1-strength)*prior+strength*((1-h)*soft+h*hard).mean(1)

    def coefficients(self):
        return {name:p.detach().cpu().tolist() for name,p in self.named_parameters()}

    @classmethod
    def restore(cls,record,device):
        model=cls().to(device)
        with torch.no_grad():
            for name,p in model.named_parameters():
                p.copy_(torch.as_tensor(record[name],device=device))
        return model.eval()


def discrete_bank(frequency,ranks):
    k=ranks.shape[1]
    soft=((frequency[:,None]-(ranks.float()+0.5)/k)/0.025).sigmoid()
    count=(k*frequency+0.5).floor()
    return (ranks<count[:,None]).float()+(soft-soft.detach())


def mean_iou(a,b):
    a,b=a.flatten(2),b.flatten(2)
    overlap=a@b.transpose(1,2)
    union=a.sum(-1)[:,:,None]+b.sum(-1)[:,None,:]-overlap
    return (overlap/union.clamp_min(1)).mean((1,2))


def distribution_loss(frequency,ranks,truth):
    samples=discrete_bank(frequency,ranks)
    ged=mean_iou(samples,samples)+mean_iou(truth,truth)-2*mean_iou(truth,samples)
    thresholds=frequency.new_tensor((0.1,0.3,0.5,0.7,0.9))[None,:,None,None]
    marginal=samples.mean(1,keepdim=True)
    smooth=((frequency[:,None]-thresholds)/0.035).sigmoid()
    pred=(marginal>thresholds).float()+(smooth-smooth.detach())
    target=(truth.mean(1,keepdim=True)>thresholds).float()
    den=(pred+target).sum((-2,-1))
    dice=torch.where(den>0,2*(pred*target).sum((-2,-1))/den.clamp_min(1),1.)
    return ged.mean()+3*(1-dice.mean())


def fit(support,proposals,ranks,truth,names,seed):
    model=FrequencyReadout().cuda()
    x,p,r,y=[torch.as_tensor(values,device="cuda") for values in (support,proposals,ranks,truth)]
    _,idx,n=np.unique(names,return_inverse=True,return_counts=True)
    weights=torch.from_numpy(1/n[idx]).cuda().float()
    rng=torch.Generator(device="cuda").manual_seed(seed)
    opt=torch.optim.Adam(model.parameters(),lr=0.035)
    for step in range(320):
        ids=torch.multinomial(weights,5,replacement=True,generator=rng)
        frequency=model(x[ids],p[ids])
        loss=distribution_loss(frequency,r[ids],y[ids].float())+0.0003*(
            model.offset.square().mean()+model.temperature.square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True)
        opt.param_groups[0]["lr"]=0.035*(1-0.9*step/319)
        opt.step()
    return model.eval()


@torch.inference_mode()
def generate(model,support,base,ranks):
    support=torch.from_numpy(support[None])
    prior=torch.from_numpy(base[None].astype(np.float32))
    freq=model(support,prior)[0].numpy()
    return ranks < np.floor(len(ranks)*freq+0.5)[None]
