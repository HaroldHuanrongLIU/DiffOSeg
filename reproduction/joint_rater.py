"""Exact four-rater count conditioning, shared translation and sample allocation."""
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import rank_contours as rc
from reproduction import vote_eval as v3


def centroid(weight):
    """Return x,y coordinates from a batch of positive spatial weights."""
    h,w=weight.shape[-2:]
    x=torch.arange(w,device=weight.device,dtype=weight.dtype)
    y=torch.arange(h,device=weight.device,dtype=weight.dtype)
    mass=weight.sum((-2,-1)).clamp_min(1e-8)
    return torch.stack(((weight.sum(-2)*x).sum(-1)/mass,(weight.sum(-1)*y).sum(-1)/mass),-1)


def translate(fields,delta):
    b,_,h,w=fields.shape
    affine=torch.eye(2,3,device=fields.device).expand(b,-1,-1).clone()
    affine[:,0,2]=-2*delta[:,0]/(w-1)
    affine[:,1,2]=-2*delta[:,1]/(h-1)
    grid=F.affine_grid(affine,fields.shape,align_corners=True)
    return F.grid_sample(fields,grid,align_corners=True,padding_mode="border")


def count_condition(expert_logits,vote_logits,mix,bits):
    # The common Bernoulli partition factor cancels within and across strata.
    odds=torch.einsum("bihw,ni->bnhw",expert_logits,bits)
    by_count=[odds[:,idx] for idx in ((bits.sum(1)==j) for j in range(5))]
    log_partition=torch.stack([group.logsumexp(1) for group in by_count],1)
    rank_law=log_partition.softmax(1)
    supports=vote_logits.sigmoid().sort(dim=1,descending=True).values
    vote_law=torch.cat((1-supports[:,:1],supports[:,:-1]-supports[:,1:],supports[:,-1:]),1)
    law=(1-mix)*vote_law+mix*rank_law
    expert=torch.zeros_like(expert_logits)
    for j in range(1,5):
        conditional=by_count[j].softmax(1)
        given_count=torch.einsum("bnhw,ni->bihw",conditional,bits[bits.sum(1)==j])
        expert=expert+law[:,j:j+1]*given_count
    cumulative=law[:,1:].flip([1]).cumsum(1).flip([1])
    return expert,cumulative


class JointDecoder(nn.Module):
    def __init__(self,origin):
        super().__init__()
        self.register_buffer("origin",torch.as_tensor(origin,dtype=torch.float32))
        self.register_buffer("bits",torch.tensor(list(product((0.,1.),repeat=4))))
        self.offset=nn.Parameter(torch.zeros(8))
        self.temperature=nn.Parameter(torch.zeros(2))
        self.mix_logit=nn.Parameter(torch.tensor(-1.0))
        self.shift_logit=nn.Parameter(torch.tensor(0.0))

    def forward(self,heads,*,location=True,joint=True):
        if location:
            center=centroid((2*heads[:,5:7]).sigmoid().mean(1))
            delta=self.origin-center
            shift=self.shift_logit.sigmoid()*delta*(-delta.square().sum(1,keepdim=True)/18).exp()
            heads=translate(heads,shift)
        if not joint:
            return heads[:,:4].sigmoid(),heads[:,4:].sigmoid()
        bias=2*self.offset.tanh()[None,:,None,None]
        inverse_temp=(0.7*self.temperature.tanh()).exp().repeat_interleave(4)
        calibrated=(heads*inverse_temp[None,:,None,None]+bias).clamp(-25,25)
        return count_condition(calibrated[:,:4],calibrated[:,4:],self.mix_logit.sigmoid(),self.bits)

    def coefficients(self):
        return {"offset":self.offset.detach().cpu().tolist(),"temperature":self.temperature.detach().cpu().tolist(),
                "mix_logit":self.mix_logit.item(),"shift_logit":self.shift_logit.item(),
                "origin":self.origin.cpu().tolist()}

    @classmethod
    def restore(cls,values,device):
        net=cls(values["origin"]).to(device)
        with torch.no_grad():
            for key,param in net.named_parameters():
                param.copy_(torch.as_tensor(values[key],device=device))
        return net.eval()


def transported_ranks(bank,ranks,expert,weight=0.2):
    if weight==0:
        return ranks
    k=len(bank)
    order=np.argsort(bank.sum((1,2)),kind="stable")
    groups=np.empty(k,dtype=np.int32)
    groups[order]=((np.arange(k)+0.5)*4/k).astype(np.int32)
    p=np.clip(expert,1e-5,1-1e-5)
    logits=np.log(p)-np.log1p(-p)
    relative=logits-logits.mean(0)
    style=1/(1+np.exp(-relative))
    priority=(1-weight)*(1-(ranks+0.5)/k)+weight*style[groups]
    best=np.argsort(-priority,axis=0,kind="stable")
    output=np.empty_like(ranks)
    np.put_along_axis(output,best,np.arange(k,dtype=np.uint8)[:,None,None],axis=0)
    return output


def preference(probabilities,prior):
    det=probabilities>0.5
    return 0.9*np.stack([rc.signed(m) for m in det])+0.1*np.stack([rc.signed(m) for m in prior])>0


def consensus(bank,ranks,expert,support,lock,order_weight):
    ranks=transported_ranks(bank,ranks,expert,order_weight)
    return v3.transport(ranks,bank.mean(0),support.mean(0),(support>0.5).mean(0),
                        lock["hardness"],lock["strength"])


def fit_layer(heads,masks,patient_names,origin,seed):
    net=JointDecoder(origin).cuda()
    values=torch.from_numpy(heads).cuda()
    truth=torch.from_numpy(masks).cuda().float()
    votes=(truth.sum(1,keepdim=True)>=torch.arange(1,5,device="cuda")[None,:,None,None]).float()
    target=torch.cat((truth,votes),1)
    _,idx,n=np.unique(patient_names,return_inverse=True,return_counts=True)
    sampling=torch.from_numpy(1/n[idx]).cuda().float()
    generator=torch.Generator(device="cuda").manual_seed(seed)
    optimizer=torch.optim.Adam(net.parameters(),lr=0.025)
    for step in range(320):
        ids=torch.multinomial(sampling,8,replacement=True,generator=generator)
        e,s=net(values[ids])
        p=torch.cat((e,s),1).clamp(1e-6,1-1e-6)
        # Sharpening makes the fit sensitive to the deployed half-probability decision.
        p=p.square()/(p.square()+(1-p).square())
        y=target[ids]
        dice=(2*(p*y).sum((-2,-1))+1)/((p+y).sum((-2,-1))+1)
        loss=1-dice.mean()+0.0005*(net.offset.square().mean()+net.temperature.square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2,error_if_nonfinite=True)
        optimizer.param_groups[0]["lr"]=0.025*(1-0.90*step/319)
        optimizer.step()
    return net.eval()
