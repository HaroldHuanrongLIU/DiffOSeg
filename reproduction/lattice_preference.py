"""Native-grid preference specialization with fixed-area shape gradients."""
import argparse
import copy
import json
import math
from pathlib import Path
import socket
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/lattice_protocol.md"
STEPS,BATCH=6000,18


class MassProjection(torch.autograd.Function):
    """Implicit derivative of a sigmoid whose intercept fixes total foreground mass."""
    @staticmethod
    def forward(ctx,logits,areas,tau):
        z=logits.flatten(2)
        a=areas.float().clamp(1,z.shape[2]-1)
        lo=z.amin(2)-12*tau;hi=z.amax(2)+12*tau
        for _ in range(16):
            mid=(lo+hi)/2
            big=((z-mid[:,:,None])/tau).sigmoid().sum(2)>a
            lo=torch.where(big,mid,lo);hi=torch.where(big,hi,mid)
        p=((z-((lo+hi)/2)[:,:,None])/tau).sigmoid().reshape_as(logits)
        ctx.save_for_backward(p);ctx.tau=tau
        return p

    @staticmethod
    def backward(ctx,grad):
        (p,)=ctx.saved_tensors
        w=p*(1-p)/ctx.tau
        avg=(grad*w).sum((-2,-1),keepdim=True)/w.sum((-2,-1),keepdim=True).clamp_min(1e-8)
        return w*(grad-avg),None,None


def fixed_area_mask(logits,areas):
    z=logits.flatten(2);a=areas.long().clamp(0,z.shape[2])
    maximum=int(a.max().item())
    hard=torch.zeros_like(z)
    if maximum:
        idx=z.topk(maximum,dim=2).indices
        value=(torch.arange(maximum,device=z.device)<a[:,:,None]).to(z.dtype)
        hard.scatter_(2,idx,value)
    return hard.reshape_as(logits)


class LatticeNet(nn.Module):
    def __init__(self,parent=None):
        super().__init__()
        self.base=vote.build() if parent is None else parent
        self.detail=nn.Sequential(nn.Conv2d(24+5,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,16,1))
        self.query=nn.Sequential(nn.Linear(160+4+4,96),nn.SiLU(),nn.Linear(96,4*17))
        self.area_head=nn.Sequential(nn.LayerNorm(160+4*24+8),nn.Linear(160+4*24+8,128),nn.SiLU(),nn.Linear(128,4))
        for head in (self.query[-1],self.area_head[-1]):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,image,area_jitter=None):
        coords=native.coordinates(128,image.device).permute(2,0,1)[None]*(2/127)-1
        x=self.base.stem(torch.cat((image*2-1,coords.expand(len(image),-1,-1,-1)),1).contiguous(memory_format=torch.channels_last))
        skips=[x]
        for layer in self.base.down:x=layer(x);skips.append(x)
        glob=x.float().mean((-2,-1))
        for layer,skip in zip(self.base.up,reversed(skips[:-1])):
            x=layer(torch.cat((F.interpolate(x,skip.shape[-2:],mode="bilinear",align_corners=False),skip),1))
        z=self.base.head(x).float()
        prob=z[:,:4].sigmoid()
        r=(prob.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
        spread=(4*prob*(1-prob)).sum((-2,-1)).sqrt()/12
        query=self.query(torch.cat((glob,r.detach()/12,spread.detach()),1)).reshape(-1,4,17)
        padded=F.pad(image,(1,1,1,1),mode="replicate")
        gx=padded[:,:,1:-1,2:]-padded[:,:,1:-1,:-2]
        gy=padded[:,:,2:,1:-1]-padded[:,:,:-2,1:-1]
        high=image-F.avg_pool2d(image,5,1,2,count_include_pad=False)
        stem=self.detail(torch.cat((x,2*image-1,4*gx,4*gy,5*high,prob.mean(1,keepdim=True)),1))
        residual=torch.einsum("brd,bdhw->brhw",query[:,:,:16].tanh(),stem)/4+query[:,:,16,None,None]
        logits=z[:,:4]+residual.float()
        cutoff=0. if area_jitter is None else area_jitter
        p=(2*(logits.detach()-cutoff)).sigmoid()
        radius=(p.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
        band=4*p*(1-p)
        pooled=torch.einsum("brp,bcp->brc",band.flatten(2),x.float().flatten(2))/band.sum((-2,-1)).clamp_min(1)[:,:,None]
        area_input=torch.cat((glob,pooled.flatten(1),radius/12,band.sum((-2,-1)).sqrt()/12),1)
        estimate=(radius+3*self.area_head(area_input).float().tanh()).clamp_min(0)
        return logits,z[:,8:],estimate


def load_net(path):
    net=LatticeNet().cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return net


def aligned_view(im,center,angle,flip,scale,identity):
    grid,back=align.transform_grids(center,scale,angle,flip)
    # Use true nearest-neighbor samples for native orthogonal lattice symmetries.
    if identity:
        inp=F.grid_sample(im,grid,mode="nearest",padding_mode="border",align_corners=True)
    else:
        inp=F.grid_sample(im,grid,mode="bicubic",padding_mode="border",align_corners=True).clamp(0,1)
    return inp,back


def train(args):
    cv2.setNumThreads(1);torch.set_num_threads(3)
    ims,labels,_,uid=native.inputs(args.data);outer,inner,valid=native.splits(uid,args.fold)
    mode="inner" if args.mode=="smoke" else args.mode
    ids=inner if mode=="inner" else outer["train_indices"]
    root=args.align/f"fold-{args.fold}-{mode}-registered";parent=json.loads((root/"fit.json").read_text())
    if json.loads((root/"plan.json").read_text())["train_indices"]!=ids or parent["checkpoint"]!=rc.hash_file(root/"final.pt"):
        raise ValueError("Registered parent provenance")
    steps=STEPS
    if args.mode=="smoke":ids,steps=ids[:48],5
    seed=351721+args.fold
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=True
    args.out.mkdir(parents=True,exist_ok=False)
    shift=parent["applied_shift"]
    image=align.registered_ct(torch.from_numpy(ims[ids,None]).cuda(),*shift)
    y=torch.from_numpy(labels[ids]).cuda().float()
    votes=vote.votes(y)
    center=native.rater_center(y)
    center=torch.where((center-64.5).abs().amax(1,keepdim=True)>4,center,torch.full_like(center,64.5))
    _,idx,n=np.unique([str(uid[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    weights=torch.from_numpy(.65/len(ids)+.35/(len(n)*n[idx])).float().cuda()
    net=LatticeNet(vote.load_net(root/"final.pt")).cuda().to(memory_format=torch.channels_last).train()
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    params,avg=list(net.parameters()),list(ema.parameters())
    old=list(net.base.parameters());known={id(p) for p in old}
    opt=torch.optim.AdamW([{"params":old},{"params":[p for p in params if id(p) not in known]}],
                          lr=.00035,weight_decay=.0002,fused=True)
    meta={"fold":args.fold,"mode":args.mode,"seed":seed,"steps":steps,"data":rc.DATA_HASH,"parent":parent["checkpoint"],
          "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"shift":shift,"gpu":torch.cuda.get_device_name(),
          "torch":str(torch.__version__),"host":socket.gethostname()}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if mode=="inner" else []})
    started=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(weights,BATCH,replacement=True);u=torch.rand(BATCH,9,device="cuda")
            exact=step%4!=0
            flip=torch.where(u[:,0]<.5,-1.,1.)
            if exact:
                a=(4*u[:,1]).floor()*math.pi/2;scale=torch.ones_like(a)
                ctr=(center[ix]-64.5).round()+64.5+(3*u[:,2:4]).floor()-1.
            else:
                a=(2*u[:,1]-1)*math.pi;scale=.73+.45*u[:,4]
                ctr=center[ix]+(u[:,2:4]-.5)*1.6
            x,back=aligned_view(image[ix].float()/255,ctr,a,flip,scale,exact)
            x=(x.clamp_min(1e-6).pow(.85+.30*u[:,5,None,None,None])*(.85+.3*u[:,6,None,None,None])+
               .08*(u[:,7,None,None,None]-.5)).clamp(0,1)
            # Hide part of the direct size signal during training.
            jitter=(torch.rand(BATCH,4,1,1,device="cuda")-.5)*2.4
            jitter=jitter*(u[:,8,None,None,None]>.35)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                logits,support,rads=net(x,jitter)
            mode_interp="nearest" if exact else "bilinear"
            def pull(z):
                out=F.grid_sample(z,back,mode=mode_interp,align_corners=True,padding_mode="border")
                return torch.where(back.abs().amax(-1)[:,None]<=1,out,-16.)
            z=pull(logits);support=pull(support);target=y[ix]
            count=target.sum((-2,-1))
            soft=MassProjection.apply(z,count,.65)
            hard=fixed_area_mask(z.detach(),count)+(soft-soft.detach())
            area_dice=(2*(hard*target).sum((-2,-1))+1)/(2*count+1)
            radius=(count/math.pi).sqrt()
            rloss=F.smooth_l1_loss(rads*scale[:,None],radius,beta=.5)
            loss=vote.mask_loss(z,target)+.55*vote.mask_loss(support,votes[ix])+.9*(1-area_dice.mean())+.2*rloss
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,4,error_if_nonfinite=True)
            rate=min(1,step/100)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            for group,lr in zip(opt.param_groups,(.00023,.0006)):group["lr"]=lr*rate
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,params,max(.004,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"shape":area_dice.mean().item(),"seconds":time.monotonic()-started}
                log.write(json.dumps(row,allow_nan=False)+"\n");print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt");rc.write_json(args.out/"fit.json",{**meta,"checkpoint":rc.hash_file(args.out/"final.pt"),
                  "seconds":time.monotonic()-started})


@torch.inference_mode()
def predict(net,images,centers,shift):
    logits=[];radii=[]
    for idx in range(0,len(images),12):
        raw=align.registered_ct(torch.as_tensor(images[idx:idx+12],device="cuda"),*shift).float()/255
        ctr=torch.as_tensor(centers[idx:idx+12],device="cuda").float()
        combined=0.;cnt=0.;r=0.;nr=0.
        for level,weight in ((1.,2.),(.8,1.),(1.12,1.)):
            exact=level==1.
            center=(ctr-64.5).round()+64.5 if exact else ctr
            scale=raw.new_full((len(raw),),level)
            for k in range(4):
                for flip in (-1.,1.):
                    x,back=aligned_view(raw,center,torch.full_like(scale,k*math.pi/2),torch.full_like(scale,flip),scale,exact)
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                        z,_,radius=net(x)
                    out=F.grid_sample(z,back,mode="nearest" if exact else "bilinear",align_corners=True,padding_mode="border")
                    coverage=(1-back.abs().amax(-1)).clamp(0,.05)[:,None]*weight
                    combined=combined+out*coverage;cnt=cnt+coverage;r=r+radius*scale[:,None]*weight;nr+=weight
        logits.append(torch.where(cnt>0,combined/cnt.clamp_min(1e-7),-16.).cpu().numpy());radii.append((r/nr).cpu().numpy())
    return np.concatenate(logits),np.concatenate(radii)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("data","align","out"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True);p.add_argument("--mode",choices=("smoke","inner","full"),required=True)
    train(p.parse_args())
