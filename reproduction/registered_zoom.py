"""Magnified CT views with native-grid area and cross-view supervision."""
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

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_90/zoom_protocol.md"
STEPS=4800


class ZoomRank(nn.Module):
    def __init__(self,base=None):
        super().__init__()
        self.base=vote.build() if base is None else base
        self.window=nn.Conv2d(2,24,3,padding=1,bias=False)
        nn.init.zeros_(self.window.weight)

    def forward(self,image):
        size=image.shape[-1]
        xy=native.coordinates(size,image.device)
        coordinate=(2*xy/(size-1)-1).permute(2,0,1)[None].expand(len(image),-1,-1,-1)
        radius=((xy-64.5).square().sum(-1)).sqrt()
        # Normalize intensities to tissue levels observed in this CT alone.
        core=image[:,0,radius<9]
        ring=image[:,0,(radius>=27)&(radius<=49)]
        hi=torch.quantile(core,0.65,dim=1)[:,None,None,None]
        lo=torch.quantile(ring,0.2,dim=1)[:,None,None,None]
        scale=(hi-lo).clamp_min(0.07)
        level=((image-lo)/scale).clamp(-.3,1.5)
        residual=((image-F.avg_pool2d(image,9,1,4,count_include_pad=False))/scale).clamp(-1,1)
        x=self.base.stem(torch.cat((2*image-1,coordinate),1).contiguous(memory_format=torch.channels_last))
        x=x+self.window(torch.cat((level,residual),1).contiguous(memory_format=torch.channels_last))
        saved=[x]
        for layer in self.base.down:
            x=layer(x);saved.append(x)
        for layer,y in zip(self.base.up,reversed(saved[:-1])):
            x=layer(torch.cat((F.interpolate(x,y.shape[-2:],mode="bilinear",align_corners=False),y),1))
        return self.base.head(x).float()


def decode(model,image,center,scale,angle,flip):
    grid,back=align.transform_grids(center,scale,angle,flip)
    crop=F.grid_sample(image,grid,mode="bicubic",padding_mode="border",align_corners=True).clamp(0,1)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        z=model(crop)
    return native.restore(z,back)


def model_from(path):
    m=ZoomRank().cuda().to(memory_format=torch.channels_last).eval()
    m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return m


@torch.inference_mode()
def predict(model,images,centers,shift):
    result=[]
    for i in range(0,len(images),16):
        im=align.registered_ct(torch.as_tensor(images[i:i+16],device="cuda"),*shift).float()/255
        center=torch.as_tensor(centers[i:i+16],device="cuda").float()
        total=0
        for s in (56/127,74/127):
            scale=im.new_full((len(im),),s)
            for k in range(4):
                for flip in (-1.,1.):
                    total=total+decode(model,im,center,scale,torch.full_like(scale,k*math.pi/2),torch.full_like(scale,flip))
        result.append((total/16).cpu().numpy())
    return np.concatenate(result)


def train(args):
    torch.set_num_threads(3);cv2.setNumThreads(1)
    ims,labels,_,uid=native.inputs(args.data)
    outer,inner,valid=native.splits(uid,args.fold)
    mode="inner" if args.mode=="smoke" else args.mode
    ids=outer["train_indices"] if mode=="full" else inner
    parent=args.parent/f"fold-{args.fold}-{mode}-registered"
    plan,fit=[json.loads((parent/f"{key}.json").read_text()) for key in ("plan","fit")]
    if plan["train_indices"]!=ids or rc.hash_file(parent/"final.pt")!=fit["checkpoint"] or fit["source"]!=rc.hash_file(align.__file__):
        raise ValueError("Registered initialization provenance")
    steps=STEPS
    if args.mode=="smoke": ids,steps=ids[:48],5
    args.out.mkdir(parents=True,exist_ok=False)
    shift=fit["applied_shift"]
    seed=165361+args.fold
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    images=align.registered_ct(torch.from_numpy(ims[ids,None]).cuda(),*shift)
    masks=torch.from_numpy(labels[ids]).cuda().float()
    center=native.rater_center(masks)
    center=torch.where((center-64.5).abs().amax(1,keepdim=True)>4,center,torch.full_like(center,64.5))
    votes=vote.votes(masks)
    fields=torch.from_numpy(np.stack([[rc.signed(m) for m in v] for v in labels[ids]]).astype("float16")).cuda()
    _,inv,n=np.unique([str(uid[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    sample_w=torch.from_numpy(.65/len(ids)+.35/(len(n)*n[inv])).float().cuda()
    model=ZoomRank(vote.load_net(parent/"final.pt")).cuda().to(memory_format=torch.channels_last).train()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params,avg=list(model.parameters()),list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=0.00045,weight_decay=1e-4,fused=True)
    meta={"fold":args.fold,"mode":args.mode,"steps":steps,"seed":seed,"source":rc.hash_file(__file__),
          "protocol":rc.hash_file(PROTOCOL),"parent":fit["checkpoint"],"applied_shift":shift,"data_hash":rc.DATA_HASH,
          "gpu":torch.cuda.get_device_name(),"host":socket.gethostname(),"torch":str(torch.__version__),"cuda":torch.version.cuda}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if mode=="inner" else []})
    started=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(sample_w,12,replacement=True).repeat(2)
            u=torch.rand(len(ix),8,device="cuda");image=images[ix].float()/255
            image=(image.clamp_min(1e-6).pow(.85+.3*u[:,5,None,None,None])*(.85+.3*u[:,6,None,None,None])+
                   (u[:,7,None,None,None]-.5)*.06).clamp(0,1)
            angle=torch.where(u[:,0]<.5,(4*u[:,1]).floor()*math.pi/2,(2*u[:,1]-1)*math.pi)
            flip=torch.where(u[:,2]<.5,-1.,1.)
            scale=(50+38*u[:,3])/127
            ctr=center[ix]+torch.randn(len(ix),2,device="cuda")*1.1
            z=decode(model,image,ctr,scale,angle,flip);target=masks[ix]
            prob=(2*z[:,:4]).sigmoid()
            hard=(z[:,:4]>0).float()+(prob-prob.detach())
            rr=(hard.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
            radii=(target.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
            size_loss=F.smooth_l1_loss(rr,radii)
            p=z[:,:4].sigmoid();band=F.max_pool2d(target.any(1,keepdim=True).float(),9,1,4)
            stability=((p[:12]-p[12:]).abs()*band[:12]).sum()/(4*band[:12].sum().clamp_min(1))
            sd=F.smooth_l1_loss(z[:,4:8].tanh(),(fields[ix].float()/8).clamp(-1,1))
            loss=vote.mask_loss(z[:,:4],target)+.65*vote.mask_loss(z[:,8:],votes[ix])+.10*sd+.14*size_loss+.18*stability
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.00045*min(1,step/80)*(.025+.975*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-started}
                log.write(json.dumps(row,allow_nan=False)+"\n");print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    rc.write_json(args.out/"fit.json",{**meta,"checkpoint":rc.hash_file(args.out/"final.pt"),"seconds":time.monotonic()-started})
    print("ZOOM_FIT_COMPLETE",args.mode,args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("out","parent","data"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--mode",choices=("inner","full","smoke"),required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    train(p.parse_args())
