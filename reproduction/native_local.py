"""Single-slice localization and native-grid multirater supervision."""
import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import socket
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny

from reproduction import rank_contours as rc
from reproduction import vote_contours as vote
from reproduction.data import patient_folds

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_90/protocol.md"
INIT="983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
STEPS={"locator":3000,"local":7500}
BATCH=16


def coordinates(n,device):
    y,x=torch.meshgrid(torch.arange(n,device=device,dtype=torch.float32),
                       torch.arange(n,device=device,dtype=torch.float32),indexing="ij")
    return torch.stack((x,y),-1)


def rater_center(mask):
    xy=coordinates(mask.shape[-1],mask.device).reshape(-1,2)
    weights=mask.flatten(2).float()
    mass=weights.sum(2,keepdim=True)
    centers=(weights@xy)/mass.clamp_min(1)
    filled=(mass>0).float()
    return torch.where(filled.sum(1)>0,(centers*filled).sum(1)/filled.sum(1).clamp_min(1),
                       torch.full_like(centers[:,0],64.5))


def grids(center,matrix,out_size):
    native=coordinates(128,center.device).reshape(1,-1,2)
    local=coordinates(out_size,center.device).reshape(1,-1,2)-(out_size-1)/2
    src=local@matrix.transpose(1,2)+center[:,None]
    back=(native-center[:,None])@torch.linalg.inv(matrix).transpose(1,2)+(out_size-1)/2
    return (src.reshape(-1,out_size,out_size,2)*(2/127)-1,
            back.reshape(-1,128,128,2)*(2/(out_size-1))-1)


def matrix_for(scale,angle,flip):
    c,s=angle.cos()*scale,angle.sin()*scale
    return torch.stack((flip*c,-s,flip*s,c),1).reshape(-1,2,2)


def ct_windows(image):
    # The quantiles come only from the current CT view.
    reduced=F.avg_pool2d(image,4).flatten(1)
    q=torch.quantile(reduced,torch.tensor([0.10,0.90],device=image.device),dim=1)
    lo,hi=q[0,:,None,None,None],q[1,:,None,None,None]
    robust=((image-lo)/(hi-lo).clamp_min(0.10)).clamp(0,1)
    blur=F.avg_pool2d(image,13,1,6,count_include_pad=False)
    return torch.cat((image,robust,(0.5+3*(image-blur)).clamp(0,1)),1)


def up(x,skip):
    return F.interpolate(x,skip.shape[-2:],mode="bilinear",align_corners=False)


class Locator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=rc.Residual(3,24)
        self.down=nn.ModuleList([nn.Sequential(rc.Residual(a,b,2),rc.Residual(b,b))
                                 for a,b in ((24,40),(40,64),(64,96))])
        self.decode=nn.ModuleList([rc.Residual(a+b,a) for a,b in ((64,96),(40,64),(24,40))])
        self.head=nn.Conv2d(24,2,1)

    def forward(self,image):
        x=self.stem(ct_windows(image).contiguous(memory_format=torch.channels_last))
        skips=[x]
        for layer in self.down:
            x=layer(x);skips.append(x)
        for layer,skip in zip(self.decode,reversed(skips[:-1])):
            x=layer(torch.cat((up(x,skip),skip),1))
        return self.head(x).float()


class LocalExperts(nn.Module):
    def __init__(self,initial=None):
        super().__init__()
        net=convnext_tiny(weights=None)
        if initial is not None:
            if rc.hash_file(initial)!=INIT: raise ValueError("ConvNeXt initialization hash mismatch")
            net.load_state_dict(torch.load(initial,map_location="cpu",weights_only=True))
        self.encoder=net.features[:6]
        self.fine=rc.Residual(5,24)
        self.half_res=rc.Residual(24,32,2)
        self.up2=rc.Residual(384+192,96)
        self.up1=rc.Residual(96+96,64)
        self.uphalf=rc.Residual(64+32,40)
        self.up0=rc.Residual(40+24,24)
        self.head=nn.Conv2d(24,9,1)
        self.register_buffer("mean",torch.tensor([0.485,0.456,0.406])[None,:,None,None])
        self.register_buffer("std",torch.tensor([0.229,0.224,0.225])[None,:,None,None])

    def forward(self,image):
        window=ct_windows(image)
        coords=coordinates(image.shape[-1],image.device).permute(2,0,1)[None]*(2/(image.shape[-1]-1))-1
        fine=self.fine(torch.cat((window,coords.expand(len(image),-1,-1,-1)),1).contiguous(memory_format=torch.channels_last))
        half=self.half_res(fine)
        x=((window-self.mean)/self.std).contiguous(memory_format=torch.channels_last)
        skips=[]
        for j,block in enumerate(self.encoder):
            x=block(x)
            if j in (1,3): skips.append(x)
        middle=self.up2(torch.cat((up(x,skips[1]),skips[1]),1))
        low=self.up1(torch.cat((up(middle,skips[0]),skips[0]),1))
        wide=self.uphalf(torch.cat((up(low,half),half),1))
        return self.head(self.up0(torch.cat((up(wide,fine),fine),1))).float()


def restore(local,back):
    logits=F.grid_sample(local,back,align_corners=True,padding_mode="border")
    # Outside the sampled view the local branch predicts background.
    return torch.where(back.abs().amax(-1)[:,None]<=1,logits,-16.)


def local_logits(net,image,center,side,angle,flip,size=144):
    grid,back=grids(center,matrix_for(side/(size-1),angle,flip),size)
    crop=F.grid_sample(image,grid,mode="bicubic",align_corners=True,padding_mode="border").clamp(0,1)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        values=net(crop)
    return restore(values,back)


def detect(heat,prior_strength=0.65):
    xy=coordinates(128,heat.device)
    x,y=xy[...,0]-64.5,xy[...,1]-64.5
    px=torch.logaddexp(-0.5*(x/1.8).square(),-3.5-0.5*(x/30).square())
    prior=px-0.5*(y/2.5).square()
    score=heat[:,0]+prior_strength*prior
    idx=score.flatten(1).argmax(1)
    apex=xy.reshape(-1,2)[idx]
    local=((xy[None]-apex[:,None,None]).square().sum(-1)<=16)
    weight=torch.where(local,((score-score.flatten(1).amax(1)[:,None,None]).clamp(-30,0)).exp(),0.)
    center=weight.flatten(1)@xy.reshape(-1,2)/weight.flatten(1).sum(1,keepdim=True)
    return center


@torch.inference_mode()
def infer(locator,expert,images,prior=0.65,force_center=False):
    outputs=[];centers=[]
    for block in images.split(12):
        im=block.float()/255
        # Heatmaps are produced directly in native coordinates.
        heat=0
        for orient in (False,True):
            x=im.flip([-1]) if orient else im
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                pred=locator(x)
            heat=heat+(pred.flip([-1]) if orient else pred)
        center=detect(heat/2,prior)
        if force_center: center=torch.full_like(center,64.5)
        z=0
        for side in (58.,80.):
            for rotate in (0,1,2,3):
                for flip in (-1.,1.):
                    v=im.new_full((len(im),),side)
                    z=z+local_logits(expert,im,center,v,torch.full_like(v,rotate*math.pi/2),torch.full_like(v,flip))
        outputs.append((z/16).cpu().numpy());centers.append(center.cpu().numpy())
    return np.concatenate(outputs),np.concatenate(centers)


def get_models(root,fold,mode):
    loc=Locator().cuda().to(memory_format=torch.channels_last).eval()
    loc.load_state_dict(torch.load(root/f"fold-{fold}-{mode}-locator/final.pt",map_location="cpu",weights_only=True))
    net=LocalExperts().cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(root/f"fold-{fold}-{mode}-local/final.pt",map_location="cpu",weights_only=True))
    return loc,net


def inputs(path):
    if rc.hash_file(path)!=rc.DATA_HASH: raise ValueError("Unexpected LIDC bytes")
    with path.open("rb") as stream: values=list(pickle.load(stream).values())
    im=np.stack([v["image"] for v in values])
    truth=np.stack([sorted(v["masks"],key=np.count_nonzero) for v in values]).astype(np.uint8)
    alpha=np.stack([v["alpha"] for v in values])
    if im.shape[1:]!=(128,128) or not np.all((truth==0)|(truth==1)) or alpha.dtype!=np.uint8:
        raise ValueError("Unsupported input")
    return im,truth,alpha,[v["series_uid"] for v in values]


def splits(uids,fold):
    class Group: pass
    group=Group();group.series_uid=uids
    return rc.split_for(group,fold)


def run(args):
    torch.set_num_threads(3);cv2.setNumThreads(1)
    ims,labels,alphas,uids=inputs(args.data)
    outer,inner,valid=splits(uids,args.fold)
    ids=outer["train_indices"] if args.mode=="full" else inner
    steps=STEPS[args.stage]
    if args.mode=="smoke": ids,steps=ids[:48],5
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    seed=120031+args.fold+(100 if args.stage=="local" else 0)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    img=torch.from_numpy(ims[ids,None]).cuda()
    truth=torch.from_numpy(labels[ids]).cuda()
    alpha=torch.from_numpy(alphas[ids,None]).cuda()
    centers=rater_center(truth)
    names=[str(uids[i]).split("_")[0] for i in ids]
    _,inv,n=np.unique(names,return_inverse=True,return_counts=True)
    sampling=torch.from_numpy(0.5/len(ids)+.5/(len(n)*n[inv])).float().cuda()
    net=(Locator() if args.stage=="locator" else LocalExperts(args.init)).cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    params,averaged=list(net.parameters()),list(ema.parameters())
    if args.stage=="local":
        enc=list(net.encoder.parameters());ei={id(p) for p in enc}
        groups=[{"params":enc},{"params":[p for p in params if id(p) not in ei]}]
        lrs=(0.00018,0.0008)
    else: groups=[{"params":params}];lrs=(0.001,)
    opt=torch.optim.AdamW(groups,fused=True,weight_decay=.0002)
    xy=coordinates(128,"cuda")
    meta={"stage":args.stage,"mode":args.mode,"fold":args.fold,"steps":steps,"seed":seed,"source":rc.hash_file(__file__),
          "protocol":rc.hash_file(PROTOCOL),"data_hash":rc.DATA_HASH,"initialization":INIT if args.stage=="local" else "scratch",
          "gpu":torch.cuda.get_device_name(),"host":socket.gethostname(),"torch":str(torch.__version__),"cuda":torch.version.cuda}
    rc.write_json(out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if args.mode=="inner" else []})
    start=time.monotonic()
    with (out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(sampling,BATCH,replacement=True)
            x=img[ix].float()/255;y=truth[ix].float();ctr=centers[ix]
            jitter=torch.rand(BATCH,3,1,1,device="cuda")
            x=(x.clamp_min(1e-6).pow(.78+.45*jitter[:,:1])*(.8+.4*jitter[:,1:2])+(jitter[:,2:]-.5)*.10).clamp(0,1)
            a=(torch.rand(BATCH,device="cuda")*2-1)*math.pi
            flip=torch.where(torch.rand(BATCH,device="cuda")<.5,-1.,1.)
            if args.stage=="locator":
                scale=.83+.34*torch.rand(BATCH,device="cuda")
                mat=matrix_for(scale,a,flip)
                offset=(torch.rand(BATCH,2,device="cuda")*2-1)*x.new_tensor([41,14])
                offset=offset*torch.where(torch.rand(BATCH,1,device="cuda")<.35,.13,1.)
                # Nodule center is mapped exactly to this synthetic location.
                target=64+offset
                src=ctr+(63.5-target)[:,None,:].bmm(mat.transpose(1,2)).squeeze(1)
                grid,_=grids(src,mat,128)
                im=F.grid_sample(x,grid,padding_mode="reflection",align_corners=True)
                gt=F.grid_sample((y.mean(1,keepdim=True)>.25).float(),grid,padding_mode="zeros",align_corners=True)
                heat=(-((xy[None]-target[:,None,None])**2).sum(-1)/(2*2.5**2)).exp()
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()): z=net(im)
                target_h=heat/heat.sum((1,2),keepdim=True)
                heat_loss=-(target_h*z[:,0].flatten(1).log_softmax(1).reshape_as(heat)).sum((1,2)).mean()
                loss=0.3*heat_loss+vote.mask_loss(z[:,1:2],gt)
            else:
                ctr=ctr+torch.randn_like(ctr)*2.0
                side=51+42*torch.rand(BATCH,device="cuda")
                z=local_logits(net,x,ctr,side,a,flip)
                v=vote.votes(y)
                A=y.sum((-1,-2));pred=z[:,:4].sigmoid()
                area_loss=F.smooth_l1_loss((pred.sum((-1,-2))+1).log(),(A+1).log())
                matte=alpha[ix].float()/255
                band=F.max_pool2d(v[:,:1],17,1,8)
                matte_loss=((z[:,8:9].sigmoid()-matte).abs()*band).sum()/band.sum().clamp_min(1)
                loss=vote.mask_loss(z[:,:4],y)+.85*vote.mask_loss(z[:,4:8],v)+.15*area_loss+.35*matte_loss
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            for group,base in zip(opt.param_groups,lrs):
                group["lr"]=base*min(1,step/100)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(averaged,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-start};log.write(json.dumps(row,allow_nan=False)+"\n");print(row,flush=True)
    torch.save(ema.state_dict(),out/"final.pt")
    rc.write_json(out/"fit.json",{**meta,"checkpoint":rc.hash_file(out/"final.pt"),"seconds":time.monotonic()-start})
    print("FIT_COMPLETE",args.stage,args.mode,args.fold,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for k in ("data","init","out"):parser.add_argument("--"+k,type=Path,required=True)
    parser.add_argument("--mode",choices=("smoke","inner","full"),required=True)
    parser.add_argument("--stage",choices=tuple(STEPS),required=True)
    parser.add_argument("--fold",choices=range(4),type=int,required=True)
    run(parser.parse_args())
