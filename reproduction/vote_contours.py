"""Joint learning of area-ranked masks and the four cumulative vote levels."""
import argparse
import copy
import math
from pathlib import Path
import socket
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets.lidc import BaseDataset
from reproduction import rank_contours as rc
from reproduction import boundary_eval as v1


PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_vote/protocol.md"
VIEWS = ("fullfield", "zoom")
STEPS, BATCH = 4000, 24


def sampling_grid(n, view, device, augment):
    axis = torch.arange(128,device=device,dtype=torch.float32)
    ref = 64.5 if view=="fullfield" else 63.5
    yy,xx = torch.meshgrid(axis-ref,axis-ref,indexing="ij")
    scale = 1.0 if view=="fullfield" else 80/128
    if augment:
        u = torch.rand(n,5,1,1,device=device)
        a = (u[:,0]*2-1)*math.pi
        flip = torch.where(u[:,1]<0.5,-1.,1.)
        scale = scale * (0.85 + 0.30*u[:,2])
        x = 64.5 + (u[:,3]-0.5)*1.5 + scale*(a.cos()*flip*xx-a.sin()*yy)
        y = 64.5 + (u[:,4]-0.5)*1.5 + scale*(a.sin()*flip*xx+a.cos()*yy)
    else:
        x = (64.5+scale*xx).expand(n,-1,-1)
        y = (64.5+scale*yy).expand(n,-1,-1)
    return torch.stack((x,y),dim=-1)*(2/127)-1, scale


def build(old=None):
    net=rc.RankContours().cuda().to(memory_format=torch.channels_last)
    if old is not None:
        net.load_state_dict(torch.load(old,map_location="cpu",weights_only=True))
    previous=net.head
    net.head=nn.Conv2d(previous.in_channels,12,1).cuda()
    with torch.no_grad():
        net.head.weight[:8].copy_(previous.weight)
        net.head.bias[:8].copy_(previous.bias)
        net.head.weight[8:].copy_(previous.weight[:4].flip([0]))
        net.head.bias[8:].copy_(previous.bias[:4].flip([0]))
    return net.to(memory_format=torch.channels_last)


def load_net(path):
    model=build()
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return model.eval()


@torch.inference_mode()
def d4(model, image):
    total=0
    for flip in (False,True):
        reflected=image.flip([-1]) if flip else image
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                out=model(torch.rot90(reflected,k,(-2,-1)))
            out=torch.rot90(out,-k,(-2,-1))
            total=total+(out.flip([-1]) if flip else out)
    return total/8


def single_view(net,image,view):
    coords,_=sampling_grid(len(image),view,image.device,False)
    crop=F.grid_sample(image,coords,align_corners=True,padding_mode="border")
    return d4(net,crop)


def combine_views(full,zoom):
    """Map the 80-native-pixel view back to its support, fading its edge."""
    out=full.clone()
    local=F.interpolate(zoom,size=(80,80),mode="bilinear",align_corners=False)
    axis=torch.arange(80,device=out.device)
    dist=torch.minimum(axis,79-axis).float()
    window=(torch.minimum(dist[:,None],dist[None,:])/6).clamp(0,1)*0.5
    crop=out[:,:,25:105,25:105]
    crop.add_((local-crop)*window)
    return out


def mask_loss(logits,y):
    p=logits.sigmoid()
    dice=(2*(p*y).sum((-2,-1))+1)/((p+y).sum((-2,-1))+1)
    bce=F.binary_cross_entropy_with_logits(logits,y,reduction="none")
    return 1-dice.mean()+(bce*(1+4*y)).mean()


def votes(masks):
    count=masks.sum(1,keepdim=True)
    cutoff=torch.arange(1,5,device=masks.device)[None,:,None,None]
    return (count>=cutoff).float()


def train(args):
    if rc.hash_file(args.data)!=rc.DATA_HASH:
        raise ValueError("Dataset hash differs")
    data=BaseDataset(str(args.data))
    outer,inner,valid=rc.split_for(data,args.fold)
    parent_mode="inner" if args.mode=="smoke" else args.mode
    old=args.prior/f"fold-{args.fold}-{parent_mode}"
    ids=outer["train_indices"] if parent_mode=="full" else inner
    plan,identity=v1.read(old/"plan.json"),v1.read(old/"result.json")
    if (plan["train_indices"]!=ids or plan["training_source"]!=rc.hash_file(rc.__file__) or
            rc.hash_file(old/"final.pt")!=identity["checkpoint_sha256"]):
        raise ValueError("Previous model provenance differs")
    steps=STEPS
    if args.mode=="smoke":
        ids,valid,steps=ids[:48],valid[:6],5
    if args.mode=="full":
        gate=v1.read(args.gate)
        if (gate["fold"],gate["view"],gate["mode"],gate["source"],gate["passed"])!=(
                args.fold,args.view,"inner",rc.hash_file(__file__),True):
            raise ValueError("Inner model gate differs")
    args.out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    seed=93071+args.fold+100*VIEWS.index(args.view)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    images,masks,fields=rc.tensors(data,ids,True)
    targets=torch.cat((masks.float(),votes(masks),fields.float()),1).half()
    net=build(old/"final.pt")
    ema=copy.deepcopy(net).requires_grad_(False).eval()
    params,avg=list(net.parameters()),list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=0.0005,weight_decay=0.0001,fused=True)
    result={"mode":args.mode,"view":args.view,"fold":args.fold,"seed":seed,"steps":steps,
            "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"parent":identity["checkpoint_sha256"],
            "data_sha256":rc.DATA_HASH,"host":socket.gethostname(),"gpu":torch.cuda.get_device_name(),
            "torch":str(torch.__version__),"cuda":torch.version.cuda}
    rc.write_json(args.out/"plan.json",{**result,"train_indices":ids,"inner_valid_indices":valid if args.mode!="full" else []})
    start=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        import json
        for step in range(1,steps+1):
            ix=torch.randint(len(ids),(BATCH,),device="cuda")
            g,scale=sampling_grid(BATCH,args.view,"cuda",True)
            x=F.grid_sample(images[ix].float()/255,g,align_corners=True,padding_mode="border")
            jitter=torch.rand(BATCH,2,1,1,device="cuda")
            x=(x*(0.85+0.30*jitter[:,0:1])+(jitter[:,1:2]-0.5)*0.08).clamp(0,1)
            y=F.grid_sample(targets[ix].float(),g,align_corners=True,padding_mode="border")
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                out=net(x)
            loss=mask_loss(out[:,:4],y[:,:4])+mask_loss(out[:,8:],y[:,4:8])
            loss=loss+0.20*F.smooth_l1_loss(out[:,4:8].tanh(),(y[:,8:]/(8*scale[:,None])).clamp(-1,1))
            rate=0.0005 if args.view=="fullfield" else 0.0007
            opt.param_groups[0]["lr"]=rate*min(1,step/80)*(0.02+0.98*(1+math.cos(math.pi*step/steps))/2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(avg,params,max(0.005,1/step))
            if step==1 or step%200==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-start}
                log.write(json.dumps(row,allow_nan=False)+"\n")
                print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    result.update(checkpoint=rc.hash_file(args.out/"final.pt"),seconds=time.monotonic()-start)
    if args.mode!="full":
        scores=[]
        images,truth=rc.tensors(data,valid,False)
        for x,t in zip(images.split(12),truth.split(12)):
            z=single_view(ema,x.float()/255,args.view)[:,:4]
            if args.view=="zoom":
                z=F.interpolate(z,(80,80),mode="bilinear",align_corners=False)
                t=t[:,:,25:105,25:105]
            pred=z>0
            den=pred.sum((-2,-1))+t.sum((-2,-1))
            scores.append(torch.where(den>0,2*(pred*t).sum((-2,-1))/den.clamp_min(1),1.).cpu())
        result["inner_view_dice"]=torch.cat(scores).mean().item()
        result["passed"]=args.mode=="smoke" or result["inner_view_dice"]>=0.78
    rc.write_json(args.out/"result.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--view",choices=VIEWS,required=True)
    p.add_argument("--mode",choices=["inner","full","smoke"],required=True)
    p.add_argument("--gate",type=Path)
    args=p.parse_args()
    if args.mode=="full" and args.gate is None:
        p.error("Full training requires --gate")
    train(args)
