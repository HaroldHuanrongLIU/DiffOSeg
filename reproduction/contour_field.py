"""CT-conditioned low-rank Gaussian contour field, trained on four masks per image."""
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
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from datasets.lidc import BaseDataset
from reproduction import joint_energy as energy
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_field/protocol.md"
WEIGHT_HASH="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
STEPS,BATCH,DIM=6000,16,7


def resize(x,skip):
    return F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False)


class ContourField(nn.Module):
    def __init__(self,initial=None):
        super().__init__()
        net=resnet18(weights=None)
        if initial is not None:
            if rc.hash_file(initial)!=WEIGHT_HASH:
                raise ValueError("Unexpected public initialization")
            net.load_state_dict(torch.load(initial,map_location="cpu",weights_only=True))
        self.encoder=nn.ModuleList([nn.Sequential(net.conv1,net.bn1,net.relu,net.layer1),net.layer2,net.layer3])
        self.fine=rc.Residual(4,24)
        self.dec2=rc.Residual(256+128,80)
        self.dec1=rc.Residual(80+64,48)
        self.dec0=nn.Sequential(rc.Residual(48+24,32),rc.Residual(32,32))
        self.head=nn.Conv2d(32,9,1)
        self.shape=nn.Sequential(nn.Conv2d(80,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,4,1))
        self.register_buffer("mean",torch.tensor([0.485,0.456,0.406])[None,:,None,None])
        self.register_buffer("std",torch.tensor([0.229,0.224,0.225])[None,:,None,None])

    def forward(self,image):
        h,w=image.shape[-2:]
        y,x=torch.meshgrid(torch.arange(h,device=image.device)-64.5,torch.arange(w,device=image.device)-64.5,indexing="ij")
        xy=torch.stack((x,y)).expand(len(image),-1,-1,-1)
        local=F.avg_pool2d(image,9,1,4,count_include_pad=False)
        full=self.fine(torch.cat((2*image-1,4*(image-local),xy/64),1))
        windows=torch.cat((image,(1.9*image).clamp(0,1),(3*image-0.2).clamp(0,1)),1)
        first=self.encoder[0](((windows-self.mean)/self.std).contiguous(memory_format=torch.channels_last))
        second=self.encoder[1](first)
        third=self.encoder[2](second)
        mid=self.dec2(torch.cat((resize(third,second),second),1))
        half=self.dec1(torch.cat((resize(mid,first),first),1))
        fine=self.dec0(torch.cat((resize(half,full),full),1))
        logits=self.head(fine).float()
        angular=xy/(xy.square().sum(1,keepdim=True)+4).sqrt()
        fixed=torch.cat((torch.ones_like(image),0.7*angular),1)
        basis=F.normalize(torch.cat((fixed,resize(self.shape(mid).float().tanh(),image)),1),dim=1)
        ranked=logits[:,:4]
        support=logits[:,4:8].sort(1,descending=True).values
        freq=(torch.logit(support.sigmoid().mean(1).clamp(1e-6,1-1e-6))+1.5*logits[:,8].tanh()).sigmoid()
        return ranked,support,freq,basis


def sample_percentiles(basis,latent):
    """A draw shares its Gaussian coefficients across all spatial positions."""
    projection=torch.einsum("bkd,bdhw->bkhw",latent,basis)
    return 0.5*(1+torch.erf(projection/math.sqrt(2)))


def population_objective(freq,basis,truth,latent):
    percentiles=sample_percentiles(basis,latent)
    smooth=((freq[:,None]-percentiles)/0.045).sigmoid()
    samples=(freq[:,None]>percentiles).float()+(smooth-smooth.detach())
    ged=energy.mean_iou(samples,samples)+energy.mean_iou(truth,truth)-2*energy.mean_iou(truth,samples)
    levels=freq.new_tensor([0.1,0.3,0.5,0.7,0.9])[None,:,None,None]
    soft=((freq[:,None]-levels)/0.045).sigmoid()
    pred=(freq[:,None]>levels).float()+(soft-soft.detach())
    target=(truth.mean(1,keepdim=True)>levels).float()
    den=(pred+target).sum((-2,-1))
    dice=torch.where(den>0,2*(pred*target).sum((-2,-1))/den.clamp_min(1),1.)
    return ged.mean()+1.6*(1-dice.mean())


def load_model(path):
    model=ContourField().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return model.eval()


@torch.inference_mode()
def predict(model,images):
    all_expert,all_frequency,all_basis=[],[],[]
    for offset in range(0,len(images),12):
        x=torch.as_tensor(images[offset:offset+12],device="cuda").float()/255
        expert=0
        frequency=0
        for flip in (False,True):
            view=x.flip([-1]) if flip else x
            for k in range(4):
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    e,_,p,basis=model(torch.rot90(view,k,(-2,-1)))
                if not flip and k==0:
                    all_basis.append(basis.cpu().numpy())
                e=torch.rot90(e,-k,(-2,-1)); p=torch.rot90(torch.logit(p.clamp(1e-6,1-1e-6)),-k,(-2,-1))
                expert=expert+(e.flip([-1]) if flip else e)
                frequency=frequency+(p.flip([-1]) if flip else p)
        all_expert.append((expert/8).cpu().numpy())
        all_frequency.append((frequency/8).sigmoid().cpu().numpy())
    out=tuple(np.concatenate(parts) for parts in (all_expert,all_frequency,all_basis))
    if not all(np.isfinite(a).all() for a in out):
        raise ValueError("Nonfinite field prediction")
    return out


def fit(args):
    if rc.hash_file(args.data)!=rc.DATA_HASH:
        raise ValueError("Dataset identity")
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    data=BaseDataset(str(args.data))
    outer,inner,valid=rc.split_for(data,args.fold)
    ids=outer["train_indices"] if args.mode=="full" else inner
    steps=STEPS
    if args.mode=="smoke":
        ids,valid,steps=inner[:48],valid[:4],5
    if args.mode=="full":
        gate=json.loads(args.gate.read_text())
        if (gate["mode"],gate["fold"],gate["source"],gate["protocol"],gate["passed"])!=(
                "inner",args.fold,rc.hash_file(__file__),rc.hash_file(PROTOCOL),True):
            raise ValueError("Matching inner gate required")
    args.out.mkdir(parents=True,exist_ok=False)
    seed=81931+args.fold
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    x,truth=rc.tensors(data,ids,False)
    target=torch.cat((truth,vote.votes(truth)),1).half()
    names=[str(data.series_uid[i]).split("_")[0] for i in ids]
    _,inv,n=np.unique(names,return_inverse=True,return_counts=True)
    sampling=torch.from_numpy(0.5/len(ids)+0.5/(n[inv]*len(n))).float().cuda()
    model=ContourField(args.init).cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).requires_grad_(False).eval()
    params,avg=list(model.parameters()),list(ema.parameters())
    enc=list(model.encoder.parameters())
    encoder_ids={id(z) for z in enc}
    nonenc=[z for z in params if id(z) not in encoder_ids]
    opt=torch.optim.AdamW([{"params":enc},{"params":nonenc}],lr=0.0006,weight_decay=0.0003,fused=True)
    meta={"mode":args.mode,"fold":args.fold,"seed":seed,"steps":steps,"batch":BATCH,"source":rc.hash_file(__file__),
          "protocol":rc.hash_file(PROTOCOL),"data_sha256":rc.DATA_HASH,"initialization":WEIGHT_HASH,
          "torch":str(torch.__version__),"cuda":torch.version.cuda,"host":socket.gethostname(),"gpu":torch.cuda.get_device_name()}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if args.mode!="full" else []})
    started=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(sampling,BATCH,replacement=True)
            g,_=vote.sampling_grid(BATCH,"fullfield","cuda",True)
            warp=F.interpolate(torch.randn(BATCH,2,6,6,device="cuda"),(128,128),mode="bicubic",align_corners=False)
            g=g+0.012*warp.permute(0,2,3,1)
            image=F.grid_sample(x[ix].float()/255,g,padding_mode="border",align_corners=True)
            yy=F.grid_sample(target[ix].float(),g,padding_mode="border",align_corners=True)
            jitter=torch.rand(BATCH,3,1,1,device="cuda")
            image=(image.clamp_min(1e-6).pow(0.84+0.32*jitter[:,:1])*(0.85+0.3*jitter[:,1:2])+
                   (jitter[:,2:]-0.5)*0.08).clamp(0,1)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                expert,support,freq,basis=model(image)
            latent=torch.randn(BATCH,6,DIM,device="cuda")
            pop=population_objective(freq,basis,(yy[:,:4]>0.5).float(),torch.cat((latent,-latent),1))
            loss=vote.mask_loss(expert,yy[:,:4])+vote.mask_loss(support,yy[:,4:])+min(1.,step/500)*pop
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            lr=min(1,step/100)*(0.025+0.975*(1+math.cos(math.pi*step/steps))/2)
            for group,base in zip(opt.param_groups,(0.00025,0.0008)):
                group["lr"]=base*lr
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(avg,params,max(0.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"population_loss":pop.item(),"seconds":time.monotonic()-started}
                log.write(json.dumps(row,allow_nan=False)+"\n")
                print(row,flush=True)
    # BatchNorm statistics belong to the matching fitted model.
    with torch.no_grad():
        for a,b in zip(ema.buffers(),model.buffers()):
            a.copy_(b)
    torch.save(ema.state_dict(),args.out/"final.pt")
    meta.update(checkpoint=rc.hash_file(args.out/"final.pt"),seconds=time.monotonic()-started)
    if args.mode!="full":
        vx=np.stack([data.images[i] for i in valid])[:,None]
        vy=np.asarray([data.mask_labels[i] for i in valid],bool)
        expert,_,_=predict(ema,vx)
        seg=expert>0
        den=seg.sum((-2,-1))+vy.sum((-2,-1))
        meta["inner_dice"]=float(np.divide(2*(seg&vy).sum((-2,-1)),den,out=np.ones(den.shape),where=den>0).mean())
        meta["passed"]=args.mode=="smoke" or meta["inner_dice"]>=0.78
    rc.write_json(args.out/"result.json",meta)
    print(meta,flush=True)
    if meta.get("passed") is False:
        raise ValueError("Inner field gate failed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("data","init","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--mode",choices=("smoke","inner","full"),required=True)
    p.add_argument("--gate",type=Path)
    args=p.parse_args()
    if args.mode=="full" and args.gate is None:
        p.error("Full fit requires a matching gate")
    fit(args)
