"""DINOv2-small adapters and ordered radius prediction with a registered CT backbone."""
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

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/semantic_protocol.md"
DINO_SHA="1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050"
STEPS,BATCH=5000,12


class TokenBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1=nn.LayerNorm(384,eps=1e-6);self.norm2=nn.LayerNorm(384,eps=1e-6)
        self.qkv=nn.Linear(384,1152);self.proj=nn.Linear(384,384)
        self.fc1=nn.Linear(384,1536);self.fc2=nn.Linear(1536,384)
        self.ls1=nn.Parameter(torch.ones(384));self.ls2=nn.Parameter(torch.ones(384))

    def forward(self,x):
        b,n,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,6,64).permute(2,0,3,1,4).unbind(0)
        attn=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,d)
        x=x+self.ls1*self.proj(attn)
        return x+self.ls2*self.fc2(F.gelu(self.fc1(self.norm2(x))))


class DinoSmall(nn.Module):
    def __init__(self,weight=None):
        super().__init__()
        self.patch=nn.Conv2d(3,384,14,14);self.cls=nn.Parameter(torch.zeros(1,1,384))
        self.pos=nn.Parameter(torch.zeros(1,1370,384));self.blocks=nn.ModuleList(TokenBlock() for _ in range(12))
        self.norm=nn.LayerNorm(384,eps=1e-6)
        if weight is not None:self.load_public(weight)
        self.requires_grad_(False)
        self.blocks[-2:].requires_grad_(True)

    def load_public(self,path):
        if rc.hash_file(path)!=DINO_SHA:raise ValueError("DINOv2 public checkpoint hash")
        d=torch.load(path,map_location="cpu",weights_only=True)
        state={"cls":d.pop("embeddings.cls_token"),"pos":d.pop("embeddings.position_embeddings"),
               "norm.weight":d.pop("layernorm.weight"),"norm.bias":d.pop("layernorm.bias")}
        d.pop("embeddings.mask_token")
        for suffix in ("weight","bias"):
            state["patch."+suffix]=d.pop("embeddings.patch_embeddings.projection."+suffix)
        for i in range(12):
            old=f"encoder.layer.{i}.";new=f"blocks.{i}."
            for suffix in ("weight","bias"):
                state[new+"qkv."+suffix]=torch.cat([d.pop(old+"attention.attention."+k+"."+suffix) for k in ("query","key","value")])
                for a,b in (("proj","attention.output.dense"),("fc1","mlp.fc1"),("fc2","mlp.fc2"),("norm1","norm1"),("norm2","norm2")):
                    state[new+a+"."+suffix]=d.pop(old+b+"."+suffix)
            for j in (1,2):state[new+f"ls{j}"]=d.pop(old+f"layer_scale{j}.lambda1")
        if d:raise ValueError("Unhandled checkpoint tensors")
        self.load_state_dict(state)

    def forward(self,image):
        x=self.patch(image);b,c,h,w=x.shape
        pos=self.pos[:,1:].reshape(1,37,37,384).permute(0,3,1,2)
        pos=F.interpolate(pos.float(),(h,w),mode="bicubic",align_corners=False).flatten(2).transpose(1,2)
        x=torch.cat((self.cls.expand(b,-1,-1)+self.pos[:,:1],x.flatten(2).transpose(1,2)+pos),1)
        stages=[]
        for i,layer in enumerate(self.blocks):
            x=layer(x)
            if i in (3,7,11):
                stages.append(self.norm(x)[:,1:].transpose(1,2).reshape(b,c,h,w))
        return stages,self.norm(x)[:,0]


class SemanticRank(nn.Module):
    def __init__(self,base=None,dino=None):
        super().__init__()
        self.base=vote.build() if base is None else base
        self.vit=DinoSmall(dino)
        self.adapters=nn.ModuleList(nn.Conv2d(384,w,1,bias=False) for w in (72,112,160))
        for layer in self.adapters:nn.init.zeros_(layer.weight)
        self.radius_head=nn.Sequential(nn.LayerNorm(384+24+4),nn.Linear(384+24+4,96),nn.SiLU(),nn.Linear(96,4))
        nn.init.zeros_(self.radius_head[-1].weight);nn.init.zeros_(self.radius_head[-1].bias)
        self.register_buffer("mean",torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer("std",torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,image):
        xy=native.coordinates(128,image.device).permute(2,0,1)[None]*(2/127)-1
        features=self.base.stem(torch.cat((2*image-1,xy.expand(len(image),-1,-1,-1)),1).contiguous(memory_format=torch.channels_last))
        windows=native.ct_windows(image)
        rgb=F.interpolate((windows-self.mean)/self.std,(224,224),mode="bilinear",align_corners=False)
        semantic,cls=self.vit(rgb)
        saved=[features]
        for j,block in enumerate(self.base.down):
            features=block(features)
            if j>=1:
                add=self.adapters[j-1](semantic[j-1])
                features=features+F.interpolate(add,features.shape[-2:],mode="bilinear",align_corners=False)
            saved.append(features)
        for block,skip in zip(self.base.up,reversed(saved[:-1])):
            features=block(torch.cat((F.interpolate(features,skip.shape[-2:],mode="bilinear",align_corners=False),skip),1))
        z=self.base.head(features).float()
        p=z[:,:4].sigmoid()
        r=(p.sum((-2,-1)).clamp_min(1)/math.pi).sqrt().detach()
        weights=(p.mean(1,keepdim=True)+.004).detach()
        pool=(features*weights).sum((-2,-1))/weights.sum((-2,-1)).clamp_min(1)
        adjustment=3*self.radius_head(torch.cat((cls,pool,r/12),1)).float().tanh()
        return z,(r+adjustment).clamp_min(0)


def load_model(path):
    model=SemanticRank().cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return model


def train(args):
    torch.set_num_threads(3);cv2.setNumThreads(1)
    ims,labels,_,uid=native.inputs(args.data);outer,inner,valid=native.splits(uid,args.fold)
    mode="inner" if args.mode=="smoke" else args.mode
    ids=inner if mode=="inner" else outer["train_indices"]
    parent=args.align/f"fold-{args.fold}-{mode}-registered"
    plan=json.loads((parent/"plan.json").read_text());fit=json.loads((parent/"fit.json").read_text())
    if plan["train_indices"]!=ids or fit["checkpoint"]!=rc.hash_file(parent/"final.pt"):raise ValueError("Registered parent identity")
    shift=fit["applied_shift"];steps=STEPS
    if args.mode=="smoke":ids,steps=ids[:48],5
    seed=31127+args.fold;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=True
    args.out.mkdir(parents=True,exist_ok=False)
    images=align.registered_ct(torch.from_numpy(ims[ids,None]).cuda(),*shift)
    y=torch.from_numpy(labels[ids]).cuda().float()
    ctr=native.rater_center(y)
    ctr=torch.where((ctr-64.5).abs().amax(1,keepdim=True)>4,ctr,torch.full_like(ctr,64.5))
    targets=torch.cat((y,vote.votes(y)),1).half()
    _,inv,n=np.unique([str(uid[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    sampling=torch.from_numpy(.6/len(ids)+.4/(len(n)*n[inv])).float().cuda()
    model=SemanticRank(vote.load_net(parent/"final.pt"),args.init).cuda().to(memory_format=torch.channels_last).train()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    named=[(k,p) for k,p in model.named_parameters() if p.requires_grad]
    params=[p for k,p in named]; average=dict(ema.named_parameters());avg=[average[k] for k,p in named]
    groups=[[p for k,p in named if k.startswith("base.")],[p for k,p in named if k.startswith("vit.")],
            [p for k,p in named if not k.startswith(("base.","vit."))]]
    opt=torch.optim.AdamW([{"params":g} for g in groups],lr=.0002,weight_decay=.0003,fused=True)
    meta={"fold":args.fold,"mode":args.mode,"steps":steps,"seed":seed,"source":rc.hash_file(__file__),
          "protocol":rc.hash_file(PROTOCOL),"parent":fit["checkpoint"],"shift":shift,"public_initialization":DINO_SHA,
          "data":rc.DATA_HASH,"gpu":torch.cuda.get_device_name(),"host":socket.gethostname(),"torch":str(torch.__version__)}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if mode=="inner" else []})
    started=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            idx=torch.multinomial(sampling,BATCH,replacement=True);u=torch.rand(BATCH,8,device="cuda")
            angle=torch.where(u[:,0]<.6,(4*u[:,1]).floor()*math.pi/2,(2*u[:,1]-1)*math.pi)
            scale=.55+.53*u[:,3];flip=torch.where(u[:,2]<.5,-1.,1.)
            center=ctr[idx]+(u[:,4:6]-.5)*1.4
            grid,_=align.transform_grids(center,scale,angle,flip)
            im=F.grid_sample(images[idx].float()/255,grid,align_corners=True,padding_mode="border")
            im=(im.clamp_min(1e-6).pow(.85+.3*u[:,6,None,None,None])+.07*(u[:,7,None,None,None]-.5)).clamp(0,1)
            target=F.grid_sample(targets[idx].float(),grid,align_corners=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                z,radius=model(im)
            p=(2*z[:,:4]).sigmoid()
            actual=(target[:,:4].sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
            radius_loss=F.smooth_l1_loss((radius-actual)*scale[:,None],torch.zeros_like(radius),beta=.6)
            loss=vote.mask_loss(z[:,:4],target[:,:4])+.65*vote.mask_loss(z[:,8:],target[:,4:])+0.16*radius_loss
            loss=loss+.025*F.smooth_l1_loss(native.rater_center(p),native.rater_center(target[:,:4]))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,4,error_if_nonfinite=True)
            lr=min(1,step/120)*(.03+.97*(1+math.cos(math.pi*step/steps))/2)
            for group,base in zip(opt.param_groups,(.00014,.000015,.0006)):group["lr"]=lr*base
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-started}
                log.write(json.dumps(row,allow_nan=False)+"\n");print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    rc.write_json(args.out/"fit.json",{**meta,"checkpoint":rc.hash_file(args.out/"final.pt"),"seconds":time.monotonic()-started})


@torch.inference_mode()
def predict(model,images,center,shift):
    logits=[];radii=[]
    for j in range(0,len(images),10):
        x=align.registered_ct(torch.as_tensor(images[j:j+10],device="cuda"),*shift).float()/255
        ctr=torch.as_tensor(center[j:j+10],device="cuda").float()
        value=0;total=0;r=0;count=0
        for scale in (.64,.83,1.02):
            s=x.new_full((len(x),),scale)
            for k in range(4):
                for flip in (-1.,1.):
                    g,back=align.transform_grids(ctr,s,torch.full_like(s,k*math.pi/2),torch.full_like(s,flip))
                    im=F.grid_sample(x,g,align_corners=True,padding_mode="border")
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                        z,rad=model(im)
                    pulled=F.grid_sample(z[:,:4],back,align_corners=True,padding_mode="border")
                    weight=(1-back.abs().amax(-1)).clamp(0,.05)[:,None]
                    value=value+weight*pulled;total=total+weight;r=r+rad*s[:,None];count+=1
        logits.append(torch.where(total>0,value/total.clamp_min(1e-6),-16.).cpu().numpy())
        radii.append((r/count).cpu().numpy())
    return np.concatenate(logits),np.concatenate(radii)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("data","init","align","out"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--mode",choices=("inner","full","smoke"),required=True);p.add_argument("--fold",type=int,choices=range(4),required=True)
    train(p.parse_args())
