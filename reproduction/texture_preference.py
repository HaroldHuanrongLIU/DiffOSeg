"""Registered, single-slice preference segmentation with training-only reader descriptors."""
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

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/protocol.md"
ATTRIBUTES = ("margin", "texture", "lobulation", "spiculation", "sphericity", "subtlety")
STEPS, BATCH = 8000, 16


def labels_for_attributes(path, indices):
    with path.open("rb") as stream:
        data = list(pickle.load(stream).values())
    values = np.zeros((len(indices),len(ATTRIBUTES)), np.float32)
    available = np.zeros_like(values, dtype=bool)
    for j,idx in enumerate(indices):
        annotations = data[idx].get("annotations") or []
        for k,name in enumerate(ATTRIBUTES):
            vals = [a.get(name) for a in annotations if isinstance(a,dict)]
            vals = [v for v in vals if isinstance(v,(int,float,np.number)) and 1 <= v <= 5]
            if vals:
                values[j,k] = (np.mean(vals)-1)/4
                available[j,k] = True
    return values,available


class TextureNet(native.LocalExperts):
    def __init__(self,initial=None):
        super().__init__(initial)
        self.attr = nn.Sequential(nn.LayerNorm(768),nn.Linear(768,96),nn.SiLU(),nn.Linear(96,len(ATTRIBUTES)))
        self.modulate = nn.Linear(len(ATTRIBUTES),48)
        nn.init.zeros_(self.modulate.weight);nn.init.zeros_(self.modulate.bias)

    def forward(self,image):
        window = native.ct_windows(image)
        size = image.shape[-1]
        xy = native.coordinates(size,image.device).permute(2,0,1)[None]*(2/(size-1))-1
        fine = self.fine(torch.cat((window,xy.expand(len(image),-1,-1,-1)),1).contiguous(memory_format=torch.channels_last))
        half = self.half_res(fine)
        x = ((window-self.mean)/self.std).contiguous(memory_format=torch.channels_last)
        features = []
        for i,layer in enumerate(self.encoder):
            x = layer(x)
            if i in (1,3): features.append(x)
        h,w = x.shape[-2:]
        center = x[:,:,h//4:h-h//4,w//4:w-w//4].mean((-2,-1))
        attr = self.attr(torch.cat((x.mean((-2,-1)),center),1)).float().sigmoid()
        mid = self.up2(torch.cat((native.up(x,features[1]),features[1]),1))
        low = self.up1(torch.cat((native.up(mid,features[0]),features[0]),1))
        near = self.uphalf(torch.cat((native.up(low,half),half),1))
        final = self.up0(torch.cat((native.up(near,fine),fine),1))
        scale,bias = self.modulate(attr)[:,:,None,None].chunk(2,dim=1)
        z = self.head(final*(1+0.25*scale.tanh())+0.25*bias.tanh()).float()
        ranks = z[:,:1]+z[:,1:5]-z[:,1:5].mean(1,keepdim=True)
        return torch.cat((ranks,z[:,5:9]),1),attr


def decode(net,im,center,side,angle,flip):
    size=144
    grid,back=native.grids(center,native.matrix_for(side/(size-1),angle,flip),size)
    crop=F.grid_sample(im,grid,mode="bicubic",padding_mode="border",align_corners=True).clamp(0,1)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        logits,attributes=net(crop)
    return native.restore(logits,back),attributes


def load_model(path):
    net=TextureNet().cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
    return net


@torch.inference_mode()
def predict(net,images,centers,shift):
    parts=[]
    for offset in range(0,len(images),12):
        im=align.registered_ct(torch.as_tensor(images[offset:offset+12],device="cuda"),*shift).float()/255
        ctr=torch.as_tensor(centers[offset:offset+12],device="cuda").float()
        z=0
        for side in (76.,108.,136.):
            s=im.new_full((len(im),),side)
            for k in range(4):
                for flip in (-1.,1.):
                    value,_=decode(net,im,ctr,s,torch.full_like(s,k*math.pi/2),torch.full_like(s,flip))
                    z=z+value
        parts.append((z/24).cpu().numpy())
    out=np.concatenate(parts)
    if not np.isfinite(out).all(): raise ValueError("nonfinite single-slice prediction")
    return out


def train(args):
    cv2.setNumThreads(1);torch.set_num_threads(3)
    images,masks,_,uid=native.inputs(args.data)
    outer,inner,valid=native.splits(uid,args.fold)
    mode="inner" if args.mode=="smoke" else args.mode
    ids=inner if mode=="inner" else outer["train_indices"]
    parent=args.align/f"fold-{args.fold}-{mode}-registered"
    ref=parent/"plan.json"
    reg_ids=ids if args.mode!="smoke" else ids[:48]
    shift,_=align.estimate_shift(images[reg_ids],masks[reg_ids],[str(uid[i]).split("_")[0] for i in reg_ids])
    shift=list(shift)
    if ref.exists() and args.mode!="smoke":
        plan=json.loads(ref.read_text())
        if plan["train_indices"]!=ids or plan["data_hash"]!=rc.DATA_HASH or plan["applied_shift"]!=shift:
            raise ValueError("Registration split/bytes or estimate differ")
    steps=STEPS
    if args.mode=="smoke": ids,steps=ids[:48],5
    seed=196031+args.fold
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    args.out.mkdir(parents=True,exist_ok=False)
    attr,known=labels_for_attributes(args.data,ids)
    _,inv,n=np.unique([str(uid[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    groups=np.where(known[:,1],np.floor(attr[:,1]*2.99),-1)
    _,gi,gn=np.unique(groups,return_inverse=True,return_counts=True)
    weights=torch.from_numpy(.50/len(ids)+.35/(len(n)*n[inv])+.15/(len(gn)*gn[gi])).float().cuda()
    im=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*shift)
    truth=torch.from_numpy(masks[ids]).cuda().float()
    attrs=torch.from_numpy(attr).cuda(); flags=torch.from_numpy(known).cuda().float()
    center=native.rater_center(truth)
    center=torch.where((center-64.5).abs().amax(1,keepdim=True)>4,center,torch.full_like(center,64.5))
    model=TextureNet(args.init).cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params,avg=list(model.parameters()),list(ema.parameters())
    enc=list(model.encoder.parameters());enc_ids={id(p) for p in enc}
    opt=torch.optim.AdamW([{"params":enc},{"params":[p for p in params if id(p) not in enc_ids]}],
                          lr=.00065,weight_decay=.0003,fused=True)
    meta={"fold":args.fold,"mode":args.mode,"steps":steps,"batch":BATCH,"seed":seed,"shift":shift,
          "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"data":rc.DATA_HASH,"initialization":native.INIT,
          "registration_plan":rc.hash_file(ref) if ref.exists() else None,"train_attribute_counts":known.sum(0).tolist(),
          "host":socket.gethostname(),"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"cuda":torch.version.cuda}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"validation_indices":valid if mode=="inner" else []})
    started=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(weights,BATCH,replacement=True)
            x=im[ix].float()/255; y=truth[ix]; ctr=center[ix]
            u=torch.rand(BATCH,10,device="cuda")
            blur=F.avg_pool2d(x,3,1,1,count_include_pad=False)
            x=x+(u[:,0,None,None,None]-.55)*(x-blur)
            x=x+torch.randn_like(x)*(.011*u[:,1,None,None,None])
            x=(x.clamp_min(1e-6).pow(.82+.36*u[:,2,None,None,None])*(.83+.34*u[:,3,None,None,None])+
               .075*(u[:,4,None,None,None]-.5)).clamp(0,1)
            angle=torch.where(u[:,5]<.6,(4*u[:,6]).floor()*math.pi/2,(2*u[:,6]-1)*math.pi)
            flip=torch.where(u[:,7]<.5,-1.,1.)
            center_view=ctr+torch.randn_like(ctr)*(.3+1.25*u[:,8,None])
            z,pred_attr=decode(model,x,center_view,65+75*u[:,9],angle,flip)
            pref=z[:,:4];v=vote.votes(y)
            p=(2*pref).sigmoid()
            hard=(pref>0).float()+(p-p.detach())
            radius=(hard.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
            target=(y.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
            loss=vote.mask_loss(pref,y)+.75*vote.mask_loss(z[:,4:],v)
            loss=loss+.11*F.smooth_l1_loss(radius,target)+.035*F.smooth_l1_loss(native.rater_center(p),native.rater_center(y))
            aloss=F.smooth_l1_loss(pred_attr,attrs[ix],reduction="none",beta=.15)
            loss=loss+.22*(aloss*flags[ix]).sum()/flags[ix].sum().clamp_min(1)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            rate=min(1,step/180)*(.025+.975*(1+math.cos(math.pi*step/steps))/2)
            for group,lr in zip(opt.param_groups,(.00014,.0007)):group["lr"]=lr*rate
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,params,max(.004,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-started}
                log.write(json.dumps(row,allow_nan=False)+"\n");print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    rc.write_json(args.out/"fit.json",{**meta,"checkpoint":rc.hash_file(args.out/"final.pt"),"seconds":time.monotonic()-started})
    print("TEXTURE_FIT_COMPLETE",args.fold,args.mode,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("data","align","init","out"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--mode",choices=("smoke","inner","full"),required=True)
    train(parser.parse_args())
