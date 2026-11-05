"""Training-fold registration before augmentation for single-slice preference prediction."""
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

from reproduction import native_local as native
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_90/alignment_protocol.md"
BATCH,STEPS=20,6000


def estimate_shift(images,labels,patients):
    shifts=[(x,y) for y in range(-2,3) for x in range(-2,3)]
    scores=[]
    for im,experts in zip(images,labels):
        m=(experts.sum(0)>=2).astype("uint8")
        inside=cv2.distanceTransform(m,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        outside=cv2.distanceTransform(1-m,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        y1,x1=np.where((inside>0)&(inside<2.9)); y0,x0=np.where((outside>0)&(outside<2.9))
        if not len(x1) or not len(x0):
            scores.append(np.zeros(len(shifts)));continue
        values=[]
        for x,y in shifts:
            a=im[np.clip(y1-y,0,127),np.clip(x1-x,0,127)]
            b=im[np.clip(y0-y,0,127),np.clip(x0-x,0,127)]
            values.append(a.mean()-b.mean())
        values=np.array(values)
        scores.append((values-values.mean())/max(np.ptp(values),1.))
    table=np.stack(scores)
    _,inv=np.unique(patients,return_inverse=True)
    surface=np.mean([table[inv==n].mean(0) for n in np.unique(inv)],0)
    best=int(np.argmax(surface))
    return shifts[best],surface.reshape(5,5).tolist()


def registered_ct(image,dx,dy):
    h,w=image.shape[-2:]
    yy=(torch.arange(h,device=image.device)-dy).clamp(0,h-1)
    xx=(torch.arange(w,device=image.device)-dx).clamp(0,w-1)
    return image.index_select(-2,yy).index_select(-1,xx)


def transform_grids(center,scale,angle,flip):
    matrix=native.matrix_for(scale,angle,flip)
    coords=native.coordinates(128,center.device).reshape(1,-1,2)
    src=(coords-64.5)@matrix.transpose(1,2)+center[:,None]
    inverse=(coords-center[:,None])@torch.linalg.inv(matrix).transpose(1,2)+64.5
    return src.reshape(-1,128,128,2)*(2/127)-1,inverse.reshape(-1,128,128,2)*(2/127)-1


def learn(args):
    torch.set_num_threads(3);cv2.setNumThreads(1)
    ims,labels,_,uid=native.inputs(args.data)
    outer,inner,valid=native.splits(uid,args.fold)
    mode="inner" if args.mode=="smoke" else args.mode
    ids=inner if mode=="inner" else outer["train_indices"]
    old=args.previous/f"fold-{args.fold}-{mode}"
    plan=json.loads((old/"plan.json").read_text())
    result=json.loads((old/"result.json").read_text())
    if plan["train_indices"]!=ids or rc.hash_file(old/"final.pt")!=result["checkpoint_sha256"]:
        raise ValueError("v1 initialization has incorrect training identity")
    steps=STEPS
    if args.mode=="smoke": ids,steps=ids[:48],5
    patients=[str(uid[i]).split("_")[0] for i in ids]
    estimated,surface=estimate_shift(ims[ids],labels[ids],patients)
    applied=(0,0) if args.identity_shift else estimated
    seed=144291+args.fold
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    args.out.mkdir(parents=True,exist_ok=False)
    image=registered_ct(torch.from_numpy(ims[ids,None]).cuda(),*applied)
    masks=torch.from_numpy(labels[ids]).cuda()
    signed=np.stack([[rc.signed(m) for m in item] for item in labels[ids]]).astype("float16")
    targets=torch.cat((masks.float(),vote.votes(masks),torch.from_numpy(signed).cuda().float()),1).half()
    ctr=native.rater_center(masks)
    ctr=torch.where((ctr-64.5).abs().amax(1,keepdim=True)>4,ctr,torch.full_like(ctr,64.5))
    _,inv,n=np.unique(patients,return_inverse=True,return_counts=True)
    weights=torch.from_numpy(.65/len(ids)+.35/(len(n)*n[inv])).float().cuda()
    model=vote.build(old/"final.pt")
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params,avg=list(model.parameters()),list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=.00055,weight_decay=1e-4,fused=True)
    meta={"fold":args.fold,"mode":args.mode,"steps":steps,"initialization":result["checkpoint_sha256"],
          "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"data_hash":rc.DATA_HASH,"seed":seed,
          "applied_shift":list(applied),"estimated_shift":list(estimated),"contrast_surface":surface,
          "host":socket.gethostname(),"gpu":torch.cuda.get_device_name(),"torch":str(torch.__version__),"cuda":torch.version.cuda}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if mode=="inner" else []})
    began=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(weights,BATCH,replacement=True)
            u=torch.rand(BATCH,9,device="cuda")
            angle=torch.where(u[:,0]<.65,(u[:,1]*4).floor()*math.pi/2,(2*u[:,1]-1)*math.pi)
            flip=torch.where(u[:,2]<.5,-1.,1.)
            scale=.65+.58*u[:,3]
            center=ctr[ix]+(u[:,4:6]-.5)*1.2
            grid,_=transform_grids(center,scale,angle,flip)
            x=F.grid_sample(image[ix].float()/255,grid,align_corners=True,padding_mode="border")
            x=(x.clamp_min(1e-6).pow(.85+.3*u[:,6,None,None,None])*(.87+.26*u[:,7,None,None,None])+
               .07*(u[:,8,None,None,None]-.5)).clamp(0,1)
            y=F.grid_sample(targets[ix].float(),grid,align_corners=True,padding_mode="zeros")
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()): z=model(x)
            pref=z[:,:4]
            prob=(2*pref).sigmoid()
            area=F.smooth_l1_loss((prob.sum((-2,-1))+1).log(),(y[:,:4].sum((-2,-1))+1).log())
            moment=F.smooth_l1_loss(native.rater_center(prob),native.rater_center(y[:,:4]))
            sd=F.smooth_l1_loss(z[:,4:8].tanh(),(y[:,8:]/(scale[:,None,None,None]*8)).clamp(-1,1))
            loss=vote.mask_loss(pref,y[:,:4])+.75*vote.mask_loss(z[:,8:],y[:,4:8])+.15*sd+.20*area+.025*moment
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.00055*min(1,step/100)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-began}
                log.write(json.dumps(row,allow_nan=False)+"\n");print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    rc.write_json(args.out/"fit.json",{**meta,"checkpoint":rc.hash_file(args.out/"final.pt"),"seconds":time.monotonic()-began})
    print("ALIGNED_FIT_COMPLETE",args.fold,args.mode,applied,flush=True)


@torch.inference_mode()
def predict(net,images,center,shift):
    out=[]
    for offset in range(0,len(images),12):
        image=registered_ct(torch.as_tensor(images[offset:offset+12],device="cuda"),*shift).float()/255
        ctr=torch.as_tensor(center[offset:offset+12],device="cuda").float()
        value=0;total=0
        for level in (.74,.94,1.10):
            for k in range(4):
                for reflection in (-1.,1.):
                    s=image.new_full((len(image),),level)
                    g,back=transform_grids(ctr,s,torch.full_like(s,k*math.pi/2),torch.full_like(s,reflection))
                    x=F.grid_sample(image,g,align_corners=True,padding_mode="border")
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()): pred=net(x).float()
                    z=F.grid_sample(pred,back,align_corners=True,padding_mode="border")
                    w=(1-back.abs().amax(-1)).clamp(0,.05)[:,None]
                    value=value+z*w; total=total+w
        out.append(torch.where(total>0,value/total.clamp_min(1e-8),-16.).cpu().numpy())
    return np.concatenate(out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("out","previous","data"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--mode",choices=("inner","full","smoke"),required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--identity-shift",action="store_true")
    learn(p.parse_args())
