"""Flow matching of four single-slice contour residuals with matched spatial-noise controls."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction.rater_area import sdf
from reproduction.semantic_fusion import ct_and_split

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/contour_flow_protocol.md"
MODES = ("white","smooth")
STEPS = 6000


def distance(masks):
    shape = masks.shape
    return np.stack([sdf(m).clip(-6,6) for m in masks.reshape(-1,*shape[-2:])]).reshape(shape).astype(np.float32)


def unit_smooth(x):
    radius = 7
    coord = torch.arange(-radius,radius+1,device=x.device,dtype=x.dtype)
    kernel = torch.exp(-.5*(coord/2.5).square())
    kernel = kernel/kernel.square().sum().sqrt()
    shape = x.shape
    planes = x.flatten(0,1)[:,None]
    y = F.conv2d(F.pad(planes,(radius,radius,0,0),mode="circular"),kernel[None,None,None,:])
    return F.conv2d(F.pad(y,(0,0,radius,radius),mode="circular"),kernel[None,None,:,None]).reshape(shape)


def noise(draw, mode):
    # Two independent banks, each with one shared and four rank-specific channels.
    local,broad = draw.chunk(2,dim=1)
    broad = unit_smooth(broad) if mode=="smooth" else broad
    mix = .5*local+math.sqrt(.75)*broad
    return (mix[:,:1]+mix[:,1:])/math.sqrt(2)


def noise_amplitude(guide):
    return 1.5*(.25+.75*torch.exp(-guide.abs()/4))


class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(13,24,3,padding=1),nn.GroupNorm(8,24),nn.SiLU(),rc.Residual(24,24))
        self.middle = rc.Residual(24,48,2)
        self.deep = nn.Sequential(rc.Residual(48,72,2),rc.Residual(72,72))
        self.up_middle = rc.Residual(120,48)
        self.up_fine = rc.Residual(72,24)
        self.out = nn.Conv2d(24,4,1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self,image,guide,residual,t):
        low = F.avg_pool2d(image,7,1,3,count_include_pad=False)
        timing = torch.cat((t,torch.cos(math.pi*t),torch.sin(math.pi*t)),1).expand(-1,-1,*image.shape[-2:])
        fine=self.stem(torch.cat((2*image-1,4*(image-low),guide/6,residual/3,timing),1))
        mid=self.middle(fine)
        deep=self.deep(mid)
        up=F.interpolate(deep,size=mid.shape[-2:],mode="bilinear",align_corners=False)
        middle=self.up_middle(torch.cat((up,mid),1))
        up=F.interpolate(middle,size=fine.shape[-2:],mode="bilinear",align_corners=False)
        return self.out(self.up_fine(torch.cat((up,fine),1))).float()


def interpolant(start,end,t):
    velocity=end-start
    return start+t*velocity,velocity


def train(args):
    images,truth,_,uids=native.inputs(args.data)
    ids,logits,bank=nested.base_bank(args,uids,args.fold)
    # Each guide teacher's own CT registration must agree with the training geometry.
    shifts=[json.loads((args.original/f"fold-{args.fold}-sub-{j}/fit.json").read_text())["shift"] for j in range(3)]
    if shifts!=[[-1,-1]]*3:
        raise ValueError("Registered CT and OOF coordinates differ")
    folder=args.run/(f"smoke-{args.mode}-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.mode}")
    folder.mkdir(exist_ok=False)
    limit = 32 if args.smoke else len(ids)
    ids=ids[:limit]
    guide=torch.from_numpy(distance(logits[:limit]>0)).cuda()
    del logits
    labels=torch.from_numpy(truth[ids]).cuda().float()
    end=torch.from_numpy(distance(truth[ids])).cuda()-guide
    ct=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),-1,-1).float()/255
    seed=385171+args.fold
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    rng_noise=torch.Generator(device="cuda").manual_seed(seed+300)
    _,inverse,counts=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    weights=.45/len(ids)+.55/(len(counts)*counts[inverse])
    model=Velocity().cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params=list(model.parameters())
    avg=list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=.00035,weight_decay=.0002)
    schedule=hashlib.sha256()
    steps=5 if args.smoke else STEPS
    with (folder/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=rng.choice(len(ids),12,p=weights)
            k=int(rng.integers(4))
            times=rng.uniform(.005,.995,size=len(ix)).astype(np.float32)
            schedule.update(ix.astype("<i4").tobytes()+bytes([k])+times.tobytes())
            x,g,y,d=(torch.rot90(z[ix],k,(-2,-1)) for z in (ct,guide,labels,end))
            t=torch.from_numpy(times).to("cuda").reshape(-1,1,1,1)
            draw=torch.randn((len(ix),10,128,128),device="cuda",generator=rng_noise)
            if step==1:
                first_noise=hashlib.sha256(draw.cpu().numpy().tobytes()).hexdigest()
            start=noise_amplitude(g)*noise(draw,args.mode)
            current,target=interpolant(start,d,t)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                velocity=model(x,g,current,t)
            band=.08+torch.exp(-g.abs()/2)
            speed=((velocity-target).square()*band).sum()/band.sum()
            prob=(2.5*(g+current+(1-t)*velocity)).sigmoid()
            dice=1-((2*(prob*y).sum((-2,-1))+1)/(prob.sum((-2,-1))+y.sum((-2,-1))+1)).mean()
            loss=speed+.4*dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,4,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.00035*min(step/100,1)*(.04+.96*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                log.write(json.dumps({"step":step,"speed":speed.item(),"dice_loss":dice.item()},allow_nan=False)+"\n")
    dst=folder/"final.pt"
    torch.save(ema.state_dict(),dst)
    rc.write_json(folder/"fit.json",{"mode":args.mode,"fold":args.fold,"train":ids,"steps":steps,"smoke":args.smoke,
        "seed":seed,"schedule":schedule.hexdigest(),"noise0":first_noise,"checkpoint":rc.hash_file(dst),
        "oof":bank["payload"],"teachers":bank["teachers"],"protocol":rc.hash_file(PROTOCOL),
        "source":rc.hash_file(__file__),"data":rc.DATA_HASH})
    print("CONTOUR_FLOW_FIT",args.fold,args.mode,steps,flush=True)


@torch.inference_mode()
def endpoint(model,image,guide,mode,draws):
    midpoints=6
    total=torch.zeros_like(guide)
    for independent in draws:
        base=noise(independent,mode)
        for sign in (-1.,1.):
            residual=sign*base*noise_amplitude(guide)
            for j in range(midpoints):
                t=guide.new_full((len(guide),1,1,1),j/midpoints)
                v=model(image,guide,residual,t)
                v=model(image,guide,residual+v/(2*midpoints),t+.5/midpoints)
                residual=residual+v/midpoints
            total+=(2.5*(guide+residual)).sigmoid()
    return total/(2*len(draws))


def lock(args):
    if (rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER
            or rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT):
        raise ValueError("E identity changed")
    images,uids=ct_and_split(args)
    _,train,valid=native.splits(uids,args.fold)
    group=type("Group",(),{"series_uid":uids})()
    library,check=polar.calibration_split(group,valid,args.fold)
    ids=library+check
    archive=args.stable/f"pred-{args.fold}.npz"
    ref=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    if ref["prediction"]!=rc.hash_file(archive) or ref["library"]!=library or ref["check"]!=check:
        raise ValueError("E patient order differs")
    with np.load(archive,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:
            raise ValueError("Incomplete frozen E comparison")
        reference=np.unpackbits(z["adapted"],axis=-1).astype(bool)
    device=torch.device(args.device)
    image=align.registered_ct(torch.from_numpy(images[ids,None]).to(device),-1,-1).float()/255
    sd=torch.from_numpy(distance(reference)).to(device)
    original=(2.5*sd).sigmoid().cpu().numpy()
    if not np.array_equal(original>.5,reference):
        raise ValueError("Zero weight must reproduce E")
    model=Velocity().to(device).eval().to(memory_format=torch.channels_last)
    rng=torch.Generator(device=device).manual_seed(15391+args.fold)
    draws=torch.randn(8,1,10,128,128,device=device,generator=rng)
    predictions={}
    records={}
    for mode in MODES:
        folder=args.run/f"fold-{args.fold}-{mode}"
        r=json.loads((folder/"fit.json").read_text())
        if (r["fold"]!=args.fold or r["mode"]!=mode or r["steps"]!=STEPS or r["train"]!=sorted(train)
                or r["smoke"] or r["data"]!=rc.DATA_HASH or r["source"]!=rc.hash_file(__file__)
                or r["protocol"]!=rc.hash_file(PROTOCOL) or r["checkpoint"]!=rc.hash_file(folder/"final.pt")):
            raise ValueError("Unverified velocity model")
        records[mode]=r
        model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
        posterior=[]
        for j in range(0,len(ids),8):
            posterior.append(endpoint(model,image[j:j+8],sd[j:j+8],mode,draws).cpu().numpy())
        predictions[mode]=np.concatenate(posterior)
        if not np.isfinite(predictions[mode]).all():
            raise ValueError("Nonfinite trajectory")
    if any(records["white"][k]!=records["smooth"][k] for k in ("schedule","noise0","oof","teachers")):
        raise ValueError("Velocity ablation uses unmatched fitting inputs")
    # Only library truth is indexed for selection.
    _,labels,_,_=native.inputs(args.data)
    bank={"reference":np.packbits(reference,axis=-1)}
    choices={}
    band=sd.abs().cpu().numpy()<=3
    for mode,post in predictions.items():
        grid=[]
        options=(0.,.25,.5,.75,1.)
        def blend(w):
            return np.where(band,(1-w)*original+w*post>.5,reference)
        for w in options:
            chosen=blend(w)[:len(library)]
            y=labels[library].astype(bool)
            denom=y.sum((2,3))+chosen.sum((2,3))
            values=np.divide(2*(y&chosen).sum((2,3)),denom,out=np.ones(denom.shape),where=denom>0)
            grid.append(float(values.mean()))
        w=options[int(np.argmax(grid))]
        choices[mode]={"weight":w,"library_scores":grid}
        bank[mode]=np.packbits(blend(w),axis=-1)
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**bank)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":check,
          "reference":ref["prediction"],"choices":choices,"fits":{k:v["checkpoint"] for k,v in records.items()},
          "prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("CONTOUR_FLOW_LOCK",args.fold,choices,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","lock"))
    for key in ("run","data"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("original","stable","formal"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--mode",choices=MODES,default="smooth")
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    {"fit":train,"lock":lock}[args.phase](args)
