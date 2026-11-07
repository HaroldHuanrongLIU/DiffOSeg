"""Train the aligned segmentation backbone through its native multi-view readout."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction import texture_study
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/native_views_protocol.md"
MODES = ("view","single","joint")
LEVELS = np.array([.74,.94,1.10],np.float32)
STEPS = 3000


def read(path):
    return json.loads(path.read_text())


def masked_preference(z,y,support):
    pi = z.sigmoid()
    weight = support.to(pi.dtype)
    overlap = (2*pi*y*weight).sum((-2,-1))+1
    mass = ((pi+y)*weight).sum((-2,-1))+1
    ce = F.binary_cross_entropy_with_logits(z,y,reduction="none")*(1+4*y)
    return 1-(overlap/mass).mean()+((ce*weight).sum((-2,-1))/weight.sum((-2,-1)).clamp_min(1)).mean()


def native_objective(view_logits,back,target,mode):
    pulled=F.grid_sample(view_logits,back,align_corners=True)
    confidence=(1-back.abs().amax(-1)).clamp(0,.05)[:,None]
    left,right=pulled.chunk(2)
    wl,wr=confidence.chunk(2)
    common=(wl>0)&(wr>0)
    if mode=="single":
        return .5*(masked_preference(left,target,common)+masked_preference(right,target,common))
    joint=(wl*left+wr*right)/(wl+wr).clamp_min(1e-8)
    return masked_preference(joint,target,common)


def warm_identity(args,ids):
    folder=args.align/f"fold-{args.fold}-inner-registered"
    fit=read(folder/"fit.json")
    plan=read(folder/"plan.json")
    if (sorted(plan["train_indices"])!=sorted(ids) or fit["data_hash"]!=rc.DATA_HASH or fit["mode"]!="inner"
            or fit["fold"]!=args.fold or fit["applied_shift"]!=[-1,-1] or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
        raise ValueError("Warm segmentation backbone has the wrong patient scope or identity")
    return folder,fit


def train(args):
    images,truth,_,uids=native.inputs(args.data)
    ids,logits,oof=nested.base_bank(args,uids,args.fold)
    warm,fit=warm_identity(args,ids)
    folder=args.run/(f"smoke-{args.mode}-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.mode}")
    folder.mkdir(exist_ok=False)
    if args.smoke:
        ids=ids[:32]
    centers=native.rater_center(torch.from_numpy(logits[:len(ids)]>0))
    centers=torch.where((centers-64.5).abs().amax(1,keepdim=True)>4,centers,64.5).cuda()
    del logits
    y=torch.from_numpy(truth[ids]).cuda().float()
    signed=np.stack([[rc.signed(m) for m in panels] for panels in truth[ids]])
    targets=torch.cat((y,vote.votes(y),torch.from_numpy(signed).cuda()),1).half()
    ct=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*fit["applied_shift"]).float()/255
    _,inverse,count=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    sampling=.65/len(ids)+.35/(len(count)*count[inverse])
    seed=307571+args.fold
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    model=vote.load_net(warm/"final.pt").train()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params=list(model.parameters())
    avg=list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=.00012,weight_decay=.0001,fused=True)
    steps=5 if args.smoke else STEPS
    schedule=hashlib.sha256()
    with (folder/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=rng.choice(len(ids),10,p=sampling)
            random=rng.random((10,12),dtype=np.float32)
            schedule.update(ix.astype("<i4").tobytes()+random.tobytes())
            first=(3*random[:,0]).astype(int)
            other=(first+1+(2*random[:,1]).astype(int))%3
            scale=torch.from_numpy(LEVELS[np.r_[first,other]]).cuda()
            u=torch.from_numpy(random).cuda()
            angle=torch.cat(((4*u[:,2]).floor(),(4*u[:,3]).floor()))*(math.pi/2)
            flip=torch.where(torch.cat((u[:,4],u[:,5]))<.5,-1.,1.)
            center=centers[ix]+1.2*(u[:,6:8]-.5)
            grid,back=align.transform_grids(center.repeat(2,1),scale,angle,flip)
            x=F.grid_sample(ct[ix].repeat(2,1,1,1),grid,align_corners=True,padding_mode="border")
            phot=u[:,8:11].repeat(2,1)[:,:,None,None]
            x=(x.clamp_min(1e-6).pow(.85+.3*phot[:,0:1])*(.87+.26*phot[:,1:2])+.07*(phot[:,2:3]-.5)).clamp(0,1)
            labels=F.grid_sample(targets[ix].float().repeat(2,1,1,1),grid,align_corners=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                output=model(x)
            z=output[:,:4].float()
            pref=vote.mask_loss(z,labels[:,:4])
            area=F.smooth_l1_loss(((2*z).sigmoid().sum((-2,-1))+1).log(),(labels[:,:4].sum((-2,-1))+1).log())
            centroid=F.smooth_l1_loss(native.rater_center((2*z).sigmoid()),native.rater_center(labels[:,:4]))
            dist=F.smooth_l1_loss(output[:,4:8].tanh(),(labels[:,8:]/(8*scale[:,None,None,None])).clamp(-1,1))
            ordinary=pref+.75*vote.mask_loss(output[:,8:],labels[:,4:8])+.15*dist+.20*area+.025*centroid
            extra=pref if args.mode=="view" else native_objective(z,back,y[ix],args.mode)
            if step==1 and args.smoke:
                gradient=torch.autograd.grad(extra,z,retain_graph=True)[0]
                if not torch.isfinite(gradient).all() or any(v.abs().sum().item()==0 for v in gradient.chunk(2)):
                    raise ValueError("Missing preference gradient in one of the views")
            loss=ordinary+.4*extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.00012*min(step/100,1)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                log.write(json.dumps({"step":step,"ordinary":ordinary.item(),"extra":extra.item(),"loss":loss.item()},allow_nan=False)+"\n")
    dst=folder/"final.pt"
    torch.save(ema.state_dict(),dst)
    rc.write_json(folder/"fit.json",{"fold":args.fold,"mode":args.mode,"steps":steps,"seed":seed,"smoke":args.smoke,
        "train":ids,"warm":fit["checkpoint"],"shift":fit["applied_shift"],"oof":oof["payload"],"teachers":oof["teachers"],
        "schedule":schedule.hexdigest(),"checkpoint":rc.hash_file(dst),"data":rc.DATA_HASH,
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("NATIVE_VIEWS_FIT",args.fold,args.mode,steps,flush=True)


def eval_context(args):
    images,truth,train,lib,check,base,mask,rec,view,model=nested.e_guidance(args)
    del view,model
    if not np.array_equal(base>0,mask):
        raise ValueError("Zero correction must reproduce E")
    locator=args.local/f"fold-{args.fold}-inner-locator"
    if sorted(read(locator/"plan.json")["train_indices"])!=sorted(train):
        raise ValueError("Wrong fitting patients for image-only locator")
    path=args.texture/f"features-{args.fold}-inner.npz"
    binding=read(path.with_suffix(".json"))
    if binding["indices"]!=lib+check or binding["payload"]!=rc.hash_file(path):
        raise ValueError("Image-only locator cache differs")
    with np.load(path,allow_pickle=False) as archive:
        if archive["ids"].tolist()!=lib+check:
            raise ValueError("Locator order differs")
        centers=archive["centers"]
    live=texture_study.centers(args,"inner",images[lib+check])
    np.testing.assert_allclose(centers,live,rtol=0,atol=1e-3)
    return images,truth,train,lib,check,base,mask,rec,centers


def insert(base,old,new,library_truth):
    difference=new-old
    scores=[]
    for weight in (0.,.25,.5,1.):
        candidate=(base[:len(library_truth)]+weight*difference[:len(library_truth)])>0
        scores.append(float(scoring.dice(library_truth,candidate).mean()))
    selected=(0.,.25,.5,1.)[int(np.argmax(scores))]
    return base+selected*difference>0,{"weight":selected,"library_scores":scores}


def lock(args):
    images,truth,train,lib,check,base,mask,rec,centers=eval_context(args)
    warm,warm_fit=warm_identity(args,train)
    ids=lib+check
    def predict(path):
        out=align.predict(vote.load_net(path),images[ids,None],centers,warm_fit["applied_shift"])[:,:4]
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite native prediction")
        return out
    old=predict(warm/"final.pt")
    arrays={"reference":np.packbits(mask,axis=-1),"initial":np.packbits(old>0,axis=-1)}
    members={};choices={};schedules=[];first=[]
    for name in MODES:
        folder=args.run/f"fold-{args.fold}-{name}"
        fit=read(folder/"fit.json")
        if (fit["fold"]!=args.fold or fit["mode"]!=name or fit["steps"]!=STEPS or fit["smoke"]
                or fit["train"]!=sorted(train) or fit["warm"]!=warm_fit["checkpoint"]
                or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or fit["source"]!=rc.hash_file(__file__)
                or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["oof"]!=rc.hash_file(args.original/f"oof-{args.fold}.npz")):
            raise ValueError("Invalid native-view training identity")
        new=predict(folder/"final.pt")
        final,choices[name]=insert(base,old,new,truth[lib].astype(bool))
        arrays[name]=np.packbits(final,axis=-1)
        arrays["standalone_"+name]=np.packbits(new>0,axis=-1)
        members[name]=fit["checkpoint"]
        schedules.append(fit["schedule"])
        with (folder/"train.jsonl").open() as st: first.append(json.loads(next(st))["ordinary"])
    if len(set(schedules))!=1 or not np.allclose(first,first[0],atol=1e-5,rtol=1e-5):
        raise ValueError("Unequal training inputs or warm first objective")
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"reference":rec["prediction"],
        "members":members,"choices":choices,"center_bank":rc.hash_file(args.texture/f"features-{args.fold}-inner.npz"),
        "warm":warm_fit["checkpoint"],"schedule":schedules[0],"prediction":rc.hash_file(dst),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("NATIVE_VIEWS_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","lock"))
    for key in ("run","data","align","original","stable","formal","parent","ensemble","public","local","texture"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--mode",choices=MODES,default="joint")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    {"fit":train,"lock":lock}[args.phase](args)
