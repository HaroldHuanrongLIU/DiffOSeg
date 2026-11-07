"""Two-stage complete-patient replicas and upstream re-decoding of their mean correction."""
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
from reproduction import canonical_guide as canonical
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import native_views as views
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/core_ensemble_protocol.md"
OFFSETS=(17000,34000)
STRENGTHS=(0.,.25,.5,1.)
MODES=("joint","output","single")


def folder(args,stage):
    prefix="smoke" if args.smoke else "inner"
    return args.run/f"{prefix}-{args.fold}-{args.member}-{stage}"


def train(args):
    images,truth,_,uids=native.inputs(args.data)
    outer,fit,valid=native.splits(uids,args.fold)
    people=lambda rows:{str(uids[i]).split("_")[0] for i in rows}
    if people(fit)&people(valid+outer["test_indices"]):
        raise ValueError("Training patients overlap withheld patients")
    stage=args.phase
    ids=fit[:48] if args.smoke else fit
    steps=5 if args.smoke else (rc.STEPS if stage=="v1" else align.STEPS)
    seed=(91103 if stage=="v1" else 144291)+args.fold+OFFSETS[args.member]
    dest=folder(args,stage)
    dest.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    raw=torch.from_numpy(images[ids,None]).cuda()
    masks=torch.from_numpy(truth[ids]).cuda()
    distances=np.stack([[rc.signed(mask) for mask in panel] for panel in truth[ids]]).astype(np.float16)
    sd=torch.from_numpy(distances).cuda()
    initialization=None
    shift=None
    if stage=="v1":
        model=rc.RankContours().cuda().to(memory_format=torch.channels_last)
    else:
        predecessor=folder(args,"v1")
        init=views.read(predecessor/"receipt.json")
        initialization=rc.hash_file(predecessor/"final.pt")
        if (init["train"]!=ids or init["checkpoint"]!=initialization or init["stage"]!="v1"
                or init["member"]!=args.member or init["fold"]!=args.fold or init["seed"]!=seed-53188
                or init["source"]!=rc.hash_file(__file__) or init["protocol"]!=rc.hash_file(PROTOCOL)):
            raise ValueError("Invalid first-stage initialization")
        patients=[str(uids[i]).split("_")[0] for i in ids]
        shift,_=align.estimate_shift(images[ids],truth[ids],patients)
        raw=align.registered_ct(raw,*shift)
        targets=torch.cat((masks.float(),vote.votes(masks),sd.float()),dim=1).half()
        centers=native.rater_center(masks)
        centers=torch.where((centers-64.5).abs().amax(1,keepdim=True)>4,centers,64.5)
        _,inverse,counts=np.unique(patients,return_inverse=True,return_counts=True)
        weights=torch.from_numpy(.65/len(ids)+.35/(len(counts)*counts[inverse])).float().cuda()
        model=vote.build(predecessor/"final.pt")
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params=list(model.parameters());mean_params=list(ema.parameters())
    rate=.001 if stage=="v1" else .00055
    opt=torch.optim.AdamW(params,lr=rate,weight_decay=.0001,fused=True)
    samples=hashlib.sha256()
    with (dest/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            if stage=="v1":
                ix=torch.randint(len(ids),(rc.BATCH,),device="cuda")
                x,y,target_distance=rc.augment(raw[ix],masks[ix],sd[ix])
            else:
                ix=torch.multinomial(weights,align.BATCH,replacement=True)
                u=torch.rand(align.BATCH,9,device="cuda")
                angle=torch.where(u[:,0]<.65,(u[:,1]*4).floor()*math.pi/2,(2*u[:,1]-1)*math.pi)
                scale=.65+.58*u[:,3]
                grid,_=align.transform_grids(centers[ix]+(u[:,4:6]-.5)*1.2,scale,angle,torch.where(u[:,2]<.5,-1.,1.))
                x=F.grid_sample(raw[ix].float()/255,grid,align_corners=True,padding_mode="border")
                x=(x.clamp_min(1e-6).pow(.85+.3*u[:,6,None,None,None])*(.87+.26*u[:,7,None,None,None])
                   +.07*(u[:,8,None,None,None]-.5)).clamp(0,1)
                y=F.grid_sample(targets[ix].float(),grid,align_corners=True)
                target_distance=(y[:,8:]/(8*scale[:,None,None,None])).clamp(-1,1)
            samples.update(ix.cpu().numpy().astype("<i4").tobytes())
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                output=model(x)
            pref=output[:,:4]
            distance=F.smooth_l1_loss(output[:,4:8].tanh(),target_distance)
            loss=vote.mask_loss(pref,y[:,:4])+(.20 if stage=="v1" else .15)*distance
            if stage=="registered":
                prob=(2*pref).sigmoid()
                area=F.smooth_l1_loss((prob.sum((-2,-1))+1).log(),(y[:,:4].sum((-2,-1))+1).log())
                center=F.smooth_l1_loss(native.rater_center(prob),native.rater_center(y[:,:4]))
                loss=loss+.75*vote.mask_loss(output[:,8:],y[:,4:8])+.20*area+.025*center
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=rate*min(1,step/100)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(mean_params,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                log.write(json.dumps({"step":step,"loss":loss.item()},allow_nan=False)+"\n")
    checkpoint=dest/"final.pt"
    torch.save(ema.state_dict(),checkpoint)
    rc.write_json(dest/"receipt.json",{"fold":args.fold,"member":args.member,"stage":stage,"seed":seed,"steps":steps,"smoke":args.smoke,
        "train":ids,"excluded_validation":valid,"shift":shift,"initialization":initialization,"sample_hash":samples.hexdigest(),
        "checkpoint":rc.hash_file(checkpoint),"data":rc.DATA_HASH,"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("CORE_MEMBER",args.fold,args.member,stage,steps,flush=True)


def lock(args):
    images,truth,fit,lib,check,e,reference,old,centers=views.eval_context(args)
    ids=lib+check;n=len(lib)
    path,fr=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Incorrect guide order")
        guide=z["adapted"].astype(np.float32)
    if fr["payload"]!=old["field"] or old["choices"]["adapted"]["weight"]!=.25:
        raise ValueError("Unexpected E fusion settings")
    warm,identity=views.warm_identity(args,fit)
    def primary(directory):
        vals=align.predict(vote.load_net(directory/"final.pt"),images[ids,None],centers,identity["applied_shift"])[:,:4]
        if not np.isfinite(vals).all():raise ValueError("Nonfinite primary logits")
        return vals
    logits=[primary(warm)];hashes=[identity["checkpoint"]]
    for member,offset in enumerate(OFFSETS):
        d=args.run/f"inner-{args.fold}-{member}-registered"
        r=views.read(d/"receipt.json")
        a=args.run/f"inner-{args.fold}-{member}-v1";first=views.read(a/"receipt.json")
        if (r["train"]!=fit or r["steps"]!=6000 or first["steps"]!=4000 or first["train"]!=fit or r["smoke"] or first["smoke"]
            or r["seed"]!=144291+args.fold+offset or first["seed"]!=91103+args.fold+offset or r["shift"]!=identity["applied_shift"]
            or r["initialization"]!=first["checkpoint"] or first["checkpoint"]!=rc.hash_file(a/"final.pt")
            or r["source"]!=rc.hash_file(__file__) or r["protocol"]!=rc.hash_file(PROTOCOL) or r["checkpoint"]!=rc.hash_file(d/"final.pt")):
            raise ValueError("Unexpected independent two-stage fit")
        logits.append(primary(d));hashes.append(r["checkpoint"])
    average=np.mean(logits,axis=0,dtype=np.float32)
    delta=average-logits[0];single=logits[1]-logits[0]
    view=cache_owner.verified_parent(args)
    models=[]
    for directory,_ in canonical.model_sources(args,fit,args.fold):
        model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
        model.load_state_dict(torch.load(directory/"final.pt",weights_only=True,map_location="cpu"))
        models.append(model)
    def decode(shifted):
        outputs=[prompt.predict(model,view.features,ids,images,shifted).astype(np.float16) for model in models]
        result=mean_saved_logits(outputs)
        if not np.isfinite(result).all():raise ValueError("Nonfinite prompt logits")
        return .75*shifted+.25*result
    initial=decode(guide)
    if not np.array_equal(initial>0,reference):raise ValueError("Zero strength must exactly reproduce E")
    arrays={"reference":np.packbits(reference,axis=-1),"primary_average":np.packbits(average>0,axis=-1)}
    for j,v in enumerate(logits):arrays[f"primary_{j}"]=np.packbits(v>0,axis=-1)
    choices={}
    for mode in MODES:
        grid=[];candidates=[]
        for w in STRENGTHS:
            if w==0:mask=reference
            elif mode=="output":mask=initial+.75*w*delta>0
            else:mask=decode(guide+w*(single if mode=="single" else delta))>0
            grid.append(float(metrics.dice(truth[lib],mask[:n]).mean()))
            candidates.append(np.packbits(mask,axis=-1))
        selected=int(np.argmax(grid));arrays[mode]=candidates[selected]
        choices[mode]={"weight":STRENGTHS[selected],"library_scores":grid}
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"reference":old["prediction"],
         "guide":fr["payload"],"members":hashes,"choices":choices,"prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL),
         "source":rc.hash_file(__file__)})
    print("CORE_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("v1","registered","lock"))
    for key in ("run","data","align","local","texture","stable","formal","parent","ensemble","public"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--member",type=int,choices=range(2),default=0);p.add_argument("--smoke",action="store_true")
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(3)
    {"v1":train,"registered":train,"lock":lock}[args.phase](args)
