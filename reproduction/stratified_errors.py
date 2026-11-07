"""Out-of-patient difficulty weighting within nodule size and reader-disagreement strata."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as score
from reproduction import native_views as views
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/stratified_errors_protocol.md"
MODES=("uniform","raw","stratified")


def percentile(values):
    return (rankdata(values,method="average")-.5)/len(values)


def difficulty(truth,teacher):
    error=1-score.dice(truth,teacher)
    spread=np.mean([1-score.dice(truth,np.roll(truth,k,axis=1)) for k in (1,2,3)],axis=0)
    area=truth.sum((2,3))
    raw=np.empty_like(error)
    stratified=np.empty_like(error)
    groups=np.zeros(error.shape,dtype=np.uint8)
    for rank in range(4):
        def bins(values):
            return np.searchsorted(np.quantile(values,[1/3,2/3]),values,side="right")
        group=3*bins(area[:,rank])+bins(spread[:,rank])
        groups[:,rank]=group
        raw[:,rank]=.5+percentile(error[:,rank])
        for value in np.unique(group):
            selected=group==value
            stratified[selected,rank]=.5+percentile(error[selected,rank])
    return error,spread,groups,raw.astype(np.float32),stratified.astype(np.float32)


def prepare(args):
    _,truth,_,uids=native.inputs(args.data)
    ids,logits,receipt=nested.base_bank(args,uids,args.fold)
    errors,spread,groups,raw,balanced=difficulty(truth[ids].astype(bool),logits>0)
    path=args.run/f"weights-{args.fold}.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path,ids=ids,error=errors,spread=spread,groups=groups,raw=raw,stratified=balanced)
    rc.write_json(path.with_suffix(".json"),{"fold":args.fold,"ids":ids,"oof":receipt["payload"],"teachers":receipt["teachers"],
        "payload":rc.hash_file(path),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),
        "changed_high_quartile":float(np.mean((raw>1.25)!=(balanced>1.25)))})
    print("ERROR_WEIGHTS",args.fold,len(ids),flush=True)


def mask_objective(logits,truth,weight):
    probability=logits.sigmoid()
    dice=1-((2*(probability*truth).sum((-2,-1))+1)/((probability+truth).sum((-2,-1))+1))
    bce=(F.binary_cross_entropy_with_logits(logits,truth,reduction="none")*(1+4*truth)).mean((-2,-1))
    return ((dice+bce)*weight).sum()/weight.sum()


def ramp(step):
    return min(1.,max((step-1)/499,0.),max((6000-step)/1500,0.))


def train(args):
    images,truth,_,uids=native.inputs(args.data)
    _,train_ids,valid=native.splits(uids,args.fold)
    warm,warm_fit=views.warm_identity(args,train_ids)
    parent=args.previous/f"fold-{args.fold}-inner"
    if (views.read(parent/"plan.json")["train_indices"]!=train_ids or
            views.read(parent/"result.json")["checkpoint_sha256"]!=warm_fit["initialization"] or
            rc.hash_file(parent/"final.pt")!=warm_fit["initialization"]):
        raise ValueError("The v1 seed model must be fitted on the original eligible patients")
    bank=args.run/f"weights-{args.fold}.npz"
    rec=views.read(bank.with_suffix(".json"))
    if (rec["oof"]!=rc.hash_file(args.original/f"oof-{args.fold}.npz") or rec["payload"]!=rc.hash_file(bank)
            or rec["source"]!=rc.hash_file(__file__) or rec["protocol"]!=rc.hash_file(PROTOCOL)):
        raise ValueError("OOF difficulty bank changed")
    with np.load(bank,allow_pickle=False) as z:
        expected=z["ids"].tolist()
        if expected!=sorted(train_ids):raise ValueError("OOF patient set differs")
        ids=train_ids[:48] if args.smoke else train_ids
        positions={i:j for j,i in enumerate(expected)}
        weights=(np.ones((len(ids),4),np.float32) if args.mode=="uniform" else
                 z[args.mode][[positions[i] for i in ids]])
    weights=torch.from_numpy(weights).cuda()
    folder=args.run/(f"smoke-{args.mode}-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.mode}")
    folder.mkdir(exist_ok=False)
    steps=5 if args.smoke else align.STEPS
    seed=144291+args.fold
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=True
    image=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*warm_fit["applied_shift"])
    masks=torch.from_numpy(truth[ids]).cuda()
    signed=np.stack([[rc.signed(m) for m in v] for v in truth[ids]]).astype(np.float16)
    targets=torch.cat((masks.float(),vote.votes(masks),torch.from_numpy(signed).cuda().float()),1).half()
    centers=native.rater_center(masks)
    centers=torch.where((centers-64.5).abs().amax(1,keepdim=True)>4,centers,64.5)
    _,inverse,count=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    sampling=torch.from_numpy(.65/len(ids)+.35/(len(count)*count[inverse])).float().cuda()
    model=vote.build(parent/"final.pt")
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    parameters=list(model.parameters());averaged=list(ema.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=.00055,weight_decay=1e-4,fused=True)
    schedule=hashlib.sha256()
    with (folder/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(sampling,20,replacement=True)
            u=torch.rand(20,9,device="cuda")
            schedule.update(ix.cpu().numpy().astype("<i4").tobytes()+u.cpu().numpy().astype("<f4").tobytes())
            angle=torch.where(u[:,0]<.65,(u[:,1]*4).floor()*math.pi/2,(2*u[:,1]-1)*math.pi)
            flip=torch.where(u[:,2]<.5,-1.,1.);scale=.65+.58*u[:,3]
            grid,_=align.transform_grids(centers[ix]+(u[:,4:6]-.5)*1.2,scale,angle,flip)
            x=F.grid_sample(image[ix].float()/255,grid,align_corners=True,padding_mode="border")
            x=(x.clamp_min(1e-6).pow(.85+.3*u[:,6,None,None,None])*(.87+.26*u[:,7,None,None,None])+
               .07*(u[:,8,None,None,None]-.5)).clamp(0,1)
            y=F.grid_sample(targets[ix].float(),grid,align_corners=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                out=model(x)
            z=out[:,:4];p=(2*z).sigmoid()
            area=F.smooth_l1_loss((p.sum((-2,-1))+1).log(),(y[:,:4].sum((-2,-1))+1).log())
            center=F.smooth_l1_loss(native.rater_center(p),native.rater_center(y[:,:4]))
            dist=F.smooth_l1_loss(out[:,4:8].tanh(),(y[:,8:]/(scale[:,None,None,None]*8)).clamp(-1,1))
            weight=1+ramp(step)*(weights[ix]-1)
            loss=mask_objective(z,y[:,:4],weight)+.75*vote.mask_loss(out[:,8:],y[:,4:8])+.15*dist+.20*area+.025*center
            if step==1 and not args.smoke:
                with (warm/"train.jsonl").open() as st:prior=json.loads(next(st))["loss"]
                if not np.isclose(loss.item(),prior,rtol=3e-4,atol=5e-5):
                    raise ValueError(f"Initial supervised objective differs from original: {loss.item()} vs {prior}")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True)
            optimizer.param_groups[0]["lr"]=.00055*min(1,step/100)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(averaged,parameters,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                log.write(json.dumps({"step":step,"loss":loss.item(),"weight_min":weight.min().item(),
                      "weight_max":weight.max().item()},allow_nan=False)+"\n")
    dst=folder/"final.pt";torch.save(ema.state_dict(),dst)
    rc.write_json(folder/"fit.json",{"fold":args.fold,"mode":args.mode,"steps":steps,"smoke":args.smoke,"seed":seed,
        "train":ids,"validation":valid,"initial":warm_fit["initialization"],"shift":warm_fit["applied_shift"],
        "bank":rec["payload"],"oof":rec["oof"],"teachers":rec["teachers"],"source":rc.hash_file(__file__),
        "protocol":rc.hash_file(PROTOCOL),"schedule":schedule.hexdigest(),"checkpoint":rc.hash_file(dst)})
    print("RANK_DIFFICULTY_FIT",args.fold,args.mode,steps,flush=True)


def lock(args):
    images,truth,train_ids,lib,check,base,mask,anchor,centers=views.eval_context(args)
    warm,warm_fit=views.warm_identity(args,train_ids)
    ids=lib+check
    def predict(path):
        value=align.predict(vote.load_net(path),images[ids,None],centers,warm_fit["applied_shift"])[:,:4]
        if not np.isfinite(value).all():raise ValueError("Nonfinite primary model output")
        return value
    old=predict(warm/"final.pt")
    arrays={"reference":np.packbits(mask,axis=-1),"initial":np.packbits(old>0,axis=-1)}
    bank=args.run/f"weights-{args.fold}.npz"
    members={};choices={};schedules=[];starts=[]
    for name in MODES:
        folder=args.run/f"fold-{args.fold}-{name}";fit=views.read(folder/"fit.json")
        if (fit["fold"]!=args.fold or fit["mode"]!=name or fit["steps"]!=6000 or fit["smoke"] or fit["train"]!=train_ids
                or fit["initial"]!=warm_fit["initialization"] or fit["source"]!=rc.hash_file(__file__)
                or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")
                or fit["bank"]!=rc.hash_file(bank) or fit["oof"]!=rc.hash_file(args.original/f"oof-{args.fold}.npz")):
            raise ValueError("Invalid retraining identity")
        new=predict(folder/"final.pt")
        candidate,choices[name]=views.insert(base,old,new,truth[lib].astype(bool))
        arrays[name]=np.packbits(candidate,axis=-1);arrays["standalone_"+name]=np.packbits(new>0,axis=-1)
        members[name]=fit["checkpoint"];schedules.append(fit["schedule"])
        with (folder/"train.jsonl").open() as st:starts.append(json.loads(next(st))["loss"])
    if len(set(schedules))!=1 or not np.allclose(starts,starts[0],rtol=1e-5,atol=1e-5):
        raise ValueError("Matched sampling or initial loss differs")
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,
        "reference":anchor["prediction"],"warm":warm_fit["checkpoint"],"bank":rc.hash_file(bank),"members":members,
        "choices":choices,"schedule":schedules[0],"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),
        "prediction":rc.hash_file(dst)})
    print("RANK_DIFFICULTY_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("prepare","fit","lock"))
    for key in ("run","data","align","original","previous","stable","formal","parent","ensemble","public","local","texture"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--mode",choices=MODES,default="stratified");p.add_argument("--smoke",action="store_true")
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(3)
    {"prepare":prepare,"fit":train,"lock":lock}[args.phase](args)
