"""Fusion of a directly trained spatial field with the frozen diffusion bank."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import boundary_eval as v1
from reproduction import contour_field as field
from reproduction import joint_energy as energy
from reproduction import joint_rater as joint
from reproduction import joint_study as v4
from reproduction import normal_study as pairing
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import vote_eval as v3

read,require=v2.read,v2.require
ARMS=("full","original_order","no_field")
NEW=(0.25,0.5,0.75,1.)
BLEND=(0.,0.5,1.)


def identity():
    return {"source":rc.hash_file(__file__),"field_source":rc.hash_file(field.__file__),
            "protocol":rc.hash_file(field.PROTOCOL),"pairing_source":rc.hash_file(pairing.__file__),
            "v4":pairing.V4_HASH}


def setup(args,mode):
    data,outer,valid,old,weights,previous=v4.inputs(args,mode)
    f=args.fold
    require(rc.hash_file(args.v4/"assessment/fourfold_comparison.json")==pairing.V4_HASH,"v4 identity")
    path=args.run/f"fold-{f}-{mode}"
    plan,result=read(path/"plan.json"),read(path/"result.json")
    expected=outer["train_indices"] if mode=="full" else rc.split_for(data,f)[1]
    require(plan["train_indices"]==expected and plan["inner_valid_indices"]==(valid if mode=="inner" else []) and
            (plan["fold"],plan["mode"],plan["initialization"])==(f,mode,field.WEIGHT_HASH) and
            result["source"]==rc.hash_file(field.__file__) and result["protocol"]==rc.hash_file(field.PROTOCOL) and
            result["checkpoint"]==rc.hash_file(path/"final.pt") and result["steps"]==field.STEPS,"field training identity")
    require(read(args.run/f"fold-{f}-inner/result.json")["passed"],"inner gate missing")
    calibration=read(args.v4/f"lock-{f}.json")
    require(all(calibration[k]==v for k,v in v4.identity().items()) and
            calibration["v3_lock"]==rc.hash_file(args.v3/f"lock-{f}.json"),"v4 parameter lock")
    return data,outer,valid,old,weights,previous,calibration,path


def field_ranks(basis,k,case):
    """Use the same Gaussian coefficient vector across the entire CT patch."""
    rng=np.random.default_rng(819733+case*100+k)
    z=rng.standard_normal((k//2,field.DIM)).astype(np.float32)
    z=np.concatenate((z,-z),axis=0)
    projected=z@basis.reshape(field.DIM,-1)
    order=np.argsort(projected,axis=0,kind="stable")
    ranks=np.empty(order.shape,np.uint8)
    np.put_along_axis(ranks,order,np.arange(k,dtype=np.uint8)[:,None],axis=0)
    return ranks.reshape(k,*basis.shape[-2:])


def mix_order(original,learned,weight):
    if weight==0:
        return original
    if weight==1:
        return learned
    # Use the diffusion order as the secondary key in exact midpoint ties.
    score=(1-weight)*original+weight*learned+original*1e-5
    indices=np.argsort(score,axis=0,kind="stable")
    ranks=np.empty_like(original)
    np.put_along_axis(ranks,indices,np.arange(len(ranks),dtype=np.uint8)[:,None,None],axis=0)
    return ranks


def preference_mask(old,new,alpha,prior):
    old_logit=np.log(np.clip(old,1e-6,1))-np.log(np.clip(1-old,1e-6,1))
    mask=((1-alpha)*old_logit+alpha*new)>0
    return 0.9*np.stack([rc.signed(m) for m in mask])+0.1*np.stack([rc.signed(m) for m in prior])>0


def inputs(data,ids,old,weights,previous,calibration,path):
    raw,cart=v4.features(data,ids,old,weights,previous["expert_alpha"])
    decoder=joint.JointDecoder.restore(calibration["parameters"],"cuda")
    both=v4.decoded(decoder,raw)
    raw_support=torch.from_numpy(raw[:,4:]).sigmoid().numpy()
    images=np.stack([data.images[i] for i in ids])[:,None]
    expert,freq,basis=field.predict(field.load_model(path/"final.pt"),images)
    return both,raw_support,cart,expert,freq,basis


@torch.inference_mode()
def anchor(readout,support,base):
    return readout(torch.from_numpy(support[None]),torch.from_numpy(base.astype(np.float32)[None]))[0].numpy()


def bank_at(q,rank):
    return rank<np.floor(len(rank)*q+0.5)[None]


def calibrate(args):
    data,_,valid,old,weights,previous,calibration,path=setup(args,"inner")
    lib,check=polar.calibration_split(data,valid,args.fold)
    require(lib==calibration["fit_indices"] and check==calibration["check_indices"],"inner identity")
    ids=lib+check
    n=len(lib)
    both,raw,cart,expert,freq,basis=inputs(data,ids,old,weights,previous,calibration,path)
    readout=energy.FrequencyReadout.restore(calibration["readout"],"cpu")
    banks,ranks,means=pairing.inner_proposals(args,data,ids,cart,lib,old)
    truth=np.array([data.mask_labels[i] for i in ids],bool)
    pref_values=[]
    for a in pairing.MIX:
        vals=[pairing.preference_scores(truth[i],preference_mask(both[i,:4],expert[i],a,cart[i]))["Dice_mean"] for i in range(n)]
        pref_values.append(float(np.mean(vals)))
    pref_weight=pairing.MIX[int(np.argmax(pref_values))]
    options=[(a,g,s) for a in NEW for g in BLEND for s in BLEND]
    scores=np.zeros((len(options),2))
    original=[]
    anchors=[]
    shape_ranks=[]
    for i,idx in enumerate(ids):
        p0=anchor(readout,raw[i],means[i]); p1=anchor(readout,both[i,4:],means[i])
        old_order=joint.transported_ranks(banks[i],ranks[i],both[i,:4])
        original.append(v4.endpoints(truth[i],bank_at(p1,old_order),joint.preference(both[i,:4],cart[i])))
        shape=field_ranks(basis[i],50,idx)
        if i>=n:
            anchors.append((p0,p1))
            shape_ranks.append(shape)
            continue
        support=truth[i].any(0)|(np.maximum.reduce([p0,p1,freq[i]])>=0.5/50)
        yy,xx=np.where(support)
        crop=np.s_[...,yy.min():yy.max()+1,xx.min():xx.max()+1] if len(yy) else np.s_[...,:1,:1]
        orders={s:mix_order(ranks[i][crop],shape[crop],s) for s in BLEND}
        for j,(a,g,s) in enumerate(options):
            q=(1-a)*((1-g)*p0[crop]+g*p1[crop])+a*freq[i][crop]
            seg=bank_at(q,orders[s])
            m=v3.active_scores(truth[i][crop],seg)
            scores[j]+=[m["GED"],m["Dice_soft"]]
    scores/=n
    original=np.stack(original)
    baseline=original[:n].mean(0)
    objective=np.minimum((baseline[0]-scores[:,0])/baseline[0],(scores[:,1]-baseline[1])/baseline[1])
    best=int(np.argmax(objective))
    a,g,s=options[best]
    after=[]
    for j,(p0,p1) in enumerate(anchors):
        i=n+j
        order=mix_order(ranks[i],shape_ranks[j],s)
        q=(1-a)*((1-g)*p0+g*p1)+a*freq[i]
        after.append(v4.endpoints(truth[i],bank_at(q,order),preference_mask(both[i,:4],expert[i],pref_weight,cart[i])))
    b=original[n:].mean(0);z=np.mean(after,axis=0)
    record={**identity(),"fold":args.fold,"fit_indices":lib,"check_indices":check,
            "inner_model":rc.hash_file(path/"final.pt"),"v4_lock":rc.hash_file(args.v4/f"lock-{args.fold}.json"),
            "pref_grid":dict(zip(map(str,pairing.MIX),pref_values)),"preference_weight":pref_weight,
            "new_weight":a,"joint_weight":g,"shape_weight":s,
            "candidates":[{"weights":list(opt),"ged":float(row[0]),"soft":float(row[1]),"objective":float(u)}
                          for opt,row,u in zip(options,scores,objective)],
            "inner_v4":dict(zip(v1.PRIMARY,b.tolist())),"inner_candidate":dict(zip(v1.PRIMARY,z.tolist()))}
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("LOCK",args.fold,pref_weight,a,g,s,"v4",b,"field",z,flush=True)


def locks(args):
    result=[read(args.run/f"lock-{i}.json") for i in range(4)]
    for i,row in enumerate(result):
        require(row["fold"]==i and all(row[k]==v for k,v in identity().items()) and
                row["v4_lock"]==rc.hash_file(args.v4/f"lock-{i}.json"),"field lock identity")
    delta={k:float(np.mean([r["inner_candidate"][k]-r["inner_v4"][k] for r in result])) for k in v1.PRIMARY}
    require(delta["GED@50"]<0 and delta["Dice_soft@50"]>0 and delta["Dice_mean"]>0,f"field inner gate failed: {delta}")
    return result


def infer(args):
    f=args.fold; rec=locks(args)[f]
    data,outer,valid,old,weights,previous,calibration,path=setup(args,"full")
    require(tuple(polar.calibration_split(data,valid,f))==(rec["fit_indices"],rec["check_indices"]),"field calibration split")
    ids=outer["test_indices"]
    both,raw,_,expert,freq,basis=inputs(data,ids,old,weights,previous,calibration,path)
    readout=energy.FrequencyReadout.restore(calibration["readout"],"cpu")
    a,g,s=(rec[k] for k in ("new_weight","joint_weight","shape_weight"))
    dest=args.run/f"assessment/fold-{f}"
    dest.mkdir(parents=True,exist_ok=False)
    for i,idx in enumerate(ids):
        banks,prior_pref=v2.old_masks(args.v1,f,i)
        p=np.packbits(preference_mask(both[i,:4],expert[i],rec["preference_weight"],prior_pref),axis=-1)
        packed={"full_pref":p,"original_order_pref":p,"no_field_pref":np.packbits(joint.preference(both[i,:4],prior_pref),axis=-1)}
        for k,bank in banks.items():
            base=bank.mean(0)
            q0=anchor(readout,raw[i],base); q1=anchor(readout,both[i,4:],base)
            previous_freq=(1-g)*q0+g*q1
            q=(1-a)*previous_freq+a*freq[i]
            original=v3.bank_ranks(bank)
            learned=field_ranks(basis[i],k,idx)
            full=mix_order(original,learned,s)
            for m,seg in (("full",bank_at(q,full)),("original_order",bank_at(q,original)),("no_field",bank_at(previous_freq,original))):
                packed[f"{m}_{k}"]=np.packbits(seg,axis=-1)
        np.savez_compressed(dest/f"{i:05d}.npz",**packed)
    rc.write_json(dest/"sealed.json",{**identity(),"fold":f,"cases":len(ids),"model":rc.hash_file(path/"final.pt"),
                  "bank_sha256":v2.digest_bank(dest,len(ids)),"lock_sha256":rc.hash_file(args.run/f"lock-{f}.json")})


def verify(args):
    locks(args)
    for f in range(4):
        seal=read(args.run/f"assessment/fold-{f}/sealed.json")
        require(seal["fold"]==f and all(seal[k]==v for k,v in identity().items()) and
                seal["lock_sha256"]==rc.hash_file(args.run/f"lock-{f}.json"),"field bank unsealed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("calibrate","gate","infer","score","report"))
    p.add_argument("--fold",type=int,choices=range(4))
    for key in ("data","run","v1","v2","v3","v4"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.phase in ("calibrate","infer","score") and args.fold is None: p.error("fold required")
    cv2.setNumThreads(1); torch.set_num_threads(3)
    if args.phase=="score":
        pairing.score(args,verify=verify,setup_fn=setup,arms=ARMS)
    elif args.phase=="report":
        pairing.report(args,verify=verify,provenance=identity,arms=ARMS)
    else:
        {"calibrate":calibrate,"gate":locks,"infer":infer}[args.phase](args)
