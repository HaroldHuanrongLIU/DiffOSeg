"""Prospectively gated object-relative inference with calibrated finite-panel evidence."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import boundary_eval as v1
from reproduction import canonical_ct as canonical
from reproduction import contour_field as field
from reproduction import field_study
from reproduction import joint_energy as energy
from reproduction import joint_rater as joint
from reproduction import joint_study as v4
from reproduction import normal_study as pair
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import vote_eval as v3

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_canonical/protocol.md"
ARMS=("full","no_geometry","no_panel","no_field")
WEIGHTS=(0.25,0.5,0.75,1.0)
MIX=(0.,0.5,1.)
KAPPA=(1.,4.,12.,None)
read,require=v2.read,v2.require


def identity():
    return {"source":rc.hash_file(__file__),"geometry":rc.hash_file(canonical.__file__),
            "protocol":rc.hash_file(PROTOCOL),"paired_scoring":rc.hash_file(pair.__file__),"v4":pair.V4_HASH}


def setup(args,mode):
    # Verify and use the fitted field as a component; this does not promote its standalone configuration.
    return field_study.setup(args,mode)


def features(args,mode):
    data,outer,valid,old,weights,previous,calibration,path=setup(args,mode)
    lib,check=polar.calibration_split(data,valid,args.fold)
    require((lib,check)==(calibration["fit_indices"],calibration["check_indices"]),"inner regions differ")
    ids=lib+check if mode=="inner" else outer["test_indices"]
    images=np.stack([data.images[i] for i in ids])[:,None]
    raw,aligned,cart=canonical.features(images,old,weights,previous["expert_alpha"])
    expert,freq,_=field.predict(field.load_model(path/"final.pt"),images)
    decoder=joint.JointDecoder.restore(calibration["parameters"],"cuda")
    readout=energy.FrequencyReadout.restore(calibration["readout"],"cpu")
    return data,ids,lib,check,old,path,raw,aligned,cart,expert,freq,decoder,readout


def panel(support):
    c=(support>0.5).mean(0).astype(np.float32)
    n=np.abs(2*support-1).sum(0)
    return c,n


def shrink(prior,count,effective,kappa):
    if kappa is None:
        return prior
    return (kappa*prior+effective*count)/(kappa+effective)


def components(decoder,raw,aligned,w):
    heads=(1-w)*raw+w*aligned
    joint_probs=v4.decoded(decoder,heads)
    vote_probs=torch.from_numpy(heads[:,4:]).sigmoid().numpy()
    return joint_probs,vote_probs


def fit(args):
    data,ids,lib,check,old,path,raw,aligned,cart,new_expert,new_freq,decoder,readout=features(args,"inner")
    n=len(lib)
    truth=np.array([data.mask_labels[i] for i in ids],bool)
    base,raw_vote=components(decoder,raw,aligned,0.)
    pref_scores=[]
    for w in pair.MIX:
        expert,_=components(decoder,raw[:n],aligned[:n],w)
        vals=[pair.preference_scores(truth[i],field_study.preference_mask(expert[i,:4],new_expert[i],0.25,cart[i]))["Dice_mean"] for i in range(n)]
        pref_scores.append(float(np.mean(vals)))
    pw=pair.MIX[int(np.argmax(pref_scores))]
    pref,_=components(decoder,raw[n:],aligned[n:],pw)
    # Each library case's proposal generator excludes its own patient's residual templates.
    options=[(w,m,h) for w in WEIGHTS for m in MIX for h in KAPPA]
    sums=np.zeros((2,len(options),2))
    prior_stats=np.zeros((2,2))
    for d,scale in enumerate((0.8,1.2)):
        banks,ranks,means=pair.inner_proposals(args,data,lib,cart[:n],lib,old,scale)
        for i in range(n):
            rank=joint.transported_ranks(banks[i],ranks[i],base[i,:4])
            ref=energy.generate(readout,base[i,4:],means[i],rank)
            stat=v3.active_scores(truth[i],ref)
            prior_stats[d]+=[stat["GED"],stat["Dice_soft"]]
        for wi,w in enumerate(WEIGHTS):
            computed,votes=components(decoder,raw[:n],aligned[:n],w)
            for i in range(n):
                p0=field_study.anchor(readout,votes[i],means[i])
                p1=field_study.anchor(readout,computed[i,4:],means[i])
                count,ne=panel(votes[i])
                active=truth[i].any(0)|(np.maximum.reduce((p0,p1,new_freq[i],count))>=0.5/50)
                yy,xx=np.where(active)
                crop=np.s_[...,yy.min():yy.max()+1,xx.min():xx.max()+1] if len(yy) else np.s_[...,:1,:1]
                order=ranks[i][crop]
                for mi,m in enumerate(MIX):
                    prior=0.75*((1-m)*p0[crop]+m*p1[crop])+0.25*new_freq[i][crop]
                    for hi,k in enumerate(KAPPA):
                        q=shrink(prior,count[crop],ne[crop],k)
                        s=v3.active_scores(truth[i][crop],field_study.bank_at(q,order))
                        sums[d,(wi*len(MIX)+mi)*len(KAPPA)+hi]+=[s["GED"],s["Dice_soft"]]
        del banks,ranks,means
    sums/=n;prior_stats/=n
    qualities=np.minimum((prior_stats[:,None,0]-sums[:,:,0])/0.004,(sums[:,:,1]-prior_stats[:,None,1])/0.0015).min(0)
    best=int(np.argmax(qualities))
    w,m,k=options[best]
    computed,votes=components(decoder,raw[n:],aligned[n:],w)
    banks,ranks,means=pair.inner_proposals(args,data,check,cart[n:],lib,old)
    before,after=[],[]
    for j,case in enumerate(check):
        i=n+j
        original=energy.generate(readout,base[i,4:],means[j],joint.transported_ranks(banks[j],ranks[j],base[i,:4]))
        before.append(v4.endpoints(truth[i],original,joint.preference(base[i,:4],cart[i])))
        p0=field_study.anchor(readout,votes[j],means[j]);p1=field_study.anchor(readout,computed[j,4:],means[j])
        count,ne=panel(votes[j]);q=shrink(0.75*((1-m)*p0+m*p1)+0.25*new_freq[i],count,ne,k)
        pred=field_study.bank_at(q,ranks[j]);expert=field_study.preference_mask(pref[j,:4],new_expert[i],0.25,cart[i])
        after.append(v4.endpoints(truth[i],pred,expert))
    b,a=np.mean(before,axis=0),np.mean(after,axis=0)
    record={**identity(),"fold":args.fold,"fit_indices":lib,"check_indices":check,
            "v4_lock":rc.hash_file(args.v4/f"lock-{args.fold}.json"),"field_checkpoint":rc.hash_file(path/"final.pt"),
            "preference_weight":pw,"canonical_weight":w,"joint_weight":m,"kappa":k,
            "pref_grid":dict(zip(map(str,pair.MIX),pref_scores)),"library_v4":prior_stats.tolist(),
            "grid":[{"options":list(opt),"dispersions":sums[:,j].tolist(),"score":float(qualities[j])} for j,opt in enumerate(options)],
            "inner_v4":dict(zip(v1.PRIMARY,b.tolist())),"inner_candidate":dict(zip(v1.PRIMARY,a.tolist()))}
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("LOCK",args.fold,pw,w,m,k,"v4",b,"candidate",a,flush=True)


def locks(args):
    rows=[read(args.run/f"lock-{f}.json") for f in range(4)]
    for f,row in enumerate(rows):
        require(row["fold"]==f and all(row[k]==v for k,v in identity().items()) and
                row["v4_lock"]==rc.hash_file(args.v4/f"lock-{f}.json"),"invalid canonical lock")
    delta={k:float(np.mean([r["inner_candidate"][k]-r["inner_v4"][k] for r in rows])) for k in v1.PRIMARY}
    require(delta["GED@50"]<0 and delta["Dice_soft@50"]>0 and delta["Dice_mean"]>0,f"canonical gate failed: {delta}")
    return rows


def infer(args):
    f=args.fold; rec=locks(args)[f]
    _,ids,lib,check,_,path,raw,aligned,_,new_expert,new_freq,decoder,readout=features(args,"full")
    require(lib==rec["fit_indices"] and check==rec["check_indices"],"calibration split")
    expert,_=components(decoder,raw,aligned,rec["preference_weight"])
    jointv,votes=components(decoder,raw,aligned,rec["canonical_weight"])
    native,raw_votes=components(decoder,raw,aligned,0.)
    m,k=rec["joint_weight"],rec["kappa"]
    dest=args.run/f"assessment/fold-{f}"
    dest.mkdir(parents=True,exist_ok=False)
    for i in range(len(ids)):
        banks,old_pref=v2.old_masks(args.v1,f,i)
        packed={}
        for arm in ARMS:
            rank=native[i,:4] if arm=="no_geometry" else expert[i,:4]
            ratio=0 if arm=="no_field" else 0.25
            packed[arm+"_pref"]=np.packbits(field_study.preference_mask(rank,new_expert[i],ratio,old_pref),axis=-1)
        for kdraw,bank in banks.items():
            order=v3.bank_ranks(bank); prior=bank.mean(0)
            for arm in ARMS:
                j,s=(native[i,4:],raw_votes[i]) if arm=="no_geometry" else (jointv[i,4:],votes[i])
                q=(1-m)*field_study.anchor(readout,s,prior)+m*field_study.anchor(readout,j,prior)
                if arm!="no_field": q=0.75*q+0.25*new_freq[i]
                if arm!="no_panel": q=shrink(q,*panel(s),k)
                packed[f"{arm}_{kdraw}"]=np.packbits(field_study.bank_at(q,order),axis=-1)
        np.savez_compressed(dest/f"{i:05d}.npz",**packed)
    rc.write_json(dest/"sealed.json",{**identity(),"fold":f,"cases":len(ids),"bank_sha256":v2.digest_bank(dest,len(ids)),
                  "lock_sha256":rc.hash_file(args.run/f"lock-{f}.json"),"field_checkpoint":rc.hash_file(path/"final.pt")})


def verify(args):
    locks(args)
    for f in range(4):
        seal=read(args.run/f"assessment/fold-{f}/sealed.json")
        require(seal["fold"]==f and all(seal[k]==v for k,v in identity().items()) and
                seal["lock_sha256"]==rc.hash_file(args.run/f"lock-{f}.json"),"unsealed canonical fold")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("calibrate","gate","infer","score","report"))
    p.add_argument("--fold",type=int,choices=range(4))
    for name in ("data","run","v1","v2","v3","v4"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.phase in ("calibrate","infer","score") and args.fold is None: p.error("fold required")
    cv2.setNumThreads(1); torch.set_num_threads(3)
    if args.phase=="score": pair.score(args,verify=verify,setup_fn=setup,arms=ARMS)
    elif args.phase=="report": pair.report(args,verify=verify,provenance=identity,arms=ARMS)
    else: {"calibrate":fit,"gate":locks,"infer":infer}[args.phase](args)
