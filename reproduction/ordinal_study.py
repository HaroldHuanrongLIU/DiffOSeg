"""Inner fit and same-cohort validation of finite-panel superlevel projection."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit,logit
import torch

from reproduction import boundary_eval as v1
from reproduction import canonical_ct as canonical
from reproduction import canonical_study as canonical_study
from reproduction import contour_field as field
from reproduction import field_study
from reproduction import joint_energy as energy
from reproduction import joint_rater as joint
from reproduction import joint_study as v4
from reproduction import normal_study as pair
from reproduction import ordinal_projection as ordinal
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import vote_eval as v3

ARMS=("full","unconstrained","direct_count","half_vote")
PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_ordinal/protocol.md"
read,require=v2.read,v2.require


def identity():
    return {"source":rc.hash_file(__file__),"projection":rc.hash_file(ordinal.__file__),
            "protocol":rc.hash_file(PROTOCOL),"pairing_source":rc.hash_file(pair.__file__),"v4":pair.V4_HASH}


@torch.inference_mode()
def field_heads(model,images):
    out=[]
    for i in range(0,len(images),12):
        x=torch.from_numpy(images[i:i+12]).cuda().float()/255
        total=0
        for flip in (False,True):
            view=x.flip([-1]) if flip else x
            for k in range(4):
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    expert,support,freq,_=model(torch.rot90(view,k,(-2,-1)))
                z=torch.cat((expert,support,torch.logit(freq.clamp(1e-6,1-1e-6))[:,None]),1)
                z=torch.rot90(z,-k,(-2,-1)); total=total+(z.flip([-1]) if flip else z)
        out.append((total/8).cpu().numpy())
    values=np.concatenate(out)
    return values[:,:4],values[:,4:8],expit(values[:,8])


def setup(args,mode):
    return field_study.setup(args,mode)


def image_features(args,mode):
    data,outer,valid,old,weights,previous,parent,path=setup(args,mode)
    lib,check=polar.calibration_split(data,valid,args.fold)
    require((lib,check)==(parent["fit_indices"],parent["check_indices"]),"patient split")
    ids=lib+check if mode=="inner" else outer["test_indices"]
    images=np.stack([data.images[i] for i in ids])[:,None]
    raw,aligned,cart=canonical.features(images,old,weights,previous["expert_alpha"])
    expert,field_support,freq=field_heads(field.load_model(path/"final.pt"),images)
    dec=joint.JointDecoder.restore(parent["parameters"],"cuda")
    original_native=v4.decoded(dec,raw)
    previous_lock=read(args.canonical/f"lock-{args.fold}.json")
    require(previous_lock["fit_indices"]==lib and previous_lock["check_indices"]==check and
            all(previous_lock[k]==v for k,v in canonical_study.identity().items()),"predeclared preference branch")
    prefs,_=canonical_study.components(dec,raw,aligned,previous_lock["preference_weight"])
    readout=energy.FrequencyReadout.restore(parent["readout"],"cpu")
    votes=expit(raw[:,4:])
    simulated=votes.mean(1)
    # This ordinal predictor never consumes a diffusion bank, same in calibration and deployment.
    evidence=np.stack([field_study.anchor(readout,s,p) for s,p in zip(original_native[:,4:],simulated)])
    centers=logit(np.array(ordinal.HALF_VOTE,dtype=np.float32))
    energy_logits=4*(logit(evidence.clip(1e-6,1-1e-6))[:,None]-centers[None,:,None,None])
    features=np.stack((raw[:,4:],aligned[:,4:],logit(original_native[:,4:].clip(1e-6,1-1e-6)),
                       field_support,energy_logits),2).clip(-16,16)
    return data,ids,lib,check,old,path,features,cart,votes,original_native,prefs[:,:4],expert,freq,readout


def calibrate(args):
    data,ids,lib,check,old,path,x,cart,votes,original,pref,new,freq,readout=image_features(args,"inner")
    truth=np.array([data.mask_labels[i] for i in ids],bool)
    n=len(lib)
    names=[str(data.series_uid[i]).split("_")[0] for i in lib]
    stack=ordinal.fit(x[:n],truth[:n],names,36193+args.fold)
    c=ordinal.counts(stack,x)
    banks,ranks,means=pair.inner_proposals(args,data,ids,cart,lib,old)
    grid=(0.,0.5,1.)
    fit_ged=np.zeros(len(grid))
    base0=[];base1=[]
    for i in range(len(ids)):
        q0=field_study.anchor(readout,votes[i],means[i]); q1=field_study.anchor(readout,original[i,4:],means[i])
        base0.append(q0);base1.append(q1)
        if i<n:
            for j,m in enumerate(grid):
                p=0.75*((1-m)*q0+m*q1)+0.25*freq[i]
                bank=ordinal.project(p,c[i],ranks[i])
                fit_ged[j]+=v3.active_scores(truth[i],bank)["GED"]
    fit_ged/=n
    mix=grid[int(np.argmin(fit_ged))]
    before,after=[],[]
    for j,case in enumerate(check):
        i=n+j
        old_order=joint.transported_ranks(banks[i],ranks[i],original[i,:4])
        before.append(v4.endpoints(truth[i],field_study.bank_at(base1[i],old_order),joint.preference(original[i,:4],cart[i])))
        prior=0.75*((1-mix)*base0[i]+mix*base1[i])+0.25*freq[i]
        after.append(v4.endpoints(truth[i],ordinal.project(prior,c[i],ranks[i]),field_study.preference_mask(pref[i],new[i],0.25,cart[i])))
    b,a=np.mean(before,axis=0),np.mean(after,axis=0)
    record={**identity(),"fold":args.fold,"fit_indices":lib,"check_indices":check,"stack":stack.coefficients(),
            "mixture":mix,"fit_ged":fit_ged.tolist(),"v4_lock":rc.hash_file(args.v4/f"lock-{args.fold}.json"),
            "canonical_lock":rc.hash_file(args.canonical/f"lock-{args.fold}.json"),"field_checkpoint":rc.hash_file(path/"final.pt"),
            "inner_v4":dict(zip(v1.PRIMARY,b.tolist())),"inner_candidate":dict(zip(v1.PRIMARY,a.tolist()))}
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("LOCK",args.fold,mix,"v4",b,"ordinal",a,flush=True)


def locks(args):
    rows=[read(args.run/f"lock-{f}.json") for f in range(4)]
    for f,r in enumerate(rows):
        require(r["fold"]==f and all(r[k]==v for k,v in identity().items()) and
                r["v4_lock"]==rc.hash_file(args.v4/f"lock-{f}.json") and
                r["canonical_lock"]==rc.hash_file(args.canonical/f"lock-{f}.json"),"ordinal lock changed")
    delta={k:float(np.mean([r["inner_candidate"][k]-r["inner_v4"][k] for r in rows])) for k in v1.PRIMARY}
    require(delta["GED@50"]<0 and delta["Dice_soft@50"]>0 and delta["Dice_mean"]>0,f"ordinal gate failed: {delta}")
    return rows


def infer(args):
    f=args.fold; lock=locks(args)[f]
    _,ids,lib,check,_,path,x,_,votes,original,pref,new,freq,readout=image_features(args,"full")
    require(lib==lock["fit_indices"] and check==lock["check_indices"],"inner partition")
    c=ordinal.counts(ordinal.OrdinalStack.restore(lock["stack"]),x)
    m=lock["mixture"]
    dest=args.run/f"assessment/fold-{f}"
    dest.mkdir(parents=True,exist_ok=False)
    for i in range(len(ids)):
        banks,old_pref=v2.old_masks(args.v1,f,i)
        expert=np.packbits(field_study.preference_mask(pref[i],new[i],0.25,old_pref),axis=-1)
        packed={arm+"_pref":expert for arm in ARMS}
        packed["count"]=c[i].astype(np.uint8)
        for k,bank in banks.items():
            q0=field_study.anchor(readout,votes[i],bank.mean(0))
            q1=field_study.anchor(readout,original[i,4:],bank.mean(0))
            prior=0.75*((1-m)*q0+m*q1)+0.25*freq[i]
            ranks=v3.bank_ranks(bank)
            for arm in ARMS:
                packed[f"{arm}_{k}"]=np.packbits(ordinal.project(prior,c[i],ranks,arm),axis=-1)
        np.savez_compressed(dest/f"{i:05d}.npz",**packed)
    rc.write_json(dest/"sealed.json",{**identity(),"fold":f,"cases":len(ids),"bank_sha256":v2.digest_bank(dest,len(ids)),
                  "lock_sha256":rc.hash_file(args.run/f"lock-{f}.json"),"field_checkpoint":rc.hash_file(path/"final.pt")})


def verify(args):
    locks(args)
    for f in range(4):
        seal=read(args.run/f"assessment/fold-{f}/sealed.json")
        require(seal["fold"]==f and all(seal[k]==v for k,v in identity().items()) and
                seal["lock_sha256"]==rc.hash_file(args.run/f"lock-{f}.json"),"ordinal predictions unsealed")


def extra_scores(truth,freq):
    target=truth.mean(0)
    dice=[]
    for t in ordinal.DIAGNOSTIC:
        a=target>t; b=freq>t; den=a.sum()+b.sum()
        dice.append(2*(a&b).sum()/den if den else 1.)
    return [float(np.mean(dice)),float(np.mean((target-freq)**2))]


def diagnostic(args):
    folds=[]
    rng=np.random.default_rng(212903);blocks=[]
    for f in range(4):
        p=args.run/f"assessment/fold-{f}"
        rows=read(p/"paired.json")["rows"]
        groups,inv=np.unique([r["patient"] for r in rows],return_inverse=True)
        vals=[]
        for i,_ in enumerate(rows):
            name=f"{i:05d}"
            base=args.v1/f"baseline/fold_{f}/lidc/fold_{f}/stage_2/cases/{name}_masks.npz"
            truth=v1.unpack(base,"truth",(4,128,128))
            with np.load(p/(name+".npz")) as z:
                new=np.unpackbits(z["full_50"],axis=-1).mean(0)
                naked=np.unpackbits(z["unconstrained_50"],axis=-1).mean(0)
            with np.load(args.v4/f"assessment/fold-{f}/{name}.npz") as z:
                prior=np.unpackbits(z["full_50"],axis=-1).mean(0)
            vals.append([extra_scores(truth,x) for x in (prior,new,naked)])
        vals=np.array(vals)
        folds.append(vals.mean(0))
        n=np.bincount(inv); draws=rng.multinomial(len(n),np.full(len(n),1/len(n)),size=5000)
        sums=np.zeros((len(n),2));np.add.at(sums,inv,vals[:,1]-vals[:,0]);blocks.append(draws@sums/(draws@n)[:,None])
    means=np.mean(folds,axis=0)
    rc.write_json(args.run/"assessment/diagnostics.json",{"names":["midpoint20_Dice","full_image_Brier"],
                  "v4":means[0].tolist(),"full":means[1].tolist(),"unconstrained":means[2].tolist(),"folds":np.asarray(folds).tolist(),
                  "paired95_v4":np.percentile(np.mean(blocks,axis=0),[2.5,97.5],axis=0).T.tolist()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("calibrate","gate","infer","score","report"))
    p.add_argument("--fold",type=int,choices=range(4))
    for name in ("data","run","v1","v2","v3","v4","canonical"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.phase in ("calibrate","infer","score") and args.fold is None: p.error("fold required")
    cv2.setNumThreads(1);torch.set_num_threads(3)
    if args.phase=="score": pair.score(args,verify=verify,setup_fn=setup,arms=ARMS)
    elif args.phase=="report":
        pair.report(args,verify=verify,provenance=identity,arms=ARMS);diagnostic(args)
    else: {"calibrate":calibrate,"gate":locks,"infer":infer}[args.phase](args)
