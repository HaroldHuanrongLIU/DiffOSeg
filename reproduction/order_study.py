"""New-seed preference replication and an independently gated order-statistic decoder."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from reproduction import boundary_eval as v1
from reproduction import field_study
from reproduction import joint_rater as joint
from reproduction import joint_study as v4
from reproduction import normal_refiner as normal
from reproduction import normal_study as pair
from reproduction import ordinal_projection as ordinal
from reproduction import ordinal_study as v8
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import rank_utility as utility
from reproduction import vote_eval as v3
from reproduction.metrics import preference_scores

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_order/protocol.md"
V5="f81ee3bdfffbe303f52c4aa149bfd6ff99ca8a6e7a43582eddefce711fdd39bd"
V8="7a2529cc04b4673285619fb563604afba511d33499cb45ac66340b443f90e21a"
OLD_TRAIN="da954421d8350f803efbbfb10ac8dd4a65c7833d2d383eb37aa4c89ec284398f"
ETA=(0.,0.25,0.50,0.65,0.80)
ARMS=("full","two_seed","rank_decision","fresh_seed","retro")


def identity():
    return {"source":rc.hash_file(__file__),"utility":rc.hash_file(utility.__file__),
            "replication":rc.hash_file(normal.__file__),"protocol":rc.hash_file(PROTOCOL),"v5":V5,"v8":V8}


def setup(args,mode):
    require(rc.hash_file(args.fixed/"assessment/fourfold_comparison.json")==V5 and
            rc.hash_file(args.ordinal/"assessment/fourfold_comparison.json")==V8,"parent summaries")
    data,outer,valid,old,weights,previous=v4.inputs(args,mode)
    parent=v1.read(args.v4/f"lock-{args.fold}.json")
    require(all(parent[k]==v for k,v in v4.identity().items()),"v4 calibration identity")
    expected=outer["train_indices"] if mode=="full" else rc.split_for(data,args.fold)[1]
    f=args.fold
    for root,seed,source in ((args.run,219071,rc.hash_file(normal.__file__)),(args.fixed,117901,OLD_TRAIN)):
        folder=root/f"fold-{f}-{mode}"
        plan,fit=v1.read(folder/"plan.json"),v1.read(folder/"fit.json")
        require((plan["fold"],plan["mode"],plan["train_indices"],plan["seed"])==(f,mode,expected,seed+f)
                and fit["data_sha256"]==rc.DATA_HASH and fit["steps"]==normal.STEPS and
                fit["source"]==source and fit["checkpoint"]==rc.hash_file(folder/"final.pt"),"refit identity")
        require(v1.read(root/f"fold-{f}-inner/result.json")["passed"],"inner gate")
    lock=v1.read(args.fixed/f"lock-{f}.json")
    require(lock["v4_lock"]==rc.hash_file(args.v4/f"lock-{f}.json"),"original preference lock")
    return data,outer,valid,old,weights,previous,parent,lock


def preference_features(args,data,ids,mode,old,weights,previous,parent,lock):
    f=args.fold
    raw,cart=v4.features(data,ids,old,weights,previous["expert_alpha"])
    guide=normal.image_guide(raw[:,4:])
    images=np.stack([data.images[i] for i in ids])[:,None]
    orig=normal.predict(normal.load_model(args.fixed/f"fold-{f}-{mode}/final.pt"),images,guide)
    fresh=normal.predict(normal.load_model(args.run/f"fold-{f}-{mode}/final.pt"),images,guide)
    dec=joint.JointDecoder.restore(parent["parameters"],"cuda")
    weight=lock["preference_weight"]
    def decode(new):
        x=raw.copy()
        x[:,:4]=(1-weight)*raw[:,:4]+weight*new[:,:4]
        return v4.decoded(dec,x)[:,:4].copy()
    return decode((orig+fresh)*0.5),decode(fresh),decode(orig),cart


def calibrate(args):
    f=args.fold
    data,_,valid,old,weights,previous,parent,lock=setup(args,"inner")
    ordinal_args=SimpleNamespace(**{**vars(args),"run":args.ordinal})
    _,ids,lib,check,_,_,x,cart,votes,original,_,_,freq,readout=v8.image_features(ordinal_args,"inner")
    require((lib,check)==(lock["fit_indices"],lock["check_indices"]),"patient partition")
    ordinal_lock=v8.locks(ordinal_args)[f]
    counts=ordinal.counts(ordinal.OrdinalStack.restore(ordinal_lock["stack"]),x)
    mix=ordinal_lock["mixture"]
    probs,_,old_ref,_=preference_features(args,data,ids,"inner",old,weights,previous,parent,lock)
    banks,ranks,means=pair.inner_proposals(args,data,ids,cart,lib,old)
    stats=[];all_dice=[];refine=[]
    for i,idx in enumerate(ids):
        truth=np.asarray(data.mask_labels[idx],bool)
        q0=field_study.anchor(readout,votes[i],means[i])
        q1=field_study.anchor(readout,original[i,4:],means[i])
        p=0.75*((1-mix)*q0+mix*q1)+0.25*freq[i]
        bank=ordinal.project(p,counts[i],ranks[i])
        anchor=joint.preference(probs[i],cart[i])
        options=utility.decisions(bank,probs[i],anchor,ETA)
        all_dice.append([preference_scores(truth,options[eta])["Dice_mean"] for eta in ETA])
        refine.append(preference_scores(truth,joint.preference(old_ref[i],cart[i]))["Dice_mean"])
        if i>=len(lib):
            old_order=joint.transported_ranks(banks[i],ranks[i],original[i,:4])
            old_bank=field_study.bank_at(q1,old_order)
            stats.append((v4.endpoints(truth,old_bank,joint.preference(original[i,:4],cart[i])),
                          v4.endpoints(truth,bank,anchor)))
    all_dice=np.array(all_dice)
    fit=all_dice[:len(lib)].mean(0); best=int(np.argmax(fit))
    before,after=np.mean(stats,axis=0)
    rank=after.copy();rank[2]=all_dice[len(lib):,best].mean()
    record={**identity(),"fold":f,"fit_indices":lib,"check_indices":check,
            "v8_lock":rc.hash_file(args.ordinal/f"lock-{f}.json"),"v5_lock":rc.hash_file(args.fixed/f"lock-{f}.json"),
            "seed2_inner":rc.hash_file(args.run/f"fold-{f}-inner/final.pt"),"eta":ETA[best],"fit_dice":fit.tolist(),
            "inner_v4":dict(zip(v1.PRIMARY,before.tolist())),"inner_ensemble":dict(zip(v1.PRIMARY,after.tolist())),
            "inner_rank":dict(zip(v1.PRIMARY,rank.tolist())),"inner_old_preference":float(np.mean(refine[len(lib):]))}
    rc.write_json(args.run/f"lock-{f}.json",record)
    print("LOCK",f,ETA[best],before,after,rank,flush=True)


def require(ok,message):
    if not ok: raise ValueError(message)


def selection(args):
    rows=[v1.read(args.run/f"lock-{f}.json") for f in range(4)]
    for f,r in enumerate(rows):
        require(r["fold"]==f and all(r[k]==v for k,v in identity().items()) and
                r["v8_lock"]==rc.hash_file(args.ordinal/f"lock-{f}.json") and
                r["v5_lock"]==rc.hash_file(args.fixed/f"lock-{f}.json"),"preference lock")
    delta=[r["inner_rank"]["Dice_mean"]-r["inner_ensemble"]["Dice_mean"] for r in rows]
    arm="inner_rank" if np.mean(delta)>0 and np.count_nonzero(np.asarray(delta)>1e-8)>=3 else "inner_ensemble"
    gain={k:float(np.mean([r[arm][k]-r["inner_v4"][k] for r in rows])) for k in v1.PRIMARY}
    require(gain["GED@50"]<0 and gain["Dice_soft@50"]>0 and gain["Dice_mean"]>0,f"replication inner gate failed: {gain}")
    return rows,arm,gain


def gate(args):
    rows,arm,gain=selection(args)
    result={"primary_preference":arm,"inner_delta":gain,"fold_eta":[r["eta"] for r in rows],
            "lock_sha256":[rc.hash_file(args.run/f"lock-{f}.json") for f in range(4)]}
    rc.write_json(args.run/"selection.json",result)
    print(result)


def infer(args):
    f=args.fold; rows,choice,_=selection(args); lock=rows[f]
    require(choice==v1.read(args.run/"selection.json")["primary_preference"],"selection receipt")
    data,outer,_,old,weights,previous,parent,original=setup(args,"full")
    ids=outer["test_indices"]
    probs,fresh,_,_=preference_features(args,data,ids,"full",old,weights,previous,parent,original)
    previous_bank=args.ordinal/f"assessment/fold-{f}"
    require(v2.digest_bank(previous_bank,len(ids))==v1.read(previous_bank/"sealed.json")["bank_sha256"],"v8 bank")
    dest=args.run/f"assessment/fold-{f}";dest.mkdir(parents=True,exist_ok=False)
    for i in range(len(ids)):
        with np.load(previous_bank/f"{i:05d}.npz",allow_pickle=False) as z:
            bank=np.unpackbits(z["full_50"],axis=-1).astype(bool)
        # v5/v4 preference fusion uses the corrected v1 bank, not the raw E11 bank.
        with np.load(args.v1/f"assessment/fold-{f}/{i:05d}_masks.npz",allow_pickle=False) as z:
            prior=np.unpackbits(z["pref"],axis=-1).astype(bool)
        anchor=joint.preference(probs[i],prior)
        decided=utility.decisions(bank,probs[i],anchor,(lock["eta"],))[lock["eta"]]
        with np.load(args.fixed/f"assessment/fold-{f}/{i:05d}.npz",allow_pickle=False) as z:
            retro=z["full_pref"]
        masks={"full":np.packbits(decided if choice=="inner_rank" else anchor,axis=-1),
               "two_seed":np.packbits(anchor,axis=-1),"rank_decision":np.packbits(decided,axis=-1),
               "fresh_seed":np.packbits(joint.preference(fresh[i],prior),axis=-1),"retro":retro}
        np.savez_compressed(dest/f"{i:05d}.npz",**masks)
    rc.write_json(dest/"sealed.json",{**identity(),"fold":f,"cases":len(ids),"bank_sha256":v2.digest_bank(dest,len(ids)),
                  "population_bank":v1.read(previous_bank/"sealed.json")["bank_sha256"],
                  "model":rc.hash_file(args.run/f"fold-{f}-full/final.pt"),"selection":rc.hash_file(args.run/"selection.json")})


def verify(args):
    selection(args)
    for f in range(4):
        seal=v1.read(args.run/f"assessment/fold-{f}/sealed.json")
        require(seal["fold"]==f and all(seal[k]==v for k,v in identity().items()) and
                seal["selection"]==rc.hash_file(args.run/"selection.json"),"unsealed v9 fold")


def score(args):
    verify(args)
    f=args.fold; data,outer,*_=setup(args,"full");ids=outer["test_indices"]
    dst=args.run/f"assessment/fold-{f}";pop=args.ordinal/f"assessment/fold-{f}"
    require(v2.digest_bank(dst,len(ids))==v1.read(dst/"sealed.json")["bank_sha256"] and
            v2.digest_bank(pop,len(ids))==v1.read(dst/"sealed.json")["population_bank"],"bank drift")
    prior=v1.read(pop/"paired.json")["rows"]; v5rows=v1.read(args.fixed/f"assessment/fold-{f}/paired.json")["rows"]
    rows=[]
    for i,idx in enumerate(ids):
        truth=np.asarray(data.mask_labels[idx],bool)
        patient=str(data.series_uid[idx]).split("_")[0]
        require(all((x[i]["index"],x[i]["patient"])==(idx,patient) for x in (prior,v5rows)),"paired identity")
        with np.load(pop/f"{i:05d}.npz",allow_pickle=False) as z:
            banks={k:np.unpackbits(z[f"full_{k}"],axis=-1).astype(bool) for k in v1.COUNTS}
        consensus={key:value for key,value in v1.metrics(truth,banks,truth).items() if "@" in key}
        require(max(abs(val-prior[i]["full"][key]) for key,val in consensus.items())<2e-6,"v8 score parity")
        row={k:prior[i][k] for k in ("index","patient","v4_full","v4_energy_only","v4_no_order","e11")}
        row["v8"]=prior[i]["full"]
        with np.load(dst/f"{i:05d}.npz",allow_pickle=False) as z:
            for arm in ARMS:
                e=preference_scores(truth,np.unpackbits(z[arm],axis=-1).astype(bool))
                pref={f"Dice_A{j+1}":float(val) for j,val in enumerate(e["Dice_each"])}
                row[arm]={**consensus,**pref,"Dice_mean":e["Dice_mean"]}
        require(abs(row["retro"]["Dice_mean"]-v5rows[i]["full"]["Dice_mean"])<2e-6,"v5 composition parity")
        rows.append(row)
    rc.write_json(dst/"paired.json",{"rows":rows,"seal_sha256":rc.hash_file(dst/"sealed.json")})


def report(args):
    verify(args)
    all_ids,all_patients=set(),set()
    folds=[]
    ctrl=("v4_full","v4_no_order","v4_energy_only","v8","retro","fresh_seed","two_seed","rank_decision")
    boots={name:[] for name in ctrl}
    rng=np.random.default_rng(212903)
    for f in range(4):
        root=args.run/f"assessment/fold-{f}"
        table=v1.read(root/"paired.json"); rows=table["rows"]
        require(table["seal_sha256"]==rc.hash_file(root/"sealed.json"),"unscored fold")
        names,inv=np.unique([r["patient"] for r in rows],return_inverse=True)
        ids=[r["index"] for r in rows]
        split=v1.read(args.v1/f"baseline/fold_{f}/lidc/fold_{f}/stage_1/split.json")
        require(ids==split["test_indices"] and set(names)==set(split["test_patients"]) and
                not all_ids.intersection(ids) and not all_patients.intersection(names),"fold coverage")
        all_ids.update(ids);all_patients.update(names)
        keys=list(rows[0]["full"]); vals={m:np.array([[row[m][k] for k in keys] for row in rows]) for m in ("e11",*ctrl,"full")}
        n=np.bincount(inv);draw=rng.multinomial(len(n),np.full(len(n),1/len(n)),size=5000)
        for m in ctrl:
            sums=np.zeros((len(n),len(keys)));np.add.at(sums,inv,vals["full"]-vals[m])
            boots[m].append(draw@sums/(draw@n)[:,None])
        folds.append({"fold":f,"patients":len(n),"cases":len(ids),"means":{m:dict(zip(keys,map(float,v.mean(0)))) for m,v in vals.items()}})
    require(len(all_ids)==1609 and len(all_patients)==214,"incomplete cohort")
    ci={m:np.percentile(np.mean(z,axis=0),[2.5,97.5],axis=0) for m,z in boots.items()}
    history=v1.read(args.ordinal/"assessment/fourfold_comparison.json")["metrics"]
    metrics={}
    for j,k in enumerate(keys):
        row={m:float(np.mean([f["means"][m][k] for f in folds])) for m in vals}
        for m,h in (("v8","full"),("v4_full","v4_full"),("e11","e11")):
            require(abs(row[m]-history[k][h])<2e-6,"reference parity")
        row["paired"]={m:{"delta":row["full"]-row[m],"ci95":ci[m][:,j].tolist()} for m in ctrl}
        metrics[k]=row
    passed={k:(metrics[k]["full"]<metrics[k]["v4_full"] if k.startswith("GED") else metrics[k]["full"]>metrics[k]["v4_full"]) for k in v1.PRIMARY}
    rc.write_json(args.run/"assessment/fourfold_comparison.json",{**identity(),"primary_pass":passed,"patients":214,"cases":1609,
                  "selection":v1.read(args.run/"selection.json"),"folds":folds,"metrics":metrics,
                  "identical_population_diagnostics":v1.read(args.ordinal/"assessment/diagnostics.json")})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("calibrate","gate","infer","score","report"))
    p.add_argument("--fold",type=int,choices=range(4))
    for name in ("data","run","v1","v2","v3","v4","canonical","ordinal","fixed"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.phase in ("calibrate","infer","score") and args.fold is None:p.error("fold required")
    cv2.setNumThreads(1);torch.set_num_threads(3)
    {"calibrate":calibrate,"gate":gate,"infer":infer,"score":score,"report":report}[args.phase](args)
