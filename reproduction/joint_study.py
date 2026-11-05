"""Four-fold calibration, blind inference and paired scoring for joint decoding."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from reproduction import boundary_eval as v1
from reproduction import joint_rater as joint
from reproduction import joint_energy as energy
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote
from reproduction import vote_eval as v3
from reproduction.metrics import preference_scores

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_joint/protocol.md"
V3_HASH="27ca97941a8a2cca095358bf6d365559c67f90e3210b436c01be6851a1a6076c"
PRIMARY=v1.PRIMARY
ARMS=("full","location","no_location","no_order","no_energy","energy_only")
read,require=v2.read,v2.require


def identity():
    return {"method":rc.hash_file(joint.__file__),"runner":rc.hash_file(__file__),
            "protocol":rc.hash_file(PROTOCOL),"energy_protocol":rc.hash_file(PROTOCOL.with_name("energy_protocol.md")),
            "energy_source":rc.hash_file(energy.__file__),"v3":V3_HASH}


def inputs(args,mode):
    require(rc.hash_file(args.v3/"assessment/fourfold_comparison.json")==V3_HASH,"v3 identity")
    old_args=SimpleNamespace(fold=args.fold,data=args.data,run=args.v3,prior=args.v1,previous=args.v2)
    data,outer,valid,old,weights=v3.setup(old_args,mode)
    lock=read(args.v3/f"lock-{args.fold}.json")
    require(lock["fold"]==args.fold and lock["protocol"]==rc.hash_file(vote.PROTOCOL),"v3 lock")
    return data,outer,valid,old,weights,lock


@torch.inference_mode()
def features(data,ids,old,weights,alpha):
    a,b=map(vote.load_net,weights)
    old_net=rc.RankContours().cuda().to(memory_format=torch.channels_last).eval()
    old_net.load_state_dict(torch.load(old/"final.pt",map_location="cpu",weights_only=True))
    raw=[]
    cart=[]
    for offset in range(0,len(ids),12):
        image=torch.from_numpy(np.stack([data.images[i] for i in ids[offset:offset+12]])[:,None]).cuda().float()/255
        base=vote.d4(old_net,image)[:,:4]
        both=vote.combine_views(vote.single_view(a,image,"fullfield"),vote.single_view(b,image,"zoom"))
        logits=torch.cat(((1-alpha)*base+alpha*both[:,:4],both[:,8:]),1)
        require(bool(torch.isfinite(logits).all()),"nonfinite image features")
        raw.append(logits.cpu().numpy())
        cart.append((base>0).cpu().numpy())
    return np.concatenate(raw),np.concatenate(cart)


def decoded(net,heads,location=True,condition=True):
    with torch.inference_mode():
        parts=[]
        for block in torch.from_numpy(heads).split(12):
            e,s=net(block.cuda(),location=location,joint=condition)
            parts.append(torch.cat((e,s),1).cpu().numpy())
        return np.concatenate(parts)


def endpoints(truth,bank,pref):
    scores=v3.active_scores(truth,bank)
    return np.array([scores["GED"],scores["Dice_soft"],preference_scores(truth,pref)["Dice_mean"]])


def calibrate(args):
    data,_,valid,old,weights,lock=inputs(args,"inner")
    inner_train=rc.split_for(data,args.fold)[1]
    lib,check=polar.calibration_split(data,valid,args.fold)
    require(check==lock["selection_indices"],"selection patients mismatch")
    ids=lib+check
    raw,cart=features(data,ids,old,weights,lock["expert_alpha"])
    truth=np.array([data.mask_labels[i] for i in ids],bool)
    # Estimate the reference landmark solely on the neural fit's inner-training cases.
    truth_train=torch.from_numpy(np.array([data.mask_labels[i] for i in inner_train],bool)).sum(1)
    weight=((truth_train>=2).float()+(truth_train>=3).float())/2
    nonempty=weight.sum((1,2))>0
    origin=joint.centroid(weight[nonempty]).median(0).values.numpy()
    names=[str(data.series_uid[i]).split("_")[0] for i in lib]
    net=joint.fit_layer(raw[:len(lib)],truth[:len(lib)],names,origin,57011+args.fold)
    control=decoded(net,raw,False,False)
    full=decoded(net,raw)
    path=args.v2/f"calibration-{args.fold}"
    vl=read(path/"lock.json")
    require(vl["library_indices"]==lib and vl["selection_indices"]==check and
            vl["library_sha256"]==rc.hash_file(path/"library.npy"),"surrogate library mismatch")
    probe=args.v2/f"fold-{args.fold}-inner"
    require(rc.hash_file(probe/"final.pt")==vl["inner_polar"] and
            rc.hash_file(old/"final.pt")==vl["inner_cart"] and read(probe/"plan.json")["train_indices"]==inner_train,
            "surrogate inner fit identity")
    library=np.load(path/"library.npy",allow_pickle=False)
    images=torch.from_numpy(np.stack([data.images[i] for i in ids])[:,None]).cuda()
    radius,sigma=polar.predict(polar.load_net(probe/"final.pt"),images)
    freqs,rank_arrays=[],[]
    check_banks=[]
    for i,case in enumerate(ids):
        field=v2.rank_fields(cart[i],radius[i],vl["fraction"])
        eligible=library
        if i<len(lib):
            eligible=library[np.asarray(names)!=names[i]]
            require(len(eligible)>0,"no patient-excluded calibration templates")
        center,noise=v2.draw_fields(field,np.maximum(0.65,sigma[i,0]),eligible,case,50)
        bank=center+vl["scale"]*noise>0
        ranks=v3.bank_ranks(bank)
        ordered=joint.transported_ranks(bank,ranks,full[i,:4])
        freqs.append(bank.mean(0).astype(np.float32))
        rank_arrays.append(ordered)
        if i>=len(lib):
            check_banks.append((bank,ranks))
    readout=energy.fit(full[:len(lib),4:],np.stack(freqs[:len(lib)]),np.stack(rank_arrays[:len(lib)]),
                       truth[:len(lib)],names,31861+args.fold).cpu()
    stats=[]
    for j,(bank,ranks) in enumerate(check_banks):
        i=j+len(lib)
        before=joint.consensus(bank,ranks,control[i,:4],control[i,4:],lock,0)
        after=energy.generate(readout,full[i,4:],freqs[i],rank_arrays[i])
        b=endpoints(truth[i],before,joint.preference(control[i,:4],cart[i]))
        a=endpoints(truth[i],after,joint.preference(full[i,:4],cart[i]))
        stats.append(np.stack((b,a)))
    b,a=np.mean(stats,axis=0)
    record={"fold":args.fold,**identity(),"fit_indices":lib,"check_indices":check,
            "v3_lock":rc.hash_file(args.v3/f"lock-{args.fold}.json"),
            "inner_checkpoints":[rc.hash_file(p) for p in weights],"parameters":net.coefficients(),"readout":readout.coefficients(),
            "inner_v3":dict(zip(PRIMARY,b.tolist())),"inner_v4":dict(zip(PRIMARY,a.tolist()))}
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("INNER",args.fold,"v3",b,"v4",a,flush=True)


def locks(args):
    records=[read(args.run/f"lock-{f}.json") for f in range(4)]
    for f,r in enumerate(records):
        require(r["fold"]==f and all(r[k]==v for k,v in identity().items()) and
                r["v3_lock"]==rc.hash_file(args.v3/f"lock-{f}.json"),"invalid lock")
    delta={k:float(np.mean([r["inner_v4"][k]-r["inner_v3"][k] for r in records])) for k in PRIMARY}
    require(delta["GED@50"]<0 and delta["Dice_soft@50"]>0 and delta["Dice_mean"]>0,
            f"inner promotion gate failed: {delta}")
    return records


def infer(args):
    f=args.fold
    record=locks(args)[f]
    data,outer,valid,old,weights,lock=inputs(args,"full")
    lib,check=polar.calibration_split(data,valid,f)
    require(record["fit_indices"]==lib and record["check_indices"]==check,"fit and check split")
    idx=outer["test_indices"]
    raw,_=features(data,idx,old,weights,lock["expert_alpha"])
    net=joint.JointDecoder.restore(record["parameters"],"cuda")
    readout=energy.FrequencyReadout.restore(record["readout"],"cpu")
    out={"energy_only":decoded(net,raw,False,False),"full":decoded(net,raw),"location":decoded(net,raw,True,False),"no_location":decoded(net,raw,False,True)}
    folder=args.run/f"assessment/fold-{f}"
    folder.mkdir(parents=True,exist_ok=False)
    for i in range(len(idx)):
        prior,pref=v2.old_masks(args.v1,f,i)
        packed={}
        for arm,values in out.items():
            packed[f"{arm}_pref"]=np.packbits(joint.preference(values[i,:4],pref),axis=-1)
        for k,bank in prior.items():
            ranks=v3.bank_ranks(bank)
            for arm in ARMS:
                x=out["full" if arm in ("no_order","no_energy") else arm][i]
                weight=0 if arm in ("location","no_order","energy_only") else 0.2
                if arm in ("location","no_energy"):
                    seg=joint.consensus(bank,ranks,x[:4],x[4:],lock,weight)
                else:
                    ordered=joint.transported_ranks(bank,ranks,x[:4],weight)
                    seg=energy.generate(readout,x[4:],bank.mean(0),ordered)
                packed[f"{arm}_{k}"]=np.packbits(seg,axis=-1)
        np.savez_compressed(folder/f"{i:05d}.npz",**packed)
    rc.write_json(folder/"sealed.json",{"fold":f,"cases":len(idx),**identity(),
                  "lock_sha256":rc.hash_file(args.run/f"lock-{f}.json"),
                  "bank_sha256":v2.digest_bank(folder,len(idx)),"full_checkpoints":[rc.hash_file(p) for p in weights]})


def verified_seals(args):
    locks(args)
    for f in range(4):
        s=read(args.run/f"assessment/fold-{f}/sealed.json")
        require(s["fold"]==f and all(s[k]==v for k,v in identity().items()) and
                s["lock_sha256"]==rc.hash_file(args.run/f"lock-{f}.json"),"unsealed fold")


def score(args):
    verified_seals(args)
    f=args.fold
    data,outer,_,_,_,_=inputs(args,"full")
    idx=outer["test_indices"]
    dst=args.run/f"assessment/fold-{f}"
    require(v2.digest_bank(dst,len(idx))==read(dst/"sealed.json")["bank_sha256"],"v4 bank identity")
    root=args.v3/f"assessment/fold-{f}"
    require(v2.digest_bank(root,len(idx))==read(root/"sealed.json")["bank_sha256"],"v3 bank identity")
    prior=read(root/"paired.json")["rows"]
    result=[]
    for i,case in enumerate(idx):
        truth=np.asarray(data.mask_labels[case],bool)
        patient=str(data.series_uid[case]).split("_")[0]
        require((prior[i]["index"],prior[i]["patient"])==(case,patient),"paired identity")
        with np.load(root/f"{i:05d}.npz",allow_pickle=False) as z:
            pref=np.unpackbits(z["pref"],axis=-1).astype(bool)
            banks={k:np.unpackbits(z[f"k{k}"],axis=-1).astype(bool) for k in v1.COUNTS}
        old=v1.metrics(truth,banks,pref)
        require(max(abs(prior[i]["v3"][k]-v) for k,v in old.items())<2e-6,"v3 per-case parity")
        row={"index":case,"patient":patient,"v3":old,"e11":prior[i]["e11"]}
        with np.load(dst/f"{i:05d}.npz",allow_pickle=False) as z:
            for arm in ARMS:
                pref_key="full" if arm in ("no_order","no_energy") else arm
                pref=np.unpackbits(z[f"{pref_key}_pref"],axis=-1).astype(bool)
                banks={k:np.unpackbits(z[f"{arm}_{k}"],axis=-1).astype(bool) for k in v1.COUNTS}
                row[arm]=v1.metrics(truth,banks,pref)
        result.append(row)
    rc.write_json(dst/"paired.json",{"rows":result,"seal_sha256":rc.hash_file(dst/"sealed.json")})


def report(args):
    verified_seals(args)
    records=locks(args)
    folds=[]
    patients,indices=set(),set()
    rng=np.random.default_rng(212903)
    blocks=[]
    for f in range(4):
        folder=args.run/f"assessment/fold-{f}"
        table=read(folder/"paired.json")
        require(table["seal_sha256"]==rc.hash_file(folder/"sealed.json"),"unscored bank")
        rows=table["rows"]
        p,inv=np.unique([r["patient"] for r in rows],return_inverse=True)
        ids=[r["index"] for r in rows]
        split=read(args.v1/f"baseline/fold_{f}/lidc/fold_{f}/stage_1/split.json")
        require(ids==split["test_indices"] and set(p)==set(split["test_patients"]) and
                not indices.intersection(ids) and not patients.intersection(p),"fold identity")
        patients.update(p)
        indices.update(ids)
        keys=list(rows[0]["full"])
        vals={m:np.array([[r[m][k] for k in keys] for r in rows]) for m in ("e11","v3",*ARMS)}
        n=np.bincount(inv)
        draw=rng.multinomial(len(n),np.full(len(n),1/len(n)),size=5000)
        sums=np.zeros((len(n),len(keys)))
        np.add.at(sums,inv,vals["full"]-vals["v3"])
        blocks.append(draw@sums/(draw@n)[:,None])
        folds.append({"fold":f,"patients":len(n),"cases":len(ids),
                      "means":{m:dict(zip(keys,map(float,a.mean(0)))) for m,a in vals.items()}})
    require(len(patients)==214 and len(indices)==1609,"LIDC coverage")
    ref=read(args.v3/"assessment/fourfold_comparison.json")["metrics"]
    ci=np.percentile(np.mean(blocks,axis=0),[2.5,97.5],axis=0)
    metrics={}
    for j,k in enumerate(keys):
        row={m:float(np.mean([x["means"][m][k] for x in folds])) for m in vals}
        require(all(abs(row[m]-ref[k][m])<2e-6 for m in ("e11","v3")),"reference parity")
        row["delta"]=row["full"]-row["v3"]
        row["ci95"]=ci[:,j].tolist()
        metrics[k]=row
    passed={k:(metrics[k]["delta"]<0 if k.startswith("GED") else metrics[k]["delta"]>0) for k in PRIMARY}
    rc.write_json(args.run/"assessment/fourfold_comparison.json",{**identity(),"patients":len(patients),"cases":len(indices),
                  "folds":folds,"metrics":metrics,"primary_pass":passed,
                  "inner_deltas":{k:float(np.mean([r["inner_v4"][k]-r["inner_v3"][k] for r in records])) for k in PRIMARY}})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("calibrate","gate","infer","score","report"))
    p.add_argument("--fold",type=int,choices=range(4))
    for name in ("data","run","v1","v2","v3"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.phase in ("calibrate","infer","score") and args.fold is None:
        p.error("fold required")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    {"calibrate":calibrate,"gate":locks,"infer":infer,"score":score,"report":report}[args.phase](args)
