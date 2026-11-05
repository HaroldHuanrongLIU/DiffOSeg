"""Patient-nested calibration and paired evaluation of the normal-profile refiner."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.special import expit

from reproduction import boundary_eval as v1
from reproduction import joint_energy as energy
from reproduction import joint_rater as joint
from reproduction import joint_study as v4
from reproduction import normal_refiner as normal
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import vote_eval as v3
from reproduction.metrics import preference_scores

V4_HASH = "1b8330667c6463b3a650465f9a42e70867a0873e5ce843f3bfbeb78336c18007"
MIX = (0.,0.25,0.5,0.75,1.)
JOINT = (0.,0.5,1.)
ARMS = ("full","center_only","no_refiner")
read, require = v2.read, v2.require


def identity():
    return {"source":rc.hash_file(__file__),"model_source":rc.hash_file(normal.__file__),
            "protocol":rc.hash_file(normal.PROTOCOL),"v4":V4_HASH}


def setup(args, mode):
    f=args.fold
    require(rc.hash_file(args.v4/"assessment/fourfold_comparison.json")==V4_HASH,"v4 summary mismatch")
    data,outer,valid,old,weights,v3lock=v4.inputs(args,mode)
    inner=rc.split_for(data,f)[1]
    folder=args.run/f"fold-{f}-{mode}"
    plan,fit=read(folder/"plan.json"),read(folder/"fit.json")
    require((plan["fold"],plan["mode"],plan["train_indices"])==(
            f,mode,inner if mode=="inner" else outer["train_indices"]) and
            plan["inner_valid_indices"]==(valid if mode=="inner" else []) and
            fit["source"]==rc.hash_file(normal.__file__) and fit["protocol"]==rc.hash_file(normal.PROTOCOL) and
            fit["data_sha256"]==rc.DATA_HASH and fit["checkpoint"]==rc.hash_file(folder/"final.pt") and
            fit["steps"]==normal.STEPS,"normal refiner identity")
    require(read(args.run/f"fold-{f}-inner/result.json")["passed"],"inner training gate")
    parent=read(args.v4/f"lock-{f}.json")
    require(parent["v3_lock"]==rc.hash_file(args.v3/f"lock-{f}.json") and
            all(parent[k]==val for k,val in v4.identity().items()),"v4 calibration provenance")
    return data,outer,valid,old,weights,v3lock,parent,folder


def branch_logits(data,ids,old,weights,v3lock,folder,ablate=False):
    raw,cart=v4.features(data,ids,old,weights,v3lock["expert_alpha"])
    guide=normal.image_guide(raw[:,4:])
    image=np.stack([data.images[i] for i in ids])[:,None]
    refiner=normal.load_model(folder/"final.pt")
    full=normal.predict(refiner,image,guide)
    removed=normal.predict(refiner,image,guide,False) if ablate else None
    return raw,cart,full,removed


def preference_field(decoder,raw,new,alpha):
    combined=raw.copy()
    combined[:,:4]=(1-alpha)*raw[:,:4]+alpha*new[:,:4]
    return v4.decoded(decoder,combined)[:,:4].copy()


def support_fields(decoder,raw,new,pref_weight,vote_weight):
    combined=raw.copy()
    combined[:,:4]=(1-pref_weight)*raw[:,:4]+pref_weight*new[:,:4]
    combined[:,4:]=(1-vote_weight)*raw[:,4:]+vote_weight*new[:,4:]
    return expit(combined[:,4:]),v4.decoded(decoder,combined)[:,4:].copy()


def inner_proposals(args,data,ids,cart,lib,old,dispersion=1.0):
    f=args.fold
    path=args.v2/f"calibration-{f}"
    vl=read(path/"lock.json")
    check=polar.calibration_split(data,rc.split_for(data,f)[2],f)[1]
    require(vl["library_indices"]==lib and vl["selection_indices"]==check and vl["library_sha256"]==rc.hash_file(path/"library.npy"),
            "surrogate library identity")
    probe=args.v2/f"fold-{f}-inner"
    require(vl["inner_cart"]==rc.hash_file(old/"final.pt") and vl["inner_polar"]==rc.hash_file(probe/"final.pt") and
            read(probe/"plan.json")["train_indices"]==rc.split_for(data,f)[1],"surrogate model identity")
    templates=np.load(path/"library.npy",allow_pickle=False)
    names=np.asarray([str(data.series_uid[i]).split("_")[0] for i in lib])
    image=torch.from_numpy(np.stack([data.images[i] for i in ids])[:,None]).cuda()
    radius,sigma=polar.predict(polar.load_net(probe/"final.pt"),image)
    banks,ranks,means=[],[],[]
    for j,idx in enumerate(ids):
        patient=str(data.series_uid[idx]).split("_")[0]
        eligible=templates[names!=patient]
        require(len(eligible)>0,"empty patient-excluded library")
        field=v2.rank_fields(cart[j],radius[j],vl["fraction"])
        center,noise=v2.draw_fields(field,np.maximum(0.65,sigma[j,0]),eligible,idx,50)
        bank=center+dispersion*vl["scale"]*noise>0
        banks.append(bank)
        ranks.append(v3.bank_ranks(bank))
        means.append(bank.mean(0).astype(np.float32))
    return banks,ranks,means


def fit_lock(args):
    data,_,valid,old,weights,v3lock,parent,folder=setup(args,"inner")
    lib,check=polar.calibration_split(data,valid,args.fold)
    require(parent["fit_indices"]==lib and parent["check_indices"]==check,"v4 calibration patient split mismatch")
    ids=lib+check
    n=len(lib)
    truth=np.array([data.mask_labels[i] for i in ids],bool)
    raw,cart,new,_=branch_logits(data,ids,old,weights,v3lock,folder)
    decoder=joint.JointDecoder.restore(parent["parameters"],"cuda")
    population=energy.FrequencyReadout.restore(parent["readout"],"cpu")
    banks,ranks,means=inner_proposals(args,data,ids,cart,lib,old)
    baseline=v4.decoded(decoder,raw)
    before=[]
    for i,bank in enumerate(banks):
        priority=joint.transported_ranks(bank,ranks[i],baseline[i,:4])
        seg=energy.generate(population,baseline[i,4:],means[i],priority)
        before.append(v4.endpoints(truth[i],seg,joint.preference(baseline[i,:4],cart[i])))
    before=np.stack(before)
    ref=before[:n].mean(0)
    pref_scores=[]
    for a in MIX:
        e=preference_field(decoder,raw[:n],new[:n],a)
        pref_scores.append(float(np.mean([preference_scores(t,joint.preference(p,c))["Dice_mean"]
                                         for p,c,t in zip(e,cart[:n],truth[:n])])))
    a=MIX[int(np.argmax(pref_scores))]
    pref=preference_field(decoder,raw,new,a)
    candidates=[]
    for b in MIX:
        raw_s,joint_s=support_fields(decoder,raw[:n],new[:n],a,b)
        for g in JOINT:
            values=[]
            for i in range(n):
                support=(1-g)*raw_s[i]+g*joint_s[i]
                seg=energy.generate(population,support,means[i],ranks[i])
                metrics=v3.active_scores(truth[i],seg)
                values.append((metrics["GED"],metrics["Dice_soft"]))
            ged,soft=np.mean(values,axis=0).tolist()
            utility=min((ref[0]-ged)/ref[0],(soft-ref[1])/ref[1])
            candidates.append({"vote_weight":b,"joint_weight":g,"GED":ged,"Dice_soft":soft,"utility":float(utility)})
    chosen=max(candidates,key=lambda row:row["utility"])
    b,g=chosen["vote_weight"],chosen["joint_weight"]
    raw_s,joint_s=support_fields(decoder,raw[n:],new[n:],a,b)
    after=[]
    for j in range(len(check)):
        i=j+n
        seg=energy.generate(population,(1-g)*raw_s[j]+g*joint_s[j],means[i],ranks[i])
        after.append(v4.endpoints(truth[i],seg,joint.preference(pref[i],cart[i])))
    reference=before[n:].mean(0); observed=np.mean(after,axis=0)
    record={**identity(),"fold":args.fold,"fit_indices":lib,"check_indices":check,"inner_model":rc.hash_file(folder/"final.pt"),
            "v4_lock":rc.hash_file(args.v4/f"lock-{args.fold}.json"),"preference_weight":a,
            "preference_grid":dict(zip(map(str,MIX),pref_scores)),"candidates":candidates,
            "vote_weight":b,"joint_weight":g,"inner_v4":dict(zip(v1.PRIMARY,reference.tolist())),
            "inner_v5":dict(zip(v1.PRIMARY,observed.tolist()))}
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("LOCK",args.fold,a,b,g,"inner v4",reference,"v5",observed,flush=True)


def locks(args):
    records=[read(args.run/f"lock-{f}.json") for f in range(4)]
    for f,rec in enumerate(records):
        require(rec["fold"]==f and all(rec[k]==v for k,v in identity().items()) and
                rec["v4_lock"]==rc.hash_file(args.v4/f"lock-{f}.json"),"v5 lock identity")
    delta={k:float(np.mean([rec["inner_v5"][k]-rec["inner_v4"][k] for rec in records])) for k in v1.PRIMARY}
    require(delta["GED@50"]<0 and delta["Dice_soft@50"]>0 and delta["Dice_mean"]>0,f"v5 inner gate failed: {delta}")
    return records


def infer(args):
    f=args.fold
    rec=locks(args)[f]
    data,outer,valid,old,weights,v3lock,parent,folder=setup(args,"full")
    lib,check=polar.calibration_split(data,valid,f)
    require((lib,check)==(rec["fit_indices"],rec["check_indices"]),"calibration identity")
    ids=outer["test_indices"]
    raw,_,new,removed=branch_logits(data,ids,old,weights,v3lock,folder,True)
    dec=joint.JointDecoder.restore(parent["parameters"],"cuda")
    readout=energy.FrequencyReadout.restore(parent["readout"],"cpu")
    a,b,g=(rec[k] for k in ("preference_weight","vote_weight","joint_weight"))
    fields={}
    for name,proposed in (("full",new),("center_only",removed),("no_refiner",raw)):
        e=preference_field(dec,raw,proposed,a)
        s,j=support_fields(dec,raw,proposed,a,b)
        fields[name]=(e,(1-g)*s+g*j)
    dest=args.run/f"assessment/fold-{f}"
    dest.mkdir(parents=True,exist_ok=False)
    for i in range(len(ids)):
        banks,old_pref=v2.old_masks(args.v1,f,i)
        packed={m+"_pref":np.packbits(joint.preference(e[i],old_pref),axis=-1) for m,(e,s) in fields.items()}
        for k,bank in banks.items():
            rank,base=v3.bank_ranks(bank),bank.mean(0)
            for m,(e,s) in fields.items():
                bank_out=energy.generate(readout,s[i],base,rank)
                packed[f"{m}_{k}"]=np.packbits(bank_out,axis=-1)
        np.savez_compressed(dest/f"{i:05d}.npz",**packed)
    rc.write_json(dest/"sealed.json",{**identity(),"fold":f,"cases":len(ids),"model":rc.hash_file(folder/"final.pt"),
                  "lock_sha256":rc.hash_file(args.run/f"lock-{f}.json"),"bank_sha256":v2.digest_bank(dest,len(ids))})


def verify_seals(args):
    locks(args)
    for f in range(4):
        receipt=read(args.run/f"assessment/fold-{f}/sealed.json")
        require(receipt["fold"]==f and all(receipt[k]==v for k,v in identity().items()) and
                receipt["lock_sha256"]==rc.hash_file(args.run/f"lock-{f}.json"),"outer bank unsealed")


def score(args,verify=verify_seals,setup_fn=setup,arms=ARMS):
    verify(args)
    f=args.fold
    data,outer=setup_fn(args,"full")[:2]
    ids=outer["test_indices"]
    root=args.run/f"assessment/fold-{f}"
    prev=args.v4/f"assessment/fold-{f}"
    for directory in (root,prev):
        require(v2.digest_bank(directory,len(ids))==read(directory/"sealed.json")["bank_sha256"],"sealed bytes changed")
    prior=read(prev/"paired.json")["rows"]
    rows=[]
    for i,idx in enumerate(ids):
        truth=np.asarray(data.mask_labels[idx],bool)
        patient=str(data.series_uid[idx]).split("_")[0]
        require((idx,patient)==(prior[i]["index"],prior[i]["patient"]),"case order")
        row={"index":idx,"patient":patient,"e11":prior[i]["e11"]}
        for directory,names in ((prev,("full","no_order","energy_only")),(root,arms)):
            with np.load(directory/f"{i:05d}.npz",allow_pickle=False) as z:
                for arm in names:
                    p="full" if arm=="no_order" else arm
                    pref=np.unpackbits(z[p+"_pref"],axis=-1).astype(bool)
                    banks={k:np.unpackbits(z[f"{arm}_{k}"],axis=-1).astype(bool) for k in v1.COUNTS}
                    values=v1.metrics(truth,banks,pref)
                    name=arm
                    if directory==prev:
                        require(max(abs(values[k]-prior[i][arm][k]) for k in values)<2e-6,"v4 per-case parity")
                        name="v4_"+arm
                    row[name]=values
        rows.append(row)
    rc.write_json(root/"paired.json",{"rows":rows,"seal_sha256":rc.hash_file(root/"sealed.json")})


def report(args,verify=verify_seals,provenance=identity,arms=ARMS):
    verify(args)
    folds=[]
    all_patients,all_indices=set(),set()
    rng=np.random.default_rng(212903)
    boots={m:[] for m in ("v4_full","v4_no_order","v4_energy_only",*arms[1:])}
    for f in range(4):
        folder=args.run/f"assessment/fold-{f}"
        table=read(folder/"paired.json")
        require(table["seal_sha256"]==rc.hash_file(folder/"sealed.json"),"unscored bank")
        rows=table["rows"]
        patients,inv=np.unique([r["patient"] for r in rows],return_inverse=True)
        ids=[r["index"] for r in rows]
        split=read(args.v1/f"baseline/fold_{f}/lidc/fold_{f}/stage_1/split.json")
        require(ids==split["test_indices"] and set(patients)==set(split["test_patients"]) and
                not all_indices.intersection(ids) and not all_patients.intersection(patients),"fold coverage")
        all_indices.update(ids)
        all_patients.update(patients)
        keys=list(rows[0]["full"])
        values={m:np.array([[r[m][k] for k in keys] for r in rows]) for m in ("e11",*boots,"full")}
        n=np.bincount(inv)
        draw=rng.multinomial(len(n),np.full(len(n),1/len(n)),size=5000)
        for m in boots:
            sums=np.zeros((len(n),len(keys)))
            np.add.at(sums,inv,values["full"]-values[m])
            boots[m].append(draw@sums/(draw@n)[:,None])
        folds.append({"fold":f,"patients":len(n),"cases":len(ids),"means":{m:dict(zip(keys,map(float,z.mean(0)))) for m,z in values.items()}})
    require(len(all_patients)==214 and len(all_indices)==1609,"incomplete cohort")
    ci={m:np.percentile(np.mean(blocks,axis=0),[2.5,97.5],axis=0) for m,blocks in boots.items()}
    historical=read(args.v4/"assessment/fourfold_comparison.json")["metrics"]
    metrics={}
    for j,k in enumerate(keys):
        row={m:float(np.mean([fold["means"][m][k] for fold in folds])) for m in values}
        for m in ("full","no_order","energy_only"):
            require(abs(row["v4_"+m]-historical[k][m])<2e-6,"v4 summary differs")
        row["paired"]={m:{"delta":row["full"]-row[m],"ci95":ci[m][:,j].tolist()} for m in boots}
        metrics[k]=row
    passed={k:(metrics[k]["full"]<metrics[k]["v4_full"] if k.startswith("GED") else metrics[k]["full"]>metrics[k]["v4_full"]) for k in v1.PRIMARY}
    rc.write_json(args.run/"assessment/fourfold_comparison.json",{**provenance(),"folds":folds,"patients":214,"cases":1609,
                  "metrics":metrics,"primary_pass":passed})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("calibrate","gate","infer","score","report"))
    p.add_argument("--fold",type=int,choices=range(4))
    for name in ("data","run","v1","v2","v3","v4"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.phase in ("calibrate","infer","score") and args.fold is None:
        p.error("fold required")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    {"calibrate":fit_lock,"gate":locks,"infer":infer,"score":score,"report":report}[args.phase](args)
