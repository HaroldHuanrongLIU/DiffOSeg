"""Adaptive combination using the predeclared normal-refinement training splits."""
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
from reproduction import normal_refiner as refiner
from reproduction import normal_reliability as reliability
from reproduction import normal_study as fixed
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import vote_eval as v3

read,require=v2.read,v2.require
PROTOCOL=refiner.PROTOCOL.with_name("adaptive_protocol.md")


def identity():
    return {**fixed.identity(),"adaptive_protocol":rc.hash_file(PROTOCOL),"gate_source":rc.hash_file(reliability.__file__),
            "adaptive_runner":rc.hash_file(__file__)}


def contrast(data,ids,raw):
    images=np.stack([data.images[i] for i in ids])[:,None]
    return reliability.contrast(images,refiner.image_guide(raw[:,4:]))


@torch.inference_mode()
def preference(model,decoded,new,raw,ct):
    output=[]
    for i in range(0,len(raw),16):
        e=torch.from_numpy(decoded[i:i+16,:4])
        proposal=torch.from_numpy(new[i:i+16,:4])
        s=torch.from_numpy(raw[i:i+16,4:]).sigmoid()
        result=model(e,proposal,s[:,:1]-s[:,-1:],torch.from_numpy(ct[i:i+16]))
        output.append(result.numpy())
    return np.concatenate(output)


@torch.inference_mode()
def bank_from_gate(model,raw,old,new,ct,base,ranks):
    values=[torch.as_tensor(x[None],dtype=torch.float32) for x in (raw,old,new,ct,base)]
    frequency=model(*values)[0].numpy()
    return ranks<np.floor(len(ranks)*frequency+0.5)[None]


def calibrate(args):
    data,_,valid,old,weights,v3lock,parent,folder=fixed.setup(args,"inner")
    lib,check=polar.calibration_split(data,valid,args.fold)
    require((lib,check)==(parent["fit_indices"],parent["check_indices"]),"v4 patient identity")
    ids=lib+check
    n=len(lib)
    names=[str(data.series_uid[i]).split("_")[0] for i in lib]
    truth=np.array([data.mask_labels[i] for i in ids],bool)
    raw,cart,new,_=fixed.branch_logits(data,ids,old,weights,v3lock,folder)
    dec=joint.JointDecoder.restore(parent["parameters"],"cuda")
    decoded=v4.decoded(dec,raw)
    ct=contrast(data,ids,raw)
    priors,orders=[],[]
    for dispersion in (0.75,1.25):
        bank,ranks,means=fixed.inner_proposals(args,data,lib,cart[:n],lib,old,dispersion)
        priors.append(np.stack(means))
        orders.append(np.stack(ranks))
        del bank,ranks,means
    expert,population=reliability.fit(parent,(raw[:n],decoded[:n]),new[:n],ct[:n],np.stack(priors),
                                      np.stack(orders),truth[:n],names,41891+args.fold)
    del priors,orders
    expert,population=expert.cpu(),population.cpu()
    pref=preference(expert,decoded[n:],new[n:],raw[n:],ct[n:])
    banks,ranks,means=fixed.inner_proposals(args,data,check,cart[n:],lib,old)
    ref=energy.FrequencyReadout.restore(parent["readout"],"cpu")
    before,after=[],[]
    for j,case in enumerate(check):
        i=n+j
        ordered=joint.transported_ranks(banks[j],ranks[j],decoded[i,:4])
        previous=energy.generate(ref,decoded[i,4:],means[j],ordered)
        seg=bank_from_gate(population,expit(raw[i,4:]),decoded[i,4:],new[i,4:],ct[i],means[j],ranks[j])
        before.append(v4.endpoints(truth[i],previous,joint.preference(decoded[i,:4],cart[i])))
        after.append(v4.endpoints(truth[i],seg,joint.preference(pref[j],cart[i])))
    b,a=np.mean(before,axis=0),np.mean(after,axis=0)
    record={**identity(),"fold":args.fold,"fit_indices":lib,"check_indices":check,
            "inner_model":rc.hash_file(folder/"final.pt"),"v4_lock":rc.hash_file(args.v4/f"lock-{args.fold}.json"),
            "preference":reliability.export(expert),"population":reliability.export(population),
            "inner_v4":dict(zip(v1.PRIMARY,b.tolist())),"inner_v5":dict(zip(v1.PRIMARY,a.tolist()))}
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("CHECK",args.fold,"v4",b,"adaptive",a,flush=True)


def locks(args):
    records=[read(args.run/f"lock-{f}.json") for f in range(4)]
    for f,r in enumerate(records):
        require(r["fold"]==f and all(r[k]==v for k,v in identity().items()) and
                r["v4_lock"]==rc.hash_file(args.v4/f"lock-{f}.json"),"invalid adaptive lock")
    delta={k:float(np.mean([r["inner_v5"][k]-r["inner_v4"][k] for r in records])) for k in v1.PRIMARY}
    require(delta["GED@50"]<0 and delta["Dice_soft@50"]>0 and delta["Dice_mean"]>0,
            f"adaptive gate failed: {delta}")
    return records


def infer(args):
    f=args.fold
    record=locks(args)[f]
    data,outer,valid,old,weights,v3lock,parent,folder=fixed.setup(args,"full")
    require(tuple(polar.calibration_split(data,valid,f))==(record["fit_indices"],record["check_indices"]),"validation partition")
    ids=outer["test_indices"]
    raw,_,new,removed=fixed.branch_logits(data,ids,old,weights,v3lock,folder,True)
    decoded=v4.decoded(joint.JointDecoder.restore(parent["parameters"],"cuda"),raw)
    ct=contrast(data,ids,raw)
    zeros=np.zeros_like(ct)
    expert=reliability.restore(reliability.PreferenceGate(),record["preference"],"cpu")
    population=reliability.restore(reliability.PopulationGate(energy.FrequencyReadout.restore(parent["readout"],"cpu")),
                                   record["population"],"cpu")
    variants={"full":(new,ct),"center_only":(removed,zeros),"no_refiner":(raw,zeros)}
    fields={m:preference(expert,decoded,x,raw,c) for m,(x,c) in variants.items()}
    raw_vote=expit(raw[:,4:])
    dest=args.run/f"assessment/fold-{f}"
    dest.mkdir(parents=True,exist_ok=False)
    for i in range(len(ids)):
        banks,prior_pref=v2.old_masks(args.v1,f,i)
        packed={m+"_pref":np.packbits(joint.preference(e[i],prior_pref),axis=-1) for m,e in fields.items()}
        for k,bank in banks.items():
            order,base=v3.bank_ranks(bank),bank.mean(0)
            for m,(x,c) in variants.items():
                seg=bank_from_gate(population,raw_vote[i],decoded[i,4:],x[i,4:],c[i],base,order)
                packed[f"{m}_{k}"]=np.packbits(seg,axis=-1)
        np.savez_compressed(dest/f"{i:05d}.npz",**packed)
    rc.write_json(dest/"sealed.json",{**identity(),"fold":f,"cases":len(ids),"model":rc.hash_file(folder/"final.pt"),
                  "lock_sha256":rc.hash_file(args.run/f"lock-{f}.json"),"bank_sha256":v2.digest_bank(dest,len(ids))})


def verify(args):
    locks(args)
    for f in range(4):
        seal=read(args.run/f"assessment/fold-{f}/sealed.json")
        require(seal["fold"]==f and all(seal[k]==v for k,v in identity().items()) and
                seal["lock_sha256"]==rc.hash_file(args.run/f"lock-{f}.json"),"adaptive bank unsealed")


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
    if args.phase=="score":
        fixed.score(args,verify=verify)
    elif args.phase=="report":
        fixed.report(args,verify=verify,provenance=identity)
    else:
        {"calibrate":calibrate,"gate":locks,"infer":infer}[args.phase](args)
