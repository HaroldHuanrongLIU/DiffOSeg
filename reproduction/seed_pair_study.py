"""Paired decoder experiments replacing one member of the frozen E combination."""
import json
import numpy as np
import torch
from reproduction import canonical_guide
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits

def lock(args, names, protocol, source, member):
    if (rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or
            rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT):
        raise ValueError("Registered E reference changed")
    images,_,_,uids = native.inputs(args.data)
    _,training,valid = native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(group,valid,args.fold)
    ids=library+checked
    field,field_receipt=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(field,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong image order")
        guide=z["adapted"]
    reference=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    pred=args.stable/f"pred-{args.fold}.npz"
    if rc.hash_file(pred)!=reference["prediction"] or reference["field"]!=field_receipt["payload"]:
        raise ValueError("Registered E mask identity changed")
    with np.load(pred,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong reference order")
        masks={"reference":z["adapted"]}
    weight=reference["choices"]["adapted"]["weight"]
    original=cache_owner.verified_parent(args).features
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    other=[]
    sources=canonical_guide.model_sources(args,training,args.fold)
    for directory,_ in sources[1:]:
        model.load_state_dict(torch.load(directory/"final.pt",map_location="cpu",weights_only=True))
        other.append(prompt.predict(model,original,ids,images,guide).astype(np.float16))
    identities={}
    for name in names:
        single,checkpoint=member(model,original,ids,images,guide,name,training)
        combined=mean_saved_logits([single.astype(np.float16),*other])
        if not np.isfinite(combined).all():raise ValueError("Nonfinite paired output")
        masks[name]=np.packbits((1-weight)*guide+weight*combined>0,axis=-1)
        masks["single_"+name]=np.packbits(single>0,axis=-1)
        identities[name]=checkpoint
    path=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(path,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,"weight":weight,
        "reference":reference["prediction"],"unchanged_members":[h for _,h in sources[1:]],"models":identities,
        "prediction":rc.hash_file(path),"protocol":rc.hash_file(protocol),"source":rc.hash_file(source)})
    print("SEED_PAIR_LOCK",args.fold,flush=True)


def score(args, names):
    seals=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    if any(s["fold"]!=f or s["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz") for f,s in enumerate(seals)):
        raise ValueError("Incomplete paired prediction seals")
    _,truth,_,_=native.inputs(args.data)
    for f,s in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=s["library"]+s["check"]:raise ValueError("Wrong paired patient order")
            scores={name:float(metrics.dice(truth[s["check"]],np.unpackbits(z[name][len(s["library"]):],axis=-1).astype(bool)).mean())
                    for name in ("reference",*names,*["single_"+name for name in names])}
        rc.write_json(args.run/f"inner-{f}.json",{**s,"scores":scores})
        print("SEED_PAIR_CHECK",f,scores,flush=True)
