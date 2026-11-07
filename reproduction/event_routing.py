"""Budgeted single-slice correction using sealed patient-excluded error rankings."""
import argparse
from pathlib import Path
import numpy as np

from reproduction import nested_guides as nested
from reproduction import panel_event
from reproduction import rank_contours as rc

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/event_routing_protocol.md"
EVENT="edd875278efe9892f45fad529eb0fe9386b54db5fafb5a511aae9a65d358c8a8"
WITNESS="3d1c5ee7607ecfc6c1f60988631e9e5176198b7d636c4b0e0608542f7b638221"
BUDGETS=("event","distance","margin")


def budget_mask(points,scores,shape):
    if len(points)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid budget scores")
    take=(len(points)+4)//5
    selected=points[np.lexsort((points,-np.asarray(scores)))[:take]]
    out=np.zeros(shape,dtype=bool)
    out.ravel()[selected]=True
    return out


def apply(reference,first,second,support):
    agreed=(first==second)&(first!=reference)&support
    return reference^agreed


def lock(args):
    if rc.hash_file(args.event/"summary.json")!=EVENT or rc.hash_file(args.witness/"summary.json")!=WITNESS:
        raise ValueError("Upstream audited fitting results changed")
    f=args.fold
    er=nested.read(args.event/f"seal-{f}.json");wr=nested.read(args.witness/f"seal-{f}.json")
    ep=args.event/f"pred-{f}.npz"
    base=args.original/f"oof-{f}.npz";strong=args.nested/"nested-training"/f"oof-{f}.npz"
    if (er["prediction"]!=rc.hash_file(ep) or er["base"]!=wr["base"] or
            er["nested"]!=wr["nested"] or er["base"]!=rc.hash_file(base) or er["nested"]!=rc.hash_file(strong)):
        raise ValueError("Inconsistent predictor identities")
    with np.load(ep,allow_pickle=False) as z,np.load(base,allow_pickle=False) as a,np.load(strong,allow_pickle=False) as b:
        ids=z["ids"].tolist();reference=z["reference"];prob=z["event"]
        if ids!=a["ids"].tolist() or ids!=b["ids"].tolist():raise ValueError("Different upstream cases")
        q=.5*(a["logits"].astype(np.float32)+b["logits"].astype(np.float32))
    np.testing.assert_array_equal(reference,np.packbits(q>0,axis=-1))
    auxiliaries={i:{} for i in ids}
    for g in range(3):
        path=args.nested/f"subbank-{f}-{g}/oof-{f}.npz"
        if rc.hash_file(path)!=wr["subbanks"][g]:raise ValueError("Wrong auxiliary bank")
        with np.load(path,allow_pickle=False) as z:
            for i,k,value in zip(z["ids"].tolist(),z["donors"].tolist(),z["logits"],strict=True):
                if k==g or k in auxiliaries[i]:raise ValueError("Duplicate or incorrect donor")
                auxiliaries[i][k]=np.packbits(value>0,axis=-1)
    masks={k:[] for k in (*BUDGETS,"all")}
    regions={k:[] for k in BUDGETS}
    cursor=0
    for j,i in enumerate(ids):
        ref=q[j]>0
        band,dist=panel_event.unanimous_band(ref);points=np.flatnonzero(band)
        scores=(prob[cursor:cursor+len(points)],-dist.ravel()[points],-abs(q[j]).min(0).ravel()[points])
        cursor+=len(points)
        if len(auxiliaries[i])!=2:raise ValueError("Expected two disjoint patient models")
        first,second=[np.unpackbits(v,axis=-1).astype(bool) for v in auxiliaries[i].values()]
        masks["all"].append(np.packbits(apply(ref,first,second,band),axis=-1))
        for name,s in zip(BUDGETS,scores,strict=True):
            selected=budget_mask(points,s,band.shape)
            regions[name].append(np.packbits(selected,axis=-1))
            masks[name].append(np.packbits(apply(ref,first,second,selected),axis=-1))
    if cursor!=len(prob):raise ValueError("Event ordering mismatch")
    dst=args.run/f"pred-{f}.npz"
    np.savez_compressed(dst,ids=ids,reference=reference,**{k:np.stack(v) for k,v in masks.items()},
        **{"selected_"+k:np.stack(v) for k,v in regions.items()})
    rc.write_json(args.run/f"seal-{f}.json",{"fold":f,"event_seal":rc.hash_file(args.event/f"seal-{f}.json"),
      "witness_seal":rc.hash_file(args.witness/f"seal-{f}.json"),"prediction":rc.hash_file(dst),
      "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("EVENT_ROUTING_LOCK",f,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","original","nested","event","witness"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    lock(p.parse_args())
