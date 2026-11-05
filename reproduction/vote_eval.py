"""Transport a frozen DiffOSeg sample bank to a learned four-rater vote frequency."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from datasets.lidc import BaseDataset
from reproduction import boundary_eval as v1
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote
from reproduction.metrics import consensus_scores, preference_scores


HARD = (0.,0.5,1.)
STRENGTH = (0.35,0.65,0.90,1.)
V2_SUMMARY = "e37024c9349726e706110ab02dea45d56c111575661aafbabb68826819b17e6f"
require, read = v2.require, v1.read


def bank_ranks(bank):
    """Rank original samples at each pixel. Closer interiors have priority."""
    k = len(bank)
    area = bank.reshape(k,-1).sum(1)
    stable_order = np.argsort(-area.astype(np.int64),kind="stable")
    distances = []
    for m in bank[stable_order].astype(np.uint8):
        if m.all() or not m.any():
            value = np.full(m.shape,182. if m.all() else -182.,np.float32)
        else:
            inside = cv2.distanceTransform(m,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            outside = cv2.distanceTransform(1-m,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            value = np.where(m,inside-0.5,0.5-outside)
        distances.append(value)
    best = stable_order[np.argsort(-np.stack(distances),axis=0,kind="stable")]
    ranks = np.empty(bank.shape,np.uint8)
    np.put_along_axis(ranks,best,np.arange(k,dtype=np.uint8)[:,None,None],axis=0)
    return ranks


def transport(ranks, original, soft, hard, hardness, strength):
    frequency = (1-hardness)*soft + hardness*hard
    target = (1-strength)*original + strength*frequency
    counts = np.clip(np.floor(len(ranks)*target+0.5),0,len(ranks))
    return ranks < counts[None]


def setup(args,mode):
    f=args.fold
    require(rc.hash_file(args.data)==rc.DATA_HASH and
            rc.hash_file(args.prior/"assessment/fourfold_comparison.json")==v2.V1_SUMMARY and
            rc.hash_file(args.previous/"assessment/fourfold_comparison.json")==V2_SUMMARY and
            rc.hash_file(args.prior/"baseline/fourfold_summary.json")==v1.SUMMARY_HASH,"input identity")
    data=BaseDataset(str(args.data))
    outer,train,valid=rc.split_for(data,f)
    expected=outer["train_indices"] if mode=="full" else train
    old=args.prior/f"fold-{f}-{mode}"
    identity=rc.hash_file(old/"final.pt")
    require(read(old/"plan.json")["train_indices"]==expected and read(old/"result.json")["checkpoint_sha256"]==identity,
            "v1 training identity")
    weights=[]
    for view in vote.VIEWS:
        folder=args.run/f"fold-{f}-{view}-{mode}"
        plan,result=read(folder/"plan.json"),read(folder/"result.json")
        require((plan["mode"],plan["view"],plan["fold"],plan["parent"],result["steps"])==(mode,view,f,identity,vote.STEPS)
                and plan["train_indices"]==expected and plan["inner_valid_indices"]==(valid if mode=="inner" else [])
                and plan["source"]==rc.hash_file(vote.__file__) and plan["protocol"]==rc.hash_file(vote.PROTOCOL)
                and result["checkpoint"]==rc.hash_file(folder/"final.pt"),"v3 fit identity")
        weights.append(folder/"final.pt")
    base=args.prior/f"baseline/fold_{f}/lidc/fold_{f}"
    for stage in (1,2):
        split=read(base/f"stage_{stage}/split.json")
        require(all(split[k]==v for k,v in outer.items()),"E11 split identity")
    return data,outer,valid,old,weights


@torch.inference_mode()
def image_heads(data,ids,old,weights):
    a,b=map(vote.load_net,weights)
    old_net=rc.RankContours().cuda().to(memory_format=torch.channels_last).eval()
    old_net.load_state_dict(torch.load(old/"final.pt",map_location="cpu",weights_only=True))
    result=[]
    for offset in range(0,len(ids),12):
        image=torch.from_numpy(np.stack([data.images[i] for i in ids[offset:offset+12]])[:,None]).cuda().float()/255
        cart=vote.d4(old_net,image)[:,:4]
        both=vote.combine_views(vote.single_view(a,image,"fullfield"),vote.single_view(b,image,"zoom"))
        votes=both[:,8:]
        packed=torch.cat((cart,both[:,:4],votes.sigmoid().mean(1,keepdims=True),(votes>0).float().mean(1,keepdims=True)),1)
        require(bool(torch.isfinite(packed).all()),"nonfinite image logits")
        result.append(packed.cpu().numpy())
    return np.concatenate(result)


def mask_for(heads,weight):
    return (1-weight)*heads[:4]+weight*heads[4:8]>0


def active_scores(truth,bank):
    support=truth.any(0)|bank.any(0)
    y,x=np.where(support)
    if len(y):
        sl=np.s_[..., y.min():y.max()+1, x.min():x.max()+1]
        return consensus_scores(truth[sl],bank[sl])
    return consensus_scores(truth[:,:1,:1],bank[:,:1,:1])


def calibrate(args):
    data,outer,valid,old,weights=setup(args,"inner")
    lib,select=polar.calibration_split(data,valid,args.fold)
    fold=args.fold
    folder=args.previous/f"calibration-{fold}"
    previous=read(folder/"lock.json")
    require(previous["library_indices"]==lib and previous["selection_indices"]==select and
            previous["library_sha256"]==rc.hash_file(folder/"library.npy") and previous["eval_source"]==rc.hash_file(v2.__file__),
            "v2 inner library identity")
    probe=args.previous/f"fold-{fold}-inner"
    require(read(probe/"plan.json")["train_indices"]==rc.split_for(data,fold)[1] and
            rc.hash_file(probe/"final.pt")==previous["inner_polar"],"v2 inner checkpoint identity")
    heads=image_heads(data,select,old,weights)
    cart,rad,sig=v2.model_outputs(data,select,old,probe)
    library=np.load(folder/"library.npy",allow_pickle=False)
    options=[(h,t) for h in HARD for t in STRENGTH]
    candidates=np.zeros((len(options),2))
    control=np.zeros(2)
    pref=np.zeros(3)
    for i,case in enumerate(select):
        truth=np.array(data.mask_labels[case],bool)
        for j,alpha in enumerate(HARD):
            pref[j]+=preference_scores(truth,mask_for(heads[i],alpha))["Dice_mean"]
        fields=v2.rank_fields(cart[i],rad[i],previous["fraction"])
        center,noise=v2.draw_fields(fields,np.maximum(0.65,sig[i]),library,case,50)
        bank=center+previous["scale"]*noise>0
        before=active_scores(truth,bank)
        control += [before["GED"],before["Dice_soft"]]
        ranks,base=bank_ranks(bank),bank.mean(0)
        for j,(h,t) in enumerate(options):
            seg=transport(ranks,base,heads[i,8],heads[i,9],h,t)
            scores=active_scores(truth,seg)
            candidates[j]+=[scores["GED"],scores["Dice_soft"]]
    pref/=len(select)
    candidates/=len(select)
    control/=len(select)
    objective=np.minimum((control[0]-candidates[:,0])/control[0],(candidates[:,1]-control[1])/control[1])
    best=int(np.argmax(objective))
    hard,strength=options[best]
    lock={"fold":fold,"source":rc.hash_file(__file__),"protocol":rc.hash_file(vote.PROTOCOL),
          "v2_lock":rc.hash_file(folder/"lock.json"),"selection_indices":select,"old_inner":rc.hash_file(old/"final.pt"),
          "view_weights":[rc.hash_file(p) for p in weights],"expert_alpha":HARD[int(np.argmax(pref))],
          "expert_dice":pref.tolist(),"hardness":hard,"strength":strength,"surrogate_control":control.tolist(),
          "candidates":[{"hardness":h,"strength":t,"ged50":row[0],"soft50":row[1],"objective":val}
                        for (h,t),row,val in zip(options,candidates,objective)]}
    rc.write_json(args.run/f"lock-{fold}.json",lock)
    print("LOCK",fold,lock["expert_alpha"],hard,strength,control,candidates[best],flush=True)


def infer(args):
    data,outer,valid,old,weights=setup(args,"full")
    f=args.fold
    lock=read(args.run/f"lock-{f}.json")
    require(lock["fold"]==f and lock["source"]==rc.hash_file(__file__) and lock["protocol"]==rc.hash_file(vote.PROTOCOL)
            and lock["selection_indices"]==polar.calibration_split(data,valid,f)[1],"v3 lock differs")
    ids=outer["test_indices"]
    heads=image_heads(data,ids,old,weights)
    where=args.run/f"assessment/fold-{f}"
    where.mkdir(parents=True,exist_ok=False)
    for i,head in enumerate(heads):
        original,prior_pref=v2.old_masks(args.prior,f,i)
        det=mask_for(head,lock["expert_alpha"])
        pref=0.9*np.stack([rc.signed(m) for m in det])+0.1*np.stack([rc.signed(m) for m in prior_pref])>0
        packed={"pref":np.packbits(pref,axis=-1),"det":np.packbits(det,axis=-1)}
        for k,bank in original.items():
            seg=transport(bank_ranks(bank),bank.mean(0),head[8],head[9],lock["hardness"],lock["strength"])
            packed[f"k{k}"]=np.packbits(seg,axis=-1)
        np.savez_compressed(where/f"{i:05d}.npz",**packed)
    rc.write_json(where/"sealed.json",{"fold":f,"cases":len(ids),"bank_sha256":v2.digest_bank(where,len(ids)),
                  "source":rc.hash_file(__file__),"protocol":rc.hash_file(vote.PROTOCOL),
                  "lock_sha256":rc.hash_file(args.run/f"lock-{f}.json"),"full_checkpoints":[rc.hash_file(p) for p in weights]})


def verify_seals(args):
    for f in range(4):
        p=args.run/f"assessment/fold-{f}/sealed.json"
        s=read(p)
        require(s["fold"]==f and s["source"]==rc.hash_file(__file__) and s["protocol"]==rc.hash_file(vote.PROTOCOL)
                and s["lock_sha256"]==rc.hash_file(args.run/f"lock-{f}.json"),"v3 fold not sealed")


def score(args):
    verify_seals(args)
    data,outer,_,_,_=setup(args,"full")
    f=args.fold
    folder=args.run/f"assessment/fold-{f}"
    ids=outer["test_indices"]
    require(v2.digest_bank(folder,len(ids))==read(folder/"sealed.json")["bank_sha256"],"v3 prediction bytes differ")
    reference=read(args.previous/f"assessment/fold-{f}/paired.json")["rows"]
    base=args.prior/f"baseline/fold_{f}/lidc/fold_{f}"
    rows,drift=[],0.
    for i,idx in enumerate(ids):
        truth=np.array(data.mask_labels[idx],bool)
        p1,p2=[base/f"stage_{s}/cases/{i:05d}_masks.npz" for s in (1,2)]
        require(all(np.array_equal(v1.unpack(p,"truth",(4,128,128)),truth) for p in (p1,p2)),"case reference mismatch")
        raw=v1.unpack(p1,"predictions",(50,128,128))
        expert=v1.unpack(p2,"predictions",(4,128,128))
        old,old_pref=v2.old_masks(args.prior,f,i)
        with np.load(folder/f"{i:05d}.npz",allow_pickle=False) as z:
            pred={k:np.unpackbits(z[f"k{k}"],axis=-1).astype(bool) for k in v1.COUNTS}
            pref,det=[np.unpackbits(z[key],axis=-1).astype(bool) for key in ("pref","det")]
        detbank={k:det[((np.arange(k)+0.5)*4/k).astype(int)] for k in v1.COUNTS}
        row={"index":idx,"patient":str(data.series_uid[idx]).split("_")[0],"v3":v1.metrics(truth,pred,pref),
             "v1":v1.metrics(truth,old,old_pref),"e11":v1.metrics(truth,{k:raw[:k] for k in v1.COUNTS},expert),
             "det":v1.metrics(truth,detbank,det)}
        require((reference[i]["index"],reference[i]["patient"])==(idx,row["patient"]),"prior row differs")
        for m in ("e11","v1"):
            drift=max(drift,max(abs(value-reference[i][m][k]) for k,value in row[m].items()))
        rows.append(row)
    require(drift<2e-6,"reference parity")
    rc.write_json(folder/"paired.json",{"rows":rows,"reference_max_abs_error":drift,"seal_sha256":rc.hash_file(folder/"sealed.json")})


def report(args):
    verify_seals(args)
    patients,indices=set(),set()
    rng=np.random.default_rng(212903)
    boots={"e11":[],"v1":[]}
    folds=[]
    for f in range(4):
        folder=args.run/f"assessment/fold-{f}"
        table=read(folder/"paired.json")
        require(table["seal_sha256"]==rc.hash_file(folder/"sealed.json"),"unscored prediction")
        rows=table["rows"]
        unique,inv=np.unique([r["patient"] for r in rows],return_inverse=True)
        ids=[r["index"] for r in rows]
        split=read(args.prior/f"baseline/fold_{f}/lidc/fold_{f}/stage_1/split.json")
        require(ids==split["test_indices"] and set(unique)==set(split["test_patients"]) and
                not indices.intersection(ids) and not patients.intersection(unique),"fold coverage")
        indices.update(ids)
        patients.update(unique)
        keys=list(rows[0]["v3"])
        values={m:np.array([[r[m][k] for k in keys] for r in rows]) for m in ("e11","v1","v3","det")}
        n=np.bincount(inv)
        multiplicity=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=5000)
        for m in boots:
            sums=np.zeros((len(n),len(keys)))
            np.add.at(sums,inv,values["v3"]-values[m])
            boots[m].append(multiplicity@sums/(multiplicity@n)[:,None])
        lock=read(args.run/f"lock-{f}.json")
        folds.append({"fold":f,"cases":len(ids),"patients":len(n),"expert_alpha":lock["expert_alpha"],
                      "hardness":lock["hardness"],"strength":lock["strength"],"reference_max_abs_error":table["reference_max_abs_error"],
                      "means":{m:dict(zip(keys,v.mean(0))) for m,v in values.items()}})
    require(len(indices)==1609 and len(patients)==214,"incomplete LIDC")
    ref=read(args.prior/"assessment/fourfold_comparison.json")["metrics"]
    ci={m:np.percentile(np.mean(v,axis=0),[2.5,97.5],axis=0) for m,v in boots.items()}
    metrics={}
    for j,k in enumerate(keys):
        row={m:float(np.mean([fold["means"][m][k] for fold in folds])) for m in values}
        for m,rkey in (("e11","baseline"),("v1","corrected")):
            require(abs(row[m]-ref[k][rkey])<2e-6,"summary parity")
            row["delta_"+m]=row["v3"]-row[m]
            row["ci95_"+m]=ci[m][:,j].tolist()
        metrics[k]=row
    passes={k:(metrics[k]["delta_v1"]<0 if k=="GED@50" else metrics[k]["delta_v1"]>0) for k in v1.PRIMARY}
    rc.write_json(args.run/"assessment/fourfold_comparison.json",{"primary_pass":passes,"patients":214,"cases":1609,
                  "folds":folds,"metrics":metrics,"source":rc.hash_file(__file__),"protocol":rc.hash_file(vote.PROTOCOL),
                  "v1_summary_sha256":v2.V1_SUMMARY,"e11_summary_sha256":v1.SUMMARY_HASH,"v2_summary_sha256":V2_SUMMARY})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["calibrate","infer","score","report"])
    p.add_argument("--fold",type=int,choices=range(4))
    for argument in ("data","run","prior","previous"):
        p.add_argument("--"+argument,type=Path,required=True)
    args=p.parse_args()
    if args.phase!="report" and args.fold is None:
        p.error("A fold is required")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    {"calibrate":calibrate,"infer":infer,"score":score,"report":report}[args.phase](args)
