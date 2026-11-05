"""Nested calibration and label-blind generation for the second LIDC study."""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets.lidc import BaseDataset
from reproduction import boundary_eval as v1
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction.metrics import consensus_scores, preference_scores


V1_SUMMARY = "382cb8465d20f23dba3f7dcc1a5cc86b03a41381bb100e3b982de2fa57859e98"
FRACTIONS = (0.25, 0.50, 0.75)
SCALES = (0.65, 1., 1.35, 1.75, 2.25)
read = v1.read


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked_inputs(args, fold, mode):
    require(rc.hash_file(args.data) == rc.DATA_HASH and
            rc.hash_file(args.prior / "assessment/fourfold_comparison.json") == V1_SUMMARY and
            rc.hash_file(args.prior / "baseline/fourfold_summary.json") == v1.SUMMARY_HASH, "input identity")
    data = BaseDataset(str(args.data))
    outer, train, valid = rc.split_for(data, fold)
    expected = outer["train_indices"] if mode == "full" else train
    for stage in (1, 2):
        manifest = read(args.prior / f"baseline/fold_{fold}/lidc/fold_{fold}/stage_{stage}/split.json")
        require(all(manifest[k] == v for k,v in outer.items()), "patient manifest differs")
    old, new = [root / f"fold-{fold}-{mode}" for root in (args.prior, args.run)]
    for path in (old, new):
        plan, result = read(path/"plan.json"), read(path/"result.json")
        digest = rc.hash_file(path/"final.pt")
        require(plan["fold"] == fold and plan["train_indices"] == expected and
                plan["inner_valid_indices"] == (valid if mode == "inner" else []) and result["steps"] == 4000,
                "training patients or budget differ")
        if path == old:
            require(plan["mode"] == mode and plan["training_source"] == rc.hash_file(rc.__file__) and
                    result["checkpoint_sha256"] == digest, "Cartesian checkpoint")
        else:
            require(plan["phase"] == mode and plan["source"] == rc.hash_file(polar.__file__) and
                    plan["protocol"] == rc.hash_file(polar.PROTOCOL) and result["checkpoint"] == digest,
                    "polar checkpoint")
    return data, outer, valid, old, new


def model_outputs(data, ids, old, new):
    images = torch.from_numpy(np.stack([data.images[i] for i in ids])[:,None]).cuda()
    cart = rc.RankContours().cuda().to(memory_format=torch.channels_last).eval()
    cart.load_state_dict(torch.load(old/"final.pt", map_location="cpu", weights_only=True))
    cart_masks = torch.cat([rc.predict(cart, part) for part in images.split(12)]).cpu().numpy()
    radii, sigma = polar.predict(polar.load_net(new/"final.pt"), images)
    return cart_masks, radii, sigma[:,0]


def rank_fields(cart, radii, fraction):
    # Absolute distances must keep increasing outside the boundary. A clipped
    # background plus a large angular displacement could create a ray to the edge.
    contours = []
    for mask in cart:
        m = np.ascontiguousarray(mask, dtype=np.uint8)
        if not m.any() or m.all():
            contours.append(np.full(m.shape, 182.0 if m.all() else -182.0, dtype=np.float32))
            continue
        inside = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outside = cv2.distanceTransform(1-m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        contours.append(np.where(m, inside-0.5, 0.5-outside))
    signed_cart = np.stack(contours)
    signed_polar = polar.angular_map(radii) - polar.RADIUS
    return fraction * signed_polar + (1-fraction) * signed_cart


def rank_order(k, prior, rng):
    groups = ((np.arange(k)+0.5)*4/k).astype(int)
    if prior is None:
        return rng.permutation(groups)
    order = np.argsort(prior.reshape(k,-1).sum(1), kind="stable")
    ranks = np.empty(k, dtype=int)
    ranks[order] = groups
    return ranks


def draw_fields(fields, sigma, library, case, k, prior=None):
    rng = np.random.default_rng(307301 + case * 100 + k)
    ranks = rank_order(k, prior, rng)
    picks = rng.integers(len(library), size=k)
    shifts = rng.integers(polar.ANGLES, size=k)
    signs = rng.choice([-1, 1], size=k)
    angles = (shifts[:,None] + signs[:,None]*np.arange(polar.ANGLES)) % polar.ANGLES
    noise = library[picks[:,None], ranks[:,None], angles] * sigma
    return fields[ranks], polar.angular_map(noise)


def calibrate(args):
    f = args.fold
    data, outer, valid, old, new = checked_inputs(args, f, "inner")
    require(read(new/"result.json")["passed"], "inner polar gate failed")
    lib_ids, select_ids = polar.calibration_split(data, valid, f)
    ids = lib_ids + select_ids
    cart, radii, sigma = model_outputs(data, ids, old, new)
    n = len(lib_ids)
    truth = np.array([data.mask_labels[i] for i in ids], dtype=bool)
    actual = torch.from_numpy(truth[:n]).cuda()
    labels = polar.mask_radii(actual, polar.grid(n, "cuda")).cpu().numpy()
    relative_error = labels - labels.mean(1,keepdims=True) - (radii[:n] - radii[:n].mean(1,keepdims=True))
    library = (relative_error / np.maximum(0.65, sigma[:n,None,:])).astype(np.float32)
    require(np.isfinite(library).all() and set(ids).isdisjoint(outer["test_indices"]), "invalid inner library")
    dice_values = []
    for w in FRACTIONS:
        values = [preference_scores(t, rank_fields(c, p, w)>0)["Dice_mean"]
                  for c,p,t in zip(cart[n:], radii[n:], truth[n:])]
        dice_values.append(float(np.mean(values)))
    w = FRACTIONS[int(np.argmax(dice_values))]
    ged_values = [[] for _ in SCALES]
    for j, case in enumerate(select_ids, start=n):
        fields = rank_fields(cart[j], radii[j], w)
        center, noise = draw_fields(fields, np.maximum(0.65,sigma[j]), library, case, 50)
        for scores, scale in zip(ged_values, SCALES):
            seg = center + scale * noise > 0
            scores.append(consensus_scores(truth[j],seg)["GED"])
    ged_values = np.mean(ged_values, axis=1).tolist()
    scale = SCALES[int(np.argmin(ged_values))]
    path = args.run / f"calibration-{f}"
    path.mkdir(exist_ok=False)
    np.save(path/"library.npy",library,allow_pickle=False)
    rc.write_json(path/"lock.json", {"fold":f, "eval_source":rc.hash_file(__file__), "protocol":rc.hash_file(polar.PROTOCOL),
                  "v1_summary":V1_SUMMARY, "inner_polar":rc.hash_file(new/"final.pt"), "inner_cart":rc.hash_file(old/"final.pt"),
                  "library_sha256":rc.hash_file(path/"library.npy"), "library_indices":lib_ids,"selection_indices":select_ids,
                  "cart_polar_validation_dice":dict(zip(map(str,FRACTIONS),dice_values)),
                  "residual_validation_ged":dict(zip(map(str,SCALES),ged_values)), "fraction":w,"scale":scale})
    print("CALIBRATED", f, w, scale, dice_values, ged_values, flush=True)


def digest_bank(path, count):
    h = hashlib.sha256()
    for case in range(count):
        name = f"{case:05d}.npz"
        h.update(name.encode())
        h.update((path/name).read_bytes())
    return h.hexdigest()


def old_masks(root, f, i):
    with np.load(root/f"assessment/fold-{f}/{i:05d}_masks.npz", allow_pickle=False) as z:
        banks = {k:np.unpackbits(z[f"k{k}"],axis=-1).astype(bool) for k in v1.COUNTS}
        pref = np.unpackbits(z["pref"],axis=-1).astype(bool)
    require(all(b.shape==(k,128,128) for k,b in banks.items()) and pref.shape==(4,128,128), "v1 bank shape")
    return banks,pref


def infer(args):
    f=args.fold
    data, outer, valid, old, new = checked_inputs(args,f,"full")
    folder = args.run/f"calibration-{f}"
    lock=read(folder/"lock.json")
    lib, select=polar.calibration_split(data,valid,f)
    require(lock["fold"]==f and lock["library_indices"]==lib and lock["selection_indices"]==select and
            lock["eval_source"]==rc.hash_file(__file__) and lock["protocol"]==rc.hash_file(polar.PROTOCOL) and
            rc.hash_file(folder/"library.npy")==lock["library_sha256"], "calibration lock mismatch")
    library = np.load(folder/"library.npy",allow_pickle=False)
    ids = outer["test_indices"]
    cart,radii,sigma = model_outputs(data,ids,old,new)
    path = args.run/"assessment"/f"fold-{f}"
    path.mkdir(parents=True,exist_ok=False)
    for i,case in enumerate(ids):
        prior,old_pref = old_masks(args.prior,f,i)
        fields = rank_fields(cart[i],radii[i],lock["fraction"])
        pred_pref = 0.85*fields + 0.15*np.stack([rc.signed(m) for m in old_pref]) > 0
        pack = {"pref":np.packbits(pred_pref,axis=-1),"det":np.packbits(fields>0,axis=-1)}
        for k in v1.COUNTS:
            center, noise = draw_fields(fields,np.maximum(0.65,sigma[i]),library,case,k,prior[k])
            new_field = center + lock["scale"] * noise
            old_field = np.stack([rc.signed(m) for m in prior[k]])
            seg = 0.85*new_field + 0.15*old_field > 0
            pack[f"k{k}"] = np.packbits(seg,axis=-1)
        np.savez_compressed(path/f"{i:05d}.npz",**pack)
    rc.write_json(path/"sealed.json", {"fold":f,"cases":len(ids),"bank_sha256":digest_bank(path,len(ids)),
                  "eval_source":rc.hash_file(__file__), "protocol":lock["protocol"],"lock_sha256":rc.hash_file(folder/"lock.json"),
                  "polar_full":rc.hash_file(new/"final.pt"),"cart_full":rc.hash_file(old/"final.pt")})


def sealed(args):
    for f in range(4):
        path=args.run/f"assessment/fold-{f}"
        s=read(path/"sealed.json")
        require(s["eval_source"]==rc.hash_file(__file__) and s["fold"]==f and
                s["lock_sha256"]==rc.hash_file(args.run/f"calibration-{f}/lock.json") and
                s["protocol"]==rc.hash_file(polar.PROTOCOL), "unsealed fold")


def score(args):
    sealed(args)
    f=args.fold
    data,outer,_,_,_ = checked_inputs(args,f,"full")
    ids=outer["test_indices"]
    path=args.run/f"assessment/fold-{f}"
    require(digest_bank(path,len(ids))==read(path/"sealed.json")["bank_sha256"],"prediction bytes differ")
    historic=read(args.prior/f"assessment/fold-{f}/paired.json")["rows"]
    base=args.prior/f"baseline/fold_{f}/lidc/fold_{f}"
    rows=[]
    error=0.
    for i,idx in enumerate(ids):
        truth=np.asarray(data.mask_labels[idx],dtype=bool)
        files=[base/f"stage_{s}/cases/{i:05d}_masks.npz" for s in (1,2)]
        require(all(np.array_equal(v1.unpack(p,"truth",(4,128,128)),truth) for p in files),"truth ordering")
        raw=v1.unpack(files[0],"predictions",(50,128,128))
        raw_pref=v1.unpack(files[1],"predictions",(4,128,128))
        old,old_pref=old_masks(args.prior,f,i)
        with np.load(path/f"{i:05d}.npz",allow_pickle=False) as z:
            bank={k:np.unpackbits(z[f"k{k}"],axis=-1).astype(bool) for k in v1.COUNTS}
            pref,det=[np.unpackbits(z[key],axis=-1).astype(bool) for key in ("pref","det")]
        detbank={k:det[((np.arange(k)+0.5)*4/k).astype(int)] for k in v1.COUNTS}
        patient=str(data.series_uid[idx]).split("_")[0]
        row={"index":idx,"patient":patient,"e11":v1.metrics(truth,{k:raw[:k] for k in v1.COUNTS},raw_pref),
             "v1":v1.metrics(truth,old,old_pref),"v2":v1.metrics(truth,bank,pref),"det":v1.metrics(truth,detbank,det)}
        require((historic[i]["index"],historic[i]["patient"])==(idx,patient),"prior row identity")
        for new_key,old_key in (("e11","baseline"),("v1","corrected")):
            error=max(error,max(abs(value-historic[i][old_key][key]) for key,value in row[new_key].items()))
        rows.append(row)
    require(error < 2e-6,"historical per-case parity failure")
    rc.write_json(path/"paired.json",{"reference_max_abs_error":error,"seal_sha256":rc.hash_file(path/"sealed.json"),"rows":rows})


def report(args):
    sealed(args)
    groups_seen, indices_seen = set(),set()
    folds=[]
    bootstrap={"v1":[],"e11":[]}
    rng=np.random.default_rng(212903)
    for f in range(4):
        path=args.run/f"assessment/fold-{f}"
        table=read(path/"paired.json")
        require(table["seal_sha256"]==rc.hash_file(path/"sealed.json"),"scored bank mismatch")
        rows=table["rows"]
        ids=[r["index"] for r in rows]
        patients,inv=np.unique([r["patient"] for r in rows],return_inverse=True)
        split=read(args.prior/f"baseline/fold_{f}/lidc/fold_{f}/stage_1/split.json")
        require(ids==split["test_indices"] and set(patients)==set(split["test_patients"]) and
                not groups_seen.intersection(patients) and not indices_seen.intersection(ids),"cross-fold identity")
        groups_seen.update(patients)
        indices_seen.update(ids)
        keys=list(rows[0]["v2"])
        arrays={m:np.array([[r[m][k] for k in keys] for r in rows]) for m in ("v2","v1","e11","det")}
        size=np.bincount(inv)
        draw=rng.multinomial(len(patients),np.full(len(patients),1/len(patients)),size=5000)
        for m in bootstrap:
            sums=np.zeros((len(patients),len(keys)))
            np.add.at(sums,inv,arrays["v2"]-arrays[m])
            bootstrap[m].append(draw@sums/(draw@size)[:,None])
        lock=read(args.run/f"calibration-{f}/lock.json")
        folds.append({"fold":f,"patients":len(patients),"cases":len(ids),"fraction":lock["fraction"],"scale":lock["scale"],
                      "score_parity_error":table["reference_max_abs_error"],
                      "means":{m:dict(zip(keys,v.mean(0))) for m,v in arrays.items()}})
    require(len(groups_seen)==214 and len(indices_seen)==1609,"incomplete outer cohort")
    ref=read(args.prior/"assessment/fourfold_comparison.json")
    metrics={}
    cis={m:np.percentile(np.mean(values,axis=0),[2.5,97.5],axis=0) for m,values in bootstrap.items()}
    for j,k in enumerate(keys):
        row={m:float(np.mean([fold["means"][m][k] for fold in folds])) for m in arrays}
        for m,rkey in (("v1","corrected"),("e11","baseline")):
            require(abs(row[m]-ref["metrics"][k][rkey])<2e-6,"summary parity")
            row["delta_"+m]=row["v2"]-row[m]
            row["ci95_"+m]=cis[m][:,j].tolist()
        metrics[k]=row
    passed={k:(metrics[k]["delta_v1"]<0 if k=="GED@50" else metrics[k]["delta_v1"]>0) for k in v1.PRIMARY}
    rc.write_json(args.run/"assessment/fourfold_comparison.json",{"patients":214,"cases":1609,"primary_pass":passed,
                  "v1_summary_sha256":V1_SUMMARY,"e11_summary_sha256":v1.SUMMARY_HASH,"protocol":rc.hash_file(polar.PROTOCOL),
                  "source":rc.hash_file(__file__),"folds":folds,"metrics":metrics})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["calibrate","infer","score","report"])
    p.add_argument("--fold",type=int,choices=range(4))
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--prior",type=Path,required=True)
    args=p.parse_args()
    if args.phase!="report" and args.fold is None:
        p.error("A fold is required")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    {"calibrate":calibrate,"infer":infer,"score":score,"report":report}[args.phase](args)
