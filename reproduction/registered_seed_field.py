"""Export a registered CT confidence field using only the frozen image components."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from reproduction import outer_preference as prior
from reproduction import registered_zoom as zoom
from reproduction import vote_contours as vote


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def model_identity(folder,expected,source):
    fit=prior.read(folder/"fit.json")
    if (prior.read(folder/"plan.json")["train_indices"]!=expected or fit["source"]!=prior.rc.hash_file(source)
            or fit["checkpoint"]!=prior.rc.hash_file(folder/"final.pt")):
        raise ValueError(f"Unregistered CT model {folder}")
    return fit


def cpu_signed(mask):
    from scipy.ndimage import distance_transform_edt
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or mask.all():
        return np.full(mask.shape, 8.0 if mask.all() else -8.0, dtype=np.float32)
    distances = np.where(mask, distance_transform_edt(mask)-0.5, 0.5-distance_transform_edt(~mask))
    return distances.clip(-8, 8).astype(np.float32)


@torch.inference_mode()
def cpu_refinement(model, images, guide):
    result = []
    for start in range(0, len(images), 16):
        x = torch.from_numpy(images[start:start+16, None]).float()/255
        g = torch.from_numpy(guide[start:start+16])
        total = 0
        for flip in (False, True):
            view = x.flip([-1]) if flip else x
            mask = g.flip([-1]) if flip else g
            for rotation in range(4):
                output = model(torch.rot90(view,rotation,(-2,-1)), torch.rot90(mask,rotation,(-2,-1)))
                output = torch.rot90(output,-rotation,(-2,-1))
                total = total + (output.flip([-1]) if flip else output)
        result.append((total/(8*0.65)).numpy())
    return np.concatenate(result)


@torch.inference_mode()
def stable_legacy(args, ids, images):
    from scipy.special import logit
    old = prior.incumbent.old
    identity, (data, _, _, previous_model, weights, previous, joint, fixed) = old.previous(args,args.scope)
    raw, _ = old.v9.v4.features(data, ids, previous_model, weights, previous["expert_alpha"])
    guide = np.stack([0.5*(cpu_signed(v[5]>0)+cpu_signed(v[6]>0)) for v in raw])[:,None]
    saved_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.backends.mkldnn.deterministic = True
        estimates = []
        for owner in (identity.fixed,identity.run):
            folder = owner/f"fold-{args.fold}-{args.scope}"
            model = old.v9.normal.BoundaryRefiner().to(memory_format=torch.channels_last).eval()
            model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
            estimates.append(cpu_refinement(model,images,guide))
        values = raw.copy()
        weight = fixed["preference_weight"]
        values[:,:4] = (1-weight)*raw[:,:4]+weight*((estimates[0]+estimates[1])*0.5)[:,:4]
        decoder = old.v9.joint.JointDecoder.restore(joint["parameters"],"cpu")
        probability = np.concatenate([decoder(batch)[0].numpy() for batch in torch.from_numpy(values).split(12)])
        base = logit(probability.clip(1e-6,1-1e-6)).astype(np.float16).astype(np.float32)
        return base, {"raw_image":digest_array(raw),"distance":digest_array(guide),
                      "normal_pair":[digest_array(x) for x in estimates],"cpu_joint":digest_array(base)}
    finally:
        torch.set_num_threads(saved_threads)


def compute(args):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        raise ValueError("The deterministic CUDA workspace must be set before starting Python")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(13411);torch.cuda.manual_seed_all(13411)
    torch.set_num_threads(3);cv2.setNumThreads(1)
    a=args.previous
    snapshot=a/"checked-source"
    if (Path(prior.__file__).resolve()!=snapshot/"reproduction/outer_preference.py"
            or prior.rc.hash_file(a/"checked-source.sha256")!="0d14e0b2641678c899f97693081cb0f9e8cd4af6ac33809a3899b9291a400abd"
            or prior.rc.hash_file(args.data)!=prior.rc.DATA_HASH
            or prior.rc.hash_file(a/"assessment/outer_report.json")!="799311c1c176f2d8239c619e4c6f35a67fa0ec9300c60c3194cb87d97ade58a3"):
        raise ValueError("Original CT inference sources or data differ")
    for k,v in prior.HISTORY.items():setattr(args,k,a.parent/v)
    f=args.fold;mode=args.scope
    with args.data.open("rb") as stream:cases=list(pickle.load(stream).values())
    images=np.stack([case["image"] for case in cases]);uids=[case["series_uid"] for case in cases]
    group=type("Groups",(),{"series_uid":uids})()
    split,training,valid=prior.native.splits(uids,f)
    library,checked=prior.polar.calibration_split(group,valid,f)
    ids=library+checked if mode=="inner" else split["test_indices"]
    expected=training if mode=="inner" else split["train_indices"]
    src=args.align/f"fold-{f}-{mode}-registered"
    registered=model_identity(src,expected,prior.align.__file__)
    zsrc=args.zoom/f"fold-{f}-{mode}";zfit=model_identity(zsrc,expected,zoom.__file__)
    loc=model_identity(args.local/f"fold-{f}-{mode}-locator",expected,prior.native.__file__)
    if zfit["parent"]!=registered["checkpoint"]:
        raise ValueError("Zoom training identity differs")
    align_lock=prior.read(args.align/f"check-{f}-registered.json")
    old_lock=prior.read(args.incumbent/f"lock-{f}.json")
    previous_selection=prior.read(args.incumbent/"selection.json")
    if (align_lock["fit_indices"]!=library or align_lock["check_indices"]!=checked
            or previous_selection["lock_hashes"][f]!=prior.rc.hash_file(args.incumbent/f"lock-{f}.json")
            or old_lock["align_lock"]!=prior.rc.hash_file(args.align/f"check-{f}-registered.json")):
        raise ValueError("Original library geometry choice differs")
    shift=registered["applied_shift"]
    x=images[ids]
    centers=prior.incumbent.centers(args,mode,x)
    whole=prior.align.predict(vote.load_net(src/"final.pt"),x[:,None],centers,shift)[:,:4].astype(np.float16).astype(np.float32)
    enlarged=zoom.predict(zoom.model_from(zsrc/"final.pt"),x[:,None],centers,shift)[:,:4].astype(np.float16).astype(np.float32)
    legacy_fingerprint = {}
    aligned_weight = align_lock["best"]["w"]
    if args.stable_refinement and aligned_weight < 1:
        base, legacy_fingerprint = stable_legacy(args,ids,x)
        whole = (1-aligned_weight)*base + aligned_weight*whole
    blocks=max(1,len(ids)//12)
    whole=np.concatenate([prior.metrics.moment_project(v,c,align_lock["best"]["s"])
                          for v,c in zip(np.array_split(whole,blocks),np.array_split(centers,blocks))])
    yy,xx=np.indices((128,128),dtype=np.float32)
    radius=np.maximum(np.abs(xx[None]-centers[:,0,None,None]),np.abs(yy[None]-centers[:,1,None,None]))
    window=np.clip((34-radius)/8,0,1)[:,None]
    field=prior.incumbent.old.blend(whole,enlarged,window,old_lock["zoom_weight"]).astype(np.float16).astype(np.float32)
    decision=prior.read(a/"selection.json")["folds"][f]
    name=decision["variant"]
    if mode=="inner":
        texture=args.texture/f"features-{f}-inner.npz"
        info=prior.read(texture.with_suffix(".json"))
        owner=args.ct if name=="ct-adapted" else args.encoder
        selected=owner/f"pred-{f}-{name}.npz"
        if (info["indices"]!=ids or prior.rc.hash_file(texture)!=info["payload"]
                or prior.rc.hash_file(selected)!=decision["inner"][name]["prediction"]):
            raise ValueError("Archived inner masks differ")
        with np.load(texture,allow_pickle=False) as bank:
            if bank["ids"].tolist()!=ids:raise ValueError("Wrong v10 inner indices")
            v10=np.unpackbits(bank["old"],axis=-1).astype(bool)
        with np.load(selected,allow_pickle=False) as bank:
            if bank["ids"].tolist()!=ids:raise ValueError("Wrong A inner indices")
            reference=np.unpackbits(bank["masks"],axis=-1).astype(bool)
        trained=owner/f"fold-{f}-{name}"
        init_dir=args.public/f"fold-{f}-inner"
    else:
        previous_folder=a/f"assessment/fold-{f}"
        seal=prior.verify_bank(previous_folder,f,prior.rc.hash_file(a/"selection.json"))
        with np.load(previous_folder/"predictions.npz",allow_pickle=False) as bank:
            if bank["ids"].tolist()!=ids:raise ValueError("Wrong A outer indices")
            v10=np.unpackbits(bank["reference"],axis=-1).astype(bool)
            reference=np.unpackbits(bank["masks"],axis=-1).astype(bool)
        trained=a/f"fold-{f}-{name}";init_dir=a/f"fold-{f}-full"
    fit=prior.read(trained/"fit.json");plan=prior.read(trained/"plan.json")
    init=prior.read(init_dir/"fit.json")
    if (fit["checkpoint"]!=prior.rc.hash_file(trained/"final.pt") or fit["plan"]!=prior.rc.hash_file(trained/"plan.json")
            or plan["train_indices"]!=sorted(expected) or init["train_indices"]!=sorted(expected)
            or plan["initial_decoder"]!=init["checkpoint"] or init["checkpoint"]!=prior.rc.hash_file(init_dir/"final.pt")):
        raise ValueError("Frozen A image model identity differs")
    state=torch.load(init_dir/"final.pt",map_location="cpu",weights_only=True)
    model=prior.encoder.AdaptedPrompts(args.public,state,False,name=="ct-adapted").cuda().eval()
    model.load_state_dict(torch.load(trained/"final.pt",map_location="cpu",weights_only=True))
    args.run=a;args.cache_run=args.encoder
    cache,_=prior.encoder.open_prefix(args)
    v10_guide=prior.incumbent.anchor_logits(field,v10)
    branch=prior.encoder.predict(model,cache,ids,images,v10_guide)
    weight=decision["weight"]
    adapted=prior.incumbent.anchor_logits((1-weight)*v10_guide+weight*branch.astype(np.float16).astype(np.float32),reference)
    direct=prior.incumbent.anchor_logits(field,reference)
    if (not np.isfinite(adapted).all() or not np.isfinite(direct).all() or not np.array_equal(adapted>0,reference)
            or not np.array_equal(direct>0,reference)):raise ValueError("Invalid guide signs or values")
    dst=args.output/f"field-{mode}-{f}.npz"
    record={"fold":f,"scope":mode,"cases":len(ids),"direct":digest_array(direct),"adapted":digest_array(adapted),
            "registered":digest_array(field),"legacy":legacy_fingerprint,"reference":digest_array(reference),"a_model":fit["checkpoint"],
            "ct_models":[loc["checkpoint"],registered["checkpoint"],zfit["checkpoint"]],
            "source":prior.rc.hash_file(__file__),"protocol":prior.rc.hash_file(args.protocol),"dataset":prior.rc.DATA_HASH}
    if args.replay:
        prev=prior.read(dst.with_suffix(".json"))
        if any(prev[k]!=v for k,v in record.items()) or prior.rc.hash_file(dst)!=prev["payload"]:
            raise ValueError(f"Independent registered field differs {mode} fold {f}: {record}")
        prior.rc.write_json(args.output/f"replay-{mode}-{f}.json",{**record,"payload":prev["payload"],"byte_equal":True})
    else:
        if dst.exists():raise FileExistsError(dst)
        tmp=dst.with_suffix(".partial.npz")
        np.savez_compressed(tmp,ids=ids,direct=direct,adapted=adapted,reference=np.packbits(reference,axis=-1))
        tmp.replace(dst)
        prior.rc.write_json(dst.with_suffix(".json"),{**record,"payload":prior.rc.hash_file(dst)})
    print("REGISTERED_FIELD",mode,f,"replayed" if args.replay else "saved",record["adapted"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("previous","texture","output","data","protocol"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--scope",choices=("inner","full"),required=True)
    parser.add_argument("--replay",action="store_true")
    parser.add_argument("--stable-refinement",action="store_true")
    compute(parser.parse_args())
