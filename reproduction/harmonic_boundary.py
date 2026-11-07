"""Patient-excluded fitting diagnostic of directional single-slice boundary errors."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from reproduction import component_utility as utility
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image, sdf

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/harmonic_boundary_protocol.md"
SECTORS = 24
ANGLES = (np.arange(SECTORS)+.5)*2*np.pi/SECTORS
MODES = ("reference", "image", "constant", "shape")


def sectors(mask):
    y,x = np.nonzero(mask.sum(0)>=2)
    cx,cy = (x.mean(),y.mean()) if len(x) else (63.5,63.5)
    yy,xx = np.indices(mask.shape[-2:])
    angle = np.arctan2(yy-cy, xx-cx) % (2*np.pi)
    return np.minimum((angle*SECTORS/(2*np.pi)).astype(np.uint8), SECTORS-1), angle


def sector_radii(mask, sector):
    return np.stack([np.sqrt(np.bincount(sector[m], minlength=SECTORS)*SECTORS/np.pi) for m in mask])


def smooth(offsets, angles, order=2):
    value = np.full(angles.shape, np.mean(offsets), dtype=np.float32)
    for frequency in range(1,order+1):
        a = 2*np.mean(offsets*np.cos(frequency*ANGLES))
        b = 2*np.mean(offsets*np.sin(frequency*ANGLES))
        value += a*np.cos(frequency*angles) + b*np.sin(frequency*angles)
    return np.clip(.6*value,-2,2)


def continuous_field(q):
    fields = []
    for logit in q:
        d = sdf(logit>0)
        near = np.abs(d)<=.51
        if np.any(near):
            amplitude = max(np.median(np.abs(logit[near])),.05)
            d[near] = np.sign(d[near])*np.clip(.5*np.abs(logit[near])/amplitude,.03,.95)
        fields.append(d)
    return np.stack(fields)


def features(image, u, s):
    q = .5*(u+s)
    ref = q>0
    sector,_ = sectors(ref)
    a,b,m = [sector_radii(mask,sector) for mask in (u>0,s>0,ref)]
    global_radii = np.sqrt(ref.sum((1,2))/np.pi)
    shape = np.column_stack((np.sin(ANGLES), np.cos(ANGLES), np.tile(global_radii,(SECTORS,1)),
        m.T, (b-a).T, m.mean(0)-global_radii.mean(), np.roll(m.mean(0),1)-np.roll(m.mean(0),-1)))
    ct = utility.normalized_ct(image,ref)
    distance = sdf(ref.sum(0)>=2)
    image_features = []
    for low,high in ((-4,-1),(-1,0),(0,1),(1,4)):
        zone = (distance>=low)&(distance<high)
        bins = sector[zone]
        counts = np.bincount(bins,minlength=SECTORS)
        image_features.append(np.column_stack([np.bincount(bins,field[zone],minlength=SECTORS) /
                                                np.maximum(counts,1) for field in ct]+[counts]))
    return shape.astype(np.float32),np.column_stack([shape,*image_features]).astype(np.float32)


def case_partition(indices, uids, seed):
    patients = sorted(nested.patients(indices,uids))
    assignment = np.random.default_rng(seed).permutation(patients)
    splits = [set(assignment[i::3]) for i in range(3)]
    return [[i for i in indices if str(uids[i]).split("_")[0] in group] for group in splits]


def fit_predict(x, y, names, train, held, seed):
    _,inv,count = np.unique(names[train],return_inverse=True,return_counts=True)
    weights = .5+.5*len(train)/(len(count)*count[inv])
    model = HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=9,min_samples_leaf=60,
                l2_regularization=8.,learning_rate=.06,early_stopping=False,random_state=seed)
    model.fit(x[train].reshape(-1,x.shape[-1]), y[train].ravel(), sample_weight=np.repeat(weights,SECTORS))
    return model.predict(x[held].reshape(-1,x.shape[-1])).reshape(len(held),SECTORS)


def lock(args):
    images,truth,_,uids = native.inputs(args.data)
    fitting,_,groups = nested.split_groups(uids,args.fold)
    ids,u,base = nested.base_bank(args,uids,args.fold)
    path = args.nested/"nested-training"/f"oof-{args.fold}.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=utility.QUALITY or
            rc.hash_file(path)!=meta["payload"] or base["payload"]!=meta["base"] or ids!=sorted(fitting)):
        raise ValueError("Training teacher provenance differs")
    with np.load(path,allow_pickle=False) as z:
        if ids!=z["ids"].tolist():
            raise ValueError("OOF order differs")
        s=z["logits"].astype(np.float32)
    row={i:j for j,i in enumerate(ids)}
    names=np.array([str(uids[i]).split("_")[0] for i in ids])
    selection=np.full(len(ids),-1)
    registrations={}
    for j,group in enumerate(groups):
        teacher=json.loads((args.original/f"fold-{args.fold}-sub-{j}/fit.json").read_text())
        for i in group:
            registrations[i]=tuple(teacher["shift"])
    geom,appearance,y=[],[],[]
    for j,i in enumerate(ids):
        ref=(u[j]+s[j])>0
        sector,_=sectors(ref)
        shape,ct=features(register_image(images[i],registrations[i]),u[j],s[j])
        geom.append(shape)
        appearance.append(ct)
        y.append(np.clip((sector_radii(truth[i].astype(bool),sector)-sector_radii(ref,sector)).mean(0),-3,3))
    geom,appearance,y=map(np.stack,(geom,appearance,y))
    targets={k:np.zeros((len(ids),SECTORS)) for k in ("image","shape")}
    identity=[]
    for g,group in enumerate(groups):
        subgroups=case_partition(group,uids,57121+37*g+args.fold)
        for j,hold in enumerate(subgroups):
            use=sorted(set(group)-set(hold))
            if not use or not hold or nested.patients(hold,uids)&nested.patients(use,uids):
                raise ValueError("Meta-fold must exclude complete patients")
            ti=np.array([row[i] for i in use])
            hi=np.array([row[i] for i in hold])
            if np.any(selection[hi]>=0):
                raise ValueError("Duplicate cross-patient prediction")
            selection[hi]=3*g+j
            for key,inputs in (("image",appearance),("shape",geom)):
                targets[key][hi]=fit_predict(inputs,y,names,ti,hi,57121+g)
            identity.append({"teacher_group":g,"meta_group":j,"fit":use,"held":hold})
    if np.any(selection<0):
        raise ValueError("Incomplete fitting-patient coverage")
    masks={k:[] for k in MODES}
    for j in range(len(ids)):
        q=.5*(u[j]+s[j])
        ref=q>0
        _,angle=sectors(ref)
        field=continuous_field(q)
        if not np.array_equal(field>0,ref):
            raise ValueError("Zero displacement must reproduce every input mask")
        masks["reference"].append(np.packbits(ref,axis=-1))
        for key,reg,order in (("image","image",2),("constant","image",0),("shape","shape",2)):
            masks[key].append(np.packbits(field+smooth(targets[reg][j],angle,order)>0,axis=-1))
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**{name:np.stack(arr) for name,arr in masks.items()})
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":identity,"base":base["payload"],
        "nested":meta["payload"],"nested_audit":utility.QUALITY,"prediction":rc.hash_file(dst),
        "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("ANGULAR_FIT_SEAL",args.fold,len(ids),len(set(names)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("run","data","original","nested"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    lock(args)
