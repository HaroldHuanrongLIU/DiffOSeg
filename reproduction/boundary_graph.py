"""Learn CT boundary affinities on out-of-fit patients and minimize a spatial Potts energy."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier

from reproduction import gated_hierarchy as public_cache
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import rater_area
from reproduction.shell_occupancy import unpack_level

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/boundary_graph_protocol.md"
STRENGTHS = (0., .125, .25, .5, 1.)
MODES = ("unary","constant","learned")
CAPACITY_SCALE = 4096


def neighbors(shape):
    pixels = np.arange(np.prod(shape)).reshape(shape)
    return (np.r_[pixels[:,:-1].ravel(),pixels[:-1,:].ravel()],
            np.r_[pixels[:,1:].ravel(),pixels[1:,:].ravel()])


def distance(mask):
    if not mask.any() or mask.all():
        return np.full(mask.shape, 8. if mask.all() else -8., np.float32)
    return np.where(mask,ndimage.distance_transform_edt(mask)-.5,
                    .5-ndimage.distance_transform_edt(~mask)).astype(np.float32)


def cut(mask, variable, unary, left, right, pair_cost):
    """Source side is foreground. Fixed neighbors contribute only a terminal cost."""
    result = mask.copy()
    if len(variable)==0:
        return result
    count = len(variable)
    node = np.full(mask.size,-1,np.intp)
    node[variable] = np.arange(count)
    a,b = node[left],node[right]
    cost = np.floor(pair_cost*CAPACITY_SCALE+.5).astype(np.int64)
    bg = np.floor(np.maximum(unary,0)*CAPACITY_SCALE+.5).astype(np.int64)
    fg = np.floor(np.maximum(-unary,0)*CAPACITY_SCALE+.5).astype(np.int64)
    flat = mask.ravel()
    for nodes,others in ((a,right),(b,left)):
        boundary = (nodes>=0)&(node[others]<0)
        np.add.at(bg,nodes[boundary&flat[others]],cost[boundary&flat[others]])
        np.add.at(fg,nodes[boundary&~flat[others]],cost[boundary&~flat[others]])
    internal = (a>=0)&(b>=0)
    pixels = np.arange(count)
    source,sink = count,count+1
    start = np.r_[np.full(count,source),pixels,a[internal],b[internal]]
    end = np.r_[pixels,np.full(count,sink),b[internal],a[internal]]
    capacity = csr_matrix((np.r_[bg,fg,cost[internal],cost[internal]],(start,end)),shape=(count+2,count+2))
    capacity.eliminate_zeros()
    residual = (capacity-maximum_flow(capacity,source,sink,method="dinic").flow).tocsr()
    residual.eliminate_zeros()
    reached = breadth_first_order(residual,source,directed=True,return_predecessors=False)
    result.ravel()[variable] = False
    result.ravel()[variable[reached[reached<count]]] = True
    return result


@dataclass
class Region:
    mask: np.ndarray
    d: np.ndarray
    variable: np.ndarray
    left: np.ndarray
    right: np.ndarray
    geom: np.ndarray
    appearance: np.ndarray
    edge: np.ndarray


def regions(image,masks,packed):
    radii = np.sqrt(masks.sum((1,2))/np.pi)
    votes = masks.sum(0)
    support = votes>=2
    outside = ndimage.distance_transform_edt(~support)
    ring = (~support)&(outside>1)&(outside<=8)
    if support.any() and ring.any():
        low,high = np.median(image[ring]),np.median(image[support])
    else:
        low,high = np.quantile(image,[.1,.9])
    normalized = ((image.astype(np.float32)-low)/max(float(high)-float(low),12.)).clip(-3,3)
    smooth = cv2.GaussianBlur(normalized,(0,0),.8)
    gradient = np.hypot(*np.gradient(smooth)).ravel()
    high_features = unpack_level(packed[512:],4).reshape(32,-1)
    a,b = neighbors(image.shape)
    image_flat,blur,support_flat = normalized.ravel(),smooth.ravel(),votes.ravel()/4
    for rank,mask in enumerate(masks):
        d = distance(mask).ravel()
        band = np.abs(d)<=3
        variable = np.flatnonzero(band)
        contact = band[a]|band[b]
        l,r = a[contact],b[contact]
        geom = np.column_stack((np.full(len(variable),rank/3),d[variable],np.abs(d[variable]),
                                support_flat[variable],np.full(len(variable),radii[rank]/16)))
        hp = high_features[:,variable].astype(np.float32)
        hp /= np.sqrt(np.square(hp).mean(0)+1e-4)
        appearance = np.column_stack((geom,image_flat[variable],blur[variable],gradient[variable],
                                      image_flat[variable]-blur[variable],hp.T))
        hl = high_features[:,l].astype(np.float32)
        hr = high_features[:,r].astype(np.float32)
        cosine = (hl*hr).sum(0)/np.maximum(np.linalg.norm(hl,axis=0)*np.linalg.norm(hr,axis=0),1e-5)
        difference = image_flat[l]-image_flat[r]
        edge = np.column_stack((np.full(len(l),rank/3),np.full(len(l),radii[rank]/16),
                                (d[l]+d[r])*.5,np.minimum(abs(d[l]),abs(d[r])),abs(d[l]-d[r]),
                                mask.ravel()[l]!=mask.ravel()[r],(support_flat[l]+support_flat[r])*.5,
                                (image_flat[l]+image_flat[r])*.5,abs(difference),difference*np.sign(d[l]-d[r]),
                                abs(blur[l]-blur[r]),(gradient[l]+gradient[r])*.5,
                                np.mean(abs(hl-hr),axis=0),1-cosine))
        yield Region(mask,d[variable],variable,l,r,geom.astype(np.float32),appearance.astype(np.float32),edge.astype(np.float32))


def fit_models(args,training,images,truth,uids,features,shift):
    bank = args.crossfit/f"oof-{args.fold}.npz"
    meta = json.loads(bank.with_suffix(".json").read_text())
    if meta["payload"]!=rc.hash_file(bank):
        raise ValueError("Teacher guide changed")
    with np.load(bank,allow_pickle=False) as z:
        ids=z["ids"].tolist()
        if ids!=sorted(training):raise ValueError("Teacher guide includes ineligible cases")
        guides=z["logits"]>0
    _,inv,nums=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    case_weight=.5+.5*len(ids)/(len(nums)*nums[inv])
    rng=np.random.default_rng(190351+args.fold)
    xs={k:[] for k in ("geom","appearance","edge")}
    ys={k:[] for k in xs}; weights={k:[] for k in xs}
    for i,masks,cw in zip(ids,guides,case_weight,strict=True):
        im=rater_area.register_image(images[i],shift)
        for rank,reg in enumerate(regions(im,masks,features[i,0])):
            if not len(reg.variable):continue
            y=truth[i,rank].ravel().astype(bool)
            pixels=rng.choice(len(reg.variable),min(160,len(reg.variable)),replace=False)
            edges=rng.choice(len(reg.left),min(160,len(reg.left)),replace=False)
            for k,values,sample,target in (
                ("geom",reg.geom,pixels,y[reg.variable]),("appearance",reg.appearance,pixels,y[reg.variable]),
                ("edge",reg.edge,edges,y[reg.left]!=y[reg.right])):
                xs[k].append(values[sample]);ys[k].append(target[sample])
                weights[k].append(np.full(len(sample),cw/len(sample)))
    models={};sizes={}
    for k in xs:
        w=np.concatenate(weights[k]);w=w/w.mean()
        y=np.concatenate(ys[k])
        if not (y.any() and (~y).any()):raise ValueError("Training needs both boundary classes")
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,min_samples_leaf=60,
                learning_rate=.07,l2_regularization=6,early_stopping=False,random_state=51971+args.fold)
        model.fit(np.concatenate(xs[k]),y,sample_weight=w)
        models[k]=model;sizes[k]=len(y)
    file=args.run/f"models-{args.fold}.joblib";joblib.dump(models,file,compress=3)
    rc.write_json(args.run/f"fit-{args.fold}.json",{"train_indices":ids,"oof":meta["payload"],"samples":sizes,
        "weights":rc.hash_file(file),"fold":args.fold,"shift":shift,"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    return models


def odds(model,features):
    return logit(model.predict_proba(features)[:,1].clip(1e-4,1-1e-4))


def candidates(reg,models,choices):
    if not len(reg.variable):
        return {key:reg.mask for key in choices}
    likelihood=(odds(models["appearance"],reg.appearance)-odds(models["geom"],reg.geom)).clip(-4,4)
    cost=(-odds(models["edge"],reg.edge)).clip(0,5)
    cost=cost/cost.mean() if cost.mean()>0 else np.ones_like(cost)
    out={}
    for name,t in choices:
        if t==0:
            new=reg.mask
        else:
            unary=reg.d+t*likelihood
            if name=="unary":
                new=reg.mask.copy()
                new.ravel()[reg.variable]=unary>0
            else:
                pair=.5*t*(cost if name=="learned" else np.ones_like(cost))
                new=cut(reg.mask,reg.variable,unary,reg.left,reg.right,pair)
        out[(name,t)]=new
    return out


def lock(args):
    if (rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT or
            rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER):
        raise ValueError("Formal E reference differs")
    images,truth,_,uids=native.inputs(args.data)
    _,training,valid=native.splits(uids,args.fold)
    groups=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(groups,valid,args.fold)
    ids=library+checked;n=len(library)
    split_record=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    ref=args.stable/f"pred-{args.fold}.npz"
    if (split_record["library"]!=library or split_record["check"]!=checked or
            split_record["prediction"]!=rc.hash_file(ref)):
        raise ValueError("Stable patient order or baseline changed")
    with np.load(ref,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Stable archive patient order differs")
        previous=np.unpackbits(z["adapted"],axis=-1).astype(bool)
    shift=json.loads((args.align/f"fold-{args.fold}-inner-registered/fit.json").read_text())["applied_shift"]
    features=public_cache.verified_parent(args).features
    models=fit_models(args,training,images,truth,uids,features,shift)
    grid=[(name,t) for name in MODES for t in STRENGTHS]
    library_scores={key:[] for key in grid}
    for j,i in enumerate(library):
        im=rater_area.register_image(images[i],shift)
        for rank,reg in enumerate(regions(im,previous[j],features[i,0])):
            out=candidates(reg,models,grid)
            for key,m in out.items():
                library_scores[key].append(float(metrics.dice(truth[i,rank],m)))
    selection={name:STRENGTHS[int(np.argmax([np.mean(library_scores[(name,t)]) for t in STRENGTHS]))] for name in MODES}
    choices=list(selection.items());masks={name:previous.copy() for name in MODES}
    for j,i in enumerate(ids):
        im=rater_area.register_image(images[i],shift)
        for rank,reg in enumerate(regions(im,previous[j],features[i,0])):
            out=candidates(reg,models,choices)
            for (name,_),m in out.items():
                exterior=np.ones(m.size,bool);exterior[reg.variable]=False
                if not np.array_equal(m.ravel()[exterior],reg.mask.ravel()[exterior]):
                    raise ValueError("Spatial cut changed protected pixels")
                masks[name][j,rank]=m
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,reference=np.packbits(previous,axis=-1),
                        **{name:np.packbits(m,axis=-1) for name,m in masks.items()})
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,
        "strength":selection,"grids":{name:[float(np.mean(library_scores[(name,t)])) for t in STRENGTHS] for name in MODES},
        "reference":split_record["prediction"],"features":public_cache.FEATURE_RECEIPT,
        "fit":rc.hash_file(args.run/f"fit-{args.fold}.json"),"prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL)})
    print("GRAPH_LOCK",args.fold,selection,flush=True)


def score(args):
    seals=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in range(4)]
    for f,r in enumerate(seals):
        if r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz"):
            raise ValueError("Four inner graph predictions must be sealed")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Graph score order differs")
            scores={name:float(metrics.dice(truth[r["check"]],np.unpackbits(z[name][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for name in ("reference",*MODES)}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores})
        print("GRAPH_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("lock","score"))
    for key in ("run","parent","public","data","formal","stable","align","crossfit"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    args=parser.parse_args();cv2.setNumThreads(1)
    {"lock":lock,"score":score}[args.phase](args)
