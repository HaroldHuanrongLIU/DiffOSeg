"""Learn expected Dice changes of foreground prefixes from out-of-fit training masks."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import rater_area as area
from reproduction import texture_study as incumbent

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/prefix_utility_protocol.md"
OFFSETS = np.arange(-4., 4.001, 0.25)
ZERO = int(np.flatnonzero(OFFSETS==0)[0])
FORMAL = "799311c1c176f2d8239c619e4c6f35a67fa0ec9300c60c3194cb87d97ade58a3"


def counts(radius, offsets, pixels):
    return np.rint(np.pi*np.maximum(radius+np.asarray(offsets),0)**2).astype(int).clip(0,pixels)


def prefix_dice(truth, order, sizes):
    actual = truth.reshape(-1)
    sizes = np.asarray(sizes,dtype=np.intp)
    hits = np.concatenate(([0],np.cumsum(actual[order],dtype=int)))
    denominator = sizes+int(actual.sum())
    return np.divide(2*hits[sizes],denominator,out=np.ones(len(sizes)),where=denominator>0)


def weights_for_mask(mask):
    return mask.reshape(32,4,32,4).mean((1,3))


def regions(logits):
    middle = (logits>0).sum(0)>=2
    distance = area.sdf(middle)
    core = distance>1.5
    ring = (distance< -1.5) & (distance>=-12)
    return middle, core if core.any() else middle, ring


def pooled_embedding(logits, feature):
    _, core, ring = regions(logits)
    def pool(mask):
        weights = weights_for_mask(mask)
        if weights.sum()==0:
            return feature.mean((1,2))
        return np.einsum("chw,hw->c",feature,weights)/weights.sum()
    difference = pool(core)-pool(ring)
    return difference/max(float(np.linalg.norm(difference)),1e-6)


@dataclass
class RankedCurve:
    order: np.ndarray
    radius: float
    sizes: np.ndarray
    features: np.ndarray


def case_curves(image, logits, pooled):
    hard = logits>0
    middle, core, ring = regions(logits)
    image = image.astype(np.float32)
    fg = float(np.median(image[core])) if core.any() else float(np.quantile(image,0.8))
    bg = float(np.median(image[ring])) if ring.any() else float(np.quantile(image,0.2))
    span = max(fg-bg,10.)
    low = cv2.GaussianBlur(image,(0,0),0.8)
    wide = cv2.GaussianBlur(image,(0,0),3)
    dy,dx = np.gradient(low)
    cy,cx = np.argwhere(middle).mean(0) if middle.any() else (64.5,64.5)
    yy,xx = np.indices(image.shape)
    radius_middle = max(float(np.sqrt(middle.sum()/np.pi)),1)
    radii = np.sqrt(hard.sum((1,2))/np.pi)
    fields = np.stack(((image-bg)/span,(image-low)/span,(image-wide)/span,np.hypot(dx,dy)/span,
                       hard.mean(0),np.hypot(xx-cx,yy-cy)/radius_middle)).reshape(6,-1)
    output = []
    for rank,radius in enumerate(radii):
        flat = logits[rank].ravel()
        order = np.argsort(-flat,kind="stable")
        sizes = counts(radius,OFFSETS,len(flat))
        left = counts(radius,OFFSETS-0.5,len(flat))
        right = counts(radius,OFFSETS+0.5,len(flat))
        cumulative = np.pad(np.cumsum(fields[:,order],axis=1),((0,0),(1,0)))
        band = (cumulative[:,right]-cumulative[:,left])/np.maximum(right-left,1)
        inside = cumulative[:,sizes]/np.maximum(sizes,1)
        ordered = flat[order]
        at = ordered[(sizes-1).clip(0,len(flat)-1)]
        near = counts(radius,[-1.,1.],len(flat))
        scale = max(float(ordered[max(near[0]-1,0)]-ordered[max(near[1]-1,0)]),0.05)
        global_features = np.array([rank,radius,*radii,radii[-1]-radii[0],radius_middle,
                                    fg/255,bg/255,span/255,*pooled],np.float32)
        matrix = np.column_stack((np.broadcast_to(global_features,(len(OFFSETS),len(global_features))),
                                  OFFSETS,sizes/128,(at-at[ZERO])/scale,band.T,inside.T))
        output.append(RankedCurve(order,radius,sizes,matrix.astype(np.float32)))
    return output


def preferred_offsets(curves, utility, point):
    order = np.argsort(abs(OFFSETS),kind="stable")
    x = np.stack([curve.features for curve in curves])
    predicted = utility.predict(x.reshape(-1,x.shape[-1])).reshape(len(curves),-1)
    optimal = OFFSETS[order[np.argmax(predicted[:,order],axis=1)]]
    average = point.predict(x[:,ZERO]).clip(OFFSETS[0],OFFSETS[-1])
    return optimal,average


def apply(curves, offset, strength):
    masks = []
    for curve, change in zip(curves, offset):
        length = int(counts(curve.radius,strength*change,len(curve.order)))
        mask = np.zeros(len(curve.order),bool)
        mask[curve.order[:length]] = True
        masks.append(mask.reshape(128,128))
    return np.stack(masks)


def accepted_inner_guide(args, library, checked):
    ids = library+checked
    if rc.hash_file(args.formal/"assessment/outer_report.json") != FORMAL:
        raise ValueError("Accepted outer reference differs")
    decision = incumbent.read(args.formal/"selection.json")["folds"][args.fold]
    owner = args.ct if decision["variant"]=="ct-adapted" else args.encoder
    name = decision["variant"]
    receipt = incumbent.read(owner/f"inner-{args.fold}-{name}.json")
    outputs = owner/f"pred-{args.fold}-{name}.npz"
    old = args.texture/f"features-{args.fold}-inner.npz"
    old_receipt = incumbent.read(old.with_suffix(".json"))
    if (receipt["library"]!=library or receipt["check"]!=checked or old_receipt["indices"]!=ids
            or rc.hash_file(outputs)!=receipt["prediction"] or rc.hash_file(old)!=old_receipt["payload"]
            or decision["weight"]!=receipt["weight"]):
        raise ValueError("Frozen inner guide or patient order differs")
    with np.load(old,allow_pickle=False) as store:
        base = incumbent.anchor_logits(store["base"].astype(np.float32),np.unpackbits(store["old"],axis=-1).astype(bool))
    with np.load(outputs,allow_pickle=False) as store:
        if store["ids"].tolist()!=ids:
            raise ValueError("Accepted model case order differs")
        original = np.unpackbits(store["masks"],axis=-1).astype(bool)
        w = decision["weight"]
        guide = incumbent.anchor_logits((1-w)*base+w*store["logits"].astype(np.float32),original)
    return original, guide, receipt, name


def run(args):
    images, truth, _, uids = native.inputs(args.data)
    outer, training, valid = native.splits(uids,args.fold)
    groups = type("Groups",(),{"series_uid":uids})()
    library,checked = polar.calibration_split(groups,valid,args.fold)
    ids = library+checked
    original, guide, receipt, name = accepted_inner_guide(args, library, checked)
    train_path = args.crossfit/f"oof-{args.fold}.npz"
    train_receipt = incumbent.read(train_path.with_suffix(".json"))
    if rc.hash_file(train_path)!=train_receipt["payload"]:
        raise ValueError("Training OOF guide differs")
    with np.load(train_path,allow_pickle=False) as store:
        fit_ids = store["ids"].tolist()
        if fit_ids!=sorted(training) or set(fit_ids)&set(outer["test_indices"]):
            raise ValueError("OOF guide includes ineligible cases")
        fit_guide = store["logits"].astype(np.float32)
    embeddings = np.load(args.public/"image_features.npy",mmap_mode="r",allow_pickle=False)
    public = incumbent.read(args.public/"image_features.json")
    if (public["data"]!=rc.DATA_HASH or rc.hash_file(args.public/"image_features.npy")!=public["array"]):
        raise ValueError("Public single-image features differ")
    shift = incumbent.read(args.align/f"fold-{args.fold}-inner-registered/fit.json")["applied_shift"]
    feature = lambda idx: embeddings[idx,0].astype(np.float32)
    pca = PCA(n_components=12,svd_solver="full")
    fit_pca = pca.fit_transform(np.stack([pooled_embedding(z,feature(i)) for i,z in zip(fit_ids,fit_guide)]))
    test_pca = pca.transform(np.stack([pooled_embedding(z,feature(i)) for i,z in zip(ids,guide)]))
    names = [str(uids[i]).split("_")[0] for i in fit_ids]
    _,inverse,n = np.unique(names,return_inverse=True,return_counts=True)
    case_weights = 0.5+0.5*len(fit_ids)/(len(n)*n[inverse])
    fit_x,fit_y,fit_r,fit_w = [],[],[],[]
    for j,(idx,z) in enumerate(zip(fit_ids,fit_guide)):
        curves = case_curves(area.register_image(images[idx],shift),z,fit_pca[j])
        for rank,curve in enumerate(curves):
            outcome = prefix_dice(truth[idx,rank],curve.order,curve.sizes)
            fit_x.append(curve.features)
            fit_y.append(100*(outcome-outcome[ZERO]))
            fit_r.append(np.sqrt(truth[idx,rank].sum()/np.pi)-curve.radius)
            fit_w.append(case_weights[j])
    x = np.stack(fit_x)
    settings = dict(max_iter=150,learning_rate=0.06,max_leaf_nodes=15,min_samples_leaf=80,
                    l2_regularization=8,early_stopping=False,random_state=94171+args.fold)
    utility = HistGradientBoostingRegressor(**settings).fit(x.reshape(-1,x.shape[-1]),np.ravel(fit_y),
                                    sample_weight=np.repeat(fit_w,len(OFFSETS)))
    point = HistGradientBoostingRegressor(**settings).fit(x[:,ZERO],fit_r,sample_weight=fit_w)
    fit_folder = args.run/f"fit-{args.fold}"
    fit_folder.mkdir(exist_ok=False)
    model_path = fit_folder/"model.joblib"
    joblib.dump({"utility":utility,"point":point,"pca":pca},model_path,compress=3)
    cases = [case_curves(area.register_image(images[i],shift),z,features) for i,z,features in zip(ids,guide,test_pca)]
    offsets = [preferred_offsets(curves,utility,point) for curves in cases]
    nlib = len(library)
    constant = []
    for rank in range(4):
        scores = np.mean([prefix_dice(truth[i,rank],c[rank].order,c[rank].sizes)
                          for i,c in zip(library,cases[:nlib])],axis=0)
        ordered = np.argsort(abs(OFFSETS),kind="stable")
        constant.append(float(OFFSETS[ordered[scores[ordered].argmax()]]))
    grids,shrink = {},{}
    for method,column in (("utility",0),("point",1)):
        scores = []
        for strength in incumbent.WEIGHTS:
            predicted = np.stack([apply(c,delta[column],strength) for c,delta in zip(cases[:nlib],offsets[:nlib])])
            scores.append(float(metrics.dice(truth[library],predicted).mean()))
        grids[method] = scores
        shrink[method] = incumbent.WEIGHTS[int(np.argmax(scores))]
    lock = {"fold":args.fold,"library":library,"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),
            "training":fit_ids,"train_payload":train_receipt["payload"],"previous_prediction":receipt["prediction"],
            "model":rc.hash_file(model_path),"constant_offset":constant,"shrink":shrink,"grids":grids,
            "previous_variant":name,"public_embedding":public["array"]}
    rc.write_json(args.run/f"lock-{args.fold}.json",lock)
    selected = {"previous":original[nlib:],"constant":np.stack([apply(c,constant,1) for c in cases[nlib:]])}
    for method,column in (("utility",0),("point",1)):
        selected[method] = np.stack([apply(c,d[column],shrink[method]) for c,d in zip(cases[nlib:],offsets[nlib:])])
    scores = {method:float(metrics.dice(truth[checked],pred).mean()) for method,pred in selected.items()}
    path = args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(path,ids=checked,**{k:np.packbits(v,axis=-1) for k,v in selected.items()})
    rc.write_json(args.run/f"inner-{args.fold}.json",{**lock,"check":checked,"scores":scores,
                  "prediction":rc.hash_file(path),"primary":"utility"})
    print("PREFIX_UTILITY",args.fold,shrink,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("run","data","formal","encoder","ct","texture","crossfit","public","align"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    cv2.setNumThreads(1)
    run(parser.parse_args())
