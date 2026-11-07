"""Estimate Dice of confidence-ordered area prefixes from native-image feature shells."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import gated_hierarchy as parent
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import rater_area
from reproduction.prefix_utility import accepted_inner_guide

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/shell_occupancy_protocol.md"
OFFSETS=np.arange(-4.,4.01,.25,dtype=np.float32)
ZERO=len(OFFSETS)//2
SHRINK=(0.,.25,.5,.75,1.)
VISUAL=32+64+256
STEPS=1800


def segments(values, counts):
    """Average sorted columns in each interval; duplicate endpoints use their nearest available pixel."""
    starts=np.r_[0,counts[:-1]]
    lengths=counts-starts
    cumulative=np.pad(np.cumsum(values,axis=1,dtype=np.float32),((0,0),(1,0)))
    pooled=(cumulative[:,counts]-cumulative[:,starts])/np.maximum(lengths,1)
    empty=lengths==0
    pooled[:,empty]=values[:,np.clip(counts[empty]-1,0,values.shape[1]-1)]
    return pooled.T.astype(np.float32)


def unpack_level(array, stride):
    channels, height, width=array.shape
    return array.reshape(channels//stride**2,stride,stride,height,width).transpose(0,3,1,4,2).reshape(
        channels//stride**2,height*stride,width*stride)


def sample_features(image,guide,packed,truth=None):
    high=unpack_level(packed[512:],4)
    middle=unpack_level(packed[256:512],2)
    coarse=packed[:256]
    image=image.astype(np.float32)/255
    gy,gx=np.gradient(image)
    gradient=np.hypot(gy,gx)
    radii=np.sqrt((guide>0).sum((1,2))/np.pi)
    features=[];all_counts=[];all_hits=[];orders=[]
    for rank,z in enumerate(guide):
        order=np.argsort(-z.ravel(),kind="stable")
        counts=np.rint(np.pi*np.maximum(radii[rank]+OFFSETS,0)**2).astype(int).clip(0,z.size)
        prefix=order[:max(1,counts[-1])]
        yy,xx=np.unravel_index(prefix,z.shape)
        features_by_pixel=np.vstack((high[:,yy,xx],middle[:,yy//2,xx//2],coarse[:,yy//4,xx//4],
                                     image[yy,xx],gradient[yy,xx],z[yy,xx].clip(-12,12),
                                     (guide[:,yy,xx]>0).mean(0)))
        pooled=segments(features_by_pixel,counts)
        constants=np.broadcast_to(np.r_[rank/3,radii/16.0],(len(OFFSETS),5))
        features.append(np.column_stack((pooled,constants,OFFSETS/4,counts/1024)).astype(np.float32))
        all_counts.append(counts)
        orders.append(order.astype(np.uint16))
        if truth is not None:
            occupied=np.r_[0,np.cumsum(truth[rank].ravel()[prefix],dtype=np.int64)]
            all_hits.append(occupied[counts])
    return np.stack(features),np.stack(all_counts),np.stack(orders),np.asarray(all_hits)


class OccupancyNet(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.local=nn.Sequential(nn.Conv1d(channels,48,1),nn.GroupNorm(6,48),nn.SiLU(),
                                 nn.Conv1d(48,48,3,padding=1),nn.GroupNorm(6,48),nn.SiLU())
        self.context=nn.Linear(48,48)
        self.occupancy=nn.Sequential(nn.Dropout(.15),nn.Conv1d(48,1,1))
        self.radius=nn.Sequential(nn.Dropout(.15),nn.Linear(48,1))

    def forward(self,x):
        batch,ranks,bands,channels=x.shape
        features=self.local(x.reshape(batch*ranks,bands,channels).transpose(1,2))
        local=features.mean(2).reshape(batch,ranks,48)
        global_context=self.context(local.mean(1))
        features=features+global_context.repeat_interleave(ranks,dim=0)[:,:,None]
        return self.occupancy(features).reshape(batch,ranks,bands),self.radius(local+global_context[:,None]).squeeze(-1)


def expected_curve(occupancy, radius_residual, counts):
    sizes=torch.diff(counts,dim=-1,prepend=torch.zeros_like(counts[...,:1]))
    hits=torch.cumsum(occupancy.sigmoid()*sizes,dim=-1)
    radius=(counts[...,ZERO]/math.pi).sqrt()+radius_residual
    area=math.pi*radius.clamp_min(0).square()
    return 2*hits/(counts+area[:,:,None]).clamp_min(1),sizes


def fit_model(args,x,counts,hits,actual,case_weights,visual,name):
    seed=117341+args.fold
    torch.manual_seed(seed)
    net=OccupancyNet(x.shape[-1])
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.005)
    inputs=x.clone()
    if not visual:inputs[...,:VISUAL]=0
    denom=counts+actual[...,None]
    true_curve=torch.where(denom>0,2*hits/denom.clamp_min(1),1.)
    radii=(actual/math.pi).sqrt()-(counts[...,ZERO]/math.pi).sqrt()
    incremental=torch.diff(hits,dim=-1,prepend=torch.zeros_like(hits[...,:1]))
    for step in range(1,STEPS+1):
        ix=torch.multinomial(case_weights,16,replacement=True)
        occupancy,residual=net(inputs[ix])
        curve,lengths=expected_curve(occupancy,residual,counts[ix])
        target=incremental[ix]/lengths.clamp_min(1)
        weights=lengths.sqrt()
        occ=F.binary_cross_entropy_with_logits(occupancy,target,reduction="none")
        curve_delta=curve-curve[...,ZERO:ZERO+1]
        actual_delta=true_curve[ix]-true_curve[ix,...,ZERO:ZERO+1]
        loss=(weights*occ).sum()/weights.sum().clamp_min(1)
        loss=loss+.15*F.smooth_l1_loss(residual,radii[ix],beta=.5)
        loss=loss+.5*F.smooth_l1_loss(10*curve_delta,10*actual_delta,beta=.2)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),3,error_if_nonfinite=True)
        opt.param_groups[0]["lr"]=.001*min(1,step/60)*(.1+.9*(1+math.cos(math.pi*step/STEPS))/2)
        opt.step()
    path=args.run/f"model-{args.fold}-{name}.pt"
    torch.save(net.state_dict(),path)
    return net.eval(),rc.hash_file(path)


def apply_sizes(orders,counts,change,strength):
    radius=np.sqrt(counts[:,:,ZERO]/np.pi)
    sizes=np.rint(np.pi*np.maximum(radius+strength*change,0)**2).astype(int).clip(0,orders.shape[-1])
    masks=np.zeros(orders.shape,dtype=bool)
    for i in range(len(masks)):
        for rank in range(4):
            masks[i,rank,orders[i,rank,:sizes[i,rank]]]=True
    return masks.reshape(-1,4,128,128)


def run(args):
    images,truth,_,uids=native.inputs(args.data)
    outer,fit,valid=native.splits(uids,args.fold)
    groups=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(groups,valid,args.fold)
    ids=library+checked
    original,guide,previous,_=accepted_inner_guide(args,library,checked)
    view=parent.verified_parent(args)
    path=args.crossfit/f"oof-{args.fold}.npz";receipt=json.loads(path.with_suffix(".json").read_text())
    if rc.hash_file(path)!=receipt["payload"]:raise ValueError("OOF payload changed")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=sorted(fit) or set(fit)&set(outer["test_indices"]):raise ValueError("Training scope changed")
        fit_guide=z["logits"].astype(np.float32)
    fit=sorted(fit)
    shift=json.loads((args.align/f"fold-{args.fold}-inner-registered/fit.json").read_text())["applied_shift"]
    xs=[];counts=[];hits=[]
    for i,g in zip(fit,fit_guide):
        item=sample_features(rater_area.register_image(images[i],shift),g,view.features[i,0],truth[i])
        xs.append(item[0]);counts.append(item[1]);hits.append(item[3])
    x=np.stack(xs);mu=x.mean((0,1,2),dtype=np.float64).astype(np.float32)
    sd=x.std((0,1,2),dtype=np.float64).astype(np.float32).clip(.01,None)
    train_x=torch.from_numpy(np.clip((x-mu)/sd,-8,8));del x,xs,fit_guide
    count_tensor=torch.tensor(np.stack(counts),dtype=torch.float32)
    hit_tensor=torch.tensor(np.stack(hits),dtype=torch.float32)
    actual=torch.tensor(truth[fit].sum((2,3)),dtype=torch.float32)
    _,inverse,n=np.unique([str(uids[i]).split("_")[0] for i in fit],return_inverse=True,return_counts=True)
    weights=torch.tensor(.5/len(fit)+.5/(len(n)*n[inverse]),dtype=torch.float32)
    eval_items=[sample_features(rater_area.register_image(images[i],shift),g,view.features[i,0]) for i,g in zip(ids,guide)]
    fields=torch.from_numpy(np.clip((np.stack([a[0] for a in eval_items])-mu)/sd,-8,8))
    eval_counts=np.stack([a[1] for a in eval_items]);orders=np.stack([a[2] for a in eval_items])
    del eval_items
    std_path=args.run/f"normalization-{args.fold}.npz";np.savez(std_path,mean=mu,std=sd)
    changes={};models={}
    for visual,name in ((True,"full"),(False,"structural")):
        net,models[name]=fit_model(args,train_x,count_tensor,hit_tensor,actual,weights,visual,name)
        feat=fields.clone()
        if not visual:feat[...,:VISUAL]=0
        with torch.no_grad():
            occ,radius=net(feat)
            curve,_=expected_curve(occ,radius,torch.as_tensor(eval_counts,dtype=torch.float32))
        tie=np.argsort(np.abs(OFFSETS),kind="stable")
        changes[name]=OFFSETS[tie[np.argmax(curve.numpy()[...,tie],axis=-1)]]
        if visual:changes["point"]=radius.numpy().clip(-4,4)
    nlib=len(library);choices={};masks={"reference":np.packbits(original,axis=-1)}
    for name,change in changes.items():
        grid=[]
        for strength in SHRINK:
            m=apply_sizes(orders[:nlib],eval_counts[:nlib],change[:nlib],strength)
            if strength==0 and not np.array_equal(m,original[:nlib]):raise ValueError("Zero strength changed the reference")
            grid.append(float(metrics.dice(truth[library],m).mean()))
        strength=SHRINK[int(np.argmax(grid))]
        final=apply_sizes(orders,eval_counts,change,strength)
        masks[name]=np.packbits(final,axis=-1)
        choices[name]={"strength":strength,"grid":grid}
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"fit":fit,"library":library,"check":checked,
                  "previous":previous["prediction"],"oof":receipt["payload"],"normalization":rc.hash_file(std_path),
                  "models":models,"choices":choices,"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),
                  "protocol":rc.hash_file(PROTOCOL)})
    print("SHELL_LOCK",args.fold,{k:v["strength"] for k,v in choices.items()},flush=True)


def score(args):
    locks=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in range(4)]
    for f,r in enumerate(locks):
        if r["fold"]!=f or rc.hash_file(args.run/f"pred-{f}.npz")!=r["prediction"]:
            raise ValueError("Incomplete patient-fold prediction locks")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(locks):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Case order changed")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference","full","structural","point")}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores})
        print("SHELL_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","score"))
    for name in ("run","parent","data","public","crossfit","formal","encoder","ct","texture","align"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    {"fit":run,"score":score}[args.phase](args)
