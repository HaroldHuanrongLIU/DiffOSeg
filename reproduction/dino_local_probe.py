"""Probe global and local tokens from the same frozen natural-image encoder."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import HistGradientBoostingRegressor

from reproduction import component_utility as utility
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/dino_local_protocol.md"
WEIGHT="8320e4778a7f8850d10f30d97e9138438e1851af1576fea789c43746140cc655"
CONFIG="12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0"
MODES=("global","local")


def bank(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,original=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz"
    rec=json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=utility.QUALITY or
            rec["payload"]!=rc.hash_file(path) or rec["base"]!=original["payload"]):
        raise ValueError("Nested teacher identity mismatch")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Incorrect OOF order")
        s=z["logits"].astype(np.float32)
    shifts={}
    for j,group in enumerate(groups):
        shift=json.loads((args.original/f"fold-{args.fold}-sub-{j}/fit.json").read_text())["shift"]
        shifts.update({i:shift for i in group})
    registered=np.stack([register_image(images[i],shifts[i]) for i in ids])
    return registered,truth[ids],uids,ids,u,s,groups,original["payload"],rec["payload"]


def centers_from_guides(u,s):
    foreground=((u+s)>0).sum(1)>=2
    ys,xs=np.indices(foreground.shape[-2:])
    total=foreground.sum((1,2))
    cx=np.divide((foreground*xs).sum((1,2)),total,out=np.full(len(total),63.5),where=total>0)
    cy=np.divide((foreground*ys).sum((1,2)),total,out=np.full(len(total),63.5),where=total>0)
    return np.column_stack((cx,cy)).astype(np.float32)


def view_image(raw,center,side):
    batch=len(raw)
    values=raw.float()
    limits=torch.quantile(values[:,:,::2,::2].flatten(1),values.new_tensor([.015,.985]),dim=1)
    scaled=((values-limits[0,:,None,None,None])/(limits[1]-limits[0]).clamp_min(35)[:,None,None,None]).clamp(0,1)
    axis=(torch.arange(448,device=raw.device)+.5)/448-.5
    y,x=torch.meshgrid(axis,axis,indexing="ij")
    coords=torch.stack((x,y),-1)[None]*side+center[:,None,None,:]
    grid=(coords+.5)*(2/128)-1
    crop=F.grid_sample(scaled,grid,mode="bilinear",align_corners=False,padding_mode="border")
    mean=crop.new_tensor([.485,.456,.406])[None,:,None,None]
    std=crop.new_tensor([.229,.224,.225])[None,:,None,None]
    return ((crop.expand(batch,3,448,448)-mean)/std).contiguous()


def load_encoder(args):
    from transformers import Dinov2Config,Dinov2Model
    import transformers
    if transformers.__version__!="4.40.2":
        raise ValueError("Unexpected transformer implementation")
    if rc.hash_file(args.run/"dinov2-large.bin")!=WEIGHT or rc.hash_file(args.run/"config.json")!=CONFIG:
        raise ValueError("The public encoder artifact changed")
    cfg=Dinov2Config.from_json_file(args.run/"config.json")
    if (cfg.hidden_size,cfg.num_hidden_layers,cfg.num_attention_heads,cfg.patch_size)!=(1024,24,16,14):
        raise ValueError("Incorrect encoder geometry")
    model=Dinov2Model(cfg)
    weights=torch.load(args.run/"dinov2-large.bin",map_location="cpu",weights_only=True)
    model.load_state_dict(weights,strict=True)
    return model.cuda().eval().requires_grad_(False)


@torch.inference_mode()
def encode(args):
    images,_,_,ids,u,s,_,base,nested_hash=bank(args)
    centers=centers_from_guides(u,s)
    del u,s
    model=load_encoder(args)
    rng=np.random.default_rng(39117)
    projection=torch.from_numpy((rng.standard_normal((1024,64))/8).astype(np.float32)).cuda()
    dst=args.run/f"maps-{args.fold}.npy"
    if dst.exists():raise FileExistsError(dst)
    maps=np.lib.format.open_memmap(dst,mode="w+",shape=(len(ids),2,64,32,32),dtype=np.float16)
    for j in range(len(ids)):
        # A fixed single-case kernel also avoids batch-size-dependent bfloat16 rounding.
        raw=torch.from_numpy(images[j:j+1,None]).cuda()
        center=torch.from_numpy(centers[j:j+1]).cuda()
        for k,side in enumerate((128,64)):
            view=view_image(raw,center,side)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                tokens=model(pixel_values=view).last_hidden_state[:,1:]
            output=(tokens.float()@projection).transpose(1,2).reshape(len(raw),64,32,32)
            if not torch.isfinite(output).all():raise ValueError("Nonfinite projected feature")
            maps[j:j+len(raw),k]=output.cpu().numpy()
        if j==0 or j%256==0:print("DINO_VIEWS",args.fold,j,len(ids),flush=True)
    maps.flush();del maps
    rc.write_json(dst.with_suffix(".json"),{"fold":args.fold,"ids":ids,"base":base,"nested":nested_hash,"weight":WEIGHT,
        "config":CONFIG,"payload":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),
        "runtime_wheels":rc.hash_file(args.run/"wheels.sha256")})


def tokens_at(tokens,x,y,center,side):
    gx=(x-center[0])*32/side+15.5
    gy=(y-center[1])*32/side+15.5
    gx=np.clip(gx,0,31);gy=np.clip(gy,0,31)
    x0=gx.astype(int);y0=gy.astype(int)
    ax=(gx-x0)[:,None];ay=(gy-y0)[:,None]
    x1=np.minimum(x0+1,31);y1=np.minimum(y0+1,31)
    return ((1-ax)*(1-ay)*tokens[:,y0,x0].T+ax*(1-ay)*tokens[:,y0,x1].T+
            (1-ax)*ay*tokens[:,y1,x0].T+ax*ay*tokens[:,y1,x1].T).astype(np.float32)


@dataclass
class Case:
    reference:np.ndarray
    index:np.ndarray
    margin:np.ndarray
    x:dict
    residual:np.ndarray
    weights:np.ndarray


def example(image,u,s,maps,center,truth):
    reference=(u+s)>0
    ct=utility.normalized_ct(image,reference)
    xs={mode:[] for mode in MODES};coords=[];margins=[];ys=[];weights=[]
    for r,m in enumerate(reference):
        distance=sdf(m)
        y,x=np.nonzero(abs(distance)<=2.5)
        if not len(x):continue
        z1=u[r,y,x];z2=s[r,y,x];margin=.5*np.tanh((z1+z2)/4)
        common=np.column_stack((np.full(len(x),r/3),np.full(len(x),np.sqrt(m.sum()/np.pi)/16),margin,
            distance[y,x],z1.clip(-12,12)/8,z2.clip(-12,12)/8,(z1>0)!=(z2>0),reference[:,y,x].mean(0),ct[:,y,x].T))
        global_tokens=tokens_at(maps[0],x,y,center,128)
        local=tokens_at(maps[1],x,y,center,64)
        outside=(abs(x-center[0])>32)|(abs(y-center[1])>32)
        local[outside]=global_tokens[outside]
        for mode,features in (("global",global_tokens),("local",local)):
            xs[mode].append(np.column_stack((common,features)).astype(np.float32))
        coords.append(r*m.size+y*m.shape[1]+x);margins.append(margin)
        ys.append(truth[r,y,x]-(.5+margin));weights.append(np.full(len(x),1/(m.sum()+1)))
    empty=lambda values:np.concatenate(values) if values else np.empty(0)
    return Case(np.packbits(reference,axis=-1),empty(coords).astype(int),empty(margins),
        {k:np.concatenate(v) if v else np.empty((0,76)) for k,v in xs.items()},empty(ys),empty(weights))


def lock(args):
    imgs,ys,uids,ids,u,s,groups,base,nested_hash=bank(args)
    path=args.run/f"maps-{args.fold}.npy";rec=json.loads(path.with_suffix(".json").read_text())
    if (rec["ids"]!=ids or rec["base"]!=base or rec["nested"]!=nested_hash or rec["source"]!=rc.hash_file(__file__)
            or rec["protocol"]!=rc.hash_file(PROTOCOL) or rec["payload"]!=rc.hash_file(path)):
        raise ValueError("Wrong DINO map identity")
    maps=np.load(path,mmap_mode="r",allow_pickle=False);centers=centers_from_guides(u,s)
    cases={i:example(imgs[j],u[j],s[j],maps[j],centers[j],ys[j]) for j,i in enumerate(ids)}
    ref=np.stack([cases[i].reference for i in ids])
    outputs={k:ref.copy() for k in MODES}
    row={i:j for j,i in enumerate(ids)};membership=[]
    for j,group in enumerate(groups):
        for hold in harmonic.case_partition(group,uids,57121+37*j+args.fold):
            fit=sorted(set(group)-set(hold))
            if nested.patients(fit,uids)&nested.patients(hold,uids):raise ValueError("Meta-validation overlap")
            _,inverse,n=np.unique([str(uids[i]).split("_")[0] for i in fit],return_inverse=True,return_counts=True)
            ww=.5+.5*len(fit)/(len(n)*n[inverse])
            w=np.concatenate([cases[i].weights*sample for i,sample in zip(fit,ww,strict=True)])
            target=np.concatenate([cases[i].residual for i in fit])
            for mode in MODES:
                x=np.concatenate([cases[i].x[mode] for i in fit if len(cases[i].index)])
                model=HistGradientBoostingRegressor(max_iter=70,max_leaf_nodes=9,min_samples_leaf=80,learning_rate=.07,
                    l2_regularization=10.,early_stopping=False,random_state=38931+j)
                model.fit(x,target,sample_weight=w/w.mean())
                hx=[cases[i].x[mode] for i in hold if len(cases[i].index)]
                pred=model.predict(np.concatenate(hx)) if hx else np.empty(0)
                offset=0
                for i in hold:
                    case=cases[i];size=len(case.index)
                    m=np.unpackbits(case.reference,axis=-1)
                    m.ravel()[case.index]=case.margin+.75*pred[offset:offset+size]>0
                    outputs[mode][row[i]]=np.packbits(m,axis=-1);offset+=size
                if offset!=len(pred):raise ValueError("Incomplete pixel assignment")
            membership.append({"upstream":j,"fit":fit,"held":hold})
    if sorted(i for g in membership for i in g["held"])!=ids:raise ValueError("Incomplete case coverage")
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,reference=ref,**outputs)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":membership,"base":base,"nested":nested_hash,
        "features":rec["payload"],"prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("DINO_LOCAL_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("encode","lock"))
    for name in ("data","run","original","nested"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(3)
    {"encode":encode,"lock":lock}[args.phase](args)
