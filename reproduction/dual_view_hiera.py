"""Train a shared Hiera decoder on global and image-guided local views of one slice."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import canonical_guide as canonical
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import roi_prompt as roi
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits
from reproduction.trajectory_sam import select_weight

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/dual_view_protocol.md"
REFERENCE="5f30f6c9500bd196c4f4f9cf8dbe8b0153f27bd8135d88254d1b079d6c9f189f"
SEED_OFFSET=44000


def fuse_registered(global_rot,local_rot,back,window,rotation):
    local_native=torch.rot90(local_rot,-rotation,(-2,-1))
    restored=F.grid_sample(local_native.float(),back,align_corners=True,padding_mode="border")
    global_native=torch.rot90(global_rot,-rotation,(-2,-1))
    mixed=global_native+.5*window*(restored-global_native)
    return torch.rot90(mixed,rotation,(-2,-1))


@torch.inference_mode()
def local_cache(args,images,ids,geometry,phase):
    path=args.run/f"local-{args.fold}-{phase}.npy"
    if path.exists():raise FileExistsError(path)
    state=hierarchy.open_public(args.public)
    encoder=hierarchy.image_encoder(state)
    _,head=hierarchy.decoder_heads(state);head=head.cuda().eval()
    no_memory=state["no_mem_embed"].reshape(1,256,1,1).cuda()
    tmp=path.with_suffix(".partial.npy")
    if tmp.exists():raise FileExistsError(tmp)
    cache=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float16,shape=(len(ids),4,1024,32,32))
    for begin in range(0,len(ids),6):
        subset=ids[begin:begin+6]
        x=align.registered_ct(torch.from_numpy(images[subset,None]).cuda(),-1,-1).float()
        ix=torch.arange(begin,begin+len(subset),device="cuda")
        crop=geometry.crop(x,ix)
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                high,middle,deep=encoder(prompt.image_input(crop,k,512))["backbone_fpn"]
                value=hierarchy.pack(head.conv_s0(high),head.conv_s1(middle),deep+no_memory)
            if not torch.isfinite(value).all():raise ValueError("Nonfinite local image features")
            cache[begin:begin+len(subset),k]=value.float().cpu().numpy()
    cache.flush();del cache
    tmp.replace(path)


class DualView:
    def __init__(self,args,images,ids,guide,phase,guide_digest):
        self.features=cache_owner.verified_parent(args).features
        self.local=args.variant=="dual"
        centers,sides=roi.geometry(guide,self.local)
        self.geometry=roi.View(centers,sides,self.local,"cuda")
        self.local_features=None
        meta={"fold":args.fold,"phase":phase,"variant":args.variant,"ids":ids,"centers":centers.tolist(),
              "sides":sides.tolist(),"guide":guide_digest,"weights":hierarchy.WEIGHT,"data":rc.DATA_HASH,
              "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),
              "global":cache_owner.FEATURE_RECEIPT}
        receipt=args.run/f"view-{args.fold}-{phase}-{args.variant}.json"
        if self.local:
            path=args.run/f"local-{args.fold}-{phase}.npy"
            if path.exists()!=receipt.exists():
                raise ValueError("Local feature cache and geometry receipt must be completed together")
            if not path.exists():local_cache(args,images,ids,self.geometry,phase)
            meta["local_payload"]=rc.hash_file(path)
        if receipt.exists():
            if json.loads(receipt.read_text())!=meta:raise ValueError("The local field or source identity changed")
        else:
            rc.write_json(receipt,meta)
        self.receipt=rc.hash_file(receipt)
        if self.local:
            self.local_features=torch.from_numpy(np.load(path,mmap_mode="r",allow_pickle=False).copy()).cuda()

    def decode(self,model,embedding,guide,image,index,rotation):
        global_fields=model(embedding,guide,image)
        if not self.local:
            second=model(embedding,guide,image)
            return tuple((x+y)*.5 for x,y in zip(global_fields,second))
        loc=self.geometry
        native_image=torch.rot90(image,-rotation,(-2,-1))
        native_guide=torch.rot90(guide,-rotation,(-2,-1))
        local_image=torch.rot90(loc.crop(native_image,index),rotation,(-2,-1))
        local_guide=torch.rot90(loc.crop(native_guide,index),rotation,(-2,-1))
        second=model(self.local_features[index,rotation].float(),local_guide,local_image)
        return tuple(fuse_registered(x,y,loc.back[index],loc.window[index],rotation) for x,y in zip(global_fields,second))


def train(args):
    images,_,_,uids=native.inputs(args.data)
    ids,guide,bank=roi.training_bank(args,uids)
    view=DualView(args,images,ids,guide,"fit",bank["payload"])
    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=args.variant,study_protocol=PROTOCOL,
                 view=view,public_weights=hierarchy.WEIGHT,seed_offset=SEED_OFFSET)


@torch.inference_mode()
def predict(model,view,ids,images,guide):
    result=[]
    for begin in range(0,len(ids),12):
        cases=ids[begin:begin+12]
        features=torch.from_numpy(view.features[cases].copy()).cuda()
        x=align.registered_ct(torch.from_numpy(images[cases,None]).cuda(),-1,-1).float()/255
        g=torch.from_numpy(guide[begin:begin+len(cases)]).cuda()
        indices=torch.arange(begin,begin+len(cases),device="cuda")
        total=0
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                z,_=view.decode(model,features[:,k].float(),torch.rot90(g,k,(-2,-1)),torch.rot90(x,k,(-2,-1)),indices,k)
            total=total+torch.rot90(z,-k,(-2,-1))
        result.append((total/4).cpu().numpy())
    return np.concatenate(result)


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,training,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(group,valid,args.fold)
    ids=library+checked;n=len(library)
    if rc.hash_file(args.stable/"summary.json")!=REFERENCE:raise ValueError("Stable inner method differs")
    field,identity=verified_field(args.stable,"inner",args.fold,PROTOCOL.with_name("stable_refinement_protocol.md"))
    with np.load(field,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong stable field order")
        guide=z["adapted"]
    previous=json.loads((args.stable/f"inner-{args.fold}.json").read_text())
    prior=args.stable/f"pred-{args.fold}.npz"
    if previous["prediction"]!=rc.hash_file(prior):raise ValueError("Stable masks differ")
    with np.load(prior,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong prior case order")
        arrays={"reference":z["adapted"]}
    seeds=canonical.model_sources(args,training,args.fold)
    features=cache_owner.verified_parent(args).features
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    other=[]
    for folder,_ in seeds[1:]:
        model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
        other.append(prompt.predict(model,features,ids,images,guide).astype(np.float16))
    choices={}
    for variant in ("full","dual"):
        args.variant=variant
        folder=args.run/f"fold-{args.fold}-{variant}"
        fit=json.loads((folder/"fit.json").read_text())
        if (fit["scope"]!="inner" or fit["train_indices"]!=sorted(training) or fit["steps"]!=5000
                or fit["seed"]!=50171+args.fold+SEED_OFFSET or fit["variant"]!=variant or fit["public_weights"]!=hierarchy.WEIGHT
                or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz") or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")
                or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["source"]!=rc.hash_file(prompt.__file__)
                or fit["view_receipt"]!=rc.hash_file(args.run/f"view-{args.fold}-fit-{variant}.json")):
            raise ValueError("A two-view member identity differs")
        view=DualView(args,images,ids,guide,"eval",identity["payload"])
        model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
        single=predict(model,view,ids,images,guide)
        mean=mean_saved_logits([single.astype(np.float16),*other])
        if not np.isfinite(mean).all():raise ValueError("Nonfinite global/local ensemble")
        weight,grid=select_weight(truth[library],guide[:n],mean[:n])
        arrays[variant]=np.packbits((1-weight)*guide+weight*mean>0,axis=-1)
        arrays["single_"+variant]=np.packbits(single>0,axis=-1)
        choices[variant]={"weight":weight,"grid":grid,"eval_view":view.receipt,"checkpoint":fit["checkpoint"]}
        del view
    path=args.run/f"pred-{args.fold}.npz";np.savez_compressed(path,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,
                  "reference":previous["prediction"],"choices":choices,"prediction":rc.hash_file(path),"source":rc.hash_file(__file__),
                  "protocol":rc.hash_file(PROTOCOL)})
    print("DUAL_LOCK",args.fold,{k:v["weight"] for k,v in choices.items()},flush=True)


def score(args):
    locks=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    if any(r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz") for f,r in enumerate(locks)):
        raise ValueError("Incomplete paired view seals")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(locks):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Wrong two-view case order")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference","full","dual","single_full","single_dual")}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores})
        print("DUAL_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("fit","lock","score"))
    for name in ("run","parent","ensemble","data","public","stable","crossfit","align"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0)
    parser.add_argument("--variant",choices=("full","dual"),default="full")
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    {"fit":train,"lock":lock,"score":score}[args.phase](args)
