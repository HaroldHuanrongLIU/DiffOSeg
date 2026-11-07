"""Compare equal-size attention and post-token adaptation of an independent CT representation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import dino_local_probe as dino
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import OFFSETS,mean_saved_logits
from reproduction.semantic_fusion import Joined,ct_and_split

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/dino_attention_protocol.md"
MODES=("post","attention")


def read(path):
    return json.loads(path.read_text())


def pack_tokens(tokens):
    if tokens.shape[1:]!=(257,1024):
        raise ValueError("Expected one class token and a 16 × 16 spatial grid")
    grid=tokens[:,1:].transpose(1,2).reshape(-1,1024,16,16)
    return torch.cat((F.pixel_shuffle(grid,2),tokens[:,0].reshape(-1,1,32,32)),1)


def unpack_tokens(packed):
    patches=F.pixel_unshuffle(packed[:,:256],2).flatten(2).transpose(1,2)
    return torch.cat((packed[:,256:].reshape(-1,1,1024),patches),1)


@torch.inference_mode()
def encode(args):
    images,uids=ct_and_split(args)
    ids=set()
    for f in (2,3):
        _,fit,valid=native.splits(uids,f)
        if read(args.align/f"fold-{f}-inner-registered/fit.json")["applied_shift"]!=[-1,-1]:
            raise ValueError("The public cache requires matching CT registration")
        ids.update(fit+valid)
    ids=sorted(ids)
    model=dino.load_encoder(type("Public",(),{"run":args.dino})())
    dst=args.run/"prefix.npy"
    if dst.exists():raise FileExistsError(dst)
    store=np.lib.format.open_memmap(dst,mode="w+",dtype=np.float16,shape=(len(images),4,257,32,32))
    error=0.
    for j,i in enumerate(ids):
        raw=align.registered_ct(torch.from_numpy(images[i:i+1,None]).cuda(),-1,-1)
        for k in range(4):
            x=prompt.image_input(raw,k,224)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                h=model.embeddings(x)
                for layer in model.encoder.layer[:-1]:h=layer(h)[0]
                packed=pack_tokens(h).half()
                if j in (0,1) and k in (0,1):
                    original=model(pixel_values=x).last_hidden_state.float()
                    restored=unpack_tokens(packed).float()
                    final=model.layernorm(model.encoder.layer[-1](restored)[0]).float()
                    diff=float(torch.linalg.vector_norm(original-final)/torch.linalg.vector_norm(original))
                    if not np.isfinite(diff) or diff>.025:raise ValueError(f"Quantized prefix error {diff}")
                    error=max(error,diff)
            if not torch.isfinite(packed).all():raise ValueError("Nonfinite local features")
            store[i,k]=packed[0].cpu().numpy()
        if j%256==0:print("DINO_PREFIX",j,len(ids),flush=True)
    store.flush();del store
    rc.write_json(dst.with_suffix(".json"),{"ids":ids,"data":rc.DATA_HASH,"public":dino.WEIGHT,"config":dino.CONFIG,
        "wheels":rc.hash_file(args.dino/"wheels.sha256"),"max_prefix_relative_error":error,"payload":rc.hash_file(dst),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("DINO_PREFIX_SEAL",len(ids),error,flush=True)


class LowRank(nn.Module):
    def __init__(self,width=1024,rank=8):
        super().__init__()
        self.down=nn.Linear(width,rank,bias=False)
        self.up=nn.Linear(rank,width,bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self,x):
        return self.up(self.down(x))


class AdaptedLinear(nn.Module):
    def __init__(self,original,update):
        super().__init__()
        self.original=original
        self.update=update

    def forward(self,x):
        return self.original(x)+self.update(x)


class TailRanks(hierarchy.HierarchicalRanks):
    def __init__(self,folder,dino_folder,mode,**kwargs):
        super().__init__(folder,**kwargs)
        from transformers import Dinov2Config
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer
        self.mode=mode
        if rc.hash_file(dino_folder/"dinov2-large.bin")!=dino.WEIGHT or rc.hash_file(dino_folder/"config.json")!=dino.CONFIG:
            raise ValueError("DINO weight/config identity differs")
        cfg=Dinov2Config.from_json_file(dino_folder/"config.json")
        state=torch.load(dino_folder/"dinov2-large.bin",map_location="cpu",weights_only=True)
        with torch.random.fork_rng(devices=[]):
            self.tail=Dinov2Layer(cfg)
            self.norm=nn.LayerNorm(1024,eps=cfg.layer_norm_eps)
            self.tail.load_state_dict(prompt.subset(state,"encoder.layer.23"),strict=True)
            self.norm.load_state_dict(prompt.subset(state,"layernorm"),strict=True)
            self.tail.requires_grad_(False);self.norm.requires_grad_(False)
            updates=[LowRank(),LowRank()]
            if mode=="attention":
                attn=self.tail.attention.attention
                attn.query=AdaptedLinear(attn.query,updates[0])
                attn.value=AdaptedLinear(attn.value,updates[1])
            elif mode=="post":
                self.post=nn.ModuleList(updates)
            else:raise ValueError("Unknown adapter position")
            self.project=nn.Sequential(nn.Conv2d(1024,48,1),nn.SiLU(),nn.Conv2d(48,256,1))
            nn.init.zeros_(self.project[-1].weight);nn.init.zeros_(self.project[-1].bias)

    def updates(self):
        if self.mode=="post":return self.post
        attn=self.tail.attention.attention
        return (attn.query.update,attn.value.update)

    def spatial_features(self,embedding,guide,image):
        primary,prefix=embedding.split((1024,257),dim=1)
        coarse,high=hierarchy.unpack(primary,True)
        hidden=self.norm(self.tail(unpack_tokens(prefix))[0])
        if self.mode=="post":hidden=hidden+self.post[0](hidden)+self.post[1](hidden)
        patches=F.layer_norm(hidden[:,1:].float(),(1024,)).transpose(1,2).reshape(-1,1024,16,16)
        extra=self.project(patches)
        return coarse+F.interpolate(extra,size=coarse.shape[-2:],mode="bilinear",align_corners=False),high


class View(hierarchy.AlreadyRotatedView):
    def __init__(self,args,needed):
        data=cache_owner.verified_parent(args)
        dst=args.run/"prefix.npy";rec=read(dst.with_suffix(".json"))
        if (rec["payload"]!=rc.hash_file(dst) or rec["public"]!=dino.WEIGHT or rec["data"]!=rc.DATA_HASH
                or rec["source"]!=rc.hash_file(__file__) or rec["protocol"]!=rc.hash_file(PROTOCOL)
                or not set(needed).issubset(rec["ids"])):raise ValueError("Unverified prefix cache")
        self.features=Joined(data.features,np.load(dst,mmap_mode="r",allow_pickle=False))
        self.receipt=rc.hash_file(dst.with_suffix(".json"))


def train(args):
    _,uids=ct_and_split(args);_,train_ids,_=native.splits(uids,args.fold)
    label=f"{args.mode}-{args.seed_index}"
    def InitializedTail(folder,**kwargs):
        return TailRanks(folder,args.dino,args.mode,**kwargs)
    view=View(args,train_ids)
    prompt.train(args,decoder_type=InitializedTail,variant=label,study_protocol=PROTOCOL,view=view,
        public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index])
    folder=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{label}")
    if not args.smoke:
        old=args.ensemble/f"fold-{args.fold}-seed-{args.seed_index}"
        def first(d):
            with (d/"train.jsonl").open() as st:return json.loads(next(st))["loss"]
        if not np.isclose(first(folder),first(old),rtol=3e-4,atol=5e-5):
            raise ValueError("Zero-start supervision failed to reproduce E")
    weights=torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
    updates=[(k,float(v.norm())) for k,v in weights.items() if k.endswith("up.weight") and ("update." in k or "post." in k)]
    if len(updates)!=2 or min(v for _,v in updates)<=0:
        raise ValueError("Both trainable low-rank heads must update")
    rc.write_json(folder/"tail.json",{"source":rc.hash_file(__file__),"fit":rc.hash_file(folder/"fit.json"),
                 "prefix":view.receipt,"updates":updates,"public":dino.WEIGHT})


def lock(args):
    if rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT:
        raise ValueError("Unexpected E reference")
    images,uids=ct_and_split(args);_,train_ids,valid=native.splits(uids,args.fold)
    lib,check=polar.calibration_split(type("Groups",(),{"series_uid":uids})(),valid,args.fold);ids=lib+check
    path,fr=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Incorrect guide order")
        guide=z["adapted"]
    old=read(args.stable/f"selection-{args.fold}.json");fp=args.stable/f"pred-{args.fold}.npz"
    if old["prediction"]!=rc.hash_file(fp) or fr["payload"]!=old["field"] or old["library"]!=lib or old["check"]!=check or old["choices"]["adapted"]["weight"]!=.25:
        raise ValueError("E masks or fusion settings differ")
    with np.load(fp,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong E order")
        arrays={"reference":z["adapted"]}
    view=View(args,ids);members={}
    for mode in MODES:
        model=TailRanks(args.public,args.dino,mode).cuda().eval()
        copies=[];members[mode]=[]
        for j,offset in enumerate(OFFSETS):
            label=f"{mode}-{j}";folder=args.run/f"fold-{args.fold}-{label}";fit=read(folder/"fit.json");tail=read(folder/"tail.json")
            if (fit["train_indices"]!=sorted(train_ids) or fit["scope"]!="inner" or fit["steps"]!=5000 or fit["seed"]!=50171+args.fold+offset
                    or fit["variant"]!=label or fit["architecture"]!="InitializedTail" or fit["view_receipt"]!=view.receipt
                    or fit["source"]!=rc.hash_file(prompt.__file__) or fit["protocol"]!=rc.hash_file(PROTOCOL)
                    or fit["public_weights"]!=hierarchy.WEIGHT or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                    or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or tail["fit"]!=rc.hash_file(folder/"fit.json")
                    or tail["source"]!=rc.hash_file(__file__) or tail["prefix"]!=view.receipt):
                raise ValueError("Invalid adaptive DINO member")
            model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
            prediction=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
            if not np.isfinite(prediction).all():raise ValueError("Nonfinite image prediction")
            copies.append(prediction);members[mode].append(fit["checkpoint"])
        mean=mean_saved_logits(copies)
        arrays[mode]=np.packbits(.75*guide+.25*mean>0,axis=-1)
        arrays["standalone_"+mode]=np.packbits(mean>0,axis=-1)
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,
        "reference":old["prediction"],"field":fr["payload"],"members":members,"prefix":view.receipt,
        "prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("DINO_ATTENTION_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("encode","fit","lock"))
    for key in ("run","data","dino","align","parent","ensemble","public","crossfit","stable","formal"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(2,3),default=2);p.add_argument("--mode",choices=MODES,default="attention")
    p.add_argument("--seed-index",type=int,choices=range(3),default=0);p.add_argument("--smoke",action="store_true")
    p.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2)
    {"encode":encode,"fit":train,"lock":lock}[args.phase](args)
