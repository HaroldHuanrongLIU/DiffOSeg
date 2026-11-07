"""Condition four SAM2 decoder calls on automatic rank-specific disagreement points."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import OFFSETS,mean_saved_logits
from reproduction.semantic_fusion import ct_and_split

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/rank_clicks_protocol.md"
SIGNAL="92cf8383fba531c6f11c8b05f2aa582e2656885e9d04ba20a5de6781dae50312"
MODES=("common","personal")


@torch.no_grad()
def disagreement_points(guide,personal):
    batch,_,h,w=guide.shape
    votes=(guide>0).sum(1)
    middle=votes>=2
    yy,xx=torch.meshgrid(torch.arange(h,device=guide.device),torch.arange(w,device=guide.device),indexing="ij")
    count=middle.sum((-2,-1))
    cx=torch.where(count>0,(middle*xx).sum((-2,-1)).double()/count.clamp_min(1),(w-1)/2)
    cy=torch.where(count>0,(middle*yy).sum((-2,-1)).double()/count.clamp_min(1),(h-1)/2)
    angle=torch.atan2(yy-cy[:,None,None],xx-cx[:,None,None]).remainder(2*math.pi)
    sector=(angle*(8/(2*math.pi))).long().remainder(8)
    eligible=((votes>0)&(votes<4))[:,None]&(sector[:,None]==torch.arange(8,device=guide.device)[None,:,None,None])
    probabilities=guide.float().sigmoid()
    gap=probabilities.amax(1)-probabilities.amin(1)
    values,index=gap[:,None].masked_fill(~eligible,-1.).flatten(2).max(2)
    xy=torch.stack((index%w,index//w),dim=-1).float()
    signs=(guide if personal else guide.mean(1,keepdim=True).expand_as(guide)).flatten(2).gather(2,index[:,None].expand(-1,4,-1))>0
    labels=torch.where(values[:,None]>=0,signs.long(),-1)
    # Pixel centers are expressed in the coordinate convention used by the public prompt encoder.
    xy=((xy+.5)*512/w-.5)[:,None].expand(batch,4,8,2)
    return xy.reshape(batch*4,8,2),labels.reshape(batch*4,8)


def select_rank(raw):
    batch=len(raw)//4
    all_ranks=raw.reshape(batch,4,4,*raw.shape[-2:])
    rank=torch.arange(4,device=raw.device)
    return all_ranks[:,rank,rank]


class CommonPoints(hierarchy.HierarchicalRanks):
    personal=False

    def forward(self,embedding,guide,image):
        coarse,high=self.spatial_features(embedding,guide,image)
        box=prompt.boxes_from_logits(guide,self.training,512).repeat_interleave(4,dim=0)
        points=disagreement_points(guide,self.personal)
        dense=guide.clamp(-10,10)
        if self.training:
            keep=torch.rand(len(guide),1,1,1,device=guide.device)>.23
            dense=dense*keep
        sparse,mask_embed=self.prompt(points=points,boxes=box,masks=dense.reshape(-1,1,*dense.shape[-2:]))
        raw,_,_,_=self.decoder.predict_masks(coarse.repeat_interleave(4,0),self.prompt.get_dense_pe(),sparse,
                       mask_embed,repeat_image=False,high_res_features=[v.repeat_interleave(4,0) for v in high])
        raw=select_rank(raw).float()
        low=F.avg_pool2d(image,7,1,3,count_include_pad=False)
        detail=self.detail(torch.cat((2*image-1,4*(image-low),guide.clamp(-12,12)/8,raw.clamp(-12,12)/8),1))
        mix=self.mix.sigmoid()[None,:,None,None]
        return (1-mix)*guide+mix*raw+detail.float(),raw


class PersonalPoints(CommonPoints):
    personal=True


def cls(mode):
    return PersonalPoints if mode=="personal" else CommonPoints


def train(args):
    if rc.hash_file(args.signal/"summary.json")!=SIGNAL:
        raise ValueError("Fitting-only automatic-prompt diagnosis changed")
    label=f"{args.mode}-{args.seed_index}"
    folder=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{label}")
    def objective(model,embedding,guide,image,labels,ids,k,step):
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            logits,raw=model(embedding,guide,image)
        loss=prompt.segmentation_loss(logits,raw,labels)
        if step==1:
            xy,point_labels=disagreement_points(guide,model.personal)
            first={"indices":ids.tolist(),"rotation":k,"coordinates":hashlib.sha256(xy.cpu().numpy().tobytes()).hexdigest(),
                   "labels":hashlib.sha256(point_labels.cpu().numpy().tobytes()).hexdigest(),
                   "positive":int((point_labels==1).sum()),"negative":int((point_labels==0).sum()),"valid":int((point_labels>=0).sum())}
            if args.smoke:
                grads=torch.autograd.grad(loss,tuple(m.weight for m in model.prompt.point_embeddings[:2]),retain_graph=True)
                if any(not torch.isfinite(g).all() or g.abs().sum().item()==0 for g in grads):
                    raise ValueError("The actual positive and negative embeddings must each receive a gradient")
            elif args.mode=="personal":
                prior=json.loads((args.run/f"fold-{args.fold}-common-{args.seed_index}/first.json").read_text())
                if (any(first[key]!=prior[key] for key in ("indices","rotation","coordinates","valid")) or
                        prior["labels"]==first["labels"]):
                    raise ValueError("The point treatments have inconsistent locations or identical labels")
            rc.write_json(folder/"first.json",first)
        return loss
    prompt.train(args,decoder_type=cls(args.mode),variant=label,study_protocol=PROTOCOL,view=cache_owner.verified_parent(args),
        public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index],paired_loss=objective)
    rc.write_json(folder/"points.json",{"fit":rc.hash_file(folder/"fit.json"),"first":rc.hash_file(folder/"first.json"),
        "diagnostic":SIGNAL,"source":rc.hash_file(__file__)})


def lock(args):
    if (rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or
            rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT):
        raise ValueError("Unexpected formal E anchor")
    images,uids=ct_and_split(args)
    _,train_ids,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    lib,check=polar.calibration_split(group,valid,args.fold);ids=lib+check
    file,field=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(file,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("E field patient order")
        guide=z["adapted"]
    old=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    archive=args.stable/f"pred-{args.fold}.npz"
    if (old["prediction"]!=rc.hash_file(archive) or old["field"]!=field["payload"] or
            old["library"]!=lib or old["check"]!=check or old["choices"]["adapted"]["weight"]!=.25):
        raise ValueError("E masks or selection changed")
    with np.load(archive,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("E archive patient order")
        masks={"reference":z["adapted"]}
    view=cache_owner.verified_parent(args);members={}
    for name in MODES:
        model=cls(name)(args.public).cuda().eval()
        copies=[];members[name]=[]
        for j,offset in enumerate(OFFSETS):
            label=f"{name}-{j}";folder=args.run/f"fold-{args.fold}-{label}"
            fit=json.loads((folder/"fit.json").read_text());info=json.loads((folder/"points.json").read_text())
            if (fit["train_indices"]!=sorted(train_ids) or fit["scope"]!="inner" or fit["steps"]!=5000 or fit["variant"]!=label
                    or fit["seed"]!=50171+args.fold+offset or fit["architecture"]!=cls(name).__name__
                    or fit["public_weights"]!=hierarchy.WEIGHT or fit["view_receipt"]!=view.receipt
                    or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["source"]!=rc.hash_file(prompt.__file__)
                    or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz") or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")
                    or info["fit"]!=rc.hash_file(folder/"fit.json") or info["first"]!=rc.hash_file(folder/"first.json")
                    or info["source"]!=rc.hash_file(__file__) or info["diagnostic"]!=SIGNAL):
                raise ValueError("Incorrect point-decoder checkpoint")
            model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
            logits=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
            if not np.isfinite(logits).all():raise ValueError("Nonfinite prompted logits")
            copies.append(logits);members[name].append(fit["checkpoint"])
        mean=mean_saved_logits(copies)
        masks[name]=np.packbits(.75*guide+.25*mean>0,axis=-1)
        masks["standalone_"+name]=np.packbits(mean>0,axis=-1)
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,
        "reference":old["prediction"],"field":field["payload"],"members":members,"prediction":rc.hash_file(dst),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("RANK_CLICK_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("fit","lock"))
    for key in ("run","data","parent","public","ensemble","crossfit","stable","formal","signal"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(2,3),required=True)
    p.add_argument("--mode",choices=MODES,default="personal")
    p.add_argument("--seed-index",type=int,choices=range(3),default=0)
    p.add_argument("--smoke",action="store_true")
    p.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2)
    {"fit":train,"lock":lock}[args.phase](args)
