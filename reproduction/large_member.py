"""Replace only the first E member by a corresponding SAM2.1 Hiera-Large decoder."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import aligned_preference as align
from reproduction import canonical_guide
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/large_member_protocol.md"
BACKBONE = "sam2.1-large"


@torch.inference_mode()
def smoke_public(args):
    import yaml
    images,_,_,uids=native.inputs(args.data)
    _,fit,_=native.splits(uids,0)
    ids=sorted(fit)[:2]
    bank=args.crossfit/"oof-0.npz"
    if rc.hash_file(bank)!=json.loads(bank.with_suffix(".json").read_text())["payload"]:
        raise ValueError("Smoke requires the fitted OOF patients")
    with np.load(bank,allow_pickle=False) as z:
        if z["ids"][:2].tolist()!=ids:raise ValueError("Wrong smoke patients")
        guide=torch.from_numpy(z["logits"][:2]).float().cuda()
    spec=hierarchy.BACKBONES[BACKBONE]
    config=yaml.safe_load((args.public/"sam2.1_hiera_l.yaml").read_text())["model"]["image_encoder"]
    trunk=config["trunk"]
    if (trunk["embed_dim"]!=spec.width or trunk["num_heads"]!=spec.heads or tuple(trunk["stages"])!=spec.stages
            or tuple(trunk["window_spec"])!=spec.windows or tuple(trunk["global_att_blocks"])!=spec.global_blocks
            or config["neck"]["backbone_channel_list"]!=[8*spec.width,4*spec.width,2*spec.width,spec.width]):
        raise ValueError("Image implementation differs from the official config")
    state=hierarchy.open_public(args.public,BACKBONE)
    encoder=hierarchy.image_encoder(state,BACKBONE)
    model=hierarchy.HierarchicalRanks(args.public,backbone=BACKBONE).cuda().eval()
    no_mem=state["no_mem_embed"].reshape(1,256,1,1).cuda()
    image=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),-1,-1)
    def evaluate(x,g):
        hs=encoder(prompt.image_input(x,0,512))["backbone_fpn"]
        embed=hierarchy.pack(model.decoder.conv_s0(hs[0]),model.decoder.conv_s1(hs[1]),hs[2]+no_mem)
        return model(embed,g,x.float()/255)[0]
    batched=evaluate(image,guide);one=evaluate(image[:1],guide[:1])
    max_diff=float((batched[:1]-one).abs().max())
    if batched.shape!=(2,4,128,128) or not torch.isfinite(batched).all() or max_diff>.01:
        raise ValueError(f"Unexpected single-image smoke output, maximum difference {max_diff}")
    rc.write_json(args.run/"public_smoke.json",{"weights":spec.digest,"parameters":sum(p.numel() for p in encoder.parameters()),
        "config":hierarchy.LARGE_CONFIG,"batch_difference":max_diff,"source":rc.hash_file(__file__)})


def train(args):
    view=hierarchy.AlreadyRotatedView(args,backbone=BACKBONE)
    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant="large",study_protocol=PROTOCOL,view=view,
                 public_weights=hierarchy.BACKBONES[BACKBONE].digest)


def lock(args):
    if (rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT or
            rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or rc.hash_file(args.data)!=rc.DATA_HASH):
        raise ValueError("Formal E identity changed")
    with args.data.open("rb") as stream:cases=list(pickle.load(stream).values())
    images=np.stack([c["image"] for c in cases]);uids=[c["series_uid"] for c in cases]
    _,fit,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(group,valid,args.fold)
    ids=library+checked
    field,receipt=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(field,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Incorrect E field indices")
        guide=z["adapted"]
    archived=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    prior=args.stable/f"pred-{args.fold}.npz"
    if (archived["prediction"]!=rc.hash_file(prior) or archived["field"]!=receipt["payload"] or
            archived["library"]!=library or archived["check"]!=checked):
        raise ValueError("Archived E order or payload differs")
    with np.load(prior,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Original prediction order differs")
        masks={"reference":z["adapted"]}
    weight=archived["choices"]["adapted"]["weight"]
    if weight!=json.loads((args.formal/"selection.json").read_text())["folds"][args.fold]["weight"]:
        raise ValueError("Unexpected E fusion weight")
    src=canonical_guide.model_sources(args,fit,args.fold)
    if [h for _,h in src]!=archived["checkpoints"]:raise ValueError("Original E member order changed")
    cache=cache_owner.verified_parent(args).features
    tiny=hierarchy.HierarchicalRanks(args.parent/"public").cuda().eval()
    tiny_logits=[]
    for directory,_ in src:
        tiny.load_state_dict(torch.load(directory/"final.pt",weights_only=True,map_location="cpu"))
        tiny_logits.append(prompt.predict(tiny,cache,ids,images,guide).astype(np.float16))
    del tiny,cache
    folder=args.run/f"fold-{args.fold}-large";record=json.loads((folder/"fit.json").read_text())
    if (record["scope"]!="inner" or record["variant"]!="large" or record["seed"]!=50171+args.fold or record["steps"]!=5000
            or record["train_indices"]!=sorted(fit) or record["backbone"]!=BACKBONE or record["public_weights"]!=hierarchy.BACKBONES[BACKBONE].digest
            or record["source"]!=rc.hash_file(prompt.__file__) or record["protocol"]!=rc.hash_file(PROTOCOL)
            or record["view_receipt"]!=rc.hash_file(args.run/"features.json") or record["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
            or record["checkpoint"]!=rc.hash_file(folder/"final.pt")):
        raise ValueError("Large fit identity differs")
    large=hierarchy.HierarchicalRanks(args.public,backbone=BACKBONE).cuda().eval()
    large.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
    cache=hierarchy.feature_cache(args,backbone=BACKBONE)
    large_logits=prompt.predict(large,cache,ids,images,guide).astype(np.float16)
    if not all(np.isfinite(z).all() for z in [large_logits,*tiny_logits]):raise ValueError("Nonfinite Hiera logits")
    for name,bank in (("tiny",tiny_logits),("hybrid",[large_logits,*tiny_logits[1:]])):
        masks[name]=np.packbits((1-weight)*guide+weight*mean_saved_logits(bank)>0,axis=-1)
    masks["single_tiny"]=np.packbits(tiny_logits[0]>0,axis=-1)
    masks["single_large"]=np.packbits(large_logits>0,axis=-1)
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,
        "weight":weight,"tiny":archived["checkpoints"],"large":record["checkpoint"],"reference":archived["prediction"],
        "field":receipt["payload"],"prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("LARGE_LOCK",args.fold,weight,flush=True)


def score(args):
    seals=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in range(4)]
    for f,r in enumerate(seals):
        if r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz"):raise ValueError("Incomplete four-fold capacity locks")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Capacity score order differs")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference","tiny","hybrid","single_tiny","single_large")}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores})
        print("LARGE_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("smoke-public","encode","fit","lock","score"))
    for name in ("run","parent","ensemble","formal","stable","data","public","crossfit","align"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone=BACKBONE)
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    if args.phase=="encode":
        hierarchy.encode(args,backbone=BACKBONE)
    else:
        {"smoke-public":smoke_public,"fit":train,"lock":lock,"score":score}[args.phase](args)
