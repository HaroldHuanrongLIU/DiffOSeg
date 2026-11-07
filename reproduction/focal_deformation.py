"""Patient-out-of-fit nodule-local deformations with native-mask supervision."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import seed_pair_study

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/focal_deformation_protocol.md"
SEED_OFFSET=75000


def geometry(masks):
    centers=np.full((len(masks),2),63.5,np.float32)
    radius=np.full(len(masks),10.,np.float32)
    for j,case in enumerate(masks):
        y,x=np.where(case.sum(0)>=2)
        if len(x):
            centers[j]=(x.mean(),y.mean())
            radius[j]=np.sqrt(len(x)/np.pi)
    return centers,radius


def sample_matrix(batch,generator):
    values=torch.rand((batch,3),generator=generator,device=generator.device)
    angle=values[:,0]*math.pi
    c,s=angle.cos(),angle.sin()
    axes=torch.stack((c,-s,s,c),dim=1).reshape(batch,2,2)
    beta=.56*(values[:,1:3]-.5)
    return (axes*beta[:,None])@axes.transpose(1,2)


def displacement(points,centers,radius,matrix,focal):
    relative=points-centers[:,None,None]
    step=torch.einsum("bij,bhwj->bhwi",matrix,relative)
    if focal:
        sigma=1.25*radius+4
        taper=torch.exp(-relative.square().sum(-1)/(2*sigma[:,None,None].square()))
        step=step*taper[:,:,:,None]
    return step


def grids(centers,radius,matrix,focal):
    pixels=native.coordinates(128,centers.device)[None].expand(len(centers),-1,-1,-1)
    source=pixels+displacement(pixels,centers,radius,matrix,focal)
    back=pixels
    # The displacement has Lipschitz constant at most ||B|| <= 0.28.
    for _ in range(12):
        back=pixels-displacement(back,centers,radius,matrix,focal)
    return source*(2/127)-1,back*(2/127)-1


class FrozenImage:
    def __init__(self,public):
        state=hierarchy.open_public(public)
        self.encoder=hierarchy.image_encoder(state)
        _,decoder=hierarchy.decoder_heads(state)
        self.high=decoder.conv_s0.cuda().eval().requires_grad_(False)
        self.middle=decoder.conv_s1.cuda().eval().requires_grad_(False)
        self.no_memory=state["no_mem_embed"].reshape(1,256,1,1).cuda()

    @torch.no_grad()
    def encode(self,image,rotation):
        high,middle,deep=self.encoder(prompt.image_input(image*255,rotation,512))["backbone_fpn"]
        return hierarchy.pack(self.high(high),self.middle(middle),deep+self.no_memory).half().float()


def train(args):
    view=cache_owner.verified_parent(args)
    path=args.crossfit/f"oof-{args.fold}.npz";bank=json.loads(path.with_suffix(".json").read_text())
    if bank["payload"]!=rc.hash_file(path):raise ValueError("Out-of-fit geometry changed")
    with np.load(path,allow_pickle=False) as archive:
        ids=archive["ids"].tolist();centers,radii=geometry(archive["logits"]>0)
    centers=torch.from_numpy(centers).cuda();radii=torch.from_numpy(radii).cuda()
    frozen=FrozenImage(args.public)
    generator=torch.Generator(device="cuda").manual_seed(215621+args.fold)
    directory=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.variant}")

    def objective(model,embedding,guide,image,labels,ix,k,step):
        if step%2:
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                logits,raw=model(embedding,guide,image)
            loss=prompt.segmentation_loss(logits,raw,labels)
            if step==1:
                first={"ids":[ids[i] for i in ix.tolist()],"rotation":k,"loss":loss.item()}
                if args.variant=="focal" and not args.smoke:
                    control=json.loads((args.run/f"fold-{args.fold}-affine/first.json").read_text())
                    if (first["ids"]!=control["ids"] or k!=control["rotation"]
                            or not np.isclose(first["loss"],control["loss"],rtol=2e-4,atol=4e-5)):
                        raise ValueError("The affine and focal initial batches differ")
                rc.write_json(directory/"first.json",first)
            return loss
        matrix=sample_matrix(len(ix),generator)
        source,back=grids(centers[ix],radii[ix],matrix,args.variant=="focal")
        native_image=torch.rot90(image,-k,(-2,-1));native_guide=torch.rot90(guide,-k,(-2,-1))
        local=F.grid_sample(native_image,source,align_corners=True,padding_mode="border")
        local_guide=F.grid_sample(native_guide,source,align_corners=True,padding_mode="border")
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            enc=frozen.encode(local,k)
            outputs=model(enc,torch.rot90(local_guide,k,(-2,-1)),torch.rot90(local,k,(-2,-1)))
        restored=[]
        covered=back.abs().amax(-1)[:,None]<=1
        for z in outputs:
            value=F.grid_sample(torch.rot90(z,-k,(-2,-1)).float(),back,align_corners=True,padding_mode="border")
            restored.append(torch.rot90(torch.where(covered,value,native_guide),k,(-2,-1)))
        loss=prompt.segmentation_loss(restored[0],restored[1],labels)
        if args.smoke and step==2:
            grad=torch.autograd.grad(loss,model.detail[-1].weight,retain_graph=True)[0]
            if not torch.isfinite(grad).all() or grad.abs().sum().item()==0:
                raise ValueError("Native-grid warped supervision has no gradient")
        return loss

    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=args.variant,study_protocol=PROTOCOL,
                 view=view,public_weights=hierarchy.WEIGHT,seed_offset=SEED_OFFSET,paired_loss=objective)
    rc.write_json(directory/"warp.json",{"fit":rc.hash_file(directory/"fit.json"),"source":rc.hash_file(__file__),
        "field_seed":215621+args.fold,"training_geometry":bank["payload"]})


def lock(args):
    def member(model,features,ids,images,guide,name,training):
        directory=args.run/f"fold-{args.fold}-{name}"
        fit=json.loads((directory/"fit.json").read_text());warp=json.loads((directory/"warp.json").read_text())
        if (fit["scope"]!="inner" or fit["variant"]!=name or fit["steps"]!=5000 or fit["train_indices"]!=sorted(training)
                or fit["seed"]!=125171+args.fold or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT or fit["protocol"]!=rc.hash_file(PROTOCOL)
                or fit["source"]!=rc.hash_file(prompt.__file__) or fit["public_weights"]!=hierarchy.WEIGHT
                or fit["checkpoint"]!=rc.hash_file(directory/"final.pt") or warp["fit"]!=rc.hash_file(directory/"fit.json")
                or warp["training_geometry"]!=fit["oof"] or warp["source"]!=rc.hash_file(__file__)):
            raise ValueError("Registered deformation fit differs")
        model.load_state_dict(torch.load(directory/"final.pt",map_location="cpu",weights_only=True))
        return prompt.predict(model,features,ids,images,guide),fit["checkpoint"]
    seed_pair_study.lock(args,("affine","focal"),PROTOCOL,__file__,member)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("fit","lock","score"))
    for key in ("run","parent","ensemble","formal","stable","data","public","crossfit"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0);parser.add_argument("--variant",choices=("affine","focal"),default="affine")
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=parser.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2)
    if args.phase=="score":seed_pair_study.score(args,("affine","focal"))
    else:{"fit":train,"lock":lock}[args.phase](args)
