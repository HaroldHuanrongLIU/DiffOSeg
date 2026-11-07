"""Decode full masks using paired whole-slice and lesion-local DINOv2 features."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import dino_local_probe as dino
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_views as views
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote
from reproduction.registered_seed_study import verified_field
from reproduction.trajectory_sam import select_weight

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/dino_dual_decoder_protocol.md"
MODES=("global","dual")


def fitting_maps(args,ids):
    path=args.dino/f"maps-{args.fold}.npy"
    r=views.read(path.with_suffix(".json"))
    if (r["ids"]!=sorted(ids) or r["payload"]!=rc.hash_file(path) or r["weight"]!=dino.WEIGHT
            or r["source"]!=rc.hash_file(dino.__file__) or r["protocol"]!=rc.hash_file(dino.PROTOCOL)):
        raise ValueError("Incorrect patient-excluded fitted token bank")
    with np.load(args.original/f"oof-{args.fold}.npz",allow_pickle=False) as a,np.load(
            args.nested/"nested-training"/f"oof-{args.fold}.npz",allow_pickle=False) as b:
        if r["base"]!=rc.hash_file(args.original/f"oof-{args.fold}.npz") or r["nested"]!=rc.hash_file(args.nested/"nested-training"/f"oof-{args.fold}.npz"):
            raise ValueError("OOF centers changed")
        if r["ids"]!=a["ids"].tolist() or r["ids"]!=b["ids"].tolist():raise ValueError("Incorrect guide order")
        centers=dino.centers_from_guides(a["logits"].astype(np.float32),b["logits"].astype(np.float32))
    positions={idx:j for j,idx in enumerate(r["ids"])}
    take=[positions[idx] for idx in ids]
    maps=np.load(path,mmap_mode="r",allow_pickle=False)
    return torch.from_numpy(maps[take].copy()).cuda(),torch.from_numpy(centers[take]).cuda(),r


@torch.inference_mode()
def encode_validation(args):
    from reproduction.semantic_fusion import ct_and_split
    from reproduction import polar_contours as polar
    images,uids=ct_and_split(args)
    _,_,valid=native.splits(uids,args.fold)
    lib,check=polar.calibration_split(type("Group",(),{"series_uid":uids})(),valid,args.fold)
    ids=lib+check
    path,info=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(path,allow_pickle=False) as archive:
        if archive["ids"].tolist()!=ids:raise ValueError("Incorrect E guide cases")
        guide=archive["adapted"]
    centers=dino.centers_from_guides(guide,guide)
    model=dino.load_encoder(type("Public",(),{"run":args.dino})())
    proj=torch.from_numpy((np.random.default_rng(39117).standard_normal((1024,64))/8).astype(np.float32)).cuda()
    dst=args.run/f"validation-{args.fold}.npy"
    if dst.exists():raise FileExistsError(dst)
    maps=np.lib.format.open_memmap(dst,mode="w+",shape=(len(ids),2,64,32,32),dtype=np.float16)
    loc=torch.from_numpy(centers).cuda()
    registered=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),-1,-1)
    for j in range(len(ids)):
        for k,side in enumerate((128,64)):
            x=dino.view_image(registered[j:j+1],loc[j:j+1],side)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                tokens=model(pixel_values=x).last_hidden_state[:,1:]
            features=(tokens.float()@proj).transpose(1,2).reshape(64,32,32)
            if not torch.isfinite(features).all():raise ValueError("Nonfinite validation tokens")
            maps[j,k]=features.cpu().numpy()
    maps.flush();del maps
    rc.write_json(dst.with_suffix(".json"),{"ids":ids,"centers":centers.tolist(),"field":info["payload"],"weight":dino.WEIGHT,
        "projection_seed":39117,"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"payload":rc.hash_file(dst)})
    print("DUAL_VALID_MAPS",args.fold,len(ids),flush=True)


def sample_tokens(maps,center,source_grid,size,dual):
    if source_grid.shape[1:3]!=(size,size):
        source_grid=F.interpolate(source_grid.permute(0,3,1,2),(size,size),mode="bilinear",align_corners=False).permute(0,2,3,1)
    delta=(source_grid+1)*63.5-center[:,None,None,:]
    def sample(index,side):
        return F.grid_sample(maps[:,index].float(),2*delta/side,mode="bilinear",padding_mode="border",align_corners=False)
    whole=sample(0,128)
    small=whole
    if dual:
        outside=delta.abs().amax(-1,keepdim=True)>32
        small=torch.where(outside.permute(0,3,1,2),whole,sample(1,64))
    return torch.cat((whole,small),1)


class TokenDecoder(nn.Module):
    def __init__(self,initial,dual):
        super().__init__()
        self.base=vote.build(initial)
        self.dual=dual
        with torch.random.fork_rng(devices=[]):
            self.adapters=nn.ModuleList(nn.Sequential(nn.Conv2d(128,24,1),nn.SiLU(),nn.Conv2d(24,width,1)) for width in (40,24))
        for layer in self.adapters:
            nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)

    def forward(self,image,maps,center,grid):
        h,w=image.shape[-2:]
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing="ij")
        current=self.base.stem(torch.cat((image*2-1,torch.stack((x,y)).expand(len(image),-1,-1,-1)),1).contiguous(memory_format=torch.channels_last))
        saved=[current]
        for layer in self.base.down:
            current=layer(current);saved.append(current)
        for k,(layer,skip) in enumerate(zip(self.base.up,reversed(saved[:-1]))):
            current=layer(torch.cat((F.interpolate(current,skip.shape[-2:],mode="bilinear",align_corners=False),skip),1))
            if k>=2:
                values=sample_tokens(maps,center,grid,current.shape[-1],self.dual)
                current=current+self.adapters[k-2](values)
        return self.base.head(current).float()


def train(args):
    images,truth,_,uids=native.inputs(args.data)
    _,ids,_=native.splits(uids,args.fold)
    warm,warm_fit=views.warm_identity(args,ids)
    parent=args.previous/f"fold-{args.fold}-inner"
    if views.read(parent/"plan.json")["train_indices"]!=ids or rc.hash_file(parent/"final.pt")!=warm_fit["initialization"]:
        raise ValueError("Invalid v1 initialization")
    maps,token_centers,record=fitting_maps(args,ids)
    if args.smoke:
        ids=ids[:48];maps=maps[:48];token_centers=token_centers[:48]
    dest=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.mode}")
    dest.mkdir(exist_ok=False)
    steps=5 if args.smoke else align.STEPS
    seed=144291+args.fold
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=True
    ct=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*warm_fit["applied_shift"])
    masks=torch.from_numpy(truth[ids]).cuda()
    distances=np.stack([[rc.signed(m) for m in panel] for panel in truth[ids]]).astype(np.float16)
    targets=torch.cat((masks.float(),vote.votes(masks),torch.from_numpy(distances).cuda().float()),1).half()
    centers=native.rater_center(masks);centers=torch.where((centers-64.5).abs().amax(1,keepdim=True)>4,centers,64.5)
    _,inv,counts=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    sampling=torch.from_numpy(.65/len(ids)+.35/(len(counts)*counts[inv])).float().cuda()
    model=TokenDecoder(parent/"final.pt",args.mode=="dual").cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params=list(model.parameters());averaged=list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=.00055,weight_decay=.0001,fused=True)
    schedule=hashlib.sha256()
    with (dest/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(sampling,align.BATCH,replacement=True)
            u=torch.rand(align.BATCH,9,device="cuda")
            schedule.update(ix.cpu().numpy().astype("<i4").tobytes()+u.cpu().numpy().astype("<f4").tobytes())
            angle=torch.where(u[:,0]<.65,(u[:,1]*4).floor()*math.pi/2,(2*u[:,1]-1)*math.pi)
            scale=.65+.58*u[:,3]
            grid,_=align.transform_grids(centers[ix]+(u[:,4:6]-.5)*1.2,scale,angle,torch.where(u[:,2]<.5,-1.,1.))
            x=F.grid_sample(ct[ix].float()/255,grid,align_corners=True,padding_mode="border")
            x=(x.clamp_min(1e-6).pow(.85+.3*u[:,6,None,None,None])*(.87+.26*u[:,7,None,None,None])+.07*(u[:,8,None,None,None]-.5)).clamp(0,1)
            y=F.grid_sample(targets[ix].float(),grid,align_corners=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                output=model(x,maps[ix],token_centers[ix],grid)
            z=output[:,:4];p=(2*z).sigmoid()
            area=F.smooth_l1_loss((p.sum((-2,-1))+1).log(),(y[:,:4].sum((-2,-1))+1).log())
            center=F.smooth_l1_loss(native.rater_center(p),native.rater_center(y[:,:4]))
            sd=F.smooth_l1_loss(output[:,4:8].tanh(),(y[:,8:]/(8*scale[:,None,None,None])).clamp(-1,1))
            loss=vote.mask_loss(z,y[:,:4])+.75*vote.mask_loss(output[:,8:],y[:,4:8])+.15*sd+.20*area+.025*center
            if step==1 and not args.smoke:
                with (warm/"train.jsonl").open() as st:expected=json.loads(next(st))["loss"]
                if not np.isclose(loss.item(),expected,rtol=3e-4,atol=5e-5):
                    raise ValueError(f"Zero adapter first objective {loss.item()} differs from {expected}")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.00055*min(1,step/100)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(averaged,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                log.write(json.dumps({"step":step,"loss":loss.item()},allow_nan=False)+"\n")
    dst=dest/"final.pt";torch.save(ema.state_dict(),dst)
    rc.write_json(dest/"fit.json",{"fold":args.fold,"mode":args.mode,"smoke":args.smoke,"steps":steps,"initial":warm_fit["initialization"],
        "train":ids,"seed":seed,"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__),"features":record["payload"],
        "schedule":schedule.hexdigest(),"checkpoint":rc.hash_file(dst)})
    print("TOKEN_DECODER_FIT",args.fold,args.mode,steps,flush=True)


@torch.inference_mode()
def predict(model,images,centers,token_centers,maps,shift):
    all_logits=[]
    for j in range(0,len(images),8):
        ct=align.registered_ct(torch.from_numpy(images[j:j+8,None]).cuda(),*shift).float()/255
        center=torch.from_numpy(centers[j:j+8]).cuda().float()
        token_center=torch.from_numpy(token_centers[j:j+8]).cuda().float()
        feature=torch.from_numpy(maps[j:j+8].copy()).cuda()
        value=0;total=0
        for scale in (.74,.94,1.10):
            for k in range(4):
                for flip in (-1.,1.):
                    g,b=align.transform_grids(center,ct.new_full((len(ct),),scale),ct.new_full((len(ct),),k*math.pi/2),ct.new_full((len(ct),),flip))
                    x=F.grid_sample(ct,g,align_corners=True,padding_mode="border")
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                        out=model(x,feature,token_center,g)[:,:4].float()
                    z=F.grid_sample(out,b,align_corners=True)
                    w=(1-b.abs().amax(-1)).clamp(0,.05)[:,None]
                    value=value+z*w;total=total+w
        all_logits.append(torch.where(total>0,value/total.clamp_min(1.e-8),-16.).cpu().numpy())
    return np.concatenate(all_logits)


def lock(args):
    images,truth,ids,lib,check,base,mask,ref,centers=views.eval_context(args)
    _,warm=views.warm_identity(args,ids);population=lib+check
    featurefile=args.run/f"validation-{args.fold}.npy";rec=views.read(featurefile.with_suffix(".json"))
    if rec["ids"]!=population or rec["field"]!=ref["field"] or rec["payload"]!=rc.hash_file(featurefile) or rec["source"]!=rc.hash_file(__file__):
        raise ValueError("Invalid single-image feature provenance")
    maps=np.load(featurefile,mmap_mode="r",allow_pickle=False)
    bank=views.read(args.dino/f"maps-{args.fold}.json")
    arrays={"reference":np.packbits(mask,axis=-1)};members={};choices={};schedules=[]
    for mode in MODES:
        folder=args.run/f"fold-{args.fold}-{mode}"
        fit=views.read(folder/"fit.json")
        if (fit["train"]!=ids or fit["steps"]!=6000 or fit["smoke"] or fit["mode"]!=mode or fit["initial"]!=warm["initialization"]
                or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["features"]!=bank["payload"] or fit["source"]!=rc.hash_file(__file__)
                or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
            raise ValueError("Invalid DINO decoder training identity")
        model=TokenDecoder(None,mode=="dual").cuda().eval().to(memory_format=torch.channels_last)
        model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
        logits=predict(model,images[population],centers,np.array(rec["centers"],np.float32),maps,warm["applied_shift"])
        if not np.isfinite(logits).all():raise ValueError("Nonfinite DINO predictions")
        w,grid=select_weight(truth[lib],base[:len(lib)],logits[:len(lib)])
        arrays[mode]=np.packbits((1-w)*base+w*logits>0,axis=-1)
        arrays["standalone_"+mode]=np.packbits(logits>0,axis=-1)
        choices[mode]={"weight":w,"grid":grid};members[mode]=fit["checkpoint"];schedules.append(fit["schedule"])
    if len(set(schedules))!=1:raise ValueError("Unequal case or augmentation stream")
    path=args.run/f"pred-{args.fold}.npz";np.savez_compressed(path,ids=population,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"reference":ref["prediction"],
          "field":ref["field"],"members":members,"choices":choices,"features":rec["payload"],"prediction":rc.hash_file(path),
          "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("TOKEN_DECODER_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("encode","fit","lock"))
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    p.add_argument("--mode",choices=MODES,default="dual");p.add_argument("--smoke",action="store_true")
    for key in ("run","data","dino","original","nested","align","previous","stable","formal","parent","ensemble","public","local","texture"):
        p.add_argument("--"+key,type=Path,required=True)
    p.set_defaults(backbone="sam2.1-tiny",image_size=512)
    args=p.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    {"encode":encode_validation,"fit":train,"lock":lock}[args.phase](args)
