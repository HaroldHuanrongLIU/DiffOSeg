"""Frozen general-image SAM features and a single-slice, automatic prompt decoder."""
import argparse
import copy
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction.crossfit_contours import training_partition
from reproduction import native_local as native
from reproduction import native_study as ns
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent
from reproduction import vote_contours as vote

WEIGHT="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
VENDOR_MANIFEST="4ff87fcf06a09a6ec95874e3d6b78cbb498333148f57e3872e9d0bc9967c2e56"
PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/prompt_protocol.md"
STEPS=5000


@dataclass(frozen=True)
class Backbone:
    filename: str
    digest: str
    width: int
    depth: int
    heads: int
    global_blocks: tuple


BACKBONES = {
    "vit_b": Backbone("sam_vit_b.pth",WEIGHT,768,12,12,(2,5,8,11)),
    "vit_h": Backbone("sam_vit_h.pth","a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
                      1280,32,16,(7,15,23,31)),
}


def vendor(root):
    manifest=root/"runtime.sha256"
    if rc.hash_file(manifest)!=VENDOR_MANIFEST:raise ValueError("Vendor manifest differs")
    for line in manifest.read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        if rc.hash_file(root/"runtime"/name)!=digest:raise ValueError("Vendor source differs")
    sys.path.insert(0,str(root/"runtime"))
    from sam91_vendor import modeling
    return modeling


def subset(state,prefix):
    return {k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+".")}


def image_encoder(root, image_size=512, backbone="vit_b"):
    lib=vendor(root)
    spec=BACKBONES[backbone]
    weights=root/spec.filename
    if rc.hash_file(weights)!=spec.digest:raise ValueError("SAM public weight hash mismatch")
    state=torch.load(weights,weights_only=True,map_location="cpu")
    model=lib.ImageEncoderViT(img_size=1024,patch_size=16,embed_dim=spec.width,depth=spec.depth,num_heads=spec.heads,mlp_ratio=4,
                              norm_layer=partial(nn.LayerNorm,eps=1e-6),use_rel_pos=True,global_attn_indexes=spec.global_blocks,
                              window_size=14,out_chans=256)
    model.load_state_dict(subset(state,"image_encoder"))
    model.pos_embed=nn.Parameter(F.interpolate(model.pos_embed.permute(0,3,1,2),size=(image_size//16,image_size//16),mode="bicubic",
                                               align_corners=False).permute(0,2,3,1).contiguous(),requires_grad=False)
    return model.requires_grad_(False).cuda().eval()


def image_input(batch,k,image_size=512):
    values=torch.rot90(batch,k,(-2,-1)).float()
    if k%2: sampled=values[:,:,24:105:2,24:105:2]
    else:sampled=values[:,:,::2,::2]
    q=torch.quantile(sampled.flatten(1),values.new_tensor([.015,.985]),dim=1)
    norm=((values-q[0,:,None,None,None])/(q[1]-q[0]).clamp_min(35)[:,None,None,None]).clamp(0,1)*255
    mean=values.new_tensor([123.675,116.28,103.53])[None,:,None,None]
    sd=values.new_tensor([58.395,57.12,57.375])[None,:,None,None]
    return (F.interpolate(norm,(image_size,image_size),mode="bilinear",align_corners=False)-mean)/sd


@torch.inference_mode()
def encode(args, verified_folds=range(4)):
    if rc.hash_file(args.data)!=rc.DATA_HASH:raise ValueError("LIDC bytes differ")
    with args.data.open("rb") as f:images=np.stack([v["image"] for v in pickle.load(f).values()])
    # This preprocessing has no patient- or cohort-fitted statistics. The registration
    # is the historical fixed training estimate, verified independently for every fold.
    for fold in verified_folds:
        if json.loads((args.align/f"fold-{fold}-inner-registered/fit.json").read_text())["applied_shift"]!=[-1,-1]:
            raise ValueError("Cache needs a per-fold registration")
    backbone=getattr(args,"backbone","vit_b")
    model=image_encoder(args.run,args.image_size,backbone)
    dst=args.run/"image_features.npy"
    if dst.exists():raise FileExistsError(dst)
    cache=np.lib.format.open_memmap(dst,mode="w+",shape=(len(images),4,256,args.image_size//16,args.image_size//16),dtype=np.float16)
    start=time.monotonic()
    for i in range(0,len(images),4):
        batch=align.registered_ct(torch.from_numpy(images[i:i+4,None]).cuda(),-1,-1)
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                feature=model(image_input(batch,k,args.image_size))
            out=feature.float().cpu().numpy()
            if not np.isfinite(out).all():raise ValueError("Nonfinite SAM image embedding")
            cache[i:i+len(batch),k]=out
        if i%100==0:print("ENCODE",i,len(images),time.monotonic()-start,flush=True)
    cache.flush();del cache
    rc.write_json(args.run/"image_features.json",{"images":len(images),"data":rc.DATA_HASH,"source":rc.hash_file(__file__),
                  "public_weights":BACKBONES[backbone].digest,"backbone":backbone,"vendor":VENDOR_MANIFEST,"image_size":args.image_size,"seconds":time.monotonic()-start,"array":rc.hash_file(dst)})


def boxes_from_logits(guide,training,image_size=512):
    mask=guide.mean(1)>0
    coords=torch.arange(128,device=guide.device)
    cols=mask.any(1);rows=mask.any(2)
    x0=torch.where(cols,coords,128).amin(1);x1=torch.where(cols,coords,-1).amax(1)
    y0=torch.where(rows,coords,128).amin(1);y1=torch.where(rows,coords,-1).amax(1)
    box=torch.stack((x0,y0,x1,y1),1).float()
    empty=~mask.flatten(1).any(1)
    box=torch.where(empty[:,None],box.new_tensor([48,48,80,80]),box)
    box=box+box.new_tensor([-2.,-2.,2.,2.])
    if training:
        box=box+torch.randn_like(box)*1.6
    return ((box.clamp(0,127)+.5)*(image_size/128)-.5)


class PromptRanks(nn.Module):
    def __init__(self,root,pretrained=True,image_size=512,backbone="vit_b"):
        super().__init__()
        lib=vendor(root)
        self.image_size=image_size
        self.mask_size=image_size//4
        self.prompt=lib.PromptEncoder(embed_dim=256,image_embedding_size=(image_size//16,image_size//16),
                                     input_image_size=(image_size,image_size),mask_in_chans=16)
        self.decoder=lib.MaskDecoder(transformer_dim=256,transformer=lib.TwoWayTransformer(depth=2,embedding_dim=256,
                                    mlp_dim=2048,num_heads=8),num_multimask_outputs=3,iou_head_depth=3,iou_head_hidden_dim=256)
        if pretrained:
            spec=BACKBONES[backbone]
            path=root/spec.filename
            if rc.hash_file(path)!=spec.digest:raise ValueError("SAM model identity")
            state=torch.load(path,weights_only=True,map_location="cpu")
            self.prompt.load_state_dict(subset(state,"prompt_encoder"));self.decoder.load_state_dict(subset(state,"mask_decoder"))
        self.detail=nn.Sequential(nn.Conv2d(32+1+4+2,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.SiLU(),
                                  rc.Residual(32,32),nn.Conv2d(32,4,1))
        self.mix=nn.Parameter(torch.zeros(4))
        nn.init.zeros_(self.detail[-1].weight);nn.init.zeros_(self.detail[-1].bias)

    def forward(self,embedding,guide,image,return_features=False):
        box=boxes_from_logits(guide,self.training,self.image_size)
        dense=guide.mean(1,keepdim=True).clamp(-10,10)
        if self.training:
            dense=dense*(torch.rand(len(dense),1,1,1,device=dense.device)>.23)
        dense=F.interpolate(dense,(self.mask_size,self.mask_size),mode="bilinear",align_corners=False)
        sparse,dense=self.prompt(points=None,boxes=box,masks=dense)
        d=self.decoder;b,c,h,w=embedding.shape
        tokens=torch.cat((d.iou_token.weight,d.mask_tokens.weight),0).expand(b,-1,-1)
        tok=torch.cat((tokens,sparse),1)
        hs,src=d.transformer(embedding+dense,self.prompt.get_dense_pe().expand(b,-1,-1,-1),tok)
        up=d.output_upscaling(src.transpose(1,2).reshape(b,c,h,w))
        hyper=torch.stack([m(hs[:,1+i]) for i,m in enumerate(d.output_hypernetworks_mlps)],1)
        sam=(hyper@up.flatten(2)).reshape(b,4,self.mask_size,self.mask_size).float()
        sam=F.interpolate(sam,(128,128),mode="bilinear",align_corners=False)
        up=F.interpolate(up,(128,128),mode="bilinear",align_corners=False)
        low=F.avg_pool2d(image,7,1,3,count_include_pad=False)
        z=self.detail(torch.cat((up,2*image-1,guide.clamp(-12,12)/8,4*(image-low),guide.sigmoid().mean(1,keepdim=True)),1)).float()
        combined=(1-self.mix.sigmoid())[None,:,None,None]*guide+self.mix.sigmoid()[None,:,None,None]*sam+z
        if return_features:
            return combined,sam,up
        return combined,sam


def features_file(args):
    owner = args.feature_cache if args.feature_cache is not None else args.run
    info = json.loads((owner / "image_features.json").read_text())
    source = Path(__file__)
    if args.feature_cache is not None:
        source = owner / "source/reproduction/prompt_preference.py"
        binding = json.loads((args.run / "cache_binding.json").read_text())
        if (binding["warm_owner"] != str(owner) or binding["warm_payload"] != info["array"]
                or binding["warm_receipt"] != rc.hash_file(owner / "image_features.json")
                or binding["runtime_prompt_source"] != rc.hash_file(__file__)):
            raise ValueError("Unverified external warm embedding cache")
    spec = BACKBONES[getattr(args,"backbone","vit_b")]
    if (info["data"] != rc.DATA_HASH or info["source"] != rc.hash_file(source)
            or info["public_weights"] != spec.digest or info.get("image_size", 512) != args.image_size):
        raise ValueError("Cache provenance")
    return np.load(owner / "image_features.npy", mmap_mode="r", allow_pickle=False)


def segmentation_loss(logits, raw, target):
    radius = (logits.sigmoid().sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
    actual = (target.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
    return vote.mask_loss(logits,target) + .4*vote.mask_loss(raw,target) + .08*F.smooth_l1_loss(radius,actual,beta=.5)


def train(args, decoder_type=PromptRanks, supervised=None, variant=None, study_protocol=None, view=None, public_weights=None, seed_offset=0, snapshot_steps=(), paired_loss=None, fit_subset=None, guide_update=None):
    ims,labels,_,uids=native.inputs(args.data)
    eligible, excluded = training_partition(uids, args.fold, args.scope)
    if fit_subset is not None:
        subset = set(fit_subset)
        if not subset or len(subset)!=len(fit_subset) or not subset.issubset(eligible):
            raise ValueError("The nested training set must be a nonempty subset of eligible cases")
        excluded = sorted(set(range(len(uids)))-subset)
        patients = lambda indices: {str(uids[i]).split("_")[0] for i in indices}
        if patients(subset)&patients(excluded):
            raise ValueError("Nested training must exclude complete patients")
        eligible = sorted(subset)
    cache=features_file(args) if view is None else view.features
    path=args.crossfit/f"oof-{args.fold}.npz"
    if rc.hash_file(path)!=json.loads(path.with_suffix(".json").read_text())["payload"]:raise ValueError("OOF bank identity")
    with np.load(path,allow_pickle=False) as z:
        ids=z["ids"].tolist(); guides=z["logits"]
    if ids!=sorted(eligible):raise ValueError("OOF training split")
    steps=STEPS
    if args.smoke:ids=ids[:32];guides=guides[:32];steps=5
    label=variant or args.scope
    out=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{label}")
    out.mkdir(exist_ok=False)
    seed=50171+args.fold+seed_offset;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    embeddings=torch.from_numpy(cache[ids].copy()).cuda()
    image=align.registered_ct(torch.from_numpy(ims[ids,None]).cuda(),-1,-1)
    truth=torch.from_numpy(labels[ids]).cuda().float()
    prior=torch.from_numpy(guides).cuda()
    _,inv,n=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    weight=torch.from_numpy(.45/len(ids)+.55/(len(n)*n[inv])).float().cuda()
    backbone=getattr(args,"backbone","vit_b")
    model=decoder_type(args.public or args.run,image_size=args.image_size,backbone=backbone).cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
    params,avg=list(model.parameters()),list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=.00016,fused=True,weight_decay=.0002)
    start=time.monotonic()
    with (out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(weight,12,replacement=True); k=int(torch.randint(4,()).item())
            native_guide = prior[ix].float()
            if guide_update is not None:
                native_guide = guide_update(native_guide, ix, step)
            g=torch.rot90(native_guide,k,(-2,-1));x=torch.rot90(image[ix].float()/255,k,(-2,-1));y=torch.rot90(truth[ix],k,(-2,-1))
            g=g*(.75+.5*torch.rand(len(g),1,1,1,device="cuda"))+(torch.rand(len(g),1,1,1,device="cuda")-.5)*2
            if paired_loss is None:
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    if view is not None:
                        z,raw=view.decode(model,embeddings[ix,k].float(),g,x,ix,k)
                        extra=0
                    elif supervised is None:
                        z,raw=model(embeddings[ix,k].float(),g,x)
                        extra=0
                    else:
                        z,raw,extra=supervised(model,embeddings[ix,k].float(),g,x,y)
                loss = segmentation_loss(z, raw, y) + extra
            else:
                loss = paired_loss(model, embeddings[ix,k].float(), g, x, y, ix, k, step)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,4,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.00016*min(1,step/80)*(.035+.965*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step in snapshot_steps:
                torch.save(ema.state_dict(), out/f"ema-{step}.pt")
            if step==1 or step%250==0 or step==steps:
                rec={"step":step,"loss":loss.item(),"seconds":time.monotonic()-start};log.write(json.dumps(rec,allow_nan=False)+"\n");print(rec,flush=True)
    torch.save(ema.state_dict(),out/"final.pt")
    protocol = PROTOCOL if args.image_size==512 else PROTOCOL.with_name("prompt_resolution_protocol.md")
    if args.scope=="full": protocol = PROTOCOL.with_name("outer_protocol.md")
    if backbone=="vit_h": protocol=PROTOCOL.with_name("sam_h_protocol.md")
    if study_protocol is not None: protocol=study_protocol
    rc.write_json(out/"fit.json",{"source":rc.hash_file(__file__),"protocol":rc.hash_file(protocol),
                  "variant":label,"architecture":decoder_type.__name__,"view_receipt":view.receipt if view is not None else None,"backbone":backbone,
                  "public_weights":public_weights if public_weights is not None else BACKBONES[backbone].digest,
                  "scope":args.scope,"excluded_indices":excluded,
                  "image_size":args.image_size,"fold":args.fold,"steps":steps,"seed":seed,
                  "train_indices":ids,"oof":rc.hash_file(path),"checkpoint":rc.hash_file(out/"final.pt"),"seconds":time.monotonic()-start})


@torch.inference_mode()
def predict(model,cache,ids,images,prior,*,reflected=False):
    out=[]
    for i in range(0,len(ids),12):
        sl=ids[i:i+12]
        emb=torch.from_numpy(cache[sl].copy()).cuda()
        x=align.registered_ct(torch.from_numpy(images[sl,None]).cuda(),-1,-1).float()/255
        g=torch.from_numpy(prior[i:i+len(sl)]).cuda().float()
        if reflected:
            x = x.flip([-1])
            g = g.flip([-1])
        sums=0
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                z,_=model(emb[:,k].float(),torch.rot90(g,k,(-2,-1)),torch.rot90(x,k,(-2,-1)))
            sums=sums+torch.rot90(z,-k,(-2,-1))
        mean = sums/4
        if reflected:
            mean = mean.flip([-1])
        out.append(mean.cpu().numpy())
    return np.concatenate(out)


def check(args):
    ims,labels,_,uids=native.inputs(args.data);_,inner,_=native.splits(uids,args.fold)
    folder=args.run/f"fold-{args.fold}-inner";fit=json.loads((folder/"fit.json").read_text())
    if fit["train_indices"]!=sorted(inner) or fit["checkpoint"]!=rc.hash_file(folder/"final.pt"):raise ValueError("Wrong decoder")
    cache=features_file(args);guide_path=args.texture/f"features-{args.fold}-inner.npz"
    info=json.loads(guide_path.with_suffix(".json").read_text());ids=info["lib"]+info["check"]
    if rc.hash_file(guide_path)!=info["payload"]:raise ValueError("Guide identity")
    with np.load(guide_path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Guide ordering")
        original=np.unpackbits(z["old"],axis=-1).astype(bool)
        guide=incumbent.anchor_logits(z["base"].astype(np.float32),original)
    backbone=getattr(args,"backbone","vit_b")
    if fit.get("backbone","vit_b") != backbone:
        raise ValueError("Warm decoder and image backbone differ")
    model=PromptRanks(args.public or args.run,False,args.image_size,backbone).cuda().eval();model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
    predicted=predict(model,cache,ids,ims,guide);n=len(info["lib"]);truth=labels[ids].astype(bool)
    grid=[]
    for w in incumbent.WEIGHTS:
        grid.append(float(ns.dice(truth[:n],(1-w)*guide[:n]+w*predicted[:n]>0).mean()))
    w=incumbent.WEIGHTS[int(np.argmax(grid))]
    result={"reference":float(ns.dice(truth[n:],original[n:]).mean()),"sam":float(ns.dice(truth[n:],predicted[n:]>0).mean()),
            "candidate":float(ns.dice(truth[n:],(1-w)*guide[n:]+w*predicted[n:]>0).mean())}
    rc.write_json(args.run/f"inner-{args.fold}.json",{"fold":args.fold,"fit_scores":grid,"weight":w,"scores":result,
                  "model":fit["checkpoint"],"fit":info["lib"],"check":info["check"]})
    np.savez_compressed(args.run/f"pred-inner-{args.fold}.npz",ids=ids,logits=predicted.astype("float16"))
    print("SAM_INNER",args.fold,w,result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("encode","fit","check"))
    for k in ("run","data","align","crossfit","texture"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),default=0);p.add_argument("--smoke",action="store_true")
    p.add_argument("--image-size",type=int,choices=(512,1024),default=512)
    p.add_argument("--scope",choices=("inner","full"),default="inner")
    p.add_argument("--backbone",choices=tuple(BACKBONES),default="vit_b")
    p.add_argument("--public",type=Path)
    p.add_argument("--feature-cache",type=Path)
    a=p.parse_args()
    if a.scope=="full" and a.phase!="fit":p.error("Full prediction and scoring use the outer study")
    torch.set_num_threads(3);cv2.setNumThreads(1)
    {"encode":encode,"fit":train,"check":check}[a.phase](a)
