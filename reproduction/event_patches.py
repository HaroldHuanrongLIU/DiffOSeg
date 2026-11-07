"""Patient-excluded spatial correction on a fixed pixel risk budget."""
import argparse
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import component_utility as components
from reproduction import gated_hierarchy as cache_owner
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction import event_routing as routing
from reproduction.panel_event import unanimous_band
from reproduction.rater_area import register_image
from reproduction.shell_occupancy import unpack_level

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/event_patch_protocol.md"
STEPS=700
BATCH=192


def patches(bank,indices,points,spatial):
    """Sample identical native locations from NHWC features, replicating border pixels."""
    offsets=torch.arange(-3,4,device=bank.device) if spatial else torch.zeros(7,dtype=torch.long,device=bank.device)
    y=(points[:,None,None]//128+offsets[None,:,None]).clamp(0,127)
    x=(points[:,None,None]%128+offsets[None,None,:]).clamp(0,127)
    return bank[indices[:,None,None],y,x].permute(0,3,1,2).float()


class ErrorPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(44,32,3),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32,32,3),nn.GroupNorm(8,32),nn.SiLU(),
            nn.Conv2d(32,24,3),nn.GroupNorm(6,24),nn.SiLU())
        self.readout=nn.Linear(24,4)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,x):
        return 4*(self.readout(self.encoder(x).flatten(1)).float()/4).tanh()


def train_and_predict(args,bank,q,truth,points,ti,hi,names,spatial,seed):
    torch.manual_seed(seed)
    model=ErrorPatch().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0002)
    _,inv,n=np.unique(names[ti],return_inverse=True,return_counts=True)
    sampling=.5/len(ti)+.5/(len(n)*n[inv])
    rng=np.random.default_rng(seed)
    w=1/(torch.count_nonzero(q>0,dim=(2,3)).float()+1)
    sequence=hashlib.sha256()
    first=None
    for step in range(1,STEPS+1):
        chosen=rng.choice(ti,size=BATCH,p=sampling)
        position=np.array([points[i][int(frac*len(points[i]))] for i,frac in zip(chosen,rng.random(BATCH),strict=True)])
        sequence.update(chosen.astype("<i4").tobytes()+position.astype("<i4").tobytes())
        ii=torch.as_tensor(chosen,device="cuda");pos=torch.as_tensor(position,device="cuda")
        y=pos//128;x=pos%128
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            delta=model(patches(bank,ii,pos,spatial))
        loss_pixels=F.binary_cross_entropy_with_logits(q[ii,:,y,x]+delta,truth[ii,:,y,x],reduction="none")+.015*delta.square()
        loss=(loss_pixels*w[ii]).sum()/w[ii].sum()
        if step==1:first=loss.item()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),4,error_if_nonfinite=True)
        opt.param_groups[0]["lr"]=.0005*(.05+.95*(1+math.cos(math.pi*step/STEPS))/2)
        opt.step()
        if step==1:
            import copy
            ema=copy.deepcopy(model).eval().requires_grad_(False)
        else:
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),max(.005,1/step))
    path=args.run/f"fold-{args.fold}-split-{seed}-{'spatial' if spatial else 'center'}.pt"
    torch.save(ema.state_dict(),path)
    output=[]
    with torch.inference_mode():
        for i in hi:
            m=(q[i]>0).cpu().numpy()
            for block in np.array_split(points[i],max(1,math.ceil(len(points[i])/512))):
                if not len(block):continue
                pos=torch.as_tensor(block,device="cuda");indices=torch.full_like(pos,int(i))
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    delta=ema(patches(bank,indices,pos,spatial))
                m.reshape(4,-1)[:,block]=(q[i,:,pos//128,pos%128]+delta.T>0).cpu().numpy()
            output.append(m)
    return output,{"hash":sequence.hexdigest(),"first":first,"checkpoint":rc.hash_file(path),"steps":STEPS}


def lock(args):
    imgs,truth,_,uids=native.inputs(args.data)
    ids,u,old=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz";rec=nested.read(path.with_suffix(".json"))
    if rec["base"]!=old["payload"] or rec["payload"]!=rc.hash_file(path) or rc.hash_file(args.nested/"independent_quality.json")!=components.QUALITY:
        raise ValueError("Nested OOF identity changed")
    with np.load(path,allow_pickle=False) as z:
        if ids!=z["ids"].tolist():raise ValueError("Case order differs")
        s=z["logits"].astype(np.float32)
    q=.5*(u+s);reference=q>0
    if (rc.hash_file(args.routing/"summary.json")!="1400b0319bf2f2f5d330df4e941e43fa76520b1c2ce478a183815d3106819de3" or
            rc.hash_file(args.event/"summary.json")!=routing.EVENT):raise ValueError("Unverified risk budgets")
    path=args.routing/f"pred-{args.fold}.npz";sel=nested.read(args.routing/f"seal-{args.fold}.json")
    es=nested.read(args.event/f"seal-{args.fold}.json")
    if sel["prediction"]!=rc.hash_file(path) or sel["event_seal"]!=rc.hash_file(args.event/f"seal-{args.fold}.json"):
        raise ValueError("Mismatched event model")
    with np.load(path,allow_pickle=False) as z:
        if ids!=z["ids"].tolist() or not np.array_equal(np.packbits(reference,axis=-1),z["reference"]):
            raise ValueError("Inconsistent routing reference")
        regions={k:np.unpackbits(z["selected_"+k],axis=-1).astype(bool) for k in ("event","margin")}
    view=cache_owner.verified_parent(args)
    bank=np.empty((len(ids),128,128,44),np.float16);points=[]
    for j,i in enumerate(ids):
        image=register_image(imgs[i],(-1,-1))
        ct=components.normalized_ct(image,reference[j])
        high=unpack_level(view.features[i,0,512:],4).astype(np.float32)
        high/=np.sqrt((high**2).mean(0)+1e-4)
        bank[j]=np.concatenate((ct,u[j].clip(-12,12)/8,s[j].clip(-12,12)/8,high)).transpose(1,2,0)
        points.append(np.flatnonzero(unanimous_band(reference[j])[0]))
    x=torch.from_numpy(bank).cuda();del bank
    qt=torch.from_numpy(q).cuda()
    labels=torch.from_numpy(truth[ids].astype(np.float32)).cuda()
    names=np.array([str(uids[i]).split("_")[0] for i in ids]);row={i:j for j,i in enumerate(ids)}
    arrays={k:np.packbits(reference,axis=-1) for k in ("spatial","center","margin","all")}
    membership=[]
    for g in es["groups"]:
        use=g["fit"];hold=g["held"]
        teacher=nested.read(args.original/f"fold-{args.fold}-sub-{g['upstream']}/plan.json")
        if (sorted(use+hold)!=sorted(teacher["held_indices"]) or nested.patients(use,uids)&nested.patients(hold,uids)
                or teacher["shift"]!=[-1,-1]):raise ValueError("Invalid patient scope or registration")
        ti=np.array([row[i] for i in use if len(points[row[i]])]);hi=np.array([row[i] for i in hold])
        seed=459191+args.fold*19+len(membership)
        records={}
        for mode in ("center","spatial"):
            predictions,records[mode]=train_and_predict(args,x,qt,labels,points,ti,hi,names,mode=="spatial",seed)
            for j,m in zip(hi,predictions,strict=True):
                arrays[mode][j]=np.packbits(np.where(regions["event"][j],m,reference[j]),axis=-1)
                if mode=="spatial":
                    arrays["margin"][j]=np.packbits(np.where(regions["margin"][j],m,reference[j]),axis=-1)
                    arrays["all"][j]=np.packbits(m,axis=-1)
        if records["center"]["hash"]!=records["spatial"]["hash"] or records["center"]["first"]!=records["spatial"]["first"]:
            raise ValueError("Unmatched minibatches or initial supervised loss")
        membership.append({"upstream":g["upstream"],"seed":seed,"fit":use,"held":hold,"records":records})
    if sorted(i for g in membership for i in g["held"])!=ids:raise ValueError("Incomplete patient holdout")
    out=args.run/f"pred-{args.fold}.npz";np.savez_compressed(out,ids=ids,reference=np.packbits(reference,axis=-1),**arrays)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":membership,"routing":sel["prediction"],
        "base":old["payload"],"nested":rec["payload"],"features":view.receipt,"prediction":rc.hash_file(out),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("EVENT_PATCH_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("data","run","original","nested","routing","event","parent","public"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True);p.set_defaults(backbone="sam2.1-tiny",image_size=512)
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2);lock(args)
