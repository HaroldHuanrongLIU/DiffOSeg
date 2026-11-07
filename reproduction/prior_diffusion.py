"""Categorical contour diffusion with a CT-predicted or uniform reset law."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ddpm.models.diffusion_denoising import cosine_schedule
from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/prior_diffusion_protocol.md"
FORMAL = "799311c1c176f2d8239c619e4c6f35a67fa0ec9300c60c3194cb87d97ade58a3"
STEPS, TIMESTEPS, SAMPLES = 8000, 32, 16


def schedule(device):
    retention = torch.cat((torch.ones(1),cosine_schedule(TIMESTEPS)[2]))
    retention[-1] = 0
    return retention.to(device)


def reset_law(guide, variant):
    if variant == "uniform":
        return torch.full_like(guide,0.5)
    return 0.02 + 0.96*guide.sigmoid()


def observation_log_ratio(observed, prior, retention):
    when_zero = (1-retention)*prior
    when_one = retention+when_zero
    yes = when_one.clamp_min(1e-10).log()-when_zero.clamp_min(1e-10).log()
    no = (1-when_one).clamp_min(1e-10).log()-(1-when_zero).clamp_min(1e-10).log()
    return torch.where(observed.bool(),yes,no)


def reverse_probability(clean, observed, prior, previous, current):
    relative = current/previous
    terminal_mass = torch.where(observed.bool(),prior,1-prior)
    common = (1-relative)*terminal_mass
    likelihood_one = relative*observed+common
    likelihood_zero = relative*(1-observed)+common
    def posterior(mass):
        selected = mass*likelihood_one
        return selected/(selected+(1-mass)*likelihood_zero).clamp_min(1e-12)
    when_zero = posterior((1-previous)*prior)
    when_one = posterior(previous+(1-previous)*prior)
    return (when_zero+clean*(when_one-when_zero)).clamp(0,1)


class Denoiser(nn.Module):
    def __init__(self, parent=None):
        super().__init__()
        self.base = vote.load_net(parent) if parent is not None else vote.build()
        self.condition = nn.Conv2d(11,24,3,padding=1)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    def forward(self, image, guide, observed, prior, retention):
        xy = native.coordinates(128,image.device).permute(2,0,1)[None]*(2/127)-1
        original = torch.cat((2*image-1,xy.expand(len(image),-1,-1,-1)),1)
        time_features = torch.cat((retention,torch.cos(math.pi*retention),torch.sin(math.pi*retention)),1)
        condition = torch.cat((2*observed-1,guide.clamp(-12,12)/8,time_features.expand(-1,-1,128,128)),1)
        x = self.base.stem(original.contiguous(memory_format=torch.channels_last))+self.condition(condition)
        skips=[x]
        for layer in self.base.down:
            x=layer(x);skips.append(x)
        for layer,skip in zip(self.base.up,reversed(skips[:-1])):
            x=layer(torch.cat((F.interpolate(x,skip.shape[-2:],mode="bilinear",align_corners=False),skip),1))
        output=self.base.head(x).float()
        odds=0.5*(output[:,:4]+guide)+observation_log_ratio(observed,prior,retention)
        return odds,output[:,4:8],output[:,8:]


def train(args):
    images,truth,_,uids=native.inputs(args.data)
    _,inner,valid=native.splits(uids,args.fold)
    folder=args.align/f"fold-{args.fold}-inner-registered"
    fitted=incumbent.read(folder/"fit.json")
    if (incumbent.read(folder/"plan.json")["train_indices"]!=inner
            or fitted["checkpoint"]!=rc.hash_file(folder/"final.pt")):
        raise ValueError("Wrong aligned initialization")
    bank=args.crossfit/f"oof-{args.fold}.npz"
    owner=incumbent.read(bank.with_suffix(".json"))
    if owner["payload"]!=rc.hash_file(bank):
        raise ValueError("OOF payload differs")
    with np.load(bank,allow_pickle=False) as stored:
        ids=stored["ids"].tolist()
        if ids!=sorted(inner):
            raise ValueError("Diffusion training guide split differs")
        initial=stored["logits"]
    steps=5 if args.smoke else STEPS
    if args.smoke:
        ids=ids[:48];initial=initial[:48]
    out=args.run/(f"smoke-{args.variant}" if args.smoke else f"fold-{args.fold}-{args.variant}")
    out.mkdir(exist_ok=False)
    seed=107071+args.fold
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    image=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*fitted["applied_shift"]).float()/255
    labels=torch.from_numpy(truth[ids]).cuda().float()
    guides=torch.from_numpy(initial).cuda().float()
    distances=torch.from_numpy(np.stack([[rc.signed(mask) for mask in case] for case in truth[ids]])).cuda()
    patients=[str(uids[i]).split("_")[0] for i in ids]
    _,inverse,n=np.unique(patients,return_inverse=True,return_counts=True)
    weight=torch.from_numpy(0.5/len(ids)+0.5/(len(n)*n[inverse])).float().cuda()
    retention=schedule("cuda")
    model=Denoiser(folder/"final.pt").cuda().train()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params,avg=list(model.parameters()),list(ema.parameters())
    optimizer=torch.optim.AdamW(params,lr=0.00025,weight_decay=0.0002,fused=True)
    rc.write_json(out/"plan.json",{"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),
                  "data":rc.DATA_HASH,"fold":args.fold,"variant":args.variant,"steps":steps,"seed":seed,
                  "train_indices":ids,"excluded_validation":valid,"oof":owner["payload"],
                  "initialization":fitted["checkpoint"],"shift":fitted["applied_shift"]})
    start=time.monotonic()
    with (out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(weight,16,replacement=True)
            x,y,g,sd=image[ix],labels[ix],guides[ix],distances[ix]
            j=torch.rand(len(ix),4,1,1,device="cuda")
            x=(x.clamp_min(1e-6).pow(0.85+0.3*j[:,:1])*(0.88+0.24*j[:,1:2])).clamp(0,1)
            g=g*(0.75+0.5*j[:,2:3])+(j[:,3:4]-0.5)*2
            x,y,g,sd=[torch.rot90(value,step%4,(-2,-1)) for value in (x,y,g,sd)]
            if (step//4)%2:
                x,y,g,sd=[value.flip([-1]) for value in (x,y,g,sd)]
            t=torch.randint(1,TIMESTEPS+1,(len(ix),),device="cuda")
            a=retention[t,None,None,None];previous=retention[t-1,None,None,None]
            prior=reset_law(g,args.variant)
            observed=(torch.rand_like(y)<a*y+(1-a)*prior).float()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                logits,pred_sd,pred_votes=model(x,g,observed,prior,a)
            probability=logits.sigmoid()
            truth_reverse=reverse_probability(y,observed,prior,previous,a)
            estimate=reverse_probability(probability,observed,prior,previous,a).clamp(1e-6,1-1e-6)
            q=truth_reverse.clamp(1e-6,1-1e-6)
            divergence=q*(q.log()-estimate.log())+(1-q)*((1-q).log()-(1-estimate).log())
            boundary=0.15+0.85*4*g.sigmoid()*(1-g.sigmoid())
            local=F.binary_cross_entropy_with_logits(logits,y,reduction="none")+0.5*TIMESTEPS*divergence
            likelihood=((local*boundary).sum((-2,-1))/boundary.sum((-2,-1))).mean()
            dice=(2*(probability*y).sum((-2,-1))+1)/((probability+y).sum((-2,-1))+1)
            loss=likelihood+0.65*(1-dice.mean())+0.35*vote.mask_loss(pred_votes,vote.votes(y))
            loss=loss+0.10*F.smooth_l1_loss(pred_sd.tanh(),(sd/8).clamp(-1,1))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            optimizer.param_groups[0]["lr"]=0.00025*min(1,step/100)*(0.035+0.965*(1+math.cos(math.pi*step/steps))/2)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(avg,params,max(0.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-start}
                log.write(json.dumps(row,allow_nan=False)+"\n")
                print(args.variant,args.fold,row,flush=True)
    torch.save(ema.state_dict(),out/"final.pt")
    rc.write_json(out/"fit.json",{"plan":rc.hash_file(out/"plan.json"),"checkpoint":rc.hash_file(out/"final.pt"),
                  "seconds":time.monotonic()-start})


def best_prefix(probability,samples):
    # samples has shape [case, rank, draw, height, width].
    shape=probability.shape
    score=probability.flatten(2)
    bank=samples.flatten(3).float()
    order=torch.argsort(score,dim=2,descending=True,stable=True)
    hits=bank.gather(3,order[:,:,None].expand_as(bank)).cumsum(3)
    areas=bank.sum(3,keepdim=True)
    sizes=torch.arange(1,score.shape[2]+1,device=score.device,dtype=score.dtype)
    expected=(2*hits/(areas+sizes)).mean(2)
    empty=(areas==0).float().mean(2)
    amount=torch.cat((empty,expected),2).argmax(2)
    selected=sizes[None,None,:]<=amount[:,:,None]
    hard=torch.zeros_like(score,dtype=torch.bool).scatter_(2,order,selected)
    logits=torch.logit(score.clamp(1e-6,1-1e-6))
    ranked=logits.gather(2,order)
    left=ranked.gather(2,(amount-1).clamp(0,score.shape[2]-1)[:,:,None])
    right=ranked.gather(2,amount.clamp(0,score.shape[2]-1)[:,:,None])
    distance=(logits-0.5*(left+right)).abs().clamp_min(0.05)
    return hard.reshape(shape),torch.where(hard,distance,-distance).reshape(shape)


@torch.inference_mode()
def predict(model, images, guide, shift, variant, fold, start_step=TIMESTEPS, band_width=None):
    if start_step not in range(4,TIMESTEPS+1,4):
        raise ValueError("Sampling start must be a registered four-step jump")
    device=next(model.parameters()).device
    retention=schedule(device)
    jumps=list(range(start_step,0,-4))
    random=torch.Generator(device=device).manual_seed(93141+fold)
    half=torch.rand((TIMESTEPS//4+1,SAMPLES//2,4,128,128),generator=random,device=device)
    noise=torch.cat((half,1-half),dim=1)
    scores=[];averages=[]
    for start in range(0,len(images),2):
        x=align.registered_ct(torch.from_numpy(images[start:start+2,None]).to(device),*shift).float()/255
        g=torch.from_numpy(guide[start:start+len(x)]).to(device).float()
        batch=len(x)
        x=x.repeat_interleave(SAMPLES,0);g=g.repeat_interleave(SAMPLES,0)
        prior=reset_law(g,variant)
        anchor=(g>0).float()
        initial=retention[start_step]*anchor+(1-retention[start_step])*prior
        observed=(noise[0].repeat(batch,1,1,1)<initial).float()
        band=None
        if band_width is not None:
            source=guide[start:start+batch]>0
            arrays=np.stack([[abs(rc.signed(mask))<=band_width for mask in case] for case in source])
            band=torch.from_numpy(arrays).to(device).repeat_interleave(SAMPLES,0)
        for t in jumps:
            a=retention[t].expand(len(x),1,1,1)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda" and torch.cuda.is_bf16_supported()):
                logits,_,_=model(x,g,observed,prior,a)
            probability=reverse_probability(logits.sigmoid(),observed,prior,retention[t-4],a)
            if band is not None:
                fixed=retention[t-4]*anchor+(1-retention[t-4])*prior
                probability=torch.where(band,probability,fixed)
            noise_index=1+(TIMESTEPS-t)//4
            observed=(noise[noise_index].repeat(batch,1,1,1)<probability).float()
        mean=probability.reshape(batch,SAMPLES,4,128,128).mean(1)
        samples=observed.reshape(batch,SAMPLES,4,128,128).transpose(1,2)
        _,field=best_prefix(mean,samples)
        scores.append(field.cpu().numpy());averages.append(mean.cpu().numpy())
    return np.concatenate(scores),np.concatenate(averages)


def check(args):
    images,truth,_,uids=native.inputs(args.data)
    _,inner,valid=native.splits(uids,args.fold)
    groups=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(groups,valid,args.fold)
    ids=library+checked
    if rc.hash_file(args.formal/"assessment/outer_report.json")!=FORMAL:
        raise ValueError("Frozen reference differs")
    selected=incumbent.read(args.formal/"selection.json")["folds"][args.fold]
    owner=args.ct if selected["variant"]=="ct-adapted" else args.encoder
    record=incumbent.read(owner/f"inner-{args.fold}-{selected['variant']}.json")
    prior=args.texture/f"features-{args.fold}-inner.npz"
    new=owner/f"pred-{args.fold}-{selected['variant']}.npz"
    if (record["library"]!=library or record["check"]!=checked or record["prediction"]!=rc.hash_file(new)
            or incumbent.read(prior.with_suffix(".json"))["payload"]!=rc.hash_file(prior)):
        raise ValueError("Accepted inner guide differs")
    with np.load(prior,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Previous IDs differ")
        base=incumbent.anchor_logits(z["base"].astype(np.float32),np.unpackbits(z["old"],axis=-1).astype(bool))
    with np.load(new,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Accepted IDs differ")
        mask=np.unpackbits(z["masks"],axis=-1).astype(bool);w=selected["weight"]
        guide=incumbent.anchor_logits((1-w)*base+w*z["logits"].astype(np.float32),mask)
    model_owner=args.model_run or args.run
    folder=model_owner/f"fold-{args.fold}-{args.variant}"
    trained_source=model_owner/"source/reproduction/prior_diffusion.py" if args.model_run else Path(__file__)
    plan,fit=incumbent.read(folder/"plan.json"),incumbent.read(folder/"fit.json")
    if (plan["train_indices"]!=sorted(inner) or plan["source"]!=rc.hash_file(trained_source) or plan["steps"]!=STEPS
            or fit["plan"]!=rc.hash_file(folder/"plan.json") or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
        raise ValueError("Diffusion model identity differs")
    net=Denoiser().cuda().eval()
    net.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
    settings={"full":(TIMESTEPS,None),"partial":(16,None),"band":(TIMESTEPS,3.),"partial-band":(16,3.)}
    start_step,band_width=settings[args.sampling]
    field,mean=predict(net,images[ids],guide,plan["shift"],args.variant,args.fold,start_step,band_width)
    if not np.isfinite(field).all() or not np.isfinite(mean).all():raise ValueError("Nonfinite diffusion output")
    count=len(library)
    grid=[float(metrics.dice(truth[library],(1-v)*guide[:count]+v*field[:count]>0).mean()) for v in incumbent.WEIGHTS]
    weight=incumbent.WEIGHTS[int(np.argmax(grid))]
    suffix=args.variant if args.sampling=="full" else f"{args.variant}-{args.sampling}"
    protocol=PROTOCOL if args.sampling=="full" else PROTOCOL.with_name("partial_diffusion_protocol.md")
    lock={"fold":args.fold,"variant":args.variant,"sampling":args.sampling,"library":library,"grid":grid,"weight":weight,
          "model":fit["checkpoint"],"protocol":rc.hash_file(protocol),"source":rc.hash_file(__file__)}
    rc.write_json(args.run/f"selection-{args.fold}-{suffix}.json",lock)
    pred=(1-weight)*guide[count:]+weight*field[count:]>0
    if band_width is not None:
        outside=np.stack([[abs(rc.signed(m))>band_width for m in row] for row in mask[count:]])
        if (not np.array_equal(pred[outside],mask[count:][outside])
                or not np.array_equal((field[count:]>0)[outside],mask[count:][outside])):
            raise ValueError("A protected pixel changed in the constrained decision")
    score=lambda m:float(metrics.dice(truth[checked],m).mean())
    summary={"reference":score(mask[count:]),"mean_threshold":score(mean[count:]>0.5),
             "prefix":score(field[count:]>0),"candidate":score(pred)}
    path=args.run/f"pred-{args.fold}-{suffix}.npz"
    np.savez_compressed(path,ids=checked,masks=np.packbits(pred,axis=-1),mean=mean[count:].astype(np.float16))
    rc.write_json(args.run/f"inner-{args.fold}-{suffix}.json",
                  {**lock,"check":checked,"scores":summary,"prediction":rc.hash_file(path)})
    print("PRIOR_DIFFUSION_CHECK",args.fold,args.variant,weight,summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","check"))
    for key in ("run","data","align","crossfit"):
        parser.add_argument("--"+key,type=Path,required=True)
    for key in ("formal","encoder","ct","texture"):
        parser.add_argument("--"+key,type=Path)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--variant",choices=("uniform","guided"),required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--model-run",type=Path)
    parser.add_argument("--sampling",choices=("full","partial","band","partial-band"),default="full")
    args=parser.parse_args()
    if args.phase=="fit" and (args.model_run is not None or args.sampling!="full"):
        parser.error("The partial diffusion study reuses an existing fitted model")
    torch.set_num_threads(3);cv2.setNumThreads(1)
    {"fit":train,"check":check}[args.phase](args)
