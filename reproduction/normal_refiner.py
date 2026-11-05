"""Refine a coarse contour from bilateral CT readings along its local normal."""
import argparse
import copy
import json
import math
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from datasets.lidc import BaseDataset
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_normal/protocol.md"
BATCH, STEPS = 20, 5000
OFFSETS = (-6., -3., -1.5, 1.5, 3., 6.)


def profile_features(image, guide, use_profile):
    """Positive normal points inward. Values outside the patch use border replication."""
    b, _, h, w = image.shape
    padded = F.pad(F.avg_pool2d(guide, 3, stride=1, padding=1), (1, 1, 1, 1), mode="replicate")
    dx = (padded[:,:,1:-1,2:] - padded[:,:,1:-1,:-2]) * 0.5
    dy = (padded[:,:,2:,1:-1] - padded[:,:,:-2,1:-1]) * 0.5
    length = (dx.square() + dy.square()).clamp_min(1e-5).sqrt()
    nx, ny = dx / length, dy / length
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device),
                            torch.linspace(-1, 1, w, device=image.device), indexing="ij")
    offsets = image.new_tensor(OFFSETS)[None,:,None,None]
    gx = xx + (2 / (w-1)) * nx * offsets
    gy = yy + (2 / (h-1)) * ny * offsets
    grid = torch.stack((gx, gy), -1).reshape(b, len(OFFSETS)*h, w, 2)
    sampled = F.grid_sample(image, grid, align_corners=True, padding_mode="border").reshape(b, len(OFFSETS), h, w)
    local = F.avg_pool2d(image, 9, stride=1, padding=4, count_include_pad=False)
    coords = torch.stack((xx, yy)).expand(b, -1, -1, -1)
    return torch.cat((2*image-1, 4*(image-local), coords, guide/8, nx, ny,
                      4*(sampled-image)*use_profile), 1)


class BoundaryRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        widths = (24, 40, 64, 96)
        self.stem = rc.Residual(7+len(OFFSETS), widths[0])
        self.down = nn.ModuleList(nn.Sequential(rc.Residual(a,b,2), rc.Residual(b,b))
                                  for a,b in zip(widths,widths[1:]))
        self.up = nn.ModuleList(rc.Residual(a+b,a) for a,b in reversed(list(zip(widths,widths[1:]))))
        self.output = nn.Conv2d(widths[0], 8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, guide, use_profile=True):
        features = self.stem(profile_features(image,guide,use_profile).contiguous(memory_format=torch.channels_last))
        saved = [features]
        for layer in self.down:
            features = layer(features)
            saved.append(features)
        for layer, skip in zip(self.up, reversed(saved[:-1])):
            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = layer(torch.cat((features,skip),1))
        return guide + 12 * (self.output(features).float()/6).tanh()


def corrupt_guide(distances):
    """Only the guide moves. The eight supervised distances stay in the image frame."""
    b, _, h, w = distances.shape
    device = distances.device
    selected = torch.randint(8, (b,), device=device)
    guide = distances[torch.arange(b,device=device),selected][:,None]
    yy, xx = torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing="ij")
    noise = F.interpolate(torch.randn(b,2,7,7,device=device), (h,w), mode="bicubic", align_corners=False)
    jitter = noise*1.6 + torch.randn(b,2,1,1,device=device)*1.3
    coords = torch.stack((xx+2*jitter[:,0]/(w-1),yy+2*jitter[:,1]/(h-1)),-1)
    warped = F.grid_sample(guide,coords,align_corners=True,padding_mode="border")
    dilation = torch.randn(b,1,1,1,device=device).clamp(-2,2)*1.35
    return (F.avg_pool2d(warped,3,1,1,count_include_pad=False)+dilation).clamp(-8,8)


def image_guide(vote_logits):
    """A single image-derived middle-support contour; no expert masks enter inference."""
    return np.stack([0.5*(rc.signed(v[1]>0)+rc.signed(v[2]>0)) for v in vote_logits])[:,None]


def load_model(path):
    model = BoundaryRefiner().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    return model.eval()


@torch.inference_mode()
def predict(model,images,guides,use_profile=True):
    result = []
    for offset in range(0,len(images),16):
        x = torch.as_tensor(images[offset:offset+16], device="cuda").float()/255
        g = torch.as_tensor(guides[offset:offset+16], device="cuda").float()
        total = 0
        for flip in (False,True):
            im,guide = (x.flip([-1]),g.flip([-1])) if flip else (x,g)
            for k in range(4):
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    value = model(torch.rot90(im,k,(-2,-1)),torch.rot90(guide,k,(-2,-1)),use_profile)
                value = torch.rot90(value,-k,(-2,-1))
                total = total + (value.flip([-1]) if flip else value)
        result.append((total / (8*0.65)).cpu().numpy())
    out = np.concatenate(result)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite boundary logits")
    return out


def fit(args):
    if rc.hash_file(args.data) != rc.DATA_HASH:
        raise ValueError("LIDC input hash differs")
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    data = BaseDataset(str(args.data))
    outer,train,valid = rc.split_for(data,args.fold)
    mode = "inner" if args.mode=="smoke" else args.mode
    ids = outer["train_indices"] if mode=="full" else train
    steps = STEPS
    if args.mode=="smoke":
        ids,steps = ids[:48],5
    seed = args.seed_base+args.fold
    if args.mode=="full":
        gate = json.loads(args.gate.read_text())
        if (gate["fold"],gate["mode"],gate["source"],gate["protocol"],gate["passed"],gate["seed"]) != (
                args.fold,"inner",rc.hash_file(__file__),rc.hash_file(PROTOCOL),True,seed):
            raise ValueError("Corresponding inner model did not pass")
    args.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    image = torch.from_numpy(np.stack([data.images[i] for i in ids])[:,None]).cuda()
    experts = np.array([data.mask_labels[i] for i in ids],bool)
    votes = experts.sum(1)[:,None]>=np.arange(1,5)[None,:,None,None]
    truth = np.concatenate((experts,votes),1)
    fields = np.stack([[rc.signed(m) for m in item] for item in truth]).astype("float16")
    targets = torch.cat((torch.from_numpy(truth).cuda().half(),torch.from_numpy(fields).cuda()),1)
    meta = {"fold":args.fold,"mode":args.mode,"steps":steps,"batch":BATCH,"seed":seed,"source":rc.hash_file(__file__),
            "protocol":rc.hash_file(PROTOCOL),"data_sha256":rc.DATA_HASH,"gpu":torch.cuda.get_device_name(),
            "host":socket.gethostname(),"torch":str(torch.__version__),"cuda":torch.version.cuda}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if mode=="inner" else []})
    net = BoundaryRefiner().cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).requires_grad_(False).eval()
    params,averaged = list(net.parameters()),list(ema.parameters())
    opt = torch.optim.AdamW(params,lr=0.0008,fused=True,weight_decay=1e-4)
    start = time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix = torch.randint(len(image),(BATCH,),device="cuda")
            g,scale = vote.sampling_grid(BATCH,"fullfield","cuda",True)
            x = F.grid_sample(image[ix].float()/255,g,align_corners=True,padding_mode="border")
            y = F.grid_sample(targets[ix].float(),g,align_corners=True,padding_mode="border")
            target = (y[:,8:]/scale[:,None]).clamp(-8,8)
            guide = corrupt_guide(target)
            jitter = torch.rand(BATCH,2,1,1,device="cuda")
            x = (x*(0.82+0.36*jitter[:,:1])+(jitter[:,1:]-0.5)*0.08).clamp(0,1)
            keep = (torch.rand(BATCH,1,1,1,device="cuda")>=0.25).float()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                estimate = net(x,guide,keep)
            overlap = vote.mask_loss(estimate/0.65,y[:,:8])
            weight = (1-target.abs()/8).clamp(0,1)
            error = F.smooth_l1_loss(estimate,target,reduction="none")
            loss = overlap+0.18*(error*weight).sum()/weight.sum().clamp_min(1)
            opt.param_groups[0]["lr"]=0.0008*min(step/100,1)*(0.025+0.975*(1+math.cos(math.pi*step/steps))/2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(averaged,params,max(0.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-start}
                log.write(json.dumps(row,allow_nan=False)+"\n")
                print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    rc.write_json(args.out/"fit.json",{**meta,"checkpoint":rc.hash_file(args.out/"final.pt"),"seconds":time.monotonic()-start})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--mode",choices=("inner","full","smoke"),required=True)
    parser.add_argument("--seed-base",type=int,default=117901)
    parser.add_argument("--gate",type=Path)
    args=parser.parse_args()
    if args.mode=="full" and args.gate is None:
        parser.error("Full training requires --gate")
    fit(args)
