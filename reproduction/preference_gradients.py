"""Paired ordinary and PCGrad continuation of the aligned preference network."""
import argparse
import copy
import json
import inspect
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/gradient_protocol.md"
STEPS = 4000


def projected_mean(gradients, orders):
    projected = gradients.clone()
    for position in range(orders.shape[1]):
        other = gradients[orders[:, position]]
        dot = (projected * other).sum(dim=1)
        coefficient = dot.clamp_max(0) / other.square().sum(dim=1).clamp_min(1e-12)
        projected = projected - coefficient[:, None] * other
    return projected.mean(dim=0)


def fit(args, augmentation=None, protocol=PROTOCOL):
    images, labels, alpha, uids = native.inputs(args.data)
    _, inner, valid = native.splits(uids, args.fold)
    previous = args.align / f"fold-{args.fold}-inner-registered"
    identity = incumbent.read(previous / "fit.json")
    plan = incumbent.read(previous / "plan.json")
    if plan["train_indices"] != inner or identity["checkpoint"] != rc.hash_file(previous / "final.pt"):
        raise ValueError("Aligned initialization training identity differs")
    ids = inner[:48] if args.smoke else inner
    steps = 5 if args.smoke else STEPS
    folder = args.run / (f"smoke-{args.variant}" if args.smoke else f"fold-{args.fold}-{args.variant}")
    folder.mkdir(exist_ok=False)
    seed = 77171 + args.fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    permutation_rng = np.random.default_rng(seed+90000)
    image = align.registered_ct(torch.from_numpy(images[ids, None]).cuda(), *identity["applied_shift"])
    masks = torch.from_numpy(labels[ids]).cuda()
    signed = np.stack([[rc.signed(mask) for mask in case] for case in labels[ids]]).astype(np.float16)
    targets = torch.cat((masks.float(), vote.votes(masks), torch.from_numpy(signed).cuda().float()), 1).half()
    centers = native.rater_center(masks)
    centers = torch.where((centers-64.5).abs().amax(1, keepdim=True)>4, centers, 64.5)
    patients = [str(uids[i]).split("_")[0] for i in ids]
    mixer = augmentation(image, labels[ids], alpha[ids], patients, seed) if augmentation else None
    _, inverse, counts = np.unique(patients, return_inverse=True, return_counts=True)
    sampling = torch.from_numpy(0.65/len(ids)+0.35/(len(counts)*counts[inverse])).float().cuda()
    model = vote.load_net(previous / "final.pt").train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    parameters, averaged = list(model.parameters()), list(ema.parameters())
    shared = [parameter for name, parameter in model.named_parameters() if not name.startswith("head.")]
    optimizer = torch.optim.AdamW(parameters, lr=0.00012, weight_decay=0.0001, fused=True)
    rc.write_json(folder / "plan.json", {
        "source": rc.hash_file(__file__), "protocol": rc.hash_file(protocol), "fold": args.fold,
        "augmentation": augmentation.__name__ if augmentation else None,
        "augmentation_source": rc.hash_file(inspect.getfile(augmentation)) if augmentation else None,
        "variant": args.variant, "seed": seed, "steps": steps, "train_indices": ids,
        "inner_valid_indices": valid, "initialization": identity["checkpoint"],
        "shift": identity["applied_shift"], "dataset": rc.DATA_HASH,
    })
    start = time.monotonic()
    with (folder / "train.jsonl").open("x", buffering=1) as log:
        for step in range(1, steps+1):
            selected = torch.multinomial(sampling, 20, replacement=True)
            random = torch.rand(20, 9, device="cuda")
            angle = torch.where(random[:, 0]<0.65, (random[:, 1]*4).floor()*math.pi/2,
                                (2*random[:, 1]-1)*math.pi)
            reflection = torch.where(random[:, 2]<0.5, -1.0, 1.0)
            scale = 0.65+0.58*random[:, 3]
            center = centers[selected]+(random[:, 4:6]-0.5)*1.2
            grid, _ = align.transform_grids(center, scale, angle, reflection)
            pixels = mixer(selected, step) if mixer else image[selected]
            x = F.grid_sample(pixels.float()/255, grid, align_corners=True, padding_mode="border")
            x = (x.clamp_min(1e-6).pow(0.85+0.3*random[:, 6, None, None, None])
                 *(0.87+0.26*random[:, 7, None, None, None])
                 +0.07*(random[:, 8, None, None, None]-0.5)).clamp(0, 1)
            y = F.grid_sample(targets[selected].float(), grid, align_corners=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                output = model(x)
            pref = output[:, :4]
            task_losses = [vote.mask_loss(pref[:, rank:rank+1], y[:, rank:rank+1]) for rank in range(4)]
            probability = (2*pref).sigmoid()
            area = F.smooth_l1_loss((probability.sum((-2, -1))+1).log(), (y[:, :4].sum((-2, -1))+1).log())
            moment = F.smooth_l1_loss(native.rater_center(probability), native.rater_center(y[:, :4]))
            distance = F.smooth_l1_loss(output[:, 4:8].tanh(),
                                        (y[:, 8:]/(scale[:, None, None, None]*8)).clamp(-1, 1))
            auxiliary = 0.75*vote.mask_loss(output[:, 8:], y[:, 4:8])+0.15*distance+0.20*area+0.025*moment
            loss = torch.stack(task_losses).mean()+auxiliary
            optimizer.zero_grad(set_to_none=True)
            correction = None
            if args.variant == "pcgrad":
                gradients = []
                for task in task_losses:
                    values = torch.autograd.grad(task, shared, retain_graph=True)
                    gradients.append(torch.cat([value.detach().flatten() for value in values]))
                gradients = torch.stack(gradients)
                orders = torch.tensor(
                    np.stack([permutation_rng.permutation([j for j in range(4) if j != rank])
                              for rank in range(4)]), device="cuda"
                )
                correction = projected_mean(gradients, orders)-gradients.mean(dim=0)
            loss.backward()
            if correction is not None:
                begin = 0
                for parameter in shared:
                    end = begin+parameter.numel()
                    parameter.grad.add_(correction[begin:end].view_as(parameter))
                    begin = end
            torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
            optimizer.param_groups[0]["lr"] = 0.00012*min(1, step/100)*(0.02+0.98*(1+math.cos(math.pi*step/steps))/2)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(averaged, parameters, max(0.005, 1/step))
            if step == 1 or step % 250 == 0 or step == steps:
                record = {"step": step, "loss": loss.item(), "seconds": time.monotonic()-start}
                if correction is not None:
                    record["projected_delta_norm"] = float(correction.norm())
                log.write(json.dumps(record, allow_nan=False)+"\n")
                print(args.fold, args.variant, record, flush=True)
    torch.save(ema.state_dict(), folder / "final.pt")
    rc.write_json(folder / "fit.json", {"plan": rc.hash_file(folder / "plan.json"),
                  "checkpoint": rc.hash_file(folder / "final.pt"), "seconds": time.monotonic()-start})


def check(args):
    images, truth, _, uids = native.inputs(args.data)
    _, inner, valid = native.splits(uids, args.fold)
    group = type("Groups", (), {"series_uid": uids})()
    library, checked = polar.calibration_split(group, valid, args.fold)
    ids = library+checked
    path = args.texture / f"features-{args.fold}-inner.npz"
    meta = incumbent.read(path.with_suffix(".json"))
    if meta["indices"] != ids or meta["payload"] != rc.hash_file(path):
        raise ValueError("Reference cache identity differs")
    with np.load(path, allow_pickle=False) as stored:
        original = np.unpackbits(stored["old"], axis=-1).astype(bool)
        guide = incumbent.anchor_logits(stored["base"].astype(np.float32), original)
        centers = stored["centers"]
    folder = args.run / f"fold-{args.fold}-{args.variant}"
    plan = incumbent.read(folder / "plan.json")
    receipt = incumbent.read(folder / "fit.json")
    if (plan["source"] != rc.hash_file(__file__) or plan["train_indices"] != inner
            or receipt["plan"] != rc.hash_file(folder / "plan.json")
            or receipt["checkpoint"] != rc.hash_file(folder / "final.pt")):
        raise ValueError("Gradient experiment model identity differs")
    model = vote.load_net(folder / "final.pt")
    logits = align.predict(model, images[ids, None], centers, plan["shift"])[:, :4]
    count = len(library)
    grid = [float(scoring.dice(truth[library], (1-weight)*guide[:count]+weight*logits[:count]>0).mean())
            for weight in incumbent.WEIGHTS]
    weight = incumbent.WEIGHTS[int(np.argmax(grid))]
    choice = {"fold": args.fold, "variant": args.variant, "weight": weight, "library": library,
              "grid": grid, "source": rc.hash_file(__file__), "checkpoint": receipt["checkpoint"]}
    rc.write_json(args.run / f"selection-{args.fold}-{args.variant}.json", choice)
    masks = (1-weight)*guide+weight*logits>0
    scores = {"reference": float(scoring.dice(truth[checked], original[count:]).mean()),
              "standalone": float(scoring.dice(truth[checked], logits[count:]>0).mean()),
              "candidate": float(scoring.dice(truth[checked], masks[count:]).mean())}
    payload = args.run / f"pred-{args.fold}-{args.variant}.npz"
    np.savez_compressed(payload, ids=ids, masks=np.packbits(masks, axis=-1), logits=logits.astype(np.float16))
    rc.write_json(args.run / f"inner-{args.fold}-{args.variant}.json",
                  {**choice, "check": checked, "scores": scores, "prediction": rc.hash_file(payload)})
    print("GRADIENT_CHECK", args.fold, args.variant, weight, scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "check"))
    for name in ("run", "data", "align"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--texture", type=Path)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--variant", choices=("ordinary", "pcgrad"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    {"fit": fit, "check": check}[args.phase](args)
