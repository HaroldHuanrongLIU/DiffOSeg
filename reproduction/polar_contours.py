"""Polar, area-ranked contour regression on fixed LIDC nodule patches."""
import argparse
import copy
import json
import math
from pathlib import Path
import socket
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets.lidc import BaseDataset
from reproduction import rank_contours as rc


ANGLES, BINS, BATCH, STEPS = 96, 112, 48, 4000
ORIGIN = 64.5
PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_polar/protocol.md"


class AngularBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layer = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, padding_mode="circular"),
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, width, 1),
        )

    def forward(self, x):
        return x + self.layer(x)


class PolarContours(nn.Module):
    def __init__(self):
        super().__init__()
        width = 96
        self.features = nn.Sequential(nn.Conv1d(BINS, width, 1),
                                      *(AngularBlock(width, d) for d in (1, 2, 4, 8, 16, 4, 1)),
                                      nn.GroupNorm(8, width), nn.SiLU())
        self.head = nn.Conv1d(width, 5, 1)

    def forward(self, rays):
        out = self.head(self.features((rays - 0.25) * 4)).float()
        return 7 * F.softplus(out[:, :4] + 1.5), 0.35 + F.softplus(out[:, 4:] + 0.5)


def grid(n, device, augment=False):
    angle = torch.arange(ANGLES, device=device) * (2 * math.pi / ANGLES)
    radius = (torch.arange(BINS, device=device) + 0.5) * 0.5
    if augment:
        u = torch.rand(n, 5, device=device)
        angle = u[:, 0, None] * 2 * math.pi + torch.where(u[:, 1, None] < 0.5, -angle, angle)
        lengths = (0.88 + 0.24 * u[:, 2, None, None]) * radius
        x0 = ORIGIN + (u[:, 3, None, None] - 0.5) * 2.5
        y0 = ORIGIN + (u[:, 4, None, None] - 0.5) * 2.5
    else:
        angle = angle.expand(n, -1)
        lengths, x0, y0 = radius, ORIGIN, ORIGIN
    x = x0 + angle.cos()[:, :, None] * lengths
    y = y0 + angle.sin()[:, :, None] * lengths
    return torch.stack((x, y), dim=-1) * (2 / 127) - 1


def read_rays(images, coords):
    return F.grid_sample(images.float() / 255, coords, align_corners=True)[:, 0].transpose(1, 2).contiguous()


def mask_radii(masks, coords):
    occupancy = F.grid_sample(masks.float(), coords, align_corners=True)
    radius = (torch.arange(BINS, device=coords.device) + 0.5) * 0.5
    return (occupancy * radius).sum(-1).clamp_min(0).sqrt()


@torch.inference_mode()
def predict(model, images):
    values = []
    for batch in images.split(32):
        x = read_rays(batch, grid(len(batch), batch.device))
        a, b = model(x)
        c, d = model(x.flip((-1,)))
        values.append(torch.cat(((a + c.flip((-1,))) / 2, (b + d.flip((-1,))) / 2), dim=1).cpu().numpy())
    out = np.concatenate(values)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite polar prediction")
    return out[:, :4], out[:, 4:]


Y, X = np.indices((128, 128), dtype=np.float32) - ORIGIN
RADIUS = np.hypot(X, Y)
ANGLE = (np.arctan2(Y, X) * (ANGLES / (2 * np.pi))) % ANGLES
LEFT = np.floor(ANGLE).astype(np.int32) % ANGLES
FRACTION = ANGLE - np.floor(ANGLE)


def angular_map(profiles):
    """Periodic, linearly interpolated angular values on the Cartesian pixel grid."""
    return profiles[..., LEFT] * (1 - FRACTION) + profiles[..., (LEFT + 1) % ANGLES] * FRACTION


def calibration_split(data, valid, fold):
    patient = lambda i: str(data.series_uid[i]).split("_")[0]
    names = sorted({patient(i) for i in valid})
    order = np.random.RandomState(4680 + fold).permutation(names)
    first = set(order[:len(order) // 2])
    return [i for i in valid if patient(i) in first], [i for i in valid if patient(i) not in first]


def load_net(path):
    model = PolarContours().cuda().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def fit(args):
    if rc.hash_file(args.data) != rc.DATA_HASH:
        raise ValueError("Unexpected LIDC input")
    data = BaseDataset(str(args.data))
    outer, inner, valid = rc.split_for(data, args.fold)
    ids = outer["train_indices"] if args.mode == "full" else inner
    steps = STEPS
    if args.mode == "smoke":
        ids, valid, steps = ids[:80], valid[:8], 5
    source, plan_hash = rc.hash_file(__file__), rc.hash_file(PROTOCOL)
    if args.mode == "full":
        gate = json.loads(args.gate.read_text())
        if (gate["fold"], gate["phase"], gate["source"], gate["protocol"], gate["passed"]) != (
                args.fold, "inner", source, plan_hash, True):
            raise ValueError("Missing corresponding inner gate")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(42901 + args.fold)
    torch.cuda.manual_seed_all(42901 + args.fold)
    torch.set_float32_matmul_precision("high")
    images, masks = rc.tensors(data, ids, False)
    model = PolarContours().cuda()
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    weights, smooth = list(model.parameters()), list(ema.parameters())
    opt = torch.optim.AdamW(weights, lr=0.001, fused=True, weight_decay=0.001)
    started = time.monotonic()
    meta = {"phase": args.mode, "fold": args.fold, "steps": steps, "source": source, "protocol": plan_hash,
            "data_sha256": rc.DATA_HASH, "seed": 42901 + args.fold, "host": socket.gethostname(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "train_indices": ids, "inner_valid_indices": valid if args.mode != "full" else []}
    rc.write_json(args.out / "plan.json", meta)
    with (args.out / "train.jsonl").open("x", buffering=1) as log:
        for step in range(1, steps + 1):
            ix = torch.randint(len(ids), (BATCH,), device="cuda")
            g = grid(BATCH, "cuda", True)
            x = read_rays(images[ix], g)
            jitter = torch.rand(BATCH, 2, device="cuda")
            x = (x * (0.85 + 0.3*jitter[:, 0, None, None]) + (jitter[:, 1, None, None]-0.5)*0.08).clamp(0, 1)
            target = mask_radii(masks[ix], g)
            radius, sigma = model(x)
            loss = F.smooth_l1_loss(radius, target) + 0.35 * F.smooth_l1_loss(
                sigma.log(), target.std(1, keepdim=True, unbiased=False).clamp_min(0.4).log())
            opt.param_groups[0]["lr"] = 0.001 * min(1, step/80) * (0.02 + 0.98 * (1+math.cos(math.pi*step/steps))/2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weights, 5, error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(smooth, weights, max(0.005, 1/step))
            if step == 1 or step % 100 == 0 or step == steps:
                row = {"step": step, "loss": loss.item(), "seconds": time.monotonic()-started}
                log.write(json.dumps(row, allow_nan=False)+"\n")
                print(row, flush=True)
    torch.save(ema.state_dict(), args.out / "final.pt")
    result = {k:meta[k] for k in ("phase","fold","steps","source","protocol","data_sha256")}
    result.update(checkpoint=rc.hash_file(args.out / "final.pt"), seconds=time.monotonic()-started)
    if args.mode != "full":
        images, truth = rc.tensors(data, valid, False)
        profiles, _ = predict(ema, images)
        scores = []
        for pred, actual in zip(profiles, truth.cpu().numpy()):
            masks = angular_map(pred) > RADIUS
            den = masks.sum((1,2)) + actual.sum((1,2))
            scores.append(np.divide(2*(masks*actual).sum((1,2)), den, out=np.ones(4), where=den>0))
        result["inner_dice"] = float(np.mean(scores))
        result["passed"] = args.mode == "smoke" or result["inner_dice"] >= 0.78
    rc.write_json(args.out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(4), required=True)
    p.add_argument("--mode", choices=["smoke", "inner", "full"], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gate", type=Path)
    args = p.parse_args()
    if args.mode == "full" and args.gate is None:
        p.error("Full training requires --gate")
    fit(args)
