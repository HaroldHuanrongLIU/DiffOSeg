"""Image-conditioned rank contours for correcting frozen LIDC diffusion masks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from datasets.lidc import BaseDataset
from reproduction.data import patient_folds


DATA_HASH = "6c401c96b0e1737d43fdccfc446cb0c3601473d3b7668ad4a68d22ec0c1ee81b"
STEPS = 4000
BATCH = 24


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def signed(mask):
    """Truncated distance to the interface between foreground and background."""
    value = np.ascontiguousarray(mask, dtype=np.uint8)
    if not value.any():
        return np.full(value.shape, -8, dtype=np.float32)
    if value.all():
        return np.full(value.shape, 8, dtype=np.float32)
    inside = cv2.distanceTransform(value, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - value, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance = np.where(value, inside - 0.5, 0.5 - outside)
    return np.clip(distance, -8, 8)


class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout),
        )
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Conv2d(cin, cout, 1, stride=stride, bias=False))

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


class RankContours(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [24, 40, 72, 112, 160]
        self.stem = Residual(3, widths[0])
        self.down = nn.ModuleList(nn.Sequential(Residual(a, b, 2), Residual(b, b))
                                  for a, b in zip(widths, widths[1:]))
        self.up = nn.ModuleList(Residual(a + b, a) for a, b in reversed(list(zip(widths, widths[1:]))))
        self.head = nn.Conv2d(widths[0], 8, 1)

    def forward(self, image):
        h, w = image.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device),
                              torch.linspace(-1, 1, w, device=image.device), indexing="ij")
        coords = torch.stack((x, y)).expand(len(image), -1, -1, -1)
        current = self.stem(torch.cat((image * 2 - 1, coords), dim=1).contiguous(memory_format=torch.channels_last))
        features = [current]
        for layer in self.down:
            current = layer(current)
            features.append(current)
        for layer, skip in zip(self.up, reversed(features[:-1])):
            current = F.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = layer(torch.cat((current, skip), dim=1))
        return self.head(current).float()


def split_for(data, fold):
    outer = patient_folds(data.series_uid)[fold]
    patients = np.array(outer["train_patients"])
    shuffled = np.random.RandomState(7310 + fold).permutation(patients)
    valid = set(shuffled[:math.ceil(0.2 * len(patients))])
    inner_train, inner_valid = [], []
    for idx in outer["train_indices"]:
        patient = str(data.series_uid[idx]).split("_")[0]
        (inner_valid if patient in valid else inner_train).append(idx)
    return outer, inner_train, inner_valid


def tensors(data, indices, distance):
    images = np.stack([data.images[i] for i in indices])
    masks = np.stack([data.mask_labels[i] for i in indices]).astype("uint8")
    image = torch.from_numpy(images[:, None]).cuda()
    truth = torch.from_numpy(masks).cuda()
    if not distance:
        return image, truth
    fields = np.stack([[signed(m) for m in item] for item in masks]).astype("float16")
    return image, truth, torch.from_numpy(fields).cuda()


def augment(images, masks, fields):
    size = len(images)
    device = images.device
    random = torch.rand(size, 8, device=device)
    angle = (random[:, 0] * 2 - 1) * math.pi
    scale = 0.88 + 0.24 * random[:, 1]
    flip = torch.where(random[:, 2] < 0.5, -1, 1)
    c, s = scale * angle.cos(), scale * angle.sin()
    affine = torch.stack((c * flip, -s, (random[:, 3] - 0.5) * 0.06,
                          s * flip, c, (random[:, 4] - 0.5) * 0.06), dim=1).reshape(-1, 2, 3)
    grid = F.affine_grid(affine, images.shape, align_corners=False)
    image = F.grid_sample(images.float() / 255, grid, padding_mode="border", align_corners=False)
    image = image * (0.85 + 0.30 * random[:, 5, None, None, None]) + (random[:, 6, None, None, None] - 0.5) * 0.08
    target = F.grid_sample(masks.float(), grid, align_corners=False)
    dist = F.grid_sample(fields.float(), grid, padding_mode="border", align_corners=False)
    return image.clamp(0, 1), target, (dist / scale[:, None, None, None] / 8).clamp(-1, 1)


@torch.inference_mode()
def predict(net, uint8_images):
    image = uint8_images.float() / 255
    logits = 0
    for flip in (False, True):
        reflected = image.flip((-1,)) if flip else image
        for k in range(4):
            view = torch.rot90(reflected, k, (-2, -1))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                pred = net(view)[:, :4]
            restored = torch.rot90(pred, -k, (-2, -1))
            logits = logits + (restored.flip((-1,)) if flip else restored)
    return logits > 0


def train(args):
    import copy

    if hash_file(args.data) != DATA_HASH:
        raise ValueError("LIDC dataset hash differs")
    args.out.mkdir(parents=True, exist_ok=False)
    data = BaseDataset(str(args.data))
    outer, inner_train, valid = split_for(data, args.fold)
    train_indices = outer["train_indices"] if args.mode == "full" else inner_train
    if args.mode == "smoke":
        train_indices, valid = train_indices[:48], valid[:8]
    nsteps = 4 if args.mode == "smoke" else STEPS
    seed = 91103 + args.fold
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    dev = torch.cuda.get_device_name()
    provenance = {"mode": args.mode, "fold": args.fold, "seed": seed, "steps": nsteps, "batch": BATCH,
                  "dataset_sha256": DATA_HASH, "training_source": hash_file(__file__),
                  "protocol_sha256": hash_file(Path(__file__).resolve().parents[1] / "docs/lidc_boundary/protocol.md"),
                  "train_indices": train_indices, "inner_valid_indices": valid if args.mode != "full" else [],
                  "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
                  "gpu": dev, "torch": str(torch.__version__), "cuda": torch.version.cuda}
    write_json(args.out / "plan.json", provenance)
    if args.mode == "full":
        gate = json.loads(args.gate.read_text())
        if (gate["mode"], gate["fold"], gate["training_source"], gate["protocol_sha256"], gate["dataset_sha256"], gate["passed"]) != (
                "inner", args.fold, provenance["training_source"], provenance["protocol_sha256"], DATA_HASH, True):
            raise ValueError("The matching inner-validation gate is required")
    images, masks, fields = tensors(data, train_indices, True)
    net = RankContours().cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    ema_params = list(ema.parameters())
    params = list(net.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=1e-4, fused=True)
    started = time.monotonic()
    with (args.out / "train.jsonl").open("x", buffering=1) as log:
        for step in range(1, nsteps + 1):
            lr = 0.001 * min(1, step / 100) * (0.02 + 0.98 * (1 + math.cos(math.pi * step / nsteps)) / 2)
            optimizer.param_groups[0]["lr"] = lr
            ids = torch.randint(len(images), (BATCH,), device="cuda")
            x, y, distance = augment(images[ids], masks[ids], fields[ids])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                out = net(x)
            logits = out[:, :4]
            prob = logits.sigmoid()
            dice = (2 * (prob * y).sum((-2, -1)) + 1) / ((prob + y).sum((-2, -1)) + 1)
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = 1 - dice.mean() + (bce * (1 + 4 * y)).mean() + 0.20 * F.smooth_l1_loss(out[:, 4:].tanh(), distance)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema_params, params, max(0.005, 1 / step))
            if step == 1 or step % 100 == 0 or step == nsteps:
                row = {"step": step, "loss": loss.item(), "seconds": time.monotonic() - started}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(row, flush=True)
    torch.save(ema.state_dict(), args.out / "final.pt")
    result = {key: provenance[key] for key in ("mode", "fold", "training_source", "dataset_sha256", "protocol_sha256")}
    result.update(steps=nsteps, checkpoint_sha256=hash_file(args.out / "final.pt"), seconds=time.monotonic()-started)
    if args.mode != "full":
        image, truth = tensors(data, valid, False)
        prediction = torch.cat([predict(ema, chunk) for chunk in image.split(12)])
        area = (prediction.float() + truth).sum((-2, -1))
        score = torch.where(area > 0, 2 * (prediction * truth).sum((-2, -1)) / area.clamp_min(1), 1.)
        result.update(aligned_dice=score.mean().item(), each=score.mean(0).tolist(), samples=len(valid))
        result["passed"] = bool(torch.isfinite(score).all()) and (args.mode == "smoke" or result["aligned_dice"] >= 0.80)
    write_json(args.out / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=range(4))
    parser.add_argument("--mode", required=True, choices=["smoke", "inner", "full"])
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.mode == "full" and args.gate is None:
        parser.error("--full training requires --gate")
    train(args)
