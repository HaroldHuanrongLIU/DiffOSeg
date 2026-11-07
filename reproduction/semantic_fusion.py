"""Fuse independent frozen visual features in the original single-slice Hiera decoder."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import semantic_preference as semantic
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import OFFSETS, mean_saved_logits

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/semantic_fusion_protocol.md"
MODES = ("self", "dino")


def ct_and_split(args):
    if rc.hash_file(args.data) != rc.DATA_HASH:
        raise ValueError("Unexpected LIDC source")
    with args.data.open("rb") as stream:
        cases = list(pickle.load(stream).values())
    return np.stack([c["image"] for c in cases]), [c["series_uid"] for c in cases]


@torch.inference_mode()
def encode(args):
    images, _ = ct_and_split(args)
    for fold in (0, 1):
        receipt = json.loads((args.align / f"fold-{fold}-inner-registered/fit.json").read_text())
        if receipt["applied_shift"] != [-1, -1]:
            raise ValueError("Different fitting-patient CT registration")
    model = semantic.DinoSmall(args.dino).cuda().eval().requires_grad_(False)
    path = args.run / "dino_tokens.npy"
    if path.exists():
        raise FileExistsError(path)
    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                      shape=(len(images), 4, 384, 32, 32))
    for start in range(0, len(images), 8):
        batch = align.registered_ct(torch.from_numpy(images[start:start+8, None]).cuda(), -1, -1)
        for rotation in range(4):
            x = prompt.image_input(batch, rotation, 448)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                levels, _ = model(x)
            last = levels[-1]
            if not torch.isfinite(last).all():
                raise ValueError("DINO token not finite")
            cache[start:start+len(batch), rotation] = last.float().cpu().numpy()
        if start % 160 == 0:
            print("DINO_TOKENS", start, flush=True)
    cache.flush()
    del cache
    rc.write_json(path.with_suffix(".json"), {"payload":rc.hash_file(path), "clinical_data":rc.DATA_HASH,
                  "model":semantic.DINO_SHA, "implementation":rc.hash_file(semantic.__file__),
                  "protocol":rc.hash_file(PROTOCOL), "producer":rc.hash_file(__file__)})


def fit_pca(features, ids, width, dims=64):
    # Half the positions sample the broader CT, half sample the central 16 × 16 token field.
    full = np.arange(2, 32, 4)
    central = np.arange(9, 25, 2)
    total = np.zeros(width, np.float64)
    second = np.zeros((width, width), np.float64)
    count = 0
    for case in ids:
        tokens = features[case, 0, :width]
        sample = np.concatenate([tokens[:, grid[:, None], grid].reshape(width, -1).T
                                 for grid in (full, central)]).astype(np.float64)
        total += sample.sum(0)
        second += sample.T @ sample
        count += len(sample)
    mean = total / count
    covariance = second / count - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(covariance)
    values = values[-dims:][::-1]
    if values[-1] <= 1e-8:
        raise ValueError("Insufficient feature rank for whitened projection")
    vectors = vectors[:, -dims:][:, ::-1]
    # Canonicalize the sign of each retained axis for independent readback.
    signs = np.sign(vectors[np.abs(vectors).argmax(0), np.arange(dims)])
    projection = (vectors * signs / np.sqrt(values)).T
    return mean.astype(np.float32), projection.astype(np.float32), values.astype(np.float32)


@torch.inference_mode()
def prepare(args):
    _, uids = ct_and_split(args)
    _, train, valid = native.splits(uids, args.fold)
    train = sorted(train)
    patients = lambda ids: {str(uids[i]).split("_")[0] for i in ids}
    if patients(train) & patients(valid):
        raise ValueError("PCA patients overlap validation")
    rawpath = args.run / "dino_tokens.npy"
    receipt = json.loads(rawpath.with_suffix(".json").read_text())
    if (receipt["payload"] != rc.hash_file(rawpath) or receipt["model"] != semantic.DINO_SHA
            or receipt["implementation"] != rc.hash_file(semantic.__file__) or receipt["producer"] != rc.hash_file(__file__)):
        raise ValueError("DINO token provenance")
    banks = {"self": cache_owner.verified_parent(args).features,
             "dino": np.load(rawpath, mmap_mode="r", allow_pickle=False)}
    for name, features in banks.items():
        width = 256 if name == "self" else 384
        mean, projection, eig = fit_pca(features, train, width)
        stem = args.run / f"projection-{args.fold}-{name}"
        np.savez(stem.with_suffix(".npz"), ids=train, mean=mean, projection=projection, eigenvalues=eig)
        filt = torch.from_numpy(projection.copy()).cuda()[:, :, None, None]
        bias = -torch.from_numpy(projection @ mean).cuda()
        dst = stem.with_suffix(".npy")
        if dst.exists():
            raise FileExistsError(dst)
        mapped = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16, shape=(len(features), 4, 64, 32, 32))
        for start in range(0, len(features), 8):
            batch = torch.from_numpy(features[start:start+8, :, :width].copy()).cuda().float()
            projected = F.conv2d(batch.flatten(0,1), filt, bias).clamp(-6, 6)
            if not torch.isfinite(projected).all():
                raise ValueError("Nonfinite frozen projection")
            mapped[start:start+len(batch)] = projected.reshape(len(batch), 4, 64, 32, 32).cpu().numpy()
        mapped.flush()
        del mapped
        rc.write_json(stem.with_suffix(".json"), {"fold":args.fold, "name":name, "train":train,
            "patients":len(patients(train)), "basis":rc.hash_file(stem.with_suffix(".npz")),
            "raw":cache_owner.FEATURE_RECEIPT if name=="self" else receipt["payload"],
            "payload":rc.hash_file(dst), "source":rc.hash_file(__file__), "protocol":rc.hash_file(PROTOCOL)})
        print("FUSION_READY", args.fold, name, len(train), eig[[0,-1]].tolist(), flush=True)


class Joined:
    def __init__(self, primary, extra):
        self.primary = primary
        self.extra = extra

    def __getitem__(self, ids):
        return np.concatenate((self.primary[ids], self.extra[ids]), axis=2)


class FusionView(hierarchy.AlreadyRotatedView):
    def __init__(self, primary, args, name):
        stem = args.run / f"projection-{args.fold}-{name}"
        r = json.loads(stem.with_suffix(".json").read_text())
        if (r["fold"] != args.fold or r["name"] != name or r["source"] != rc.hash_file(__file__)
                or r["payload"] != rc.hash_file(stem.with_suffix(".npy"))
                or r["basis"] != rc.hash_file(stem.with_suffix(".npz")) or r["protocol"] != rc.hash_file(PROTOCOL)):
            raise ValueError("Fold-specific fusion projection changed")
        with np.load(args.crossfit/f"oof-{args.fold}.npz", allow_pickle=False) as oof:
            if r["train"] != oof["ids"].tolist():
                raise ValueError("Projection and OOF training patients differ")
        self.receipt = rc.hash_file(stem.with_suffix(".json"))
        self.features = Joined(primary, np.load(stem.with_suffix(".npy"), mmap_mode="r", allow_pickle=False))


class FusionRanks(hierarchy.HierarchicalRanks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.adapter = nn.Sequential(nn.Conv2d(64,48,1), nn.SiLU(), nn.Conv2d(48,256,1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def spatial_features(self, embedding, guide, image):
        primary, extra = embedding.split((1024,64), dim=1)
        coarse, high = hierarchy.unpack(primary, True)
        return coarse + self.adapter(extra), high


def train(args):
    view = FusionView(cache_owner.verified_parent(args).features, args, args.variant)
    prompt.train(args, decoder_type=FusionRanks, variant=f"{args.variant}-{args.seed_index}",
                 study_protocol=PROTOCOL, view=view, public_weights=hierarchy.WEIGHT,
                 seed_offset=OFFSETS[args.seed_index])
    if not args.smoke:
        folder = args.run / f"fold-{args.fold}-{args.variant}-{args.seed_index}"
        source = (args.parent / f"fold-{args.fold}-pyramid" if args.seed_index==0 else
                  args.ensemble / f"fold-{args.fold}-seed-{args.seed_index}")
        initial = lambda f: json.loads((f/"train.jsonl").read_text().splitlines()[0])["loss"]
        if not np.isclose(initial(folder), initial(source), rtol=2e-4, atol=4e-5):
            raise ValueError("Zero-start projection did not reproduce E initial supervision")


def lock(args):
    if (rc.hash_file(args.formal/"assessment/outer_report.json") != mirror_views.E_REPORT
            or rc.hash_file(args.stable/"summary.json") != mirror_views.E_INNER):
        raise ValueError("E identity changed")
    images, uids = ct_and_split(args)
    _, fitting, valid = native.splits(uids, args.fold)
    group = type("Groups", (), {"series_uid":uids})()
    lib, check = polar.calibration_split(group, valid, args.fold)
    ids = lib + check
    field, field_rec = verified_field(args.stable, "inner", args.fold, mirror_views.STABLE)
    with np.load(field, allow_pickle=False) as z:
        if z["ids"].tolist() != ids:
            raise ValueError("Guide order differs")
        guide = z["adapted"]
    previous = json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    archive = args.stable/f"pred-{args.fold}.npz"
    if (previous["prediction"] != rc.hash_file(archive) or previous["library"] != lib or previous["check"] != check
            or previous["field"] != field_rec["payload"]):
        raise ValueError("E reference changed")
    with np.load(archive, allow_pickle=False) as z:
        if z["ids"].tolist() != ids:
            raise ValueError("E reference order differs")
        masks = {"reference":z["adapted"]}
    weight = previous["choices"]["adapted"]["weight"]
    if weight != .25:
        raise ValueError("Unexpected E weight")
    original = cache_owner.verified_parent(args).features
    model = FusionRanks(args.public).cuda().eval()
    members = {}
    for name in MODES:
        view = FusionView(original, args, name)
        copies, hashes = [], []
        for i, offset in enumerate(OFFSETS):
            label = f"{name}-{i}"
            folder = args.run/f"fold-{args.fold}-{label}"
            fit = json.loads((folder/"fit.json").read_text())
            if (fit["train_indices"] != sorted(fitting) or fit["scope"] != "inner" or fit["steps"] != 5000
                    or fit["seed"] != 50171+args.fold+offset or fit["variant"] != label or fit["architecture"] != "FusionRanks"
                    or fit["public_weights"] != hierarchy.WEIGHT or fit["protocol"] != rc.hash_file(PROTOCOL)
                    or fit["source"] != rc.hash_file(prompt.__file__) or fit["view_receipt"] != view.receipt
                    or fit["oof"] != rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                    or fit["checkpoint"] != rc.hash_file(folder/"final.pt")):
                raise ValueError("Frozen fusion member differs")
            model.load_state_dict(torch.load(folder/"final.pt", weights_only=True, map_location="cpu"))
            prediction = prompt.predict(model, view.features, ids, images, guide).astype(np.float16)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite SAM fusion output")
            copies.append(prediction)
            hashes.append(fit["checkpoint"])
        masks[name] = np.packbits((1-weight)*guide + weight*mean_saved_logits(copies) > 0, axis=-1)
        members[name] = hashes
    path = args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(path, ids=ids, **masks)
    rc.write_json(args.run/f"selection-{args.fold}.json", {"fold":args.fold, "library":lib, "check":check,
        "reference":previous["prediction"], "guide":field_rec["payload"], "weight":weight, "members":members,
        "prediction":rc.hash_file(path), "protocol":rc.hash_file(PROTOCOL), "source":rc.hash_file(__file__)})
    print("FUSION_LOCK", args.fold, flush=True)


def score(args):
    seals = [json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    for f, r in enumerate(seals):
        if r["fold"] != f or r["prediction"] != rc.hash_file(args.run/f"pred-{f}.npz"):
            raise ValueError("Both validation predictions must be sealed before evaluation")
    _, truth, _, _ = native.inputs(args.data)
    for f, r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz", allow_pickle=False) as z:
            if z["ids"].tolist() != r["library"]+r["check"]:
                raise ValueError("Unexpected validation order")
            scores = {name:float(metrics.dice(truth[r["check"]], np.unpackbits(z[name][len(r["library"]):], axis=-1)).mean())
                      for name in ("reference", *MODES)}
        rc.write_json(args.run/f"inner-{f}.json", {**r, "scores":scores})
        print("FUSION_CHECK", f, scores, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("encode", "prepare", "fit", "lock", "score"))
    for field in ("run", "parent", "ensemble", "formal", "stable", "align", "crossfit", "data", "public", "dino"):
        p.add_argument("--"+field, type=Path, required=True)
    p.add_argument("--variant", choices=MODES, default="dino")
    p.add_argument("--seed-index", type=int, choices=range(3), default=0)
    p.add_argument("--fold", type=int, choices=(0,1), default=0)
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(scope="inner", image_size=512, backbone="sam2.1-tiny")
    args = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    {"encode":encode, "prepare":prepare, "fit":train, "lock":lock, "score":score}[args.phase](args)
