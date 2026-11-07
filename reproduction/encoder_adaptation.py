"""Adapt the last original SAM image blocks using existing LIDC training patients."""
import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction.early_attention import ImagePrefix
from reproduction.lesion_window import LesionWindow
from reproduction.crossfit_contours import training_partition
from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import geometric_sam, rank_mixture
from reproduction import texture_study as incumbent
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/encoder_protocol.md"
STEPS = 5000
PREFIX_BLOCKS = 8


def prefix_forward(encoder, image):
    features = encoder.patch_embed(image)
    if encoder.pos_embed is not None:
        features = features + encoder.pos_embed
    for block in encoder.blocks[:-4]:
        features = block(features)
    return features


class EncoderTail(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.blocks = nn.ModuleList(encoder.blocks[-4:])
        self.neck = encoder.neck

    def forward(self, features):
        for block in self.blocks:
            features = block(features)
        return self.neck(features.permute(0, 3, 1, 2)).float()


class CTInputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(rc.Residual(3, 24, 2), rc.Residual(24, 48, 2),
                                     rc.Residual(48, 48), nn.Conv2d(48, 768, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, image):
        return self.network(native.ct_windows(image)).permute(0, 2, 3, 1)


def variant_name(args):
    if getattr(args, "lesion_window", False):
        return "lesion-window"
    if getattr(args, "geometric_views", False):
        return "geometric-views"
    if getattr(args,"rank_mixture",False):
        return "rank-mixture"
    prefix_mode=getattr(args,"prefix_mode","cache")
    if prefix_mode!="cache":
        return prefix_mode
    if args.ct_adapter:
        return "ct-adapted"
    return "adapted" if args.adapt else "frozen"


class AdaptedPrompts(nn.Module):
    def __init__(self, public, decoder_state, adapt, ct_adapter=False, backbone="vit_b", prefix_mode="cache", lesion_window=False):
        super().__init__()
        encoder = prompt.image_encoder(public, 512, backbone)
        self.front = None if prefix_mode=="cache" else ImagePrefix(encoder,prefix_mode=="live-lora")
        self.encoder = EncoderTail(encoder).requires_grad_(adapt)
        self.decoder = prompt.PromptRanks(public, pretrained=False, image_size=512, backbone=backbone)
        self.decoder.load_state_dict(decoder_state)
        self.window = LesionWindow() if lesion_window else None
        self.ct_adapter = None
        if ct_adapter:
            # Keep the paired augmentation stream identical when adding random adapter weights.
            with torch.random.fork_rng(devices=[]):
                self.ct_adapter = CTInputAdapter()

    def forward(self, prefix, guide, image, rotation=0):
        if self.front is not None:
            original = torch.rot90(image,-rotation,(-2,-1))*255
            normalized = prompt.image_input(original,rotation,512)
            if self.window is not None:
                normalized = self.window(normalized, image, guide)
            prefix = self.front(normalized).half().float()
        if self.ct_adapter is not None:
            prefix = prefix + self.ct_adapter(image).to(prefix.dtype)
        return self.decoder(self.encoder(prefix), guide, image)


@torch.inference_mode()
def encode(args):
    if rc.hash_file(args.data) != rc.DATA_HASH:
        raise ValueError("LIDC dataset identity differs")
    with args.data.open("rb") as stream:
        images = np.stack([case["image"] for case in pickle.load(stream).values()])
    for fold in range(4):
        fit = json.loads((args.align / f"fold-{fold}-inner-registered/fit.json").read_text())
        if fit["applied_shift"] != [-1, -1]:
            raise ValueError("Prefix cache needs fold-specific registration")
    backbone = getattr(args,"backbone","vit_b")
    spec = prompt.BACKBONES[backbone]
    encoder = prompt.image_encoder(args.public, 512, backbone)
    tail = EncoderTail(encoder).eval()
    example = align.registered_ct(torch.from_numpy(images[:2, None]).cuda(), -1, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
        image = prompt.image_input(example, 0, 512)
        full = encoder(image).float()
        prefix = prefix_forward(encoder, image)
        split = tail(prefix)
        compressed = tail(prefix.half().float())
    torch.testing.assert_close(split, full, rtol=1e-5, atol=1e-6)
    error = float((compressed-full).square().mean().sqrt() / full.square().mean().sqrt().clamp_min(1e-6))
    if error > 0.02:
        raise ValueError(f"Prefix cache quantization error is excessive: {error}")
    path = args.run / "prefix.npy"
    if path.exists():
        raise FileExistsError(path)
    final_path = args.run / "image_features.npy"
    store_final = getattr(args,"cache_final",False)
    if store_final and final_path.exists():
        raise FileExistsError(final_path)
    cache = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(len(images),4,32,32,spec.width)
    )
    final_cache = None
    if store_final:
        final_cache = np.lib.format.open_memmap(final_path,mode="w+",dtype=np.float16,
                                               shape=(len(images),4,256,32,32))
    start = time.monotonic()
    for begin in range(0, len(images), 4):
        batch = align.registered_ct(torch.from_numpy(images[begin:begin+4, None]).cuda(), -1, -1)
        for rotation in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                features = prefix_forward(encoder, prompt.image_input(batch, rotation, 512))
                if final_cache is not None:
                    final = tail(features)
                    if not torch.isfinite(final).all():
                        raise ValueError("Nonfinite complete image embedding")
                    final_cache[begin:begin+len(batch),rotation] = final.cpu().numpy()
            if not torch.isfinite(features).all():
                raise ValueError("Nonfinite prefix cache")
            cache[begin:begin+len(batch), rotation] = features.float().cpu().numpy()
        if begin % 100 == 0:
            print("PREFIX", begin, len(images), time.monotonic()-start, flush=True)
    cache.flush()
    del cache
    if final_cache is not None:
        final_cache.flush()
        del final_cache
        rc.write_json(args.run/"image_features.json", {
            "source":rc.hash_file(prompt.__file__),"producer":rc.hash_file(__file__),
            "data":rc.DATA_HASH,"public_weights":spec.digest,"backbone":backbone,
            "vendor":prompt.VENDOR_MANIFEST,"image_size":512,"images":len(images),
            "array":rc.hash_file(final_path),
        })
    rc.write_json(args.run / "prefix.json", {
        "source": rc.hash_file(__file__), "prompt_source": rc.hash_file(prompt.__file__),
        "public_weights": spec.digest, "vendor": prompt.VENDOR_MANIFEST,"backbone":backbone,
        "encoder_parameters":sum(p.numel() for p in encoder.parameters()),
        "tail_parameters":sum(p.numel() for p in tail.parameters()),
        "dataset": rc.DATA_HASH, "prefix_blocks": spec.depth-4, "payload": rc.hash_file(path),
        "split_forward_max_error": float((split-full).abs().max()),
        "cache_relative_rmse": error, "seconds": time.monotonic()-start,
    })


def open_prefix(args):
    owner = args.cache_run if args.cache_run is not None else args.run
    receipt = json.loads((owner / "prefix.json").read_text())
    source = owner / "source/reproduction/encoder_adaptation.py" if args.cache_run is not None else Path(__file__)
    if args.cache_run is not None:
        binding = json.loads((args.run / "cache_binding.json").read_text())
        if (binding["payload"] != receipt["payload"] or binding["owner"] != str(owner)
                or binding["receipt"] != rc.hash_file(owner / "prefix.json")):
            raise ValueError("External prefix does not match the verified cache binding")
    prompt_source = owner / "source/reproduction/prompt_preference.py" if args.cache_run is not None else Path(prompt.__file__)
    if args.scope == "full" and binding["runtime_prompt_source"] != rc.hash_file(prompt.__file__):
        raise ValueError("Full run has a different prompt implementation")
    spec = prompt.BACKBONES[getattr(args,"backbone","vit_b")]
    if (receipt["source"] != rc.hash_file(source) or receipt["prefix_blocks"] != spec.depth-4
            or receipt["prompt_source"] != rc.hash_file(prompt_source)
            or receipt["public_weights"] != spec.digest or receipt["dataset"] != rc.DATA_HASH):
        raise ValueError("Prefix provenance differs")
    return np.load(owner / "prefix.npy", mmap_mode="r", allow_pickle=False), receipt


def decoder_initialization(args, expected):
    warm_variant = getattr(args,"warm_variant",None) or args.scope
    folder = args.previous / f"fold-{args.fold}-{warm_variant}"
    receipt = json.loads((folder / "fit.json").read_text())
    source = Path(prompt.__file__) if args.scope == "full" else args.previous / "source/reproduction/prompt_preference.py"
    if receipt.get("backbone","vit_b") != getattr(args,"backbone","vit_b"):
        raise ValueError("Warm initialization and adaptation backbones differ")
    if receipt.get("variant",warm_variant) != warm_variant:
        raise ValueError("Warm decoder variant differs")
    if receipt.get("scope", "inner") != args.scope:
        raise ValueError("Initial decoder training scope differs")
    if (receipt["train_indices"] != sorted(expected)
            or receipt["checkpoint"] != rc.hash_file(folder / "final.pt")
            or receipt["source"] != rc.hash_file(source) or receipt["steps"] != 5000):
        raise ValueError("Initial prompt decoder has a different training identity")
    state = torch.load(folder / "final.pt", weights_only=True, map_location="cpu")
    return state, receipt


def train(args):
    images, labels, _, uids = native.inputs(args.data)
    eligible, excluded = training_partition(uids, args.fold, args.scope)
    state, initialization = decoder_initialization(args, eligible)
    path = args.crossfit / f"oof-{args.fold}.npz"
    guide_receipt = json.loads(path.with_suffix(".json").read_text())
    if rc.hash_file(path) != guide_receipt["payload"]:
        raise ValueError("OOF guide payload differs")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
        guides = archive["logits"]
    if ids != sorted(eligible) or guide_receipt.get("scope", "inner") != args.scope:
        raise ValueError("OOF guide contains ineligible training patients")
    prefix_mode=getattr(args,"prefix_mode","cache")
    if prefix_mode=="cache":
        features,feature_receipt=open_prefix(args)
    else:
        features=None
        feature_receipt={"payload":None}
        registration=incumbent.read(args.align/f"fold-{args.fold}-inner-registered/fit.json")
        if registration["applied_shift"]!=[-1,-1]:
            raise ValueError("Online image prefix needs the registered training convention")
    steps = 5 if args.smoke else STEPS
    if args.smoke:
        ids, guides = ids[:32], guides[:32]
    variant = variant_name(args)
    directory = args.run / (f"smoke-{variant}" if args.smoke else f"fold-{args.fold}-{variant}")
    directory.mkdir(exist_ok=False)
    seed = 60171 + args.fold
    view_generator = None
    if getattr(args, "geometric_views", False):
        view_generator = torch.Generator(device="cuda").manual_seed(289231 + args.fold)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prefix = torch.from_numpy(features[ids]).cuda() if features is not None else None
    image = align.registered_ct(torch.from_numpy(images[ids, None]).cuda(), -1, -1).float() / 255
    truth = torch.from_numpy(labels[ids]).cuda().float()
    assignment = rank_mixture.assignment_weights(truth) if getattr(args,"rank_mixture",False) else None
    prior = torch.from_numpy(guides).cuda().float()
    names = [str(uids[i]).split("_")[0] for i in ids]
    _, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
    weight = torch.from_numpy(0.45/len(ids) + 0.55/(len(counts)*counts[inverse])).float().cuda()
    backbone = getattr(args,"backbone","vit_b")
    window = getattr(args, "lesion_window", False)
    model = AdaptedPrompts(args.public, state, args.adapt, args.ct_adapter, backbone, prefix_mode, window).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    groups = [{"params": list(model.decoder.parameters()), "lr": 0.00008,
               "base_lr": 0.00008, "weight_decay": 0.0002}]
    if args.adapt:
        groups.append({"params": list(model.encoder.parameters()), "lr": 0.00002,
                       "base_lr": 0.00002, "weight_decay": 0.003})
    if model.ct_adapter is not None:
        groups.append({"params": list(model.ct_adapter.parameters()), "lr": 0.00016,
                       "base_lr": 0.00016, "weight_decay": 0.0002})
    if model.window is not None:
        groups.append({"params": list(model.window.parameters()), "lr":0.0025, "base_lr":0.0025, "weight_decay":0.0})
    low_rank=[] if model.front is None else [p for p in model.front.parameters() if p.requires_grad]
    if low_rank:
        groups.append({"params":low_rank,"lr":0.00012,"base_lr":0.00012,"weight_decay":0.0002})
    optimizer = torch.optim.AdamW(groups, fused=True)
    pairs=[(p,shadow) for p,shadow in zip(model.parameters(),ema.parameters()) if p.requires_grad]
    parameters=[pair[0] for pair in pairs]
    averaged=[pair[1] for pair in pairs]
    protocol = PROTOCOL.with_name("ct_adapter_protocol.md") if args.ct_adapter else PROTOCOL
    if backbone == "vit_h":
        protocol = PROTOCOL.with_name("sam_h_protocol.md")
    if prefix_mode!="cache":
        protocol=PROTOCOL.with_name("early_attention_protocol.md")
    if assignment is not None:
        protocol=PROTOCOL.with_name("rank_mixture_protocol.md")
    if view_generator is not None:
        protocol = PROTOCOL.with_name("geometric_sam_protocol.md")
    if window:
        protocol = PROTOCOL.with_name("lesion_window_protocol.md")
    if args.scope == "full":
        protocol = PROTOCOL.with_name("outer_protocol.md")
    rc.write_json(directory / "plan.json", {
        "scope": args.scope, "excluded_indices": excluded, "prefix_mode":prefix_mode,
        "view_augmentation_source": rc.hash_file(geometric_sam.__file__) if view_generator is not None else None,
        "lesion_window": window,
        "early_attention_parameters":sum(p.numel() for p in low_rank),
        "rank_assignment_source":rc.hash_file(rank_mixture.__file__) if assignment is not None else None,
        "mean_off_diagonal_mass":float(1-assignment.diagonal(dim1=1,dim2=2).mean()) if assignment is not None else 0.,
        "fold": args.fold, "variant": variant, "steps": steps, "seed": seed, "train_indices": ids,
        "source": rc.hash_file(__file__), "prompt_source": rc.hash_file(prompt.__file__),
        "protocol": rc.hash_file(protocol),
        "initial_decoder": initialization["checkpoint"],
        "adapter_parameters": sum(p.numel() for p in model.ct_adapter.parameters()) if args.ct_adapter else 0,
        "public_weights": prompt.BACKBONES[backbone].digest, "backbone":backbone, "prefix": feature_receipt["payload"],
        "oof": guide_receipt["payload"],
    })
    start = time.monotonic()
    with (directory / "train.jsonl").open("x", buffering=1) as log:
        for step in range(1, steps+1):
            ix = torch.multinomial(weight, 12, replacement=True)
            rotation = int(torch.randint(4, ()).item())
            guide = torch.rot90(prior[ix], rotation, (-2, -1))
            x = torch.rot90(image[ix], rotation, (-2, -1))
            y = torch.rot90(truth[ix], rotation, (-2, -1))
            guide = guide * (0.75+0.5*torch.rand(len(ix), 1, 1, 1, device="cuda"))
            guide = guide + (torch.rand(len(ix), 1, 1, 1, device="cuda")-0.5)*2
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                sampled = prefix[ix,rotation].float() if prefix is not None else None
                if view_generator is not None and step % 2 == 0:
                    with torch.autocast("cuda", enabled=False):
                        grid, back = geometric_sam.random_grids(len(ix), view_generator)
                    logits, raw = geometric_sam.native_predictions(model, x, guide, rotation, grid, back)
                else:
                    logits, raw = model(sampled,guide,x,rotation)
            radius = (logits.sigmoid().sum((-2, -1)).clamp_min(1)/math.pi).sqrt()
            actual = (y.sum((-2, -1)).clamp_min(1)/math.pi).sqrt()
            area_loss=0.08*F.smooth_l1_loss(radius,actual,beta=0.5)
            if assignment is None:
                loss=vote.mask_loss(logits,y)+0.4*vote.mask_loss(raw,y)+area_loss
            else:
                law=assignment[ix]
                loss=rank_mixture.expected_mask_loss(logits,y,law)+0.4*rank_mixture.expected_mask_loss(raw,y,law)+area_loss
                if step==1 and not args.smoke:
                    with (args.control_run/f"fold-{args.fold}-live-frozen/train.jsonl").open() as stream:
                        control_loss=json.loads(stream.readline())["loss"]
                    ordinary=(vote.mask_loss(logits,y)+0.4*vote.mask_loss(raw,y)+area_loss).item()
                    if not np.isclose(ordinary,control_loss,rtol=1e-4,atol=2e-5):
                        raise ValueError(f"Ordinary first losses differ: {ordinary} versus {control_loss}")
                    print("PAIRED_ORDINARY_COMPONENT",args.fold,ordinary,control_loss,flush=True)
            if step == 1 and args.ct_adapter and not args.smoke and args.scope == "inner":
                control = args.cache_run / f"fold-{args.fold}-adapted/train.jsonl"
                with control.open() as stream:
                    initial_loss = json.loads(stream.readline())["loss"]
                if not np.isclose(loss.item(), initial_loss, rtol=1e-4, atol=2e-5):
                    raise ValueError(f"Paired first losses differ: {loss.item()} and {initial_loss}")
                print("PAIRED_INITIAL_LOSS", args.fold, loss.item(), initial_loss, flush=True)
            if step == 1 and (prefix_mode == "live-lora" or view_generator is not None or window) and not args.smoke:
                paired_run = args.control_run if view_generator is not None or window else args.run
                with (paired_run/f"fold-{args.fold}-live-frozen/train.jsonl").open() as stream:
                    paired=json.loads(stream.readline())["loss"]
                if not np.isclose(loss.item(),paired,rtol=1e-4,atol=2e-5):
                    raise ValueError(f"First losses differ: {loss.item()} versus {paired}")
                print("PAIRED_INITIAL_LOSS",args.fold,loss.item(),paired,flush=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.smoke and args.adapt:
                gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
                if not gradients or sum(float(g.abs().sum()) for g in gradients) <= 0:
                    raise ValueError("Encoder adaptation has no gradient")
                if window and (model.window.gates.grad is None or model.window.gates.grad.abs().max().item() == 0):
                    raise ValueError("Input contrast gates have no gradient")
                if low_rank and not any(p.grad is not None and p.grad.abs().sum().item()>0 for p in low_rank):
                    raise ValueError("Early attention updates have no gradient")
                if args.ct_adapter and float(model.ct_adapter.network[-1].weight.grad.abs().sum()) <= 0:
                    raise ValueError("CT adapter has no gradient")
            torch.nn.utils.clip_grad_norm_(parameters, 4, error_if_nonfinite=True)
            schedule = min(1, step/80)*(0.035+0.965*(1+math.cos(math.pi*step/steps))/2)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * schedule
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(averaged, parameters, max(0.005, 1/step))
            if step == 1 or step % 250 == 0 or step == steps:
                record = {"step": step, "loss": loss.item(), "seconds": time.monotonic()-start}
                log.write(json.dumps(record, allow_nan=False)+"\n")
                print(variant, args.fold, record, flush=True)
    torch.save(ema.state_dict(), directory / "final.pt")
    rc.write_json(directory / "fit.json", {
        "checkpoint": rc.hash_file(directory / "final.pt"),
        "plan": rc.hash_file(directory / "plan.json"), "seconds": time.monotonic()-start,
    })


@torch.inference_mode()
def predict(model, cache, ids, images, guide):
    output = []
    for begin in range(0, len(ids), 12):
        cases = ids[begin:begin+12]
        feature = torch.from_numpy(cache[cases].copy()).cuda() if cache is not None else None
        image = align.registered_ct(torch.from_numpy(images[cases, None]).cuda(), -1, -1).float()/255
        prior = torch.from_numpy(guide[begin:begin+len(cases)]).cuda()
        total = 0
        for rotation in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                sampled = feature[:,rotation].float() if feature is not None else None
                logits, _ = model(sampled,torch.rot90(prior,rotation,(-2,-1)),
                                  torch.rot90(image,rotation,(-2,-1)),rotation)
            total = total + torch.rot90(logits, -rotation, (-2, -1))
        output.append((total/4).cpu().numpy())
    return np.concatenate(output)


def check(args):
    images, truth, _, uids = native.inputs(args.data)
    _, inner, valid = native.splits(uids, args.fold)
    group = type("Groups", (), {"series_uid": uids})()
    library, checked = polar.calibration_split(group, valid, args.fold)
    ids = library + checked
    path = args.texture / f"features-{args.fold}-inner.npz"
    owner = json.loads(path.with_suffix(".json").read_text())
    if owner["indices"] != ids or owner["payload"] != rc.hash_file(path):
        raise ValueError("Incumbent patient order or payload differs")
    with np.load(path, allow_pickle=False) as archive:
        original = np.unpackbits(archive["old"], axis=-1).astype(bool)
        guide = incumbent.anchor_logits(archive["base"].astype(np.float32), original)
    variant = variant_name(args)
    directory = args.run / f"fold-{args.fold}-{variant}"
    plan = json.loads((directory / "plan.json").read_text())
    receipt = json.loads((directory / "fit.json").read_text())
    if (plan["train_indices"] != sorted(inner) or plan["source"] != rc.hash_file(__file__)
            or receipt["plan"] != rc.hash_file(directory / "plan.json")
            or receipt["checkpoint"] != rc.hash_file(directory / "final.pt")):
        raise ValueError("Adapted model identity differs")
    initial, _ = decoder_initialization(args, inner)
    backbone = getattr(args,"backbone","vit_b")
    if plan.get("backbone","vit_b") != backbone:
        raise ValueError("Check and training backbones differ")
    prefix_mode=getattr(args,"prefix_mode","cache")
    if plan.get("prefix_mode","cache")!=prefix_mode:
        raise ValueError("Training and inference prefix modes differ")
    window = getattr(args, "lesion_window", False)
    if plan.get("lesion_window", False) != window:
        raise ValueError("Window training and evaluation differ")
    model = AdaptedPrompts(args.public, initial, False, args.ct_adapter, backbone, prefix_mode, window).cuda().eval()
    model.load_state_dict(torch.load(directory / "final.pt", weights_only=True, map_location="cpu"))
    cache = open_prefix(args)[0] if prefix_mode=="cache" else None
    predicted = predict(model, cache, ids, images, guide)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite adapted prediction")
    count = len(library)
    grid = [float(scoring.dice(truth[library], ((1-w)*guide[:count]+w*predicted[:count]) > 0).mean())
            for w in incumbent.WEIGHTS]
    weight = incumbent.WEIGHTS[int(np.argmax(grid))]
    choice = {"weight": weight, "library_scores": grid, "library": library,
              "source": rc.hash_file(__file__), "model": receipt["checkpoint"]}
    rc.write_json(args.run / f"selection-{args.fold}-{variant}.json", choice)
    masks = (1-weight)*guide + weight*predicted > 0
    scores = {
        "reference": float(scoring.dice(truth[checked], original[count:]).mean()),
        "standalone": float(scoring.dice(truth[checked], predicted[count:] > 0).mean()),
        "candidate": float(scoring.dice(truth[checked], masks[count:]).mean()),
    }
    payload = args.run / f"pred-{args.fold}-{variant}.npz"
    np.savez_compressed(payload, ids=ids, logits=predicted.astype(np.float16), masks=np.packbits(masks, axis=-1))
    rc.write_json(args.run / f"inner-{args.fold}-{variant}.json", {
        **choice, "fold": args.fold, "variant": variant, "check": checked, "scores": scores,
        "prediction": rc.hash_file(payload), "protocol": plan["protocol"],
    })
    print("ENCODER_CHECK", args.fold, variant, weight, scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("encode", "fit", "check"))
    for name in ("run", "data", "public", "align", "previous", "crossfit", "texture"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    parser.add_argument("--adapt", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ct-adapter", action="store_true")
    parser.add_argument("--cache-run", type=Path)
    parser.add_argument("--scope", choices=("inner", "full"), default="inner")
    parser.add_argument("--backbone",choices=tuple(prompt.BACKBONES),default="vit_b")
    parser.add_argument("--cache-final",action="store_true")
    parser.add_argument("--prefix-mode",choices=("cache","live-frozen","live-lora"),default="cache")
    parser.add_argument("--warm-variant",choices=("control",))
    parser.add_argument("--rank-mixture",action="store_true")
    parser.add_argument("--geometric-views", action="store_true")
    parser.add_argument("--lesion-window", action="store_true")
    parser.add_argument("--control-run",type=Path)
    args = parser.parse_args()
    if args.lesion_window and (args.prefix_mode != "live-frozen" or args.control_run is None or args.rank_mixture or args.geometric_views):
        parser.error("Lesion windows require the ordinary online frozen-prefix paired control")
    if args.geometric_views and (args.prefix_mode != "live-frozen" or args.control_run is None or args.rank_mixture):
        parser.error("Geometric views require the unmodified live-frozen paired control")
    if args.rank_mixture and (args.prefix_mode!="live-frozen" or args.control_run is None):
        parser.error("Rank mixture uses the paired online frozen-prefix control")
    if args.prefix_mode!="cache" and (args.scope!="inner" or args.ct_adapter or args.backbone!="vit_b" or not args.adapt or args.phase=="encode"):
        parser.error("The early-attention study uses inner ViT-B tail adaptation")
    if args.scope == "full" and (args.phase != "fit" or args.cache_run is None):
        parser.error("Full fits require a verified prefix cache; outer inference has a separate entry")
    if args.backbone=="vit_h" and (args.ct_adapter or args.scope=="full"):
        parser.error("The ViT-H study is registered for inner tail adaptation")
    if args.ct_adapter and (not args.adapt or args.cache_run is None):
        parser.error("The CT adapter experiment requires --adapt and --cache-run")
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    {"encode": encode, "fit": train, "check": check}[args.phase](args)
