"""Compare high-resolution single-image features of the public SAM 2.1 encoder."""
import argparse
from dataclasses import dataclass
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.prefix_utility import accepted_inner_guide

WEIGHT = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SOURCE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/hierarchical_sam_protocol.md"


@dataclass(frozen=True)
class HieraSpec:
    filename: str
    digest: str
    width: int
    heads: int
    stages: tuple
    global_blocks: tuple
    windows: tuple


BACKBONES = {
    "sam2.1-tiny": HieraSpec("sam2.1_hiera_tiny.pt", WEIGHT,96,1,(1,2,7,2),(5,7,9),(8,4,14,7)),
    "sam2.1-large": HieraSpec("sam2.1_hiera_large.pt",
        "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
        144,2,(2,6,36,4),(23,33,43),(8,4,16,8)),
}
LARGE_CONFIG = "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"


def open_public(folder, backbone="sam2.1-tiny"):
    spec = BACKBONES[backbone]
    receipt = json.loads((folder / "receipt.json").read_text())
    manifest = folder / "runtime.sha256"
    if receipt["official_source"] != SOURCE_REVISION or rc.hash_file(manifest) != receipt["runtime_manifest"]:
        raise ValueError("Official runtime identity changed")
    runtime = folder / "runtime"
    for row in manifest.read_text().splitlines():
        expected, name = row.split(maxsplit=1)
        if rc.hash_file(runtime / name) != expected:
            raise ValueError(f"Public runtime file changed: {name}")
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    if backbone=="sam2.1-large" and (receipt.get("config_sha256")!=LARGE_CONFIG or rc.hash_file(folder/"sam2.1_hiera_l.yaml")!=LARGE_CONFIG):
        raise ValueError("Large Hiera configuration differs")
    checkpoint = folder / spec.filename
    if receipt["weights_sha256"] != spec.digest or rc.hash_file(checkpoint) != spec.digest:
        raise ValueError(f"Public {backbone} checkpoint differs")
    return torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]


def decoder_heads(state):
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    from sam2.modeling.sam.prompt_encoder import PromptEncoder
    from sam2.modeling.sam.transformer import TwoWayTransformer

    prompts = PromptEncoder(embed_dim=256, image_embedding_size=(32, 32),
                            input_image_size=(512, 512), mask_in_chans=16)
    decoder = MaskDecoder(transformer_dim=256, num_multimask_outputs=3,
                          transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                          iou_head_depth=3, iou_head_hidden_dim=256, use_high_res_features=True,
                          iou_prediction_use_sigmoid=True, pred_obj_scores=True, pred_obj_scores_mlp=True,
                          use_multimask_token_for_obj_ptr=True)
    prompts.load_state_dict(prompt.subset(state, "sam_prompt_encoder"))
    decoder.load_state_dict(prompt.subset(state, "sam_mask_decoder"))
    return prompts, decoder


def image_encoder(state, backbone="sam2.1-tiny"):
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
    from sam2.modeling.position_encoding import PositionEmbeddingSine

    spec = BACKBONES[backbone]
    trunk = Hiera(embed_dim=spec.width, num_heads=spec.heads, stages=spec.stages, global_att_blocks=spec.global_blocks,
                  window_spec=spec.windows, window_pos_embed_bkg_spatial_size=(7, 7))
    position = PositionEmbeddingSine(256, normalize=True, warmup_cache=False)
    neck = FpnNeck(position_encoding=position, d_model=256, backbone_channel_list=[spec.width*scale for scale in (8,4,2,1)],
                   fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    encoder = ImageEncoder(trunk, neck, scalp=1)
    encoder.load_state_dict(prompt.subset(state, "image_encoder"))
    return encoder.cuda().eval().requires_grad_(False)


def pack(high, middle, coarse):
    return torch.cat((coarse, F.pixel_unshuffle(middle, 2), F.pixel_unshuffle(high, 4)), dim=1)


def unpack(packed, use_high):
    coarse, middle, high = packed.split((256, 256, 512), dim=1)
    middle = F.pixel_shuffle(middle, 2)
    high = F.pixel_shuffle(high, 4)
    if not use_high:
        high, middle = torch.zeros_like(high), torch.zeros_like(middle)
    return coarse, [high, middle]


class HierarchicalRanks(nn.Module):
    use_high = True

    def __init__(self, folder, image_size=512, backbone="sam2.1-tiny"):
        super().__init__()
        if image_size != 512 or backbone not in BACKBONES:
            raise ValueError("Unexpected hierarchical SAM geometry")
        state = open_public(folder, backbone)
        self.prompt, self.decoder = decoder_heads(state)
        self.decoder.conv_s0.requires_grad_(False)
        self.decoder.conv_s1.requires_grad_(False)
        self.detail = nn.Sequential(nn.Conv2d(10, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32),
                                    nn.SiLU(), rc.Residual(32, 32), nn.Conv2d(32, 4, 1))
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)
        self.mix = nn.Parameter(torch.zeros(4))

    def spatial_features(self, embedding, guide, image):
        return unpack(embedding, self.use_high)

    def forward(self, embedding, guide, image):
        coarse, high = self.spatial_features(embedding, guide, image)
        box = prompt.boxes_from_logits(guide, self.training, 512)
        dense = guide.mean(1, keepdim=True).clamp(-10, 10)
        if self.training:
            keep = torch.rand(len(guide), 1, 1, 1, device=guide.device) > 0.23
            dense = dense * keep
        sparse, mask_embed = self.prompt(points=None, boxes=box, masks=dense)
        raw, _, _, _ = self.decoder.predict_masks(coarse, self.prompt.get_dense_pe(), sparse, mask_embed,
                                                   repeat_image=False, high_res_features=high)
        raw = raw.float()
        low = F.avg_pool2d(image, 7, 1, 3, count_include_pad=False)
        detail = self.detail(torch.cat((2*image-1, 4*(image-low), guide.clamp(-12,12)/8, raw.clamp(-12,12)/8), dim=1))
        mix = self.mix.sigmoid()[None, :, None, None]
        return (1-mix)*guide + mix*raw + detail.float(), raw


class CoarseRanks(HierarchicalRanks):
    use_high = False


@torch.inference_mode()
def encode(args, *, reflected=False, backbone="sam2.1-tiny"):
    if rc.hash_file(args.data) != rc.DATA_HASH:
        raise ValueError("LIDC identity changed")
    for f in range(4):
        if json.loads((args.align / f"fold-{f}-inner-registered/fit.json").read_text())["applied_shift"] != [-1,-1]:
            raise ValueError("Different registration requires a separate cache")
    with args.data.open("rb") as stream:
        images = np.stack([case["image"] for case in pickle.load(stream).values()])
    state = open_public(args.public, backbone)
    encoder = image_encoder(state, backbone)
    _, decoder = decoder_heads(state)
    decoder = decoder.cuda().eval()
    no_memory = state["no_mem_embed"].reshape(1, 256, 1, 1).cuda()
    path = args.run / "features.npy"
    if path.exists():
        raise FileExistsError(path)
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                      shape=(len(images), 4, 1024, 32, 32))
    start = time.monotonic()
    for i in range(0, len(images), 6):
        batch = align.registered_ct(torch.from_numpy(images[i:i+6, None]).cuda(), -1, -1)
        if reflected:
            batch = batch.flip([-1])
        for k in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                high, middle, coarse = encoder(prompt.image_input(batch, k, 512))["backbone_fpn"]
                packed = pack(decoder.conv_s0(high), decoder.conv_s1(middle), coarse + no_memory)
            if not torch.isfinite(packed).all():
                raise ValueError("Nonfinite multiscale features")
            store[i:i+len(batch), k] = packed.float().cpu().numpy()
        if i % 120 == 0:
            print("PYRAMID", i, len(images), time.monotonic()-start, flush=True)
    store.flush()
    del store
    orientation = {"horizontal_reflection": True} if reflected else {}
    rc.write_json(args.run / "features.json", {**orientation, "dataset":rc.DATA_HASH, "weights":BACKBONES[backbone].digest,
                  "source":rc.hash_file(__file__), "runtime":rc.hash_file(args.public/"runtime.sha256"),
                  "images":len(images), "payload":rc.hash_file(path), "seconds":time.monotonic()-start})


def feature_cache(args, producer=None, *, reflected=False, backbone="sam2.1-tiny"):
    record = json.loads((args.run / "features.json").read_text())
    path = args.run / "features.npy"
    if (record.get("horizontal_reflection", False) != reflected
            or record["source"] != rc.hash_file(producer or __file__) or record["payload"] != rc.hash_file(path)
            or record["dataset"] != rc.DATA_HASH or record["weights"] != BACKBONES[backbone].digest
            or record["runtime"] != rc.hash_file(args.public / "runtime.sha256")):
        raise ValueError("Frozen single-slice feature identity differs")
    return np.load(path, mmap_mode="r", allow_pickle=False)


class AlreadyRotatedView:
    def __init__(self, args, producer=None, *, backbone="sam2.1-tiny"):
        self.features = feature_cache(args, producer, backbone=backbone)
        self.receipt = rc.hash_file(args.run / "features.json")

    @staticmethod
    def decode(model, embedding, guide, image, indices, rotation):
        return model(embedding, guide, image)


def train(args):
    cls = HierarchicalRanks if args.variant == "pyramid" else CoarseRanks
    prompt.train(args, decoder_type=cls, variant=args.variant, study_protocol=PROTOCOL,
                 view=AlreadyRotatedView(args), public_weights=WEIGHT)


def check(args):
    images, truth, _, uids = native.inputs(args.data)
    _, training, valid = native.splits(uids, args.fold)
    groups = type("Groups", (), {"series_uid":uids})()
    library, checked = polar.calibration_split(groups, valid, args.fold)
    ids = library + checked
    original, guide, previous, _ = accepted_inner_guide(args, library, checked)
    folder = args.run / f"fold-{args.fold}-{args.variant}"
    fit = json.loads((folder / "fit.json").read_text())
    cls = HierarchicalRanks if args.variant == "pyramid" else CoarseRanks
    if (fit["train_indices"] != sorted(training) or fit["steps"] != prompt.STEPS or fit["variant"] != args.variant
            or fit["scope"] != "inner" or fit["backbone"] != args.backbone or fit["architecture"] != cls.__name__
            or fit["protocol"] != rc.hash_file(PROTOCOL) or fit["source"] != rc.hash_file(prompt.__file__)
            or fit["public_weights"] != WEIGHT or fit["view_receipt"] != rc.hash_file(args.run/"features.json")
            or fit["checkpoint"] != rc.hash_file(folder/"final.pt")):
        raise ValueError("Paired model identity differs")
    cache = feature_cache(args)
    model = cls(args.public).cuda().eval()
    model.load_state_dict(torch.load(folder/"final.pt", weights_only=True, map_location="cpu"))
    logits = prompt.predict(model, cache, ids, images, guide)
    record_check(args, truth, library, checked, original, guide, previous, fit, logits, PROTOCOL)


def record_check(args, truth, library, checked, original, guide, previous, fit, logits, protocol):
    ids = library + checked
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite check prediction")
    n = len(library)
    grid = [float(metrics.dice(truth[library], (1-w)*guide[:n]+w*logits[:n]>0).mean()) for w in (0,.25,.5,.75,1)]
    weight = (0,.25,.5,.75,1)[int(np.argmax(grid))]
    selection = {"fold":args.fold, "library":library, "weight":weight, "grid":grid, "variant":args.variant,
                 "previous":previous["prediction"], "checkpoint":fit["checkpoint"], "protocol":rc.hash_file(protocol)}
    rc.write_json(args.run/f"selection-{args.fold}-{args.variant}.json",selection)
    masks = (1-weight)*guide+weight*logits > 0
    scores = {"reference":float(metrics.dice(truth[checked],original[n:]).mean()),
              "standalone":float(metrics.dice(truth[checked],logits[n:]>0).mean()),
              "candidate":float(metrics.dice(truth[checked],masks[n:]).mean())}
    path = args.run/f"pred-{args.fold}-{args.variant}.npz"
    np.savez_compressed(path,ids=ids,masks=np.packbits(masks,axis=-1),logits=logits.astype(np.float16))
    rc.write_json(args.run/f"inner-{args.fold}-{args.variant}.json",{**selection,
                  "check":checked,"scores":scores,"prediction":rc.hash_file(path)})
    print("HIERARCHICAL_CHECK",args.fold,args.variant,weight,scores,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("encode","fit","check"))
    for field in ("run","data","public","align","crossfit","formal","encoder","ct","texture"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--variant",choices=("coarse","pyramid"),default="coarse")
    parser.set_defaults(scope="inner", image_size=512, backbone="sam2.1-tiny")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args = parser.parse_args()
    {"encode":encode,"fit":train,"check":check}[args.phase](args)
