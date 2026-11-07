"""Learn a second automatic mask prompt on the same CT slice."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import gated_hierarchy as parent
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.prefix_utility import accepted_inner_guide
from reproduction.trajectory_sam import select_weight

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/iterative_prompt_protocol.md"
SEED_OFFSET = 30000


def two_pass(decode, embedding, guide, image, feedback):
    # The intermediate mask is an automatic prompt. Only the supervised call carries gradients.
    with torch.no_grad():
        first, _ = decode(embedding, guide, image)
    second_prompt = (guide+first)/2 if feedback else guide
    return decode(embedding, second_prompt, image)


class MatchedRanks(hierarchy.HierarchicalRanks):
    feedback = False

    def forward(self, embedding, guide, image):
        return two_pass(super().forward, embedding, guide, image, self.feedback)


class FeedbackRanks(MatchedRanks):
    feedback = True


def train(args):
    cls = FeedbackRanks if args.variant=="feedback" else MatchedRanks
    view = parent.verified_parent(args)
    prompt.train(args, decoder_type=cls, variant=args.variant, study_protocol=PROTOCOL, view=view,
                 public_weights=hierarchy.WEIGHT, seed_offset=SEED_OFFSET)


def lock(args):
    images, truth, _, uids = native.inputs(args.data)
    _, training, valid = native.splits(uids,args.fold)
    group = type("Groups",(),{"series_uid":uids})()
    library, checked = polar.calibration_split(group,valid,args.fold)
    ids = library+checked
    original, guide, previous, _ = accepted_inner_guide(args,library,checked)
    n = len(library)
    view = parent.verified_parent(args)
    packed = {"reference":np.packbits(original,axis=-1)}
    raw = {}
    choices = {}
    for name in ("matched","feedback","inference-only"):
        fit_name = "matched" if name=="inference-only" else name
        cls = FeedbackRanks if fit_name=="feedback" else MatchedRanks
        folder = args.run/f"fold-{args.fold}-{fit_name}"
        fit = json.loads((folder/"fit.json").read_text())
        if (fit["steps"]!=prompt.STEPS or fit["seed"]!=50171+args.fold+SEED_OFFSET or fit["scope"]!="inner"
                or fit["train_indices"]!=sorted(training) or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                or fit["variant"]!=fit_name or fit["architecture"]!=cls.__name__ or fit["protocol"]!=rc.hash_file(PROTOCOL)
                or fit["public_weights"]!=hierarchy.WEIGHT or fit["view_receipt"]!=parent.FEATURE_RECEIPT
                or fit["source"]!=rc.hash_file(prompt.__file__) or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
            raise ValueError("Two-pass decoder training identity differs")
        model = cls(args.public).cuda().eval()
        model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
        if name=="inference-only":
            model.feedback = True
        logits = prompt.predict(model,view.features,ids,images,guide)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite feedback prediction")
        weight, grid = select_weight(truth[library],guide[:n],logits[:n])
        choices[name]={"weight":weight,"library_scores":grid,"checkpoint":fit["checkpoint"]}
        packed[name]=np.packbits((1-weight)*guide+weight*logits>0,axis=-1)
        raw[name]=np.packbits(logits>0,axis=-1)
    payload = args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(payload,ids=ids,**packed,**{"raw_"+k:v for k,v in raw.items()})
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,
                  "choices":choices,"previous":previous["prediction"],"prediction":rc.hash_file(payload),
                  "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("ITERATIVE_LOCK",args.fold,[choices[name]["weight"] for name in choices],flush=True)


def score(args):
    locks = [json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    for f,record in enumerate(locks):
        if record["fold"]!=f or record["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz"):
            raise ValueError("Incomplete paired prediction locks")
    _, truth, _, _ = native.inputs(args.data)
    for record in locks:
        f = record["fold"]
        n,checked = len(record["library"]),record["check"]
        scores = {}
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=record["library"]+checked:
                raise ValueError("Locked patient order differs")
            for name in ("reference","matched","feedback","inference-only","raw_matched","raw_feedback","raw_inference-only"):
                masks = np.unpackbits(z[name][n:],axis=-1).astype(bool)
                scores[name]=float(metrics.dice(truth[checked],masks).mean())
        rc.write_json(args.run/f"inner-{f}.json",{**record,"scores":scores})
        print("ITERATIVE_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","lock","score"))
    for field in ("run","parent","data","public","crossfit","formal","encoder","ct","texture"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0)
    parser.add_argument("--variant",choices=("matched","feedback"),default="matched")
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner", image_size=512, backbone="sam2.1-tiny")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    args=parser.parse_args()
    {"fit":train,"lock":lock,"score":score}[args.phase](args)
