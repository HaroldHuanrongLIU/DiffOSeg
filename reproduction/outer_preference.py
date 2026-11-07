"""Full single-slice training, four-fold prediction sealing and primary assessment."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch

from reproduction import aligned_preference as align
from reproduction import boundary_eval as baseline
from reproduction import crossfit_residual as crossfit
from reproduction import encoder_adaptation as encoder
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import polar_eval
from reproduction import prompt_preference as warm
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent

DOCS = Path(__file__).resolve().parents[1] / "docs/lidc_91"
TARGET = 0.91
MIN_CT_GAIN = 0.00030
HISTORY = {
    "v1":"20261106_rank_boundary_v1", "v2":"20261106_polar_residual_v2",
    "v3":"20261106_vote_transport_v3", "v4":"20261106_joint_energy_v4",
    "v5":"20261106_normal_refine_v5", "v7":"20261106_canonical_panel_v7",
    "v8":"20261106_ordinal_projection_v8", "v9":"20261106_order_preference_v9",
    "local":"20261106_single_slice_90_native_v10c", "align":"20261106_single_slice_90_aligned",
    "zoom":"20261106_single_slice_90_zoom", "area":"20261106_single_slice_90_area",
    "incumbent":"20261106_preference90_comparisonb", "encoder":"20261106_preference91_encoder",
    "ct":"20261106_preference91_ctadapter", "public":"20261106_preference91_prompt",
}


def read(path):
    return json.loads(path.read_text())


def fixed_record(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing record differs: {path}")
    else:
        rc.write_json(path, value)


def select_fold(tail, ct):
    return "ct-adapted" if ct-tail >= MIN_CT_GAIN else "adapted"


def selections(args, audit):
    if rc.hash_file(args.incumbent/"assessment/preference_report.json") != incumbent.REFERENCE:
        raise ValueError("Accepted comparator identity differs")
    truth = uids = None
    if audit:
        _, truth, _, uids = native.inputs(args.data)
    folds = []
    for f in range(4):
        records = {}
        for variant, owner in (("adapted", args.encoder), ("ct-adapted", args.ct)):
            path = owner/f"inner-{f}-{variant}.json"
            row = read(path)
            choice = read(owner/f"selection-{f}-{variant}.json")
            if (row["source"] != rc.hash_file(owner/"source/reproduction/encoder_adaptation.py")
                    or row["weight"] != choice["weight"] or row["model"] != choice["model"]
                    or row["weight"] != incumbent.WEIGHTS[int(np.argmax(choice["library_scores"]))]):
                raise ValueError("Inner selection identity differs")
            if audit:
                outer, train, valid = native.splits(uids, f)
                groups = type("Groups", (), {"series_uid": uids})()
                library, checked = polar.calibration_split(groups, valid, f)
                if row["library"] != library or row["check"] != checked:
                    raise ValueError("Inner patient order differs")
                def patients(indices):
                    return {str(uids[i]).split("_")[0] for i in indices}
                if (patients(library+checked) & set(outer["test_patients"])
                        or patients(library) & patients(checked) or patients(train) & patients(valid)):
                    raise ValueError("Inner selection patients overlap excluded patients")
                bank = owner/f"pred-{f}-{variant}.npz"
                if rc.hash_file(bank) != row["prediction"]:
                    raise ValueError("Inner prediction changed")
                with np.load(bank, allow_pickle=False) as data:
                    if data["ids"].tolist() != library+checked:
                        raise ValueError("Inner prediction order differs")
                    masks = np.unpackbits(data["masks"][len(library):], axis=-1).astype(bool)
                score = float(metrics.dice(truth[checked], masks).mean())
                if abs(score-row["scores"]["candidate"]) > 1e-12:
                    raise ValueError("Recorded inner score differs from masks")
            records[variant] = {"weight":row["weight"], "score":row["scores"]["candidate"],
                                "receipt":rc.hash_file(path), "prediction":row["prediction"]}
        variant = select_fold(records["adapted"]["score"], records["ct-adapted"]["score"])
        folds.append({"fold":f, "variant":variant, "weight":records[variant]["weight"], "inner":records})
    return {"dataset":rc.DATA_HASH, "reference":incumbent.REFERENCE, "target":TARGET,
            "protocol":rc.hash_file(DOCS/"outer_protocol.md"),
            "amendment":rc.hash_file(DOCS/"outer_selection.md"), "folds":folds}


def lock(args):
    result = selections(args, True)
    fixed_record(args.run/"selection.json", result)
    print("LOCKED", [(row["fold"], row["variant"], row["weight"]) for row in result["folds"]], flush=True)


def locked(args):
    value = read(args.run/"selection.json")
    if value != selections(args, False):
        raise ValueError("Architecture or blend lock changed")
    return value


@torch.inference_mode()
def bind(args):
    prefix_owner, warm_owner = args.encoder, args.public
    prefix_meta, warm_meta = read(prefix_owner/"prefix.json"), read(warm_owner/"image_features.json")
    prefix_path, warm_path = prefix_owner/"prefix.npy", warm_owner/"image_features.npy"
    if (prefix_meta["payload"] != rc.hash_file(prefix_path) or warm_meta["array"] != rc.hash_file(warm_path)
            or prefix_meta["public_weights"] != warm.WEIGHT or warm_meta["public_weights"] != warm.WEIGHT
            or prefix_meta["dataset"] != rc.DATA_HASH or warm_meta["data"] != rc.DATA_HASH
            or prefix_meta["prefix_blocks"] != encoder.PREFIX_BLOCKS
            or prefix_meta["source"] != rc.hash_file(prefix_owner/"source/reproduction/encoder_adaptation.py")
            or warm_meta["source"] != rc.hash_file(warm_owner/"source/reproduction/prompt_preference.py")):
        raise ValueError("Public feature cache verification failed")
    images, _, _, uids = native.inputs(args.data)
    for f in range(4):
        outer, _, _ = native.splits(uids, f)
        directory = args.align/f"fold-{f}-full-registered"
        if (read(directory/"plan.json")["train_indices"] != outer["train_indices"]
                or read(directory/"fit.json")["applied_shift"] != [-1,-1]):
            raise ValueError("Full training registration differs from the cache")
    eligible = native.splits(uids, 0)[0]["train_indices"]
    cases = [eligible[0], eligible[len(eligible)//2]]
    x = align.registered_ct(torch.from_numpy(images[cases,None]).cuda(), -1, -1)
    vision = warm.image_encoder(warm_owner, 512)
    prefix_cache = np.load(prefix_path, mmap_mode="r", allow_pickle=False)
    warm_cache = np.load(warm_path, mmap_mode="r", allow_pickle=False)
    errors = []
    for rotation in range(4):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            image = warm.image_input(x, rotation, 512)
            prefix = encoder.prefix_forward(vision, image).float()
            full = vision(image).float()
        row = []
        for actual, cached in ((prefix, prefix_cache), (full, warm_cache)):
            stored = torch.from_numpy(cached[cases,rotation].copy()).cuda().float()
            relative = float((actual-stored).square().mean().sqrt()/actual.square().mean().sqrt().clamp_min(1e-6))
            if relative > 0.02:
                raise ValueError(f"Current single-image encoder differs from cached values: {relative}")
            row.append(relative)
        errors.append(row)
    binding = {"owner":str(prefix_owner), "payload":prefix_meta["payload"],
               "receipt":rc.hash_file(prefix_owner/"prefix.json"), "warm_owner":str(warm_owner),
               "warm_payload":warm_meta["array"], "warm_receipt":rc.hash_file(warm_owner/"image_features.json"),
               "runtime_prompt_source":rc.hash_file(warm.__file__), "fiducial_cases":cases,
               "cache_relative_rmse":errors, "dataset":rc.DATA_HASH}
    fixed_record(args.run/"cache_binding.json", binding)
    print("CACHE_VERIFIED", max(max(row) for row in errors), flush=True)


def check_fit(args, eligible):
    choice = locked(args)["folds"][args.fold]
    directory = args.run/f"fold-{args.fold}-{choice['variant']}"
    plan, fit = read(directory/"plan.json"), read(directory/"fit.json")
    oof = read(args.run/f"oof-{args.fold}.json")
    initial, initial_receipt = encoder.decoder_initialization(args, eligible)
    if (plan["scope"] != "full" or plan["train_indices"] != sorted(eligible) or plan["steps"] != encoder.STEPS
            or plan["variant"] != choice["variant"] or plan["source"] != rc.hash_file(encoder.__file__)
            or plan["initial_decoder"] != initial_receipt["checkpoint"] or plan["oof"] != oof["payload"]
            or initial_receipt["oof"] != oof["payload"] or oof["scope"] != "full"
            or plan["protocol"] != rc.hash_file(DOCS/"outer_protocol.md")
            or fit["plan"] != rc.hash_file(directory/"plan.json") or fit["checkpoint"] != rc.hash_file(directory/"final.pt")):
        raise ValueError("Full model or training identity differs")
    model = encoder.AdaptedPrompts(args.public, initial, False, choice["variant"]=="ct-adapted").cuda().eval()
    model.load_state_dict(torch.load(directory/"final.pt", weights_only=True, map_location="cpu"))
    return model, fit, choice


def infer(args):
    selection = locked(args)
    f = args.fold
    folder = args.run/f"assessment/fold-{f}"
    if (folder/"sealed.json").exists():
        verify_bank(folder, f, rc.hash_file(args.run/"selection.json"))
        return
    data, ids, _, _, _, raw, reference, shift = incumbent.comparator(args, "full")
    outer, _, _ = rc.split_for(data, f)
    if ids != outer["test_indices"] or shift != [-1,-1]:
        raise ValueError("Outer prediction order or registration differs")
    archive = args.incumbent/f"assessment/fold-{f}"
    frozen = read(archive/"sealed.json")
    if frozen["bank"] != polar_eval.digest_bank(archive, len(ids)):
        raise ValueError("Archived reference predictions differ")
    guide = incumbent.anchor_logits(raw.astype(np.float16).astype(np.float32), reference)
    model, fit, choice = check_fit(args, outer["train_indices"])
    cache, _ = encoder.open_prefix(args)
    logits = encoder.predict(model, cache, ids, np.stack(data.images), guide)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite outer logits")
    weight = choice["weight"]
    masks = (1-weight)*guide+weight*logits > 0
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder/"predictions.partial.npz"
    np.savez_compressed(partial, ids=np.asarray(ids), masks=np.packbits(masks, axis=-1),
                        reference=np.packbits(reference, axis=-1))
    path = folder/"predictions.npz"
    partial.replace(path)
    rc.write_json(folder/"sealed.json", {
        "fold":f, "cases":len(ids), "selection":rc.hash_file(args.run/"selection.json"),
        "weight":weight, "variant":choice["variant"], "checkpoint":fit["checkpoint"],
        "reference_bank":frozen["bank"], "payload":rc.hash_file(path), "source":rc.hash_file(__file__),
        "guide_sha256":hashlib.sha256(np.ascontiguousarray(guide).tobytes()).hexdigest(),
        "protocol":selection["protocol"], "amendment":selection["amendment"],
    })
    print("OUTER_SEALED", f, len(ids), flush=True)


def verify_bank(folder, fold, selection_hash):
    seal = read(folder/"sealed.json")
    if (seal["fold"] != fold or seal["selection"] != selection_hash
            or seal["source"] != rc.hash_file(__file__) or seal["payload"] != rc.hash_file(folder/"predictions.npz")):
        raise ValueError("Outer prediction seal differs")
    return seal


def score(args):
    selection = locked(args)
    lock_hash = rc.hash_file(args.run/"selection.json")
    folders = [args.run/f"assessment/fold-{f}" for f in range(4)]
    seals = [verify_bank(folder,f,lock_hash) for f, folder in enumerate(folders)]
    # Open the evaluation labels only after all four new prediction banks are sealed.
    _, truth, _, uids = native.inputs(args.data)
    base_root = args.v1/"baseline"
    if rc.hash_file(base_root/"fourfold_summary.json") != baseline.SUMMARY_HASH:
        raise ValueError("E11 reference summary differs")
    original = read(base_root/"fourfold_summary.json")
    historical = read(args.incumbent/"assessment/preference_report.json")
    patients, all_ids, folds, bootstrap = set(), set(), [], []
    rng = np.random.default_rng(9101337)
    names = ("e11", "reference", "candidate")
    for f, (folder, seal, choice) in enumerate(zip(folders, seals, selection["folds"])):
        outer, _, _ = native.splits(uids, f)
        stage = base_root/f"fold_{f}/lidc/fold_{f}/stage_2"
        split = read(stage/"split.json")
        if any(split[key] != outer[key] for key in outer):
            raise ValueError("Original E11 patient manifest differs")
        with np.load(folder/"predictions.npz", allow_pickle=False) as archive:
            ids = archive["ids"].tolist()
            pred = np.unpackbits(archive["masks"], axis=-1).astype(bool)
            ref = np.unpackbits(archive["reference"], axis=-1).astype(bool)
        if (ids != split["test_indices"] or seal["cases"] != len(ids) or choice["variant"] != seal["variant"]
                or pred.shape != (len(ids),4,128,128) or ref.shape != pred.shape):
            raise ValueError("Test prediction identity differs")
        expected_patients = set(split["test_patients"])
        if all_ids.intersection(ids) or patients & expected_patients:
            raise ValueError("Outer cases or patients repeated")
        all_ids.update(ids); patients.update(expected_patients)
        e11 = []
        for j, idx in enumerate(ids):
            path = stage/f"cases/{j:05d}_masks.npz"
            y = baseline.unpack(path, "truth", (4,128,128))
            if not np.array_equal(y, truth[idx]):
                raise ValueError("E11 archived reference masks differ")
            e11.append(baseline.unpack(path, "predictions", (4,128,128)))
        values = np.stack([metrics.dice(truth[ids].astype(bool), m) for m in (np.stack(e11), ref, pred)], axis=1)
        means = values.mean((0,2))
        if (abs(means[0]-original["folds"][f]["stages"]["2"]["metrics"]["Dice_mean"]) > 1e-6
                or abs(means[1]-historical["folds"][f]["means"]["full"]["Dice_mean"]) > 1e-12):
            raise ValueError("Original comparator parity failed")
        group, inverse = np.unique([str(uids[i]).split("_")[0] for i in ids], return_inverse=True)
        if set(group) != expected_patients:
            raise ValueError("Incomplete patient coverage")
        rows = [{"index":i, "patient":str(uids[i]).split("_")[0],
                 **{name:values[j,k].tolist() for k,name in enumerate(names)}} for j,i in enumerate(ids)]
        rc.write_json(folder/"paired.json", {"seal":rc.hash_file(folder/"sealed.json"),"rows":rows})
        folds.append({"fold":f, "patients":len(group), "cases":len(ids), "variant":seal["variant"],
                      "weight":seal["weight"], "means":dict(zip(names, means.tolist())),
                      "rank_means":dict(zip(names, values.mean(0).tolist()))})
        sums = np.zeros((len(group),3))
        np.add.at(sums, inverse, values.mean(2))
        count = np.bincount(inverse)
        draws = rng.multinomial(len(group), np.full(len(group),1/len(group)), size=5000)
        bootstrap.append((draws@sums)/(draws@count)[:,None])
    if len(patients)!=214 or all_ids != set(range(1609)):
        raise ValueError("Original dataset coverage differs")
    means = {name:float(np.mean([row["means"][name] for row in folds])) for name in names}
    draws = np.mean(bootstrap, axis=0)
    report = {"folds":folds, "patients":len(patients), "cases":len(all_ids), "means":means,
              "delta_vs_reference_pp":100*(means["candidate"]-means["reference"]),
              "delta_vs_e11_pp":100*(means["candidate"]-means["e11"]),
              "absolute_ci95":np.percentile(draws[:,2],[2.5,97.5]).tolist(),
              "paired_reference_ci95_pp":np.percentile(100*(draws[:,2]-draws[:,1]),[2.5,97.5]).tolist(),
              "paired_e11_ci95_pp":np.percentile(100*(draws[:,2]-draws[:,0]),[2.5,97.5]).tolist(),
              "selection":lock_hash,"target":TARGET,"target_pass":means["candidate"]>=TARGET,
              "source":rc.hash_file(__file__),"scope":"locked original four-fold repeated LIDC cohort"}
    rc.write_json(args.run/"assessment/outer_report.json",report)
    print("OUTER_RESULT",json.dumps(means),report["target_pass"],flush=True)


def main(args):
    for name, folder in HISTORY.items():
        setattr(args, name, args.runs_root/folder)
    args.scope = "full"
    args.crossfit = args.run
    args.previous = args.run
    args.cache_run = args.encoder
    args.feature_cache = args.public
    args.image_size = 512
    args.adapt = True
    if args.phase in ("warm","fit","infer"):
        choice = locked(args)["folds"][args.fold]
        args.ct_adapter = choice["variant"]=="ct-adapted"
    phases = {"bind":bind,"lock":lock,"oof":crossfit.prepare,"warm":warm.train,"fit":encoder.train,
              "infer":infer,"score":score}
    phases[args.phase](args)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("bind","lock","oof","warm","fit","infer","score"))
    for key in ("run","data","runs-root"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    main(args)
