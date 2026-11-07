"""Learn which single-slice predictor to use on their disagreement pixels."""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import normal_refiner as normal
from reproduction import normal_reliability as reliability
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import rater_area as area
from reproduction import texture_study as incumbent

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/disagreement_protocol.md"
SCORE_COLUMNS = 11


def component_fraction(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    portions = stats[:, cv2.CC_STAT_AREA].astype(np.float32) / max(int(mask.sum()), 1)
    portions[0] = 0
    return portions[labels]


def features(image, old, new, contrast):
    a, b = old > 0, new > 0
    rank, yy, xx = np.nonzero(a != b)
    points = rank, yy, xx
    values = [rank.astype(np.float32), old[points], new[points]]
    values.extend(z[:, yy, xx][k] for z in (old, new) for k in range(4))
    middle = a.sum(0) >= 2
    ring = cv2.dilate(middle.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & ~middle
    im = image.astype(np.float32)
    foreground = float(np.median(im[middle])) if middle.any() else float(np.quantile(im, 0.8))
    background = float(np.median(im[ring])) if ring.any() else float(np.quantile(im, 0.2))
    span = max(foreground-background, 10.0)
    low = cv2.GaussianBlur(im, (0, 0), 1.2)
    smooth = cv2.GaussianBlur(im, (0, 0), 4.0)
    dy, dx = np.gradient(low)
    samples = [im/255, (im-background)/span, (im-low)/span, (im-smooth)/span,
               np.hypot(dx, dy)/span, contrast, a.sum(0), b.sum(0)]
    values.extend(field[yy, xx] for field in samples)
    for masks in (a, b):
        distance = np.stack([rc.signed(mask) for mask in masks])
        mass = np.stack([component_fraction(mask) for mask in masks])
        values.extend((distance[points], mass[points]))
    a_area, b_area = a.sum((1, 2)), b.sum((1, 2))
    cy, cx = np.argwhere(middle).mean(0) if middle.any() else (64.5, 64.5)
    radius = np.sqrt(max(int(middle.sum()), 1)/np.pi)
    values.extend((np.log1p(a_area[rank]), np.log1p(b_area[rank]),
                   np.log((b_area[rank]+1)/(a_area[rank]+1)),
                   (xx-cx)/radius, (yy-cy)/radius, np.full(len(rank), span/255)))
    return np.clip(np.stack(values, axis=1), -24, 24).astype(np.float32), points


def select_mask(old, new, scalar, points, probability):
    result = old.copy()
    fallback = scalar[points]
    result[points] = np.where(probability > 0.6, new[points],
                              np.where(probability < 0.4, old[points], fallback))
    return result


def run(args):
    images, truth, _, uids = native.inputs(args.data)
    _, _, valid = native.splits(uids, args.fold)
    group = type("Groups", (), {"series_uid": uids})()
    library, checked = polar.calibration_split(group, valid, args.fold)
    ids = library+checked
    cache = args.texture / f"features-{args.fold}-inner.npz"
    prior = incumbent.read(cache.with_suffix(".json"))
    if prior["indices"] != ids or prior["payload"] != rc.hash_file(cache):
        raise ValueError("Incumbent cache differs")
    with np.load(cache, allow_pickle=False) as archive:
        original = np.unpackbits(archive["old"], axis=-1).astype(bool)
        base = incumbent.anchor_logits(archive["base"].astype(np.float32), original)
    parent = incumbent.read(args.encoder / f"inner-{args.fold}-adapted.json")
    path = args.encoder / f"pred-{args.fold}-adapted.npz"
    if parent["library"] != library or parent["check"] != checked or rc.hash_file(path) != parent["prediction"]:
        raise ValueError("Adapted prediction identity differs")
    with np.load(path, allow_pickle=False) as archive:
        if archive["ids"].tolist() != ids:
            raise ValueError("Adapted case order differs")
        adapted = archive["logits"].astype(np.float32)
        scalar = np.unpackbits(archive["masks"], axis=-1).astype(bool)
    shift = incumbent.read(args.align / f"fold-{args.fold}-inner-registered/fit.json")["applied_shift"]
    registered = np.stack([area.register_image(images[idx], shift) for idx in ids])
    contrast = reliability.contrast(registered[:, None], normal.image_guide(base))[:, 0]
    cases = [features(image, old, new, ct) for image, old, new, ct in zip(registered, base, adapted, contrast)]
    names = [str(uids[idx]).split("_")[0] for idx in library]
    _, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
    patient_weight = 0.5+0.5*len(library)/(len(counts)*counts[inverse])
    xs, ys, weights = [], [], []
    for j, idx in enumerate(library):
        x, points = cases[j]
        new = adapted[j] > 0
        label = truth[idx].astype(bool)[points]
        size = original[j].sum((1, 2))+new.sum((1, 2))+1
        xs.append(x)
        ys.append(new[points] == label)
        weights.append(patient_weight[j]*np.where(label, 1.1, 0.9)/size[points[0]])
    x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    weight /= weight.mean()
    directory = args.run / f"fit-{args.fold}"
    directory.mkdir(exist_ok=False)
    models, receipts = {}, {}
    for name, width in (("scores_only", SCORE_COLUMNS), ("context", x.shape[1])):
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                  max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=15,
                  early_stopping=False, random_state=88171+args.fold)
        model.fit(x[:, :width], y, sample_weight=weight)
        model_path = directory / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        receipts[name] = {"sha256": rc.hash_file(model_path), "columns": width}
        models[name] = model
    lock = {"fold": args.fold, "library": library, "check_indices": checked,
            "source": rc.hash_file(__file__), "protocol": rc.hash_file(PROTOCOL),
            "fit_pixels": len(y), "models": receipts, "adaptive_source": parent["prediction"],
            "reference_source": prior["payload"], "fixed_blend_weight": parent["weight"]}
    rc.write_json(args.run / f"lock-{args.fold}.json", lock)
    count = len(library)
    score = lambda masks: float(scoring.dice(truth[checked], masks).mean())
    results = {"reference": score(original[count:]), "fixed_blend": score(scalar[count:])}
    selected = {}
    for name, model in models.items():
        width = receipts[name]["columns"]
        masks = []
        for j in range(count, len(ids)):
            rows, points = cases[j]
            new = adapted[j] > 0
            probability = model.predict_proba(rows[:, :width])[:, 1] if len(rows) else np.empty(0)
            mask = select_mask(original[j], new, scalar[j], points, probability)
            agree = original[j] == new
            if not np.array_equal(mask[agree], new[agree]):
                raise ValueError("Agreement pixels changed")
            masks.append(mask)
        masks = np.stack(masks)
        results[name] = score(masks)
        selected[name] = np.packbits(masks, axis=-1)
    payload = args.run / f"pred-inner-{args.fold}.npz"
    np.savez_compressed(payload, ids=checked, **selected)
    rc.write_json(args.run / f"inner-{args.fold}.json", {
        **lock, "scores": results, "primary": "context", "prediction": rc.hash_file(payload)
    })
    print("DISAGREEMENT_CHECK", args.fold, len(y), results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "data", "texture", "encoder", "align"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    cv2.setNumThreads(1)
    run(parser.parse_args())
