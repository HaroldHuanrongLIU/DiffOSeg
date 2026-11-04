"""Foreground metrics; empty/empty masks match and expert identities stay fixed."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def pairwise_scores(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float32).reshape(len(b), -1)
    intersection = a @ b.T
    totals = a.sum(1)[:, None] + b.sum(1)[None, :]
    union = totals - intersection
    dice = np.divide(2 * intersection, totals, out=np.ones_like(totals), where=totals > 0)
    iou = np.divide(intersection, union, out=np.ones_like(union), where=union > 0)
    return dice, iou


def consensus_scores(labels, predictions):
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    dice, cross_iou = pairwise_scores(labels, predictions)
    _, label_iou = pairwise_scores(labels, labels)
    _, pred_iou = pairwise_scores(predictions, predictions)
    # The published implementation uses the empirical V-statistic, including diagonals.
    ged_empty_match = 2 * (1 - cross_iou).mean() - (1 - label_iou).mean() - (1 - pred_iou).mean()
    # The released evaluation uses IoU=0 for empty/empty foreground masks.
    # Retain that convention for comparison, and report the metric-correct
    # empty/empty IoU=1 variant separately rather than silently changing it.
    empty_labels = (~labels.reshape(len(labels), -1).any(1)).mean()
    empty_predictions = (~predictions.reshape(len(predictions), -1).any(1)).mean()
    ged = ged_empty_match - (empty_labels - empty_predictions) ** 2
    rows, columns = linear_sum_assignment(1 - dice)
    soft = []
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        value, _ = pairwise_scores(
            (labels.mean(0) > threshold)[None], (predictions.mean(0) > threshold)[None]
        )
        soft.append(value.item())
    return {
        "GED": float(ged), "GED_empty_match": float(ged_empty_match),
        "Dice_soft": float(np.mean(soft)),
        "Dice_match": float(dice[rows, columns].mean()),
        "Dice_max": float(dice.max(axis=1).mean()),
    }


def preference_scores(labels, predictions):
    if len(labels) != 4 or len(predictions) != 4:
        raise ValueError("Preference evaluation requires all four aligned experts")
    dice, _ = pairwise_scores(labels, predictions)
    each = np.diag(dice)
    return {"Dice_each": each.tolist(), "Dice_mean": float(each.mean())}
