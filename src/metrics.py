"""Pure scoring functions: overlap, matching, rates, intervals, calibration and error types.

Every number in the results table comes from here. No file I/O, no model calls and no
knowledge of splits. Callers pass in the random generator. Spec: docs/metrics_spec.md.

Boxes are [x1, y1, x2, y2] in continuous pixel coordinates. Masks are bool arrays.
NaN means undefined, never zero.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils

NAN = float("nan")

# The kind of negative that turns a false positive into an absent-attribute error.
ATTRIBUTE_NEGATIVE = {
    "damaged_cereal_box": "intact_cereal_box",
    "misaligned_label": "straight_label",
}


# ---------------------------------------------------------------- overlap

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union)


def mask_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Returns (iou, empty_union). The caller counts empty unions."""
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0, True
    return float(np.logical_and(a, b).sum() / union), False


def box_from_mask(m: np.ndarray) -> list[float] | None:
    """Tight box around the True pixels, or None for an empty mask."""
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if rows.size == 0:
        return None
    # A pixel at column c covers [c, c + 1), so the far edge is one past the last index.
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


def decode_rle(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


# ---------------------------------------------------------------- ties and ordering

def count_exact_ties(scores: Sequence[float]) -> int:
    """How many scores equal another score exactly."""
    _, counts = np.unique(np.asarray(scores, dtype=float), return_counts=True)
    return int(counts[counts > 1].sum())


def _score_order(scores: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    """Indices from highest score to lowest. Exact ties fall in the order rng shuffled them."""
    scores = np.asarray(scores, dtype=float)
    shuffled = rng.permutation(len(scores))
    return shuffled[np.argsort(-scores[shuffled], kind="stable")]


# ---------------------------------------------------------------- matching

def match(preds: Sequence[Mapping], gts: Sequence[Mapping], iou_fn: Callable[[Mapping, Mapping], float],
          thr: float, rng: np.random.Generator):
    """Greedy one-to-one matching, highest score first.

    preds need "score", gts need "instance_id", and iou_fn(pred, gt) returns a float.
    Returns (pairs, fp_idx, fn_idx, n_score_ties, n_iou_ties), where pairs holds
    (pred index, gt index, iou).
    """
    scores = [p["score"] for p in preds]
    unmatched = list(range(len(gts)))
    pairs, fp_idx, n_iou_ties = [], [], 0
    for i in map(int, _score_order(scores, rng)):
        best_iou, best = 0.0, []
        for j in unmatched:
            iou = iou_fn(preds[i], gts[j])
            if iou > best_iou:
                best_iou, best = iou, [j]
            elif best and iou == best_iou:
                best.append(j)
        if best and best_iou >= thr:
            n_iou_ties += len(best) > 1
            j = min(best, key=lambda g: gts[g]["instance_id"])
            unmatched.remove(j)
            pairs.append((i, j, best_iou))
        else:
            fp_idx.append(i)
    return pairs, sorted(fp_idx), unmatched, count_exact_ties(scores), n_iou_ties


# ---------------------------------------------------------------- counts and rates

def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = float(tp / (tp + fp)) if tp + fp else NAN
    recall = float(tp / (tp + fn)) if tp + fn else NAN
    # Written on counts, so it stays defined when precision is not.
    f1 = float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else NAN
    return precision, recall, f1


def wilson(k: float, n: float, z: float) -> tuple[float, float]:
    if n == 0:
        return NAN, NAN
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def miou(ious: Sequence[float]) -> tuple[float, int]:
    """Mean IoU over matched pairs only, with its n. Report it beside recall."""
    ious = list(ious)
    return (float(np.mean(ious)), len(ious)) if ious else (NAN, 0)


def macro(values: Mapping[str, float]) -> tuple[float, list[str]]:
    """Mean over categories. Any NaN category makes the result NaN and is named."""
    undefined = [name for name, v in values.items() if math.isnan(v)]
    if undefined or not values:
        return NAN, undefined
    return float(np.mean(list(values.values()))), []


# ---------------------------------------------------------------- bootstrap

def bootstrap_stats(per_image_counts, stat_fn: Callable[[np.ndarray], float], n_resamples: int,
                    rng: np.random.Generator) -> np.ndarray:
    """One statistic per resample. Images are resampled, their counts pooled, then stat_fn applied."""
    counts = np.asarray(per_image_counts, dtype=float)
    n_images = counts.shape[0]
    if n_images == 0:
        return np.full(n_resamples, NAN)
    picks = rng.integers(0, n_images, size=(n_resamples, n_images))
    pooled = counts[picks].sum(axis=1)
    return np.array([stat_fn(row) for row in pooled], dtype=float)


def bootstrap_ci(per_image_counts, stat_fn: Callable[[np.ndarray], float], n_resamples: int,
                 rng: np.random.Generator) -> tuple[float, float, int]:
    """Returns (2.5th percentile, 97.5th percentile, resamples dropped as undefined).

    per_image_counts has one row per image, for example [tp, fp, fn].
    """
    stats = bootstrap_stats(per_image_counts, stat_fn, n_resamples, rng)
    defined = stats[~np.isnan(stats)]
    dropped = int(stats.size - defined.size)
    if defined.size == 0:
        return NAN, NAN, dropped
    lo, hi = np.percentile(defined, [2.5, 97.5])
    return float(lo), float(hi), dropped


# ---------------------------------------------------------------- ensemble

def nms(boxes: Sequence[Sequence[float]], scores: Sequence[float], iou: float,
        rng: np.random.Generator) -> tuple[list[int], int]:
    """Greedy NMS, highest score first. Returns (kept indices, tie count).

    A box is dropped when it reaches `iou` with a box already kept.
    """
    kept: list[int] = []
    for i in map(int, _score_order(scores, rng)):
        if all(box_iou(boxes[i], boxes[k]) < iou for k in kept):
            kept.append(i)
    return kept, count_exact_ties(scores)  # counted the same way as match(): two equal scores are 2


# ---------------------------------------------------------------- calibration

def ece(conf: Sequence[float], correct: Sequence[float], n_bins: int) -> tuple[float, list[dict]]:
    """Expected calibration error over fixed equal-width bins on [0, 1], plus the per-bin table."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)  # the last bin includes 1.0
    total, table = 0.0, []
    for b in range(n_bins):
        in_bin = bins == b
        count = int(in_bin.sum())
        row = {"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "count": count,
               "mean_conf": NAN, "accuracy": NAN}
        if count:
            row["mean_conf"] = float(conf[in_bin].mean())
            row["accuracy"] = float(correct[in_bin].mean())
            total += count / conf.size * abs(row["accuracy"] - row["mean_conf"])
        table.append(row)
    return (total if conf.size else NAN), table


# ---------------------------------------------------------------- error types

def fp_type(pred: Mapping, category: str, gts: Sequence[Mapping], negatives: Sequence[Mapping],
            matched_ids, match_iou: float, localization_iou: float) -> str:
    """Why an unmatched box is wrong. First rule that applies wins.

    pred needs "box" and "mask". gts are this category's ground truth with "instance_id",
    "box" and "mask". negatives need "box" and "kind".
    """
    overlaps = [(box_iou(pred["box"], gt["box"]), gt) for gt in gts]
    on_gt = [gt for iou, gt in overlaps if iou >= match_iou]
    if any(gt["instance_id"] in matched_ids for gt in on_gt):
        return "duplicate"
    if any(mask_iou(pred["mask"], gt["mask"])[0] < match_iou for gt in on_gt):
        return "bad_mask"
    kinds_hit = {neg["kind"] for neg in negatives if box_iou(pred["box"], neg["box"]) >= match_iou}
    if ATTRIBUTE_NEGATIVE.get(category) in kinds_hit:
        return "attribute"
    if kinds_hit:
        return "wrong_product"
    best = max((iou for iou, _ in overlaps), default=0.0)
    if localization_iou <= best < match_iou:
        return "localization"
    return "background"


def stage(item: Mapping, detections: Sequence[Mapping], match_iou: float, rng: np.random.Generator,
          fp_label: str | None = None) -> tuple[str, bool]:
    """Which stage started the error. Returns (stage, verifier_also_failed).

    False positive: pass the prediction as item and its fp_type as fp_label.
    Missed ground truth: pass the ground truth as item and, as detections, the DINO boxes
    above threshold, each with "box", "score", "mask" and "kept" (None when the verifier is off).
    """
    if fp_label is not None:
        return ("SAM" if fp_label == "bad_mask" else "DINO"), item.get("kept") is True
    reaching = [d for d in detections if box_iou(d["box"], item["box"]) >= match_iou]
    if not reaching:
        return "DINO", False
    judged = reaching[int(_score_order([d["score"] for d in reaching], rng)[0])]
    if judged.get("kept") is False:
        return "CLIP", False
    if mask_iou(judged["mask"], item["mask"])[0] < match_iou:
        return "SAM", False
    return "other", False
