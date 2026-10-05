"""Hand-worked checks for src/metrics.py. The table is in docs/metrics_spec.md.

Run: python -m pytest tests/test_metrics.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import metrics  # noqa: E402

THR = 0.5
SEEDS = range(40)


def rng(seed=42):
    return np.random.default_rng(seed)


def by_box(pred, gt):
    return metrics.box_iou(pred["box"], gt["box"])


def mask_of(n_true, shape=(10, 10)):
    """A mask whose first n_true pixels, in row order, are True."""
    m = np.zeros(shape, dtype=bool)
    m.flat[:n_true] = True
    return m


FULL = mask_of(100)


# ---------------------------------------------------------------- overlap

def test_box_iou_half_overlap():
    assert metrics.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(0.3333, abs=1e-4)


def test_box_iou_no_overlap():
    assert metrics.box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_exactly_at_threshold_is_a_hit():
    preds = [{"score": 0.9, "box": [0, 0, 10, 5]}]
    gts = [{"instance_id": 1, "box": [0, 0, 10, 10]}]
    assert by_box(preds[0], gts[0]) == 0.5
    pairs, fp_idx, fn_idx, _, _ = metrics.match(preds, gts, by_box, THR, rng())
    assert pairs == [(0, 0, 0.5)] and fp_idx == [] and fn_idx == []


def test_mask_iou_empty():
    empty = np.zeros((4, 4), dtype=bool)
    assert metrics.mask_iou(empty, empty) == (0.0, True)


def test_mask_iou_value():
    assert metrics.mask_iou(mask_of(30), FULL) == (0.3, False)


def test_box_from_mask_is_tight_and_continuous():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 3:9] = True
    assert metrics.box_from_mask(m) == [3.0, 2.0, 9.0, 5.0]
    assert metrics.box_from_mask(np.zeros((10, 10), dtype=bool)) is None


def test_decode_rle_round_trip_with_json_style_counts():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 3:9] = True
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")  # what json.load gives back
    assert np.array_equal(metrics.decode_rle(rle), m)


# ---------------------------------------------------------------- counts and rates

def test_precision_recall_f1():
    p, r, f1 = metrics.prf(8, 2, 4)
    assert p == pytest.approx(0.8000, abs=1e-4)
    assert r == pytest.approx(0.6667, abs=1e-4)
    assert f1 == pytest.approx(0.7273, abs=1e-4)


def test_no_predictions():
    p, r, f1 = metrics.prf(0, 0, 5)
    assert math.isnan(p) and r == 0.0 and f1 == 0.0


def test_nothing_at_all():
    assert all(math.isnan(v) for v in metrics.prf(0, 0, 0))


@pytest.mark.parametrize("k, n, lo, hi", [
    (87, 100, 0.7902, 0.9224),
    (20, 30, 0.4878, 0.8077),
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
])
def test_wilson(k, n, lo, hi):
    got_lo, got_hi = metrics.wilson(k, n, 1.96)
    assert got_lo == pytest.approx(lo, abs=1e-4)
    assert got_hi == pytest.approx(hi, abs=1e-4)
    assert 0.0 <= got_lo <= got_hi <= 1.0


def test_wilson_zero_of_zero():
    assert all(math.isnan(v) for v in metrics.wilson(0, 0, 1.96))


def test_miou_carries_its_n():
    assert metrics.miou([0.6, 0.8]) == (pytest.approx(0.7), 2)
    value, n = metrics.miou([])
    assert math.isnan(value) and n == 0


def test_macro_with_nan_names_the_category():
    value, undefined = metrics.macro(
        {"cereal_box": 0.8, "damaged_cereal_box": float("nan"), "misaligned_label": 0.6})
    assert math.isnan(value) and undefined == ["damaged_cereal_box"]
    assert metrics.macro({"a": 0.8, "b": 0.6}) == (pytest.approx(0.7), [])


# ---------------------------------------------------------------- calibration

def test_ece():
    value, table = metrics.ece([0.95] * 4 + [0.55] * 2, [1, 1, 1, 0, 1, 0], 10)
    assert value == pytest.approx(0.1500, abs=1e-4)
    assert [row["count"] for row in table] == [0, 0, 0, 0, 0, 2, 0, 0, 0, 4]
    assert math.isnan(table[0]["accuracy"])


def test_ece_last_bin_includes_one():
    value, table = metrics.ece([1.0, 1.0], [1, 1], 10)
    assert table[9]["count"] == 2 and value == 0.0


# ---------------------------------------------------------------- matching and ties

def test_matching_duplicate():
    gts = [{"instance_id": 1, "box": [0, 0, 10, 10], "mask": FULL}]
    preds = [{"score": 0.9, "box": [0, 0, 10, 10], "mask": FULL},
             {"score": 0.8, "box": [0, 0, 10, 9], "mask": mask_of(90)}]
    pairs, fp_idx, fn_idx, _, _ = metrics.match(preds, gts, by_box, THR, rng())
    assert len(pairs) == 1 and fp_idx == [1] and fn_idx == []
    assert metrics.fp_type(preds[1], "cereal_box", gts, [], {1}, THR, 0.1) == "duplicate"


def test_score_tie_separate_objects():
    gts = [{"instance_id": 1, "box": [0, 0, 10, 10]}, {"instance_id": 2, "box": [20, 0, 30, 10]}]
    preds = [{"score": 0.7, "box": [0, 0, 10, 10]}, {"score": 0.7, "box": [20, 0, 30, 10]}]
    for seed in SEEDS:
        pairs, fp_idx, fn_idx, n_score_ties, _ = metrics.match(preds, gts, by_box, THR, rng(seed))
        assert len(pairs) == 2 and fp_idx == [] and fn_idx == []
        assert n_score_ties == 2


def test_score_tie_same_object():
    gts = [{"instance_id": 1, "box": [0, 0, 10, 10]}]
    preds = [{"score": 0.7, "box": [0, 0, 10, 10]}, {"score": 0.7, "box": [0, 0, 10, 9]}]
    winners = set()
    for seed in SEEDS:
        pairs, fp_idx, fn_idx, n_score_ties, _ = metrics.match(preds, gts, by_box, THR, rng(seed))
        assert len(pairs) == 1 and len(fp_idx) == 1 and fn_idx == []
        assert n_score_ties == 2
        again = metrics.match(preds, gts, by_box, THR, rng(seed))[0]
        assert again == pairs  # same seed, same winner
        winners.add(pairs[0][0])
    assert winners == {0, 1}  # the seed decides, which is why ties are logged


def test_iou_tie_takes_lower_instance_id():
    gts = [{"instance_id": 7, "box": [0, 0, 10, 10]}, {"instance_id": 3, "box": [10, 0, 20, 10]}]
    preds = [{"score": 0.9, "box": [0, 0, 20, 10]}]  # IoU 0.5 with each
    pairs, _, fn_idx, _, n_iou_ties = metrics.match(preds, gts, by_box, THR, rng())
    assert pairs == [(0, 1, 0.5)] and fn_idx == [0] and n_iou_ties == 1


def test_count_exact_ties():
    assert metrics.count_exact_ties([0.7, 0.7, 0.2]) == 2
    assert metrics.count_exact_ties([0.9, 0.5, 0.2]) == 0
    assert metrics.count_exact_ties([]) == 0


def test_nms_tie():
    boxes, scores = [[0, 0, 10, 10], [0, 0, 10, 9]], [0.7, 0.7]
    winners = set()
    for seed in SEEDS:
        kept, n_ties = metrics.nms(boxes, scores, 0.5, rng(seed))
        assert len(kept) == 1 and n_ties == 2  # the same count match() gives for two equal scores
        assert metrics.nms(boxes, scores, 0.5, rng(seed))[0] == kept  # same seed, same box
        winners.add(kept[0])
    assert winners == {0, 1}


def test_nms_keeps_separate_boxes_highest_first():
    boxes = [[0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 9]]
    kept, n_ties = metrics.nms(boxes, [0.5, 0.9, 0.8], 0.5, rng())
    assert kept == [1, 2] and n_ties == 0


# ---------------------------------------------------------------- bootstrap

def f1_of(counts):
    return metrics.prf(*counts)[2]


def test_bootstrap_pools_counts():
    per_image = [(1, 0, 0), (0, 0, 9)]
    assert f1_of(np.sum(per_image, axis=0)) == pytest.approx(2 / 11)
    stats = metrics.bootstrap_stats(per_image, f1_of, 200, rng())
    # Pooled resamples can only be both-first (1.0), mixed (2/11) or both-second (0.0).
    # The mean of per-image F1 for a mixed draw would be 0.5, and it never shows up.
    assert set(np.round(stats, 4)) == {0.0, 0.1818, 1.0}


def test_bootstrap_ci_drops_undefined_resamples():
    per_image = [(1, 0, 0), (0, 0, 0)]  # drawing the empty image twice leaves F1 undefined
    lo, hi, dropped = metrics.bootstrap_ci(per_image, f1_of, 200, rng())
    assert (lo, hi) == (1.0, 1.0) and 0 < dropped < 200
    assert metrics.bootstrap_ci(per_image, f1_of, 200, rng()) == (lo, hi, dropped)


# ---------------------------------------------------------------- error types and stages

def test_bad_mask_type_and_stage():
    gts = [{"instance_id": 1, "box": [0, 0, 10, 10], "mask": FULL}]
    pred = {"score": 0.9, "box": [0, 0, 10, 8], "mask": mask_of(30), "kept": True}
    assert by_box(pred, gts[0]) == pytest.approx(0.8)
    assert metrics.mask_iou(pred["mask"], FULL)[0] == pytest.approx(0.3)
    label = metrics.fp_type(pred, "cereal_box", gts, [], set(), THR, 0.1)
    assert label == "bad_mask"
    assert metrics.stage(pred, [], THR, rng(), fp_label=label) == ("SAM", True)


def test_miss_precedence():
    gt = {"instance_id": 1, "box": [0, 0, 10, 10], "mask": FULL}
    rejected = {"score": 0.6, "box": [0, 0, 10, 10], "mask": FULL, "kept": False}
    bad_mask = {"score": 0.4, "box": [0, 0, 10, 9], "mask": mask_of(30), "kept": True}
    for detections in ([rejected, bad_mask], [bad_mask, rejected]):
        assert metrics.stage(gt, detections, THR, rng()) == ("CLIP", False)


def test_miss_stages():
    gt = {"instance_id": 1, "box": [0, 0, 10, 10], "mask": FULL}
    far = {"score": 0.9, "box": [50, 50, 60, 60], "mask": FULL, "kept": True}
    bad_mask = {"score": 0.4, "box": [0, 0, 10, 9], "mask": mask_of(30), "kept": None}
    good = {"score": 0.4, "box": [0, 0, 10, 10], "mask": FULL, "kept": True}
    assert metrics.stage(gt, [], THR, rng()) == ("DINO", False)
    assert metrics.stage(gt, [far], THR, rng()) == ("DINO", False)
    assert metrics.stage(gt, [bad_mask], THR, rng()) == ("SAM", False)
    assert metrics.stage(gt, [good], THR, rng()) == ("other", False)


def test_fp_type_order():
    pred = {"score": 0.9, "box": [0, 0, 20, 10], "mask": mask_of(200, (10, 20))}
    dented = {"instance_id": 1, "box": [0, 0, 10, 10], "mask": np.zeros((10, 20), dtype=bool)}
    intact = {"kind": "intact_cereal_box", "box": [10, 0, 20, 10]}
    args = ("damaged_cereal_box", [dented], [intact])
    assert metrics.fp_type(pred, *args, {1}, THR, 0.1) == "duplicate"
    # Without the matched ground truth, the same box is the absent-attribute error.
    assert metrics.fp_type(pred, "damaged_cereal_box", [], [intact], set(), THR, 0.1) == "attribute"


def test_fp_type_remaining_labels():
    def box_pred(box):
        return {"score": 0.9, "box": box, "mask": FULL}

    gts = [{"instance_id": 1, "box": [0, 0, 10, 10], "mask": FULL}]
    intact = {"kind": "intact_cereal_box", "box": [40, 0, 50, 10]}
    soup = {"kind": "distractor", "box": [60, 0, 70, 10]}
    negatives = [intact, soup]

    def label(box, category="damaged_cereal_box"):
        return metrics.fp_type(box_pred(box), category, gts, negatives, {1}, THR, 0.1)

    assert label([60, 0, 70, 10]) == "wrong_product"
    assert label([40, 0, 50, 10], category="cereal_box") == "wrong_product"  # no attribute for this category
    assert label([6, 0, 16, 10]) == "localization"  # IoU 0.25 with the ground truth
    assert label([80, 80, 90, 90]) == "background"
    assert metrics.stage(box_pred([80, 80, 90, 90]), [], THR, rng(), fp_label="background") == ("DINO", False)
