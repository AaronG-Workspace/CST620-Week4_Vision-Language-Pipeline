"""Make every choice the pipeline needs on val, then freeze it so eval.py has nothing left to decide.

Spec: docs/tune_val_spec.md. Grids, the objective, the order and the tie-breaks all come from
config.yaml under tuning. Nothing here opens the test split.

    python src/tune_val.py                 # cache if needed, select, report, freeze
    python src/tune_val.py --stage select  # reuse caches, redo selection and the report only
    python src/tune_val.py --quick         # 3 val images, a smoke test; never writes frozen.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # before anything imports transformers

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

import data  # noqa: E402
import metrics  # noqa: E402
import models as models_module  # noqa: E402
import pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTE_KIND = metrics.ATTRIBUTE_NEGATIVE


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid(spec) -> list[float]:
    start, stop, step = spec
    return [round(start + i * step, 10) for i in range(round((stop - start) / step) + 1)]  # integer steps, no drift


def _rng(seed: int, key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(row: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


def _load(path: Path):
    return torch.load(path, weights_only=False)  # our own cache files: tensors plus plain lists and arrays


def _json_safe(value):
    """NaN is not valid JSON, so an undefined number is written as null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


# ---------------------------------------------------------------- stage 1: cache

def build_cache(rows: list[dict], cfg: dict, prompts: dict, truth: dict) -> None:
    """One Grounding DINO pass per image and wording, one SAM encode per image, masks and CLIP crops per box."""
    m = models_module.get_models()
    tuning, categories = cfg["tuning"], prompts["grounding_dino"]["categories"]
    lowest = _grid(tuning["dino_threshold_grid"])[0]
    base_key = [models_module.fingerprint_sha256(), tuning["cache_floor"], lowest, tuning["clip_crop_pad_frac"]]
    seconds = {"dino": 0.0, "sam_encode": 0.0, "sam_decode": 0.0, "clip": 0.0}
    for d in ("dino", "sam", "oracle"):
        _p("cache", "val", d).mkdir(parents=True, exist_ok=True)

    for n, row in enumerate(rows, 1):
        image = None
        sam_path = _p("cache", "val", "sam", f"{row['asset_id']}.pt")
        embedding = None

        def need_image():
            nonlocal image, embedding
            if image is None:
                image = pipeline.prepare(Image.open(_p(row["path"])))
            if embedding is None:
                if sam_path.exists() and _load(sam_path)["key"] == [row["md5"], base_key[0]]:
                    embedding = _load(sam_path)["embedding"]
                else:
                    start = time.perf_counter()
                    embedding = pipeline.sam_embed(m, image)
                    seconds["sam_encode"] += time.perf_counter() - start
                    torch.save({"key": [row["md5"], base_key[0]], "embedding": embedding.float()}, sam_path)

        for category, entry in categories.items():
            for k, wording in enumerate(entry["wordings"]):
                path = _p("cache", "val", "dino", f"{row['asset_id']}__{category}__{k}.pt")
                key = [row["md5"], wording, *base_key]
                if path.exists() and _load(path)["key"] == key:
                    continue  # a changed wording recomputes only that wording
                need_image()
                start = time.perf_counter()
                raw = pipeline.forward_dino(m, image, wording)
                seconds["dino"] += time.perf_counter() - start
                keep = torch.sigmoid(raw["logits"]).max(dim=-1).values >= tuning["cache_floor"]
                raw = {"logits": raw["logits"][keep].float(), "pred_boxes": raw["pred_boxes"][keep].float(),
                       "input_ids": raw["input_ids"], "query_index": raw["query_index"][keep]}  # float32, not float16
                detections = pipeline.select(m, raw, image.size, wording, lowest)
                start = time.perf_counter()
                masks, ious = pipeline.segment(m, image, embedding, [d.box for d in detections])
                seconds["sam_decode"] += time.perf_counter() - start
                start = time.perf_counter()
                crops = pipeline.embed_crops(m, image, [d.box for d in detections], tuning["clip_crop_pad_frac"])
                seconds["clip"] += time.perf_counter() - start
                torch.save({"key": key, "raw": raw, "size": image.size, "sam_iou": ious, "clip": crops.float(),
                            "masks": np.packbits(np.array(masks, dtype=bool).reshape(len(masks), -1), axis=1),
                            "boxes": [{"box": d.box, "score": d.score, "query_index": d.query_index,
                                       "word_scores": d.word_scores, "other": d.other_token_score} for d in detections]}, path)

        oracle_path = _p("cache", "val", "oracle", f"{row['asset_id']}.pt")
        if not (oracle_path.exists() and _load(oracle_path)["key"] == [row["md5"], base_key[0]]):
            need_image()
            gts = {g["instance_id"]: g for group in truth[row["asset_id"]]["gts"].values() for g in group}
            masks, _ = pipeline.segment(m, image, embedding, [g["box"] for g in gts.values()])
            ious = {i: metrics.mask_iou(mask, metrics.record_mask(g))[0] for (i, g), mask in zip(gts.items(), masks)}
            torch.save({"key": [row["md5"], base_key[0]], "iou": ious}, oracle_path)
        print(f"[cache] image {n}/{len(rows)}")
    for stage, value in seconds.items():
        print(f"[time] cache {stage:<10} {value:7.1f} s")


def selftest(rows: list[dict], cfg: dict, prompts: dict) -> None:
    """On 2 images: our selection equals the library's, and selection from the cache equals a live pass."""
    m = models_module.get_models()
    category, entry = next(iter(prompts["grounding_dino"]["categories"].items()))
    wording, lowest = entry["wordings"][0], _grid(cfg["tuning"]["dino_threshold_grid"])[0]
    worst = 0.0
    for row in rows[:2]:
        image = pipeline.prepare(Image.open(_p(row["path"])))
        worst = max(worst, pipeline.selftest(m, [image], wording, lowest, cfg["tuning"]["dino_text_threshold"]))
        live = pipeline.detect(m, image, wording, lowest, cfg["tuning"]["dino_text_threshold"])
        cached = _load(_p("cache", "val", "dino", f"{row['asset_id']}__{category}__0.pt"))["boxes"]
        if len(live) != len(cached):
            sys.exit(f"Self-test failed: {len(live)} live boxes, {len(cached)} cached, on {row['asset_id']}.")
        for a, b in zip(live, cached):
            worst = max(worst, abs(a.score - b["score"]), *(abs(x - y) for x, y in zip(a.box, b["box"])))
    if worst > 1e-4:
        sys.exit(f"Self-test failed: cached and live post-processing differ by {worst}.")
    print(f"[selftest] cached vs live vs library on {min(2, len(rows))} images: largest difference {worst:.2e}")


def load_records(rows: list[dict], cfg: dict, prompts: dict, truth: dict) -> list[dict]:
    """Per image: ground truth, and every cached box with its mask overlap against that ground truth."""
    records = []
    for row in rows:
        image_truth = truth[row["asset_id"]]
        record = {"name": row["asset_id"], "gts": image_truth["gts"], "negatives": image_truth["negatives"], "boxes": {},
                  "oracle": _load(_p("cache", "val", "oracle", f"{row['asset_id']}.pt"))["iou"]}
        gt_masks = {g["instance_id"]: metrics.record_mask(g) for group in image_truth["gts"].values() for g in group}
        for category, entry in prompts["grounding_dino"]["categories"].items():
            for k, wording in enumerate(entry["wordings"]):
                cached = _load(_p("cache", "val", "dino", f"{row['asset_id']}__{category}__{k}.pt"))
                width, height = cached["size"]
                masks = np.unpackbits(cached["masks"], axis=1)[:, :width * height].reshape(-1, height, width).astype(bool)
                boxes = []
                for box, mask, sam_iou, clip in zip(cached["boxes"], masks, cached["sam_iou"], cached["clip"]):
                    overlap = {g["instance_id"]: metrics.mask_iou(mask, gt_masks[g["instance_id"]])[0]
                               for g in image_truth["gts"][category]}
                    boxes.append({**box, "wording": wording, "sam_iou_pred": sam_iou, "clip": clip, "miou": overlap,
                                  "image": row["asset_id"], "kept": None})
                record["boxes"][(category, k)] = boxes
        records.append(record)
    return records


# ---------------------------------------------------------------- stage 2: select

def _overlap(pred, gt) -> float:
    return pred["miou"].get(gt["instance_id"], 0.0)


class Selector:
    def __init__(self, records, cfg, prompts):
        self.records, self.cfg, self.tuning, self.prompts = records, cfg, cfg["tuning"], prompts
        self.categories = {c: e["wordings"] for c, e in prompts["grounding_dino"]["categories"].items()}
        seed = int(cfg["seed"])
        self.rng = {"match": _rng(seed, 201), "bootstrap": _rng(seed, 202), "nms": _rng(seed, 203), "stage": _rng(seed, 204)}
        self.ties: list[dict] = []
        self.thr = self.tuning["match_iou"]

    def preds(self, record, category, setup, threshold, keep=None):
        """One image's boxes for a category under a setup (a wording index, or 'ensemble') above the threshold."""
        if setup == "ensemble":
            pooled = [b for k in range(len(self.categories[category])) for b in record["boxes"][(category, k)]
                      if b["score"] > threshold]
            kept, _ = metrics.nms([b["box"] for b in pooled], [b["score"] for b in pooled],
                                  self.tuning["ensemble"]["nms_iou"], self.rng["nms"])
            chosen = [pooled[i] for i in kept]
        else:
            chosen = [b for b in record["boxes"][(category, setup)] if b["score"] > threshold]
        return [b for b in chosen if keep is None or keep(b)]

    def score(self, category, setup, threshold, keep=None):
        images = [(self.preds(r, category, setup, threshold, keep), r["gts"][category]) for r in self.records]
        counts, details, score_ties, iou_ties = metrics.score_images(images, self.thr, self.rng["match"], _overlap)
        return counts, details, images, score_ties + iou_ties

    def f1(self, category, setup, threshold, keep=None):
        counts, _, _, ties = self.score(category, setup, threshold, keep)
        return metrics.prf(*counts[:, :3].sum(axis=0)), ties

    def choose(self, step: str, options: list, values: list[float], prefer_last: bool):
        """The best value, with exact ties logged and broken by the rule config.yaml fixes for this step."""
        finite = [v for v in values if not math.isnan(v)]
        if not finite:
            sys.exit(f"Selection step {step} has no defined value to compare. Val may have no boxes or no ground truth.")
        best = max(finite)
        winners = [o for o, v in zip(options, values) if v == best]
        self.ties.append({"step": step, "exact_ties_at_best": len(winners) - 1,
                          "exact_ties_anywhere": metrics.count_exact_ties([v for v in values if not math.isnan(v)])})
        return winners[-1] if prefer_last else winners[0]

    def run(self) -> dict:
        t = self.tuning
        # 1. One global Grounding DINO threshold: macro F1, every wording averaged, verifier off.
        sweep = []
        for threshold in _grid(t["dino_threshold_grid"]):
            per_category, precisions, recalls, n_ties = [], [], [], 0
            for category, wordings in self.categories.items():
                scores = []
                for k in range(len(wordings)):
                    (p, r, f1), ties = self.f1(category, k, threshold)
                    scores.append(f1), precisions.append(p), recalls.append(r)
                    n_ties += ties
                per_category.append(float(np.mean(scores)))
            sweep.append({"threshold": threshold, "macro_f1": float(np.mean(per_category)),
                          "mean_precision": float(np.nanmean(precisions)), "mean_recall": float(np.nanmean(recalls)),
                          "match_ties": n_ties})
        threshold = self.choose("dino_threshold", [s["threshold"] for s in sweep], [s["macro_f1"] for s in sweep], True)

        # 2. Headline wording per category: each single wording, or the ensemble. Ties: ensemble, then list order.
        headline = {}
        for category, wordings in self.categories.items():
            options = ["ensemble"] + list(range(len(wordings)))
            values = [self.f1(category, o, threshold)[0][2] for o in options]
            pick = self.choose(f"headline_{category}", options, values, False)
            headline[category] = pick if pick == "ensemble" else wordings[pick]
        setups = {c: ("ensemble" if h == "ensemble" else self.categories[c].index(h)) for c, h in headline.items()}

        # 3. Temperature: lowest negative log likelihood of the positive-class probability.
        m = models_module.get_models()
        verifier = self.prompts["clip_verifier"]
        banks = {c: pipeline.contrast_bank(m, verifier["contrasts"][c], verifier["templates"]) for c in self.categories}
        owned = {c: [(r, b) for r in self.records for b in self.preds(r, c, setups[c], threshold)] for c in self.categories}
        kept = {c: [b for _, b in owned[c]] for c in self.categories}
        # Label by box IoU with any ground truth of the category: the verifier judges what is inside the crop.
        labels = {c: np.array([float(any(metrics.box_iou(b["box"], g["box"]) >= self.thr for g in r["gts"][c]))
                               for r, b in owned[c]]) for c in self.categories}

        def verifier_at(temperature):
            out = {}
            for c in self.categories:
                clips = torch.stack([b["clip"] for b in kept[c]]) if kept[c] else torch.zeros((0, 512))
                out[c] = pipeline.verifier_scores(m, clips, banks[c][0], banks[c][1], temperature)
            return out

        def nll(temperature):
            scores = verifier_at(temperature)
            p = np.clip(np.concatenate([scores[c][0] for c in self.categories]), 1e-7, 1 - 1e-7)
            y = np.concatenate([labels[c] for c in self.categories])
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(y) else float("nan")

        temperatures = _grid(t["temperature_grid"])
        losses = [nll(temperature) for temperature in temperatures]
        temperature = self.choose("clip_temperature", temperatures, [-v for v in losses], False)
        calibration = {}
        for label, value in (("T=1", 1.0), ("chosen", temperature)):
            scores = verifier_at(value)
            conf = np.concatenate([scores[c][0] for c in self.categories])
            calibration[label] = metrics.ece(conf, np.concatenate([labels[c] for c in self.categories]), t["ece_bins"])
        scores = verifier_at(temperature)
        for c in self.categories:
            for b, p, margin in zip(kept[c], scores[c][0], scores[c][1]):
                b["p_positive"], b["margin"] = float(p), float(margin)

        # 4. Margin: macro F1 with the verifier on, at the headline setups. Ties go to the higher margin.
        margins = _grid(t["margin_grid"])
        macro = [float(np.mean([self.f1(c, setups[c], threshold, lambda b, mm=mm: b["margin"] >= mm)[0][2]
                                for c in self.categories])) for mm in margins]
        margin = self.choose("clip_margin", margins, macro, True)
        for c in self.categories:
            for b in kept[c]:
                b["kept"] = bool(b["margin"] >= margin)
        return {"threshold": threshold, "headline": headline, "setups": setups, "temperature": temperature,
                "margin": margin, "sweep": sweep, "nll": dict(zip(temperatures, losses)), "calibration": calibration,
                "margin_sweep": dict(zip(margins, macro))}


# ---------------------------------------------------------------- stage 3: report

def report(sel: Selector, chosen: dict, out: Path, cfg: dict) -> dict:
    z, n_boot = cfg["reporting"]["wilson_z"], cfg["reporting"]["bootstrap_resamples"]
    threshold, margin = chosen["threshold"], chosen["margin"]
    on = lambda b: b.get("kept") is True  # noqa: E731
    for s in chosen["sweep"]:
        s["chosen"] = s["threshold"] == threshold
    _write_csv(out / "sweep.csv", [_fmt(s) for s in chosen["sweep"]])
    plt.figure(figsize=(6, 3.5))
    plt.plot([s["threshold"] for s in chosen["sweep"]], [s["macro_f1"] for s in chosen["sweep"]], marker="o")
    plt.axvline(threshold, linestyle="--", color="gray")
    plt.xlabel("Grounding DINO threshold"), plt.ylabel("macro F1 on val, verifier off"), plt.tight_layout()
    plt.savefig(out / "sweep.png", dpi=120), plt.close()

    # Wording sensitivity: every wording and the ensemble, verifier off and on.
    sensitivity, summary = [], {}
    for category, wordings in sel.categories.items():
        groups = len({g["group_id"] for r in sel.records for g in r["gts"][category]})
        for setup in list(range(len(wordings))) + ["ensemble"]:
            name = "ensemble" if setup == "ensemble" else wordings[setup]
            # The verifier's decision was computed for the headline boxes, so "on" rows exist for that setup only.
            for state, keep in (("off", None), ("on", on)) if setup == chosen["setups"][category] else (("off", None),):
                counts, _, _, _ = sel.score(category, setup, threshold, keep)
                row = {"category": category, "setup": name, "verifier": state, "headline": setup == chosen["setups"][category],
                       "groups": groups, **metrics.summarize(counts, z, n_boot, sel.rng["bootstrap"])}
                sensitivity.append(_fmt(row))
                if row["headline"]:
                    summary[(category, state)] = row
        single = [r["f1"] for r in sensitivity if r["category"] == category and r["verifier"] == "off" and r["setup"] != "ensemble"]
        for r in sensitivity:
            if r["category"] == category:
                r["f1_spread_across_wordings"] = round(max(single) - min(single), 4)
    _write_csv(out / "sensitivity.csv", sensitivity)

    # Stage table and hallucination candidates, at the headline setup with the verifier on.
    stage_rows, candidates, rates = [], [], []
    loc = cfg["reporting"]["fp_localization_iou"]
    for category in sel.categories:
        setup, tally = chosen["setups"][category], {}
        kind = ATTRIBUTE_KIND.get(category)
        for record in sel.records:
            proposed = sel.preds(record, category, setup, threshold)
            kept = [b for b in proposed if on(b)]
            gts = record["gts"][category]
            pairs, fp_idx, fn_idx, _, _ = metrics.match(kept, gts, _overlap, sel.thr, sel.rng["match"])
            matched = {gts[j]["instance_id"] for _, j, _ in pairs}
            typed = lambda b: metrics.fp_type(b, category, gts, record["negatives"], matched, sel.thr, loc, _overlap)  # noqa: E731
            for i in fp_idx:
                label = typed(kept[i])
                stage, also = metrics.stage(kept[i], [], sel.thr, sel.rng["stage"], fp_label=label)
                tally[("fp", stage)] = tally.get(("fp", stage), 0) + 1
                tally["verifier_also_failed"] = tally.get("verifier_also_failed", 0) + also
            for j in fn_idx:
                stage, _ = metrics.stage(gts[j], proposed, sel.thr, sel.rng["stage"], mask_iou_of=_overlap)
                tally[("miss", stage)] = tally.get(("miss", stage), 0) + 1
            # False positives the verifier caught: proposed, rejected, and wrong without it.
            _, fp_off, _, _, _ = metrics.match(proposed, gts, _overlap, sel.thr, sel.rng["match"])
            tally["fp_caught_by_verifier"] = tally.get("fp_caught_by_verifier", 0) + sum(not on(proposed[i]) for i in fp_off)
            if kind:
                for b in proposed:
                    if any(n["kind"] == kind and metrics.box_iou(b["box"], n["box"]) >= sel.thr for n in record["negatives"]):
                        candidates.append({"image_id": record["name"], "category": category, "wording": b["wording"],
                                           "score": round(b["score"], 4), "box": [round(v, 1) for v in b["box"]],
                                           "word_scores": json.dumps({w: round(s, 3) for w, s in b["word_scores"].items()}),
                                           "verifier_margin": round(b["margin"], 4), "kept_by_verifier": on(b)})
        ious = [iou for r in sel.records for g in r["gts"][category] for iou in [r["oracle"].get(g["instance_id"])] if iou is not None]
        pipeline_miou = summary[(category, "on")]["miou"]
        stage_rows.append({"category": category, **{f"{a}_{b}": tally.get((a, b), 0) for a in ("miss", "fp")
                                                    for b in ("DINO", "SAM", "CLIP", "other") if not (a == "fp" and b in ("CLIP", "other"))},
                           "verifier_also_failed": tally.get("verifier_also_failed", 0),
                           "fp_caught_by_verifier": tally.get("fp_caught_by_verifier", 0),
                           "oracle_box_sam_miou": round(float(np.mean(ious)), 4) if ious else "", "oracle_n": len(ious),
                           "pipeline_miou": pipeline_miou})
        if kind:
            for state, keep in (("off", None), ("on", on)):
                images = [(sel.preds(r, category, setup, threshold, keep), r["negatives"]) for r in sel.records]
                rates.append(_fmt({"category": category, "negative_kind": kind, "verifier": state,
                                   **metrics.attribute_rate(images, kind, sel.thr, z)}))
    _write_csv(out / "stage_table.csv", stage_rows)
    _write_csv(out / "hallucination_candidates.csv", sorted(candidates, key=lambda c: -c["score"]))
    _write_csv(out / "hallucination_rates.csv", rates)

    rows = []
    for label, (value, table) in chosen["calibration"].items():
        rows += [{"temperature": label, "ece": round(value, 4), **_fmt(b)} for b in table]
    _write_csv(out / "calibration.csv", rows)
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for label, (value, table) in chosen["calibration"].items():
        filled = [b for b in table if b["count"]]
        plt.plot([b["mean_conf"] for b in filled], [b["accuracy"] for b in filled], marker="o", label=f"{label}, ECE {value:.3f}")
    plt.xlabel("verifier P(positive)"), plt.ylabel("share that are right"), plt.legend(), plt.tight_layout()
    plt.savefig(out / "reliability.png", dpi=120), plt.close()
    _write_csv(out / "ties.csv", sel.ties)

    macro = {state: metrics.macro({c: summary[(c, state)]["f1"] for c in sel.categories})[0] for state in ("off", "on")}
    return {"macro_f1_verifier_on": macro["on"], "macro_f1_verifier_off": macro["off"],
            "per_category": {c: {state: {k: summary[(c, state)][k] for k in ("precision", "recall", "f1", "miou", "tp", "fp", "fn")}
                                 for state in ("off", "on")} for c in sel.categories},
            "absent_attribute": rates, "stage_table": stage_rows,
            "ece": {label: round(value, 4) for label, (value, _) in chosen["calibration"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["all", "select"], default="all")
    parser.add_argument("--quick", action="store_true", help="3 val images, a smoke test; never writes frozen.json")
    args = parser.parse_args()
    with open(_p("config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_p("prompts", "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    rows = [r for r in data.load_split("val") if r["row_type"] == "synthetic"]
    if args.quick:
        rows = rows[:3]
    truth = metrics.truth_from_coco(json.loads(_p("data", "synth", "val", "annotations.json").read_text(encoding="utf-8")))

    if args.stage == "all":
        start = time.perf_counter()
        build_cache(rows, cfg, prompts, truth)
        selftest(rows, cfg, prompts)
        print(f"[time] stage 1 cache {time.perf_counter() - start:7.1f} s")
    start = time.perf_counter()
    selector = Selector(load_records(rows, cfg, prompts, truth), cfg, prompts)
    chosen = selector.run()
    print(f"[time] stage 2 select {time.perf_counter() - start:6.1f} s")
    start = time.perf_counter()
    out = _p("results", "val_quick" if args.quick else "val")
    out.mkdir(parents=True, exist_ok=True)
    val_metrics = report(selector, chosen, out, cfg)
    print(f"[time] stage 3 report {time.perf_counter() - start:6.1f} s")

    at = next(s for s in chosen["sweep"] if s["threshold"] == chosen["threshold"])
    print(f"[select] threshold {chosen['threshold']:.2f} kept {100 * at['mean_recall']:.0f}% of real objects and "
          f"{100 * at['mean_precision']:.0f}% of its boxes were right (averaged over every wording, verifier off)")
    print(f"[select] headline wording: {chosen['headline']}")
    print(f"[select] verifier temperature {chosen['temperature']:.2f}, margin {chosen['margin']:.2f}; "
          f"macro F1 {val_metrics['macro_f1_verifier_off']:.3f} without the verifier, {val_metrics['macro_f1_verifier_on']:.3f} with it")
    print("[select] These are val numbers, and the choices were made on val, so they are optimistic.")
    if args.quick:
        print("[quick] Smoke test on 3 val images. frozen.json and results/val/runs.log were not written.")
        return

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    now = datetime.now().isoformat(timespec="seconds")
    shas = {"prompts_sha256": _sha(_p("prompts", "prompts.yaml")), "config_sha256": _sha(_p("config.yaml"))}
    with open(out / "runs.log", "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{now}\t{commit}\t{shas['prompts_sha256']}\t{shas['config_sha256']}\n")
    frozen = {"created": now, "git_commit": commit, "val_list_sha256": data.val_list_sha256(), **shas,
              "models": models_module.fingerprint(),
              "dino": {"threshold": chosen["threshold"], "text_threshold": cfg["tuning"]["dino_text_threshold"]},
              "headline": chosen["headline"], "ensemble": cfg["tuning"]["ensemble"],
              "clip": {"temperature": chosen["temperature"], "margin": chosen["margin"],
                       "crop_pad_frac": cfg["tuning"]["clip_crop_pad_frac"]},
              "objective": "macro F1 at mask IoU 0.5, every wording averaged", "val_metrics_at_frozen": val_metrics}
    _p("frozen.json").write_text(json.dumps(_json_safe(frozen), indent=1) + "\n", encoding="utf-8", newline="\n")  # last
    print("[freeze] frozen.json written. Commit it before running eval.py.")


if __name__ == "__main__":
    main()
