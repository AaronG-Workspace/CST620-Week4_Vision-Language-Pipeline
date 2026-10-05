"""Score the test set once, with everything already decided in frozen.json.

Spec: docs/eval_spec.md. The only file allowed to load test images or test annotations.
It never sweeps, tunes or picks a best anything: every wording in prompts.yaml is run and
reported, and the frozen headline choice is marked.

    python src/eval.py --preflight               # checks 1 to 5, then exits before reading test
    python src/eval.py --preflight --log <path>  # check 5 reads a sample log instead
    python src/eval.py --dry-run                 # the full code path on 3 val images; never touches test
    python src/eval.py
    python src/eval.py --rescore "reason of at least 10 characters"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # check 1: set here, before anything imports transformers

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import data  # noqa: E402
import metrics  # noqa: E402
import models as models_module  # noqa: E402
import pipeline  # noqa: E402
import tune_val  # noqa: E402  (the shared Selector and report; it never opens test)

ROOT = Path(__file__).resolve().parents[1]
LIMITATIONS = [
    '"Held out" means the boxes and labels are newly rendered, while the backgrounds and real distractors are public '
    "Open Images photos with possible pretraining overlap, and rendered boxes differ in style from the real photo objects.",
    'The rendered cereal boxes print "Cereal" and the look-alikes print their own product names, so the models may '
    "partly read the text.",
]


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _log_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


# ---------------------------------------------------------------- before anything from test is read

def preflight(args, frozen_path: Path, log_path: Path) -> tuple[bool, dict, str]:
    """Checks 1 to 5. Prints each result. Returns (all passed, frozen settings, reason for the log)."""
    results, frozen, reason = [], {}, "first run"

    def record(number, ok, text):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] preflight {number} {text}")

    record(1, os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE=1 was set before transformers was imported")
    if not frozen_path.exists():
        record(2, False, "frozen.json is missing. Run src/tune_val.py first.")
    else:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        now = {"val_list_sha256": data.val_list_sha256(), "prompts_sha256": _sha(_p("prompts", "prompts.yaml")),
               "config_sha256": _sha(_p("config.yaml"))}
        changed = [k.replace("_sha256", "") for k, v in now.items() if frozen.get(k) != v]
        if frozen.get("models") != models_module.fingerprint():
            changed.append("model fingerprint")
        record(2, not changed, "frozen.json matches the val list, prompts.yaml, config.yaml and the models"
               if not changed else f"changed since freezing: {', '.join(changed)}")
    if args.dry_run:
        print("[SKIP] preflight 3 to 5 (check_splits, git and the run log are skipped in a dry run)")
        return all(results), frozen, "dry run"

    audit = subprocess.run([sys.executable, "src/check_splits.py", "--fast"], cwd=ROOT, capture_output=True, text=True)
    record(3, audit.returncode == 0, "check_splits.py --fast" + ("" if audit.returncode == 0 else ": " + (audit.stdout.strip().splitlines() or ["failed"])[-1]))
    status = subprocess.run(["git", "status", "--porcelain", "--", "src", "prompts", "config.yaml", "frozen.json"],
                            cwd=ROOT, capture_output=True, text=True)
    dirty = status.stdout.strip()
    tracked = _git("ls-files", "frozen.json") == "frozen.json"
    record(4, status.returncode == 0 and not dirty and tracked,
           "no uncommitted changes in src/, prompts/, config.yaml or frozen.json" if status.returncode == 0 and not dirty and tracked
           else "git status failed (not a repository?)" if status.returncode else
           "frozen.json is not committed" if not dirty else "uncommitted: " + ", ".join(line[3:] for line in dirty.splitlines()))
    entries = _log_entries(log_path)
    completed = sum(e.get("event") == "completed" for e in entries)
    crashed = bool(entries) and entries[-1].get("event") == "started" and not _p("results", "test").exists()
    if completed and not (args.rescore and len(args.rescore) >= 10):
        record(5, False, f"{log_path.name} already has a completed run. A second needs --rescore with a reason of 10 or more characters.")
    else:
        reason = args.rescore or ("rerun after crash" if crashed else "first run")
        record(5, True, f"test may be scored ({reason})")
    return all(results), frozen, reason


# ---------------------------------------------------------------- one pass over the images

def analyze(rows: list[dict], truth: dict, frozen: dict, cfg: dict, prompts: dict) -> tuple[list[dict], dict, dict]:
    """Every wording on every image at the frozen settings. Prints a progress counter and nothing else."""
    m = models_module.get_models()
    threshold, text_threshold = frozen["dino"]["threshold"], frozen["dino"]["text_threshold"]
    clip, verifier = frozen["clip"], prompts["clip_verifier"]
    categories = prompts["grounding_dino"]["categories"]
    banks = {c: pipeline.contrast_bank(m, verifier["contrasts"][c], verifier["templates"]) for c in categories}
    records, pictures, seconds = [], {}, {"dino": 0.0, "sam_encode": 0.0, "sam_decode": 0.0, "clip": 0.0}
    for n, row in enumerate(rows, 1):
        image = pipeline.prepare(Image.open(_p(row["path"])))  # each image is opened once
        pictures[row["asset_id"]] = image
        image_truth = truth[row["asset_id"]]
        gt_masks = {g["instance_id"]: metrics.record_mask(g) for group in image_truth["gts"].values() for g in group}
        start = time.perf_counter()
        embedding = pipeline.sam_embed(m, image)  # once per image
        seconds["sam_encode"] += time.perf_counter() - start
        record = {"name": row["asset_id"], "condition": image_truth["condition"], "gts": image_truth["gts"],
                  "negatives": image_truth["negatives"], "boxes": {}}
        for category, entry in categories.items():
            for k, wording in enumerate(entry["wordings"]):
                start = time.perf_counter()
                detections = pipeline.detect(m, image, wording, threshold, text_threshold)
                seconds["dino"] += time.perf_counter() - start
                start = time.perf_counter()
                masks, ious = pipeline.segment(m, image, embedding, [d.box for d in detections])
                seconds["sam_decode"] += time.perf_counter() - start
                start = time.perf_counter()
                crops = pipeline.embed_crops(m, image, [d.box for d in detections], clip["crop_pad_frac"])
                p_positive, margins, _ = pipeline.verifier_scores(m, crops, banks[category][0], banks[category][1], clip["temperature"])
                seconds["clip"] += time.perf_counter() - start
                record["boxes"][(category, k)] = [
                    {"box": d.box, "score": d.score, "query_index": d.query_index, "word_scores": d.word_scores,
                     "other": d.other_token_score, "wording": wording, "sam_iou_pred": iou, "image": row["asset_id"],
                     "miou": {g["instance_id"]: metrics.mask_iou(mask, gt_masks[g["instance_id"]])[0] for g in image_truth["gts"][category]},
                     "p_positive": float(p), "margin": float(mg), "kept": bool(mg >= clip["margin"]), "mask": np.packbits(mask)}
                    for d, mask, iou, p, mg in zip(detections, masks, ious, p_positive, margins)]
        gts = {g["instance_id"]: g for group in image_truth["gts"].values() for g in group}
        oracle, _ = pipeline.segment(m, image, embedding, [g["box"] for g in gts.values()])  # SAM on the true boxes
        record["oracle"] = {i: metrics.mask_iou(mask, gt_masks[i])[0] for i, mask in zip(gts, oracle)}
        records.append(record)
        print(f"image {n}/{len(rows)}")  # the only thing printed until every output is written
    return records, pictures, seconds


# ---------------------------------------------------------------- outputs

def write_outputs(records, pictures, seconds, frozen, cfg, prompts, out: Path) -> dict:
    sel = tune_val.Selector(records, cfg, prompts, seed_base=300)  # matching 301, bootstrap 302, NMS 303, stage ties 304
    threshold, margin, thr = frozen["dino"]["threshold"], frozen["clip"]["margin"], cfg["tuning"]["match_iou"]
    z, n_boot = cfg["reporting"]["wilson_z"], cfg["reporting"]["bootstrap_resamples"]
    setups = {c: ("ensemble" if h == "ensemble" else sel.categories[c].index(h)) for c, h in frozen["headline"].items()}
    on = lambda b: b["kept"]  # noqa: E731

    # Calibration at the frozen temperature, on the headline boxes, labelled by box IoU with any ground truth.
    conf, raw, labels = [], [], []
    for c in sel.categories:
        for r in records:
            for b in sel.preds(r, c, setups[c], threshold):
                conf.append(b["p_positive"]), raw.append(b["score"])
                labels.append(float(any(metrics.box_iou(b["box"], g["box"]) >= thr for g in r["gts"][c])))
    bins = cfg["tuning"]["ece_bins"]
    chosen = {"threshold": threshold, "margin": margin, "setups": setups, "sweep": [],
              "calibration": {"verifier at the frozen temperature": metrics.ece(conf, labels, bins),
                              "raw Grounding DINO score (information only)": metrics.ece(raw, labels, bins)}}
    summary = tune_val.report(sel, chosen, out, cfg)  # sensitivity, stage table, hallucination, calibration, ties

    headline = [{"category": c, **summary["per_category"][c]["on"]} for c in sel.categories]
    headline.append({"category": "macro", "f1": summary["macro_f1_verifier_on"]})
    tune_val._write_csv(out / "headline.csv", [tune_val._fmt(h) for h in headline])

    # Slices at the headline setup: conditions, and damaged recall by dent severity.
    slices = []
    for c in sel.categories:
        for condition in cfg["conditions"]:
            subset = [r for r in records if r["condition"] == condition]
            counts, _, _, _ = sel.score(c, setups[c], threshold, on, records=subset)
            slices.append(tune_val._fmt({"slice": f"condition {condition}", "category": c, "verifier": "on",
                                         **metrics.summarize(counts, z, n_boot, sel.rng["bootstrap"])}))
    for state, keep in (("off", None), ("on", on)):
        _, details, images, _ = sel.score("damaged_cereal_box", setups["damaged_cereal_box"], threshold, keep)
        found: dict[str, list] = {}
        for detail, (_, gts) in zip(details, images):
            hit = {j for _, j, _ in detail["pairs"]}
            for j, g in enumerate(gts):
                found.setdefault(g.get("dent_severity", "unrated"), []).append((j in hit, g["group_id"]))
        for severity, items in sorted(found.items()):
            k, n = sum(h for h, _ in items), len(items)
            lo, hi = metrics.wilson(k, n, z)
            slices.append(tune_val._fmt({"slice": f"dent severity {severity}", "category": "damaged_cereal_box", "verifier": state,
                                         "recall": k / n, "recall_lo": lo, "recall_hi": hi, "n_truth": n,
                                         "groups": len({g for _, g in items})}))
    fields = sorted({k for s in slices for k in s}, key=lambda k: (k not in ("slice", "category", "verifier"), k))
    tune_val._write_csv(out / "slices.csv", [{f: s.get(f, "") for f in fields} for s in slices])

    # Every box of every wording, its match and its error type, and an overlay for each headline failure.
    per_box, confident, loc = [], [], cfg["reporting"]["fp_localization_iou"]
    (out / "failures").mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=13)
    for c, wordings in sel.categories.items():
        for k, wording in enumerate(wordings):
            for r in records:
                proposed, gts = sel.preds(r, c, k, threshold), r["gts"][c]
                kept = [b for b in proposed if on(b)]
                pairs, fp_idx, fn_idx, _, _ = metrics.match(kept, gts, tune_val._overlap, thr, sel.rng["match"])
                matched = {gts[j]["instance_id"] for _, j, _ in pairs}
                status = {id(kept[i]): ("TP", "", "") for i, _, _ in pairs}
                for i in fp_idx:
                    label = metrics.fp_type(kept[i], c, gts, r["negatives"], matched, thr, loc, tune_val._overlap)
                    status[id(kept[i])] = ("FP", label, metrics.stage(kept[i], [], thr, sel.rng["stage"], fp_label=label)[0])
                    confident.append({"image_id": r["name"], "category": c, "wording": wording, "score": round(kept[i]["score"], 4),
                                      "box": [round(v, 1) for v in kept[i]["box"]], "fp_type": label,
                                      "word_scores": json.dumps({w: round(s, 3) for w, s in kept[i]["word_scores"].items()}),
                                      "verifier_margin": round(kept[i]["margin"], 4)})
                for b in proposed:
                    match, label, stage = status.get(id(b), ("rejected by verifier", "", ""))
                    per_box.append({"image_id": r["name"], "category": c, "wording": wording, "score": round(b["score"], 4),
                                    "word_scores": json.dumps({w: round(s, 3) for w, s in b["word_scores"].items()}),
                                    "best_mask_iou": round(max(b["miou"].values(), default=0.0), 4), "sam_iou_pred": round(b["sam_iou_pred"], 4),
                                    "p_positive": round(b["p_positive"], 4), "margin": round(b["margin"], 4), "kept": b["kept"],
                                    "match": match, "fp_type": label, "stage": stage})
        # One overlay per failure at the headline setup, so a failure can be explained without reopening test.
        for r in records:
            proposed, gts = sel.preds(r, c, setups[c], threshold), r["gts"][c]
            kept = [b for b in proposed if on(b)]
            pairs, fp_idx, fn_idx, _, _ = metrics.match(kept, gts, tune_val._overlap, thr, sel.rng["match"])
            matched = {gts[j]["instance_id"] for _, j, _ in pairs}
            failures = []
            for i in fp_idx:
                label = metrics.fp_type(kept[i], c, gts, r["negatives"], matched, thr, loc, tune_val._overlap)
                stage = metrics.stage(kept[i], [], thr, sel.rng["stage"], fp_label=label)[0]
                failures.append(("fp", kept[i], f"false positive | '{kept[i]['wording']}' | score {kept[i]['score']:.2f} | "
                                                f"margin {kept[i]['margin']:+.2f} | {label} | stage {stage}"))
            for j in fn_idx:
                stage = metrics.stage(gts[j], proposed, thr, sel.rng["stage"], mask_iou_of=tune_val._overlap)[0]
                failures.append(("miss", gts[j], f"missed {c} | headline {frozen['headline'][c]} | stage {stage}"))
            for n, (kind, item, text) in enumerate(failures):
                picture = pictures[r["name"]].copy()
                if kind == "fp":
                    mask = np.unpackbits(item["mask"])[:picture.width * picture.height].reshape(picture.height, picture.width)
                    picture.paste(Image.new("RGB", picture.size, (255, 60, 60)), mask=Image.fromarray((mask * 110).astype(np.uint8)))
                draw = ImageDraw.Draw(picture, "RGBA")
                draw.rectangle(item["box"], outline=(255, 0, 0, 255) if kind == "fp" else (0, 220, 0, 255), width=3)
                draw.rectangle([0, 0, picture.width, 20], fill=(0, 0, 0, 200))
                draw.text((4, 3), text, fill="white", font=font)
                picture.save(out / "failures" / f"{r['name']}__{c}__{kind}_{n}.png")
    tune_val._write_csv(out / "per_box.csv", per_box)
    confident.sort(key=lambda row: -row["score"])
    tune_val._write_csv(out / "hallucination.csv", confident[:cfg["reporting"]["top_confident_fp"]])
    n_images = len(records)
    tune_val._write_csv(out / "timing.csv", [{"stage": s, "seconds": round(v, 1), "seconds_per_image": round(v / n_images, 2)}
                                             for s, v in seconds.items()])

    def plain(c):
        row = summary["per_category"][c]["on"]
        return (f"{c}: when it flags one, it is right {100 * row['precision']:.0f}% of the time "
                f"({row['tp']} of {row['tp'] + row['fp']}), and it finds {100 * row['recall']:.0f}% of the real ones "
                f"({row['tp']} of {row['tp'] + row['fn']}). F1 {row['f1']:.2f}.")

    lines = ["# Results table (numbers only)", "", *[f"- {note}" for note in LIMITATIONS], "",
             f"Headline setup: {frozen['headline']}, threshold {threshold}, verifier margin {margin}.", ""]
    lines += [f"- {plain(c)}" for c in sel.categories]
    lines += [f"- Macro F1 {summary['macro_f1_verifier_on']:.3f} with the verifier, {summary['macro_f1_verifier_off']:.3f} without."]
    article = lambda kind: "an" if kind[0] in "aeiou" else "a"  # noqa: E731
    lines += [f"- {r['category']} on {article(r['negative_kind'])} {r['negative_kind'].replace('_', ' ')}, verifier {r['verifier']}: {r['negatives_hit']} of "
              f"{r['negatives']} hit ({r['boxes']} boxes)." for r in summary["absent_attribute"]]
    (out / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out / "metrics.json").write_text(json.dumps(tune_val._json_safe({"frozen": frozen, **summary, "slices": slices}), indent=1) + "\n",
                                      encoding="utf-8", newline="\n")
    return {"summary": summary, "lines": lines[5:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preflight", action="store_true", help="run checks 1 to 5 and exit before reading test")
    parser.add_argument("--log", type=Path, help="with --preflight only: read this sample log for check 5")
    parser.add_argument("--dry-run", action="store_true", help="the full code path on 3 val images; never touches test")
    parser.add_argument("--rescore", metavar="REASON", help="score again, with a reason of at least 10 characters")
    args = parser.parse_args()
    if args.log and not args.preflight:
        parser.error("--log is refused without --preflight. The real run always uses results/test_runs.log.")
    with open(_p("config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_p("prompts", "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    log_path = args.log or _p("results", "test_runs.log")

    ok, frozen, reason = preflight(args, _p("frozen.json"), log_path)
    if args.preflight:
        print("preflight passed. Nothing from test was read." if ok else "preflight failed. Nothing from test was read.")
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("eval.py refuses to start. Nothing from test was read.")

    if args.dry_run:
        rows = [r for r in data.load_split("val") if r["row_type"] == "synthetic"][:3]
        truth = metrics.truth_from_coco(json.loads(_p("data", "synth", "val", "annotations.json").read_text(encoding="utf-8")))
        final, tmp = _p("results", "dry_run"), _p("results", "dry_run_tmp")
    else:
        frozen_sha, commit = _sha(_p("frozen.json")), _git("rev-parse", "--short", "HEAD")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:  # logged before the test list is read
            f.write(json.dumps({"event": "started", "time": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
                                "frozen_sha256": frozen_sha, "reason": reason}) + "\n")
        rows = [r for r in data.load_split("test", allow_test=True) if r["row_type"] == "synthetic"]  # read once
        truth = metrics.truth_from_coco(json.loads(_p("data", "synth", "test", "annotations.json").read_text(encoding="utf-8")))
        final, tmp = _p("results", "test"), _p("results", "test_tmp")

    if tmp.exists():
        shutil.rmtree(tmp)  # a scratch folder from a crashed run; it never held a finished result
    tmp.mkdir(parents=True)
    records, pictures, seconds = analyze(rows, truth, frozen, cfg, prompts)
    written = write_outputs(records, pictures, seconds, frozen, cfg, prompts, tmp)
    if final.exists():
        if args.dry_run:
            shutil.rmtree(final)  # a dry run overwrites the last dry run
        else:
            n = 1
            while _p("results", f"test_run{n}").exists():
                n += 1
            final.rename(_p("results", f"test_run{n}"))  # old runs are kept, never deleted
    tmp.rename(final)
    if not args.dry_run:
        listing = {p.relative_to(final).as_posix(): _sha(p) for p in sorted(final.rglob("*")) if p.is_file()}
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"event": "completed", "time": datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
                                "frozen_sha256": frozen_sha, "outputs_sha256": _sha_of(listing)}) + "\n")

    # Only now, with every output in place, do numbers reach the console.
    print(f"\n{'Dry run on 3 val images (not a test result)' if args.dry_run else 'Test set, scored once'}: {final.relative_to(ROOT).as_posix()}")
    for line in written["lines"]:
        print(line.replace("- ", "", 1))
    stages = {k: sum(row[k] for row in written["summary"]["stage_table"]) for k in ("miss_DINO", "miss_SAM", "miss_CLIP", "fp_DINO", "fp_SAM")}
    print(f"Most failures start at: {max(stages, key=stages.get)} ({stages})")
    for stage, value in seconds.items():
        print(f"[time] {stage:<10} {value:7.1f} s")


def _sha_of(listing: dict) -> str:
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
