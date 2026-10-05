"""Reproduce one hallucination and show why it happens, with numbers.

Spec: docs/probe_spec.md. It takes one object that does NOT have the attribute (an intact box, a
straight label), asks about it with several wordings, and shows which words drove the match, what
SAM did with the box, and whether CLIP caught it. Showcase images and data/showcase_live/ only.
The rate of this failure comes from eval.py, never from here.

    python src/probe.py --image s_showcase_0001 --family cereal_box_attribute [--target <instance_id>]
    python src/probe.py --image s_showcase_0001 --family cereal_box_attribute --compare results/probe/<file>.csv
    python src/probe.py --scan --family cereal_box_attribute
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # before anything imports transformers

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
FAMILIES = {  # family -> (the kind of negative probed, the category whose verifier contrast applies)
    "cereal_box_attribute": ("intact_cereal_box", "damaged_cereal_box"),
    "label_attribute": ("straight_label", "misaligned_label"),
}
ATTRIBUTE_WORDS = {"damaged", "dented", "undamaged", "misaligned", "straight"}
NUMBERS = ["score", "attribute_word_score", "best_object_word_score", "box_iou_with_target", "p_positive", "margin",
           "sam_iou_pred", "sam_mask_iou"]


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def probe(m, image, target_box, family: str, frozen: dict, prompts: dict, embedding, true_mask=None) -> list[dict]:
    """One row per wording, all about the same object, even where its score falls below the threshold."""
    wordings = prompts["probe"][family]
    _, category = FAMILIES[family]
    verifier = prompts["clip_verifier"]
    masks, sam_ious = pipeline.segment(m, image, embedding, [target_box])
    bank, n_positive = pipeline.contrast_bank(m, verifier["contrasts"][category], verifier["templates"])
    crop = pipeline.embed_crops(m, image, [target_box], frozen["clip"]["crop_pad_frac"])
    p_positive, margin, _ = pipeline.verifier_scores(m, crop, bank, n_positive, frozen["clip"]["temperature"])
    rows = []
    for wording in wordings:
        raw = pipeline.forward_dino(m, image, wording)
        boxes = pipeline.boxes_from_raw(raw, image.size)
        scores = torch.sigmoid(raw["logits"]).max(dim=-1).values
        overlaps = torch.tensor([metrics.box_iou(b.tolist(), target_box) for b in boxes])
        best = float(overlaps.max())
        # Among the queries tied for the best overlap, the higher score; then that one query's per-word scores.
        q = int(max(torch.nonzero(overlaps == overlaps.max()).flatten().tolist(), key=lambda i: float(scores[i])))
        one = {"logits": raw["logits"][[q]], "pred_boxes": raw["pred_boxes"][[q]], "input_ids": raw["input_ids"],
               "query_index": raw["query_index"][[q]]}
        detection = pipeline.select(m, one, image.size, wording, -1.0)[0]
        attribute = [s for w, s in detection.word_scores.items() if w in ATTRIBUTE_WORDS]
        rows.append({"wording": wording, "found_same_object": best >= 0.5, "box_iou_with_target": best,
                     "score": detection.score, "over_threshold": detection.score > frozen["dino"]["threshold"],
                     "word_scores": detection.word_scores,
                     "attribute_word_score": max(attribute) if attribute else float("nan"),
                     "best_object_word_score": max(s for w, s in detection.word_scores.items() if w not in ATTRIBUTE_WORDS),
                     "p_positive": float(p_positive[0]), "margin": float(margin[0]),
                     "kept_by_verifier": bool(margin[0] >= frozen["clip"]["margin"]), "sam_iou_pred": sam_ious[0],
                     "sam_mask_iou": metrics.mask_iou(masks[0], true_mask)[0] if true_mask is not None else float("nan")})
    return rows


def verdict(rows: list[dict]) -> str:
    """Picked by rules fixed in the spec, before any result. It says what the numbers say."""
    with_attribute, opposite = rows[1], rows[2]  # for example "damaged cereal box" and "undamaged cereal box"
    if opposite["score"] - with_attribute["score"] >= 0.10:
        return "attribute used"
    if (with_attribute["attribute_word_score"] < 0.5 * with_attribute["best_object_word_score"]
            and abs(with_attribute["score"] - opposite["score"]) < 0.05):
        return "attribute ignored"
    return "mixed"


def summary(rows: list[dict], family: str, frozen: dict, live: bool) -> str:
    a, b, word = rows[1], rows[2], rows[1]["wording"].split()[0]
    thing = "an intact cereal box" if family == "cereal_box_attribute" else "a straight label"
    if live:
        thing = "the chosen object in a live photo (no ground truth, so whether it really lacks the attribute is yours to say)"
    result = verdict(rows)
    words = ", ".join(f"'{w}' {s:.2f}" for w, s in a["word_scores"].items())
    text = [f"Verdict: {result}.",
            f"On {thing}, '{a['wording']}' scored {a['score']:.2f} and '{b['wording']}' scored {b['score']:.2f}.",
            f"Word by word for '{a['wording']}': {words}."]
    if result == "attribute ignored":
        text.append("So the match came from the object words, and the attribute word barely counted.")
    elif result == "attribute used":
        text.append(f"The expected pattern did not appear here: the detector did respond to '{word}'.")
    else:
        text.append("The pattern is mixed: neither clearly ignored nor clearly used.")
    text.append(f"At the frozen threshold of {frozen['dino']['threshold']:.2f} this object is "
                f"{'reported' if a['over_threshold'] else 'not reported'} for '{a['wording']}'.")
    if not np.isnan(a["sam_mask_iou"]):
        text.append(f"SAM masked it at IoU {a['sam_mask_iou']:.2f} with the real outline. SAM never sees the words.")
    text.append(f"The CLIP check gave it probability {a['p_positive']:.2f} (margin {a['margin']:+.2f}), so it was "
                f"{'kept' if a['kept_by_verifier'] else 'rejected'}.")
    gaps = [r["wording"] for r in rows if not r["found_same_object"]]
    if gaps:
        text.append(f"Gap: no candidate box overlapped the object by 0.5 for {gaps}; the nearest one is reported.")
    return "\n".join(text)


def write(rows, text, image, target_box, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({**r, "word_scores": json.dumps(r["word_scores"])} for r in rows)
    (out / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, len(rows) + 1, figsize=(4 + 3 * len(rows), 4))
    axes[0].imshow(image)
    x1, y1, x2, y2 = target_box
    axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
    axes[0].set_axis_off(), axes[0].set_title("the object probed")
    for ax, r in zip(axes[1:], rows):
        ax.bar(list(r["word_scores"]), list(r["word_scores"].values()),
               color=["tab:red" if w in ATTRIBUTE_WORDS else "tab:blue" for w in r["word_scores"]])
        ax.set_ylim(0, 1), ax.set_title(f"'{r['wording']}'\nbox score {r['score']:.2f}", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout(), fig.savefig(out / f"{stem}.png", dpi=110), plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="a showcase image id")
    parser.add_argument("--photo", help="a file in data/showcase_live/")
    parser.add_argument("--family", required=True, choices=list(FAMILIES))
    parser.add_argument("--target", help="instance_id of the intact box or straight label to probe")
    parser.add_argument("--box", help="x1,y1,x2,y2 for a live photo")
    parser.add_argument("--compare", type=Path, help="a saved .csv from an earlier run; reports the largest difference")
    parser.add_argument("--scan", action="store_true", help="probe every showcase negative and rank them")
    args = parser.parse_args()
    if not _p("frozen.json").exists():
        sys.exit("frozen.json is missing. The probe only runs at the settings tune_val.py froze.")
    frozen = json.loads(_p("frozen.json").read_text(encoding="utf-8"))
    with open(_p("prompts", "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    kind, _ = FAMILIES[args.family]
    showcase = {r["asset_id"]: _p(r["path"]) for r in data.load_split("showcase") if r["row_type"] == "synthetic"}
    truth = metrics.truth_from_coco(json.loads(_p("data", "synth", "showcase", "annotations.json").read_text(encoding="utf-8")))
    m = models_module.get_models()
    start = time.perf_counter()

    if args.scan:  # choosing an example to show is fine here: nothing from showcase is ever scored
        ranked = []
        for name, path in sorted(showcase.items()):
            image = pipeline.prepare(Image.open(path))
            embedding = pipeline.sam_embed(m, image)
            for negative in [n for n in truth[name]["negatives"] if n["kind"] == kind]:
                rows = probe(m, image, negative["box"], args.family, frozen, prompts, embedding, metrics.record_mask(negative))
                ranked.append((rows[1]["score"], name, negative["instance_id"], verdict(rows), rows[1]["kept_by_verifier"]))
        for score, name, instance, result, kept in sorted(ranked, reverse=True):
            print(f"{score:.2f}  {name}  --target {instance}  {result}  verifier {'kept' if kept else 'rejected'} it")
        print(f"[time] scan {time.perf_counter() - start:.1f} s")
        return

    if args.photo:
        path, live = Path(args.photo).resolve(), True
        if path.parent != _p("data", "showcase_live").resolve() or not path.is_file():
            sys.exit("--photo takes a file in data/showcase_live/, which is in no split and never scored.")
        name, image, true_mask = path.stem, pipeline.prepare(Image.open(path)), None
        if args.box:
            target_box = [float(v) for v in args.box.split(",")]
        else:
            first = pipeline.detect(m, image, prompts["probe"][args.family][0], 0.0, frozen["dino"]["text_threshold"])
            target_box = max(first, key=lambda d: d.score).box
    else:
        if args.image not in showcase:
            sys.exit(f"{args.image!r} is not a showcase image. The probe refuses val and test.")
        name, live, image = args.image, False, pipeline.prepare(Image.open(showcase[args.image]))
        negatives = [n for n in truth[name]["negatives"] if n["kind"] == kind and args.target in (None, n["instance_id"])]
        if not negatives:
            sys.exit(f"{name} has no {kind.replace('_', ' ')}" + (f" with id {args.target}." if args.target else "."))
        target_box, true_mask = negatives[0]["box"], metrics.record_mask(negatives[0])

    rows = probe(m, image, target_box, args.family, frozen, prompts, pipeline.sam_embed(m, image), true_mask)
    text = summary(rows, args.family, frozen, live)
    stem = f"{name}__{args.family}"
    if args.compare:  # the reference is read and never rewritten
        with open(args.compare, encoding="utf-8", newline="") as f:
            saved = list(csv.DictReader(f))
        gap = max(abs(float(old[k]) - r[k]) for old, r in zip(saved, rows) for k in NUMBERS
                  if not (np.isnan(r[k]) and old[k] in ("nan", "")))
        when = datetime.fromtimestamp(args.compare.stat().st_mtime).strftime("%H:%M")
        text += f"\nMatches the saved run from {when}: largest difference {gap:.6f}"
        stem += f"__compare_{datetime.now():%Y%m%d_%H%M%S}"
    write(rows, text, image, target_box, _p("results", "probe"), stem)
    print(text)
    print(f"Saved results/probe/{stem}.png, .csv and .txt")
    print(f"[time] probe {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
