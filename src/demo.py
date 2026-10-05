"""Type a description, get boxes and masks back on a shelf image, with no retraining.

Spec: docs/demo_spec.md. Uses frozen.json and the same pipeline functions as eval.py, so what is
shown is what was measured. Showcase images and data/showcase_live/ only. There is no threshold flag.

    python src/demo.py --image s_showcase_0001 --prompt "dented cereal box" [--open]
    python src/demo.py --image s_showcase_0001 --interactive
    python src/demo.py --photo data/showcase_live/shelf.jpg --prompt "cereal box"
    python src/demo.py --grid --prompt "cereal box"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # before anything imports transformers

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

import data  # noqa: E402
import models as models_module  # noqa: E402
import pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = [(66, 135, 245), (245, 150, 40), (60, 190, 110), (220, 70, 200), (240, 210, 50), (90, 210, 220)]


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def showcase_images() -> dict[str, Path]:
    return {r["asset_id"]: _p(r["path"]) for r in data.load_split("showcase") if r["row_type"] == "synthetic"}


def resolve(args) -> tuple[str, Path]:
    """The image to use. Anything that is not a showcase image or a file in data/showcase_live/ is refused."""
    if args.photo:
        path, live = Path(args.photo).resolve(), _p("data", "showcase_live").resolve()
        if path.parent != live or not path.is_file():
            sys.exit("--photo takes a file in data/showcase_live/, which is in no split and never scored.")
        return path.stem, path
    images = showcase_images()
    if args.image not in images:
        sys.exit(f"{args.image!r} is not a showcase image. The demo refuses val and test. Showcase ids: {', '.join(sorted(images))}")
    return args.image, images[args.image]


def _dashed(draw, box, color, width=3, dash=10):
    x1, y1, x2, y2 = box
    for a, b, fixed, horizontal in ((x1, x2, y1, True), (x1, x2, y2, True), (y1, y2, x1, False), (y1, y2, x2, False)):
        for start in np.arange(a, b, dash * 2):
            end = min(start + dash, b)
            draw.line([(start, fixed), (end, fixed)] if horizontal else [(fixed, start), (fixed, end)], fill=color, width=width)


def overlay(image: Image.Image, result: dict, prompt: str, threshold: float) -> Image.Image:
    """Masks tinted, a solid box if kept, a dashed box if the verifier rejected it, and a footer saying what this is."""
    picture, font = image.copy(), ImageFont.load_default(size=14)
    for i, d in enumerate(result["detections"]):
        tint = Image.new("RGB", picture.size, COLORS[i % len(COLORS)])
        picture.paste(tint, mask=Image.fromarray((d.mask * 105).astype(np.uint8)))
    draw = ImageDraw.Draw(picture)
    for i, d in enumerate(result["detections"]):
        color = COLORS[i % len(COLORS)]
        if d.kept is False:
            _dashed(draw, d.box, color)
        else:
            draw.rectangle(d.box, outline=color, width=3)
        caption = f"{d.score:.2f}" + ("" if d.p_positive is None else f" | check {d.p_positive:.2f} {'kept' if d.kept else 'rejected'}")
        x, y = d.box[0], max(0, d.box[1] - 18)
        draw.rectangle([x, y, x + draw.textlength(caption, font=font) + 6, y + 18], fill=color)
        draw.text((x + 3, y + 1), caption, fill="black", font=font)
    kind = "evaluated prompt" if result["category"] else "PREVIEW ONLY: this prompt was not part of the evaluation"
    footer = f"'{prompt}' | threshold {threshold:.2f} | {kind}"
    framed = Image.new("RGB", (picture.width, picture.height + 24), "black")
    framed.paste(picture, (0, 0))
    ImageDraw.Draw(framed).text((6, picture.height + 4), footer, fill="white", font=font)
    return framed


def summary(result: dict, prompt: str, threshold: float, seconds: float) -> str:
    detections = result["detections"]
    if not detections:
        return f"Found nothing above the threshold for '{prompt}'."
    found = f"Found {len(detections)} region{'s' if len(detections) != 1 else ''} for '{prompt}' (threshold {threshold:.2f})."
    if result["category"] is None:
        return f"{found} Preview only: not part of the evaluation, and no check is set up for it. {seconds:.1f} s."
    kept = sum(d.kept for d in detections)
    return f"{found} {kept} passed the check, {len(detections) - kept} rejected. {seconds:.1f} s."


def run_prompt(m, image, name, prompt, frozen, prompts, embedding, save=True):
    start = time.perf_counter()
    result = pipeline.run(m, image, prompt, frozen, prompts, sam_embedding=embedding)
    seconds = time.perf_counter() - start
    threshold = frozen["dino"]["threshold"]
    picture = overlay(image, result, prompt, threshold)
    print(summary(result, prompt, threshold, seconds))
    path = None
    if save:
        out = _p("results", "demo")
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y%m%d_%H%M%S}__{name}__{re.sub(r'[^a-z0-9]+', '-', prompt.lower()).strip('-')}"
        path = out / f"{stem}.png"
        picture.save(path)
        detections = []
        for d in result["detections"]:
            rle = mask_utils.encode(np.asfortranarray(d.mask.astype(np.uint8)))
            detections.append({"box": [round(v, 1) for v in d.box], "score": round(d.score, 4), "word_scores": d.word_scores,
                               "mask": {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")},
                               "mask_area": int(d.mask.sum()), "sam_iou_pred": d.sam_iou_pred, "p_positive": d.p_positive,
                               "margin": d.margin, "kept": d.kept})
        (out / f"{stem}.json").write_text(json.dumps({
            "prompt": prompt, "image": name, "status": "evaluated" if result["category"] else "preview",
            "category": result["category"], "verifier": result["verifier"], "threshold": threshold,
            "timing_seconds": {k: round(v, 2) for k, v in result["timing"].items()}, "detections": detections}, indent=1) + "\n",
            encoding="utf-8")
        print(f"Saved {path.relative_to(ROOT).as_posix()}")
    return result, picture, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", help="a showcase image id")
    parser.add_argument("--photo", help="a file in data/showcase_live/")
    parser.add_argument("--prompt", help="what to look for, in plain words")
    parser.add_argument("--interactive", action="store_true", help="type prompts one after another against the same image")
    parser.add_argument("--grid", action="store_true", help="one prompt across every showcase image, as a contact sheet")
    parser.add_argument("--open", action="store_true", help="open the saved overlay")
    args = parser.parse_args()
    if not _p("frozen.json").exists():
        sys.exit("frozen.json is missing. The demo only runs at the settings tune_val.py froze.")
    if not (args.grid or args.image or args.photo) or not (args.prompt or args.interactive):
        parser.error("give --image or --photo (or --grid), and --prompt or --interactive")
    frozen = json.loads(_p("frozen.json").read_text(encoding="utf-8"))
    with open(_p("prompts", "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    m = models_module.get_models()

    if args.grid:
        tiles = []
        for name, path in sorted(showcase_images().items()):
            image = pipeline.prepare(Image.open(path))
            _, picture, _ = run_prompt(m, image, name, args.prompt, frozen, prompts, None, save=False)
            picture.thumbnail((420, 420))
            tiles.append(picture)
        sheet = Image.new("RGB", (3 * 430, -(-len(tiles) // 3) * 450), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 3) * 430 + 5, (i // 3) * 450 + 5))
        out = _p("results", "demo")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{datetime.now():%Y%m%d_%H%M%S}__grid__{re.sub(r'[^a-z0-9]+', '-', args.prompt.lower()).strip('-')}.png"
        sheet.save(path)
        print(f"Saved {path.relative_to(ROOT).as_posix()}")
    else:
        name, path = resolve(args)
        image = pipeline.prepare(Image.open(path))
        embedding = pipeline.sam_embed(m, image)  # once; every prompt on this image reuses it
        queue = [args.prompt] if args.prompt else []
        while True:
            if not queue:
                typed = input("prompt (empty line to stop): ").strip() if args.interactive else ""
                if not typed:
                    break
                queue.append(typed)
            _, _, path = run_prompt(m, image, name, queue.pop(0), frozen, prompts, embedding)
            if not args.interactive:
                break
    if args.open and path:
        os.startfile(path)  # noqa: S606 - opens the overlay on screen while recording


if __name__ == "__main__":
    main()
