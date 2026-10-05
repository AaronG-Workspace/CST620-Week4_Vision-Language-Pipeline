"""Build the val, test and showcase images by pasting cutouts and drawn labels onto shelf backgrounds.

Spec: docs/make_synth_spec.md. Ground truth is the paste mask itself, so nothing here may
use SAM, Grounding DINO, CLIP, transformers or torch. Every split is built from its own pools only.

    python src/make_synth.py --split val|test|showcase|all [--force] [--preview] [--verify]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import hashlib
import io
import json
import math
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from pycocotools import mask as mask_utils

from data import PROVENANCE_FIELDS, SPLITS, _read_csv

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = {"cereal_box": 1, "damaged_cereal_box": 2, "misaligned_label": 3}
STOP_WORDS = {"a", "an", "of", "on", "at", "in", "and", "the", "up", "with"}
MANIFEST_FIELDS = ["image_id", "path", "condition", "background_id", "asset_ids", "md5", "phash", "seed_entropy",
                   "font_sha256"]
MAX_LAYOUTS = 200          # whole-image layout retries before giving up on an image
LABEL_SUPERSAMPLE = 3      # labels are drawn large, rotated, then shrunk, so their edges come out clean
PREVIEW_COLORS = {"cereal_box": (40, 200, 60), "damaged_cereal_box": (230, 40, 40), "distractor": (150, 150, 150),
                  "misaligned_label": (255, 0, 255), "straight_label": (0, 200, 255)}


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _half_up(share: float, total: int) -> int:
    return math.floor(Fraction(str(share)) * total + Fraction(1, 2))


# ---------------------------------------------------------------- inputs

def prompt_words(node, skip=("gt", "written", "version", "change_log")) -> set[str]:
    """Content words of every phrase in prompts.yaml: 3 or more letters, minus the stop words."""
    if isinstance(node, dict):
        return set().union(*[prompt_words(v) for k, v in node.items() if k not in skip])
    if isinstance(node, list):
        return set().union(*[prompt_words(v) for v in node])
    if isinstance(node, str):
        return {w for w in re.findall(r"[a-z]+", node.lower()) if len(w) >= 3 and w not in STOP_WORDS}
    return set()


@dataclasses.dataclass
class Env:
    cfg: dict
    words: list[str]
    font_path: str
    font_sha256: str

    @functools.lru_cache(maxsize=None)
    def background(self, source_path: str, shelf_lines: str) -> tuple[np.ndarray, tuple[int, ...]]:
        image = ImageOps.exif_transpose(Image.open(_p(source_path))).convert("RGB")
        scale = self.cfg["scene"]["canvas_long_side"] / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        lines = tuple(round(int(y) * scale) for y in shelf_lines.split("|") if y)
        return np.asarray(image), lines

    @functools.lru_cache(maxsize=None)
    def cutout(self, asset_id: str) -> Image.Image:
        return Image.open(_p("data", "assets", "cutouts", f"{asset_id}.png")).convert("RGBA")

    @functools.lru_cache(maxsize=None)
    def native_height(self, asset_id: str) -> int:
        """Height of the object itself in its cutout, without the transparent padding."""
        rows = np.flatnonzero((np.asarray(self.cutout(asset_id))[..., 3] > 127).any(axis=1))
        return int(rows[-1] - rows[0] + 1)

    def scaled(self, asset_id: str, height: int) -> tuple[np.ndarray, np.ndarray, bool]:
        """The cutout with its object at the given height: (rgb, hard mask, capped), cropped tight to the mask.

        A source too small for the target is placed smaller, never enlarged past scene.max_upscale.
        """
        cut = self.cutout(asset_id)
        wanted = height / self.native_height(asset_id)
        scale = min(wanted, float(self.cfg["scene"]["max_upscale"]))
        size = (max(1, round(cut.width * scale)), max(1, round(cut.height * scale)))
        arr = np.asarray(cut.resize(size, Image.Resampling.LANCZOS))  # Pillow resizes RGBA premultiplied
        mask = arr[..., 3] >= 128
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        box = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        return arr[..., :3][box], mask[box], scale < wanted

    def __hash__(self):  # lru_cache on methods needs a hashable self
        return id(self)


def load_env() -> Env:
    with open(_p("config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(_p("prompts", "prompts.yaml"), encoding="utf-8") as f:
        banned = prompt_words(yaml.safe_load(f))
    words = [w.strip().lower() for w in _p(cfg["labels"]["word_list"]).read_text(encoding="utf-8").split() if w.strip()]
    # CLIP-style models can read printed text, so a label must never spell out a prompt word.
    clashes = sorted({f"{w} (contains {b})" for w in words for b in banned if b in w})
    if clashes:
        sys.exit("label_words.txt has words that contain a prompt word:\n  " + "\n  ".join(clashes))
    font = Path(cfg["labels"]["font"])
    if not font.is_file():
        sys.exit(f"Font not found: {font}")
    return Env(cfg, words, str(font), hashlib.sha256(font.read_bytes()).hexdigest())


def load_pool(split: str) -> dict:
    """This split's backgrounds, cereal-box groups and distractor groups. Never another split's."""
    assets = {a["asset_id"]: a for a in _read_csv(_p("data", "assets", "assets.csv")) if a["status"] == "ok"}
    mine = [assets[p["asset_id"]] for p in _read_csv(_p("splits", "pools.csv")) if p["split"] == split]
    pool = {"assets": {a["asset_id"]: a for a in mine}, "backgrounds": [], "cereal": {}, "distractor": {}}
    for a in sorted(mine, key=lambda a: a["asset_id"]):
        if a["kind"] == "background":
            if a["bg_clean"] != "true":
                sys.exit(f"{a['source_path']} is not marked bg_clean, so it is refused as a background.")
            pool["backgrounds"].append(a["asset_id"])
        elif a["kind"] == "cereal_box":
            pool["cereal"].setdefault(a["group_id"], {"true": [], "false": []})[a["damaged"]].append(a["asset_id"])
        else:
            pool["distractor"].setdefault(a["group_id"], []).append(a["asset_id"])
    if not (pool["backgrounds"] and pool["cereal"] and pool["distractor"]):
        sys.exit(f"{split}: the pool needs at least one background, one cereal-box group and one distractor group.")
    return pool


# ---------------------------------------------------------------- the split-level plan

@dataclasses.dataclass
class Plan:
    cereal: list[int]        # per image
    damaged: list[int]       # per image, how many of its cereal boxes are dented
    distractors: list[int]
    labels: list[int]
    misaligned: list[int]    # per image, how many of its labels are crooked
    conditions: list[str]
    backgrounds: list[str]
    capped: dict[str, int]


def _spread(total_true: int, sizes: list[int], rng: np.random.Generator) -> list[int]:
    """Shuffle a list holding exactly total_true True values and cut it into runs of the given sizes."""
    flags = np.zeros(sum(sizes), dtype=bool)
    flags[:total_true] = True
    flags = flags[rng.permutation(len(flags))]
    edges = np.cumsum([0] + sizes)
    return [int(flags[a:b].sum()) for a, b in zip(edges[:-1], edges[1:])]


def make_plan(split: str, pool: dict, cfg: dict) -> Plan:
    """Every draw that is fixed for the whole split, made before any image is laid out."""
    scene, lab, n = cfg["scene"], cfg["labels"], int(cfg["images_per_split"][split])
    rngs = [np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), SPLITS.index(split), 1000 + k]))
            for k in range(5)]  # 0 counts, 1 damaged list, 2 misaligned list, 3 condition list, 4 background order
    limit = {"cereal": len(pool["cereal"]), "distractors": len(pool["distractor"])}
    if not scene["cap_at_available_groups"]:
        limit = {key: 10 ** 9 for key in limit}
    cereal, distractors, labels, capped = [], [], [], {"cereal_boxes": 0, "distractors": 0, "labels": 0}
    for _ in range(n):
        drawn = [int(rngs[0].integers(lo, hi + 1)) for lo, hi in
                 (scene["cereal_boxes_per_image"], scene["distractors_per_image"], lab["per_image"])]
        c, d = min(drawn[0], limit["cereal"]), min(drawn[1], limit["distractors"])
        lb = min(drawn[2], c + d)  # one label per product at most
        capped["cereal_boxes"] += c < drawn[0]
        capped["distractors"] += d < drawn[1]
        capped["labels"] += lb < drawn[2]
        cereal.append(c), distractors.append(d), labels.append(lb)

    # Quotas, not coin flips: exactly round(share x total) across the split.
    damaged = _spread(_half_up(scene["damaged_share_of_cereal_boxes"], sum(cereal)), cereal, rngs[1])
    n_dented = sum(1 for g in pool["cereal"].values() if g["true"])
    n_intact = sum(1 for g in pool["cereal"].values() if g["false"])
    lo = [max(0, c - n_intact) for c in cereal]   # an image can't show more intact boxes than intact groups
    hi = [min(c, n_dented) for c in cereal]       # nor more dented boxes than groups with a dented cutout
    if not sum(lo) <= sum(damaged) <= sum(hi):
        sys.exit(f"{split}: the damaged quota is {sum(damaged)} boxes, but this split's groups can show only "
                 f"{sum(lo)} to {sum(hi)}. Shortfall: {max(sum(damaged) - sum(hi), sum(lo) - sum(damaged))}.")
    for i in range(n):  # move a flag to the first image with room; the split total never changes
        while damaged[i] > hi[i]:
            j = next(j for j in range(n) if damaged[j] < hi[j])
            damaged[i], damaged[j] = damaged[i] - 1, damaged[j] + 1
        while damaged[i] < lo[i]:
            j = next(j for j in range(n) if damaged[j] > lo[j])
            damaged[i], damaged[j] = damaged[i] + 1, damaged[j] - 1
    misaligned = _spread(_half_up(lab["misaligned_share"], sum(labels)), labels, rngs[2])

    names = list(cfg["conditions"])
    balanced = [names[i % len(names)] for i in range(n)]
    conditions = [balanced[i] for i in rngs[3].permutation(n)]
    order = [pool["backgrounds"][i] for i in rngs[4].permutation(len(pool["backgrounds"]))]
    return Plan(cereal, damaged, distractors, labels, misaligned, conditions,
                [order[i % len(order)] for i in range(n)], capped)


# ---------------------------------------------------------------- one image

@functools.lru_cache(maxsize=64)
def _font(path: str, size: int):
    return ImageFont.truetype(path, size)


def render_label(rng: np.random.Generator, host_width: int, misaligned: bool, env: Env) -> dict:
    lab, ss = env.cfg["labels"], LABEL_SUPERSAMPLE
    width = max(16, round(float(rng.uniform(*lab["width_frac_of_host"])) * host_width))
    height = max(8, round(width / float(rng.uniform(*lab["aspect"]))))
    color = lab["palette"][int(rng.integers(len(lab["palette"])))]
    n_lines = int(rng.integers(lab["text_lines"][0], lab["text_lines"][1] + 1))
    lines = [env.words[int(rng.integers(len(env.words)))] for _ in range(n_lines)]
    lo, hi = lab["misaligned_deg"] if misaligned else lab["straight_deg"]
    angle = float(rng.uniform(lo, hi)) * (1 if int(rng.integers(2)) else -1)  # either direction

    big = Image.new("RGBA", (width * ss, height * ss), color)
    draw = ImageDraw.Draw(big)
    size = max(6, int(height * ss * 0.8 / n_lines))
    while size > 6 and max(draw.textlength(line, font=_font(env.font_path, size)) for line in lines) > 0.9 * width * ss:
        size -= 1
    for i, line in enumerate(lines):
        draw.text((width * ss / 2, height * ss * (i + 0.5) / n_lines), line, fill=(25, 25, 25, 255),
                  font=_font(env.font_path, size), anchor="mm")
    turned = big.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    small = np.asarray(turned.resize((max(1, round(turned.width / ss)), max(1, round(turned.height / ss))),
                                     Image.Resampling.LANCZOS))
    mask = small[..., 3] >= 128
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    box = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    mask = mask[box]
    return {"rgb": small[..., :3][box], "mask": mask, "alpha": np.where(mask, small[..., 3][box] / 255.0, 0.0),
            "angle": angle, "misaligned": misaligned, "text": " ".join(lines)}


def _try_layout(rng, slots, n_labels, n_misaligned, canvas_shape, shelf, max_overlap, pool, env):
    """One attempt at placing every product and label. Returns (products, labels), or None to retry."""
    scene = env.cfg["scene"]
    H, W = canvas_shape
    tries, used, placed = int(scene["max_placement_tries"]), set(), []
    for slot in rng.permutation(len(slots)):  # also the paste order: later products sit in front
        kind, dented = slots[int(slot)]
        if kind == "cereal_box":
            want = "true" if dented else "false"
            options = sorted(g for g, by in pool["cereal"].items() if g not in used and by[want])
        else:
            options = sorted(g for g in pool["distractor"] if g not in used)
        if not options:
            return None  # for example a dented slot with no dented group left for this image
        group = options[int(rng.integers(len(options)))]
        ids = pool["cereal"][group][want] if kind == "cereal_box" else pool["distractor"][group]
        asset_id = ids[int(rng.integers(len(ids)))]
        used.add(group)  # one asset per group per image: a box never stands beside its own dented copy
        rgb, mask, capped = env.scaled(asset_id, round(float(rng.uniform(*scene["object_height_frac"])) * H))
        h, w = mask.shape
        if w > W or h > H:
            return None
        area = int(mask.sum())
        lines = [y for y in shelf if y - h >= 0]
        for _ in range(tries):
            x = int(rng.integers(0, W - w + 1))
            bottom = lines[int(rng.integers(len(lines)))] if lines else int(rng.integers(math.ceil(2 * H / 3), H + 1))
            y = bottom - h
            if all(int((p["full"][y:y + h, x:x + w] & mask).sum()) <= max_overlap * min(area, p["area"]) for p in placed):
                break
        else:
            return None
        full = np.zeros((H, W), dtype=bool)
        full[y:y + h, x:x + w] = mask
        b = float(scene["brightness_jitter"])
        placed.append({"kind": kind, "dented": bool(dented), "asset_id": asset_id, "x": x, "y": y, "w": w, "h": h,
                       "rgb": rgb, "mask": mask, "full": full, "area": area, "capped": capped,
                       "bright": 1 + float(rng.uniform(-b, b))})

    covered = np.zeros((H, W), dtype=bool)
    for obj in reversed(placed):  # later pastes hide earlier ones
        obj["visible"] = obj["full"] & ~covered
        obj["visible_frac"] = float(obj["visible"].sum() / obj["area"])
        if obj["visible_frac"] < scene["min_visible_frac"]:
            return None
        covered |= obj["full"]

    labels = []
    hosts = rng.permutation(len(placed))[:n_labels]
    crooked = (np.arange(n_labels) < n_misaligned)[rng.permutation(n_labels)]
    for host_index, misaligned in zip(hosts, crooked):
        host = placed[int(host_index)]
        label = render_label(rng, host["w"], bool(misaligned), env)
        lh, lw = label["mask"].shape
        if lw > host["w"] or lh > host["h"]:
            return None
        for _ in range(tries):
            x = int(rng.integers(host["x"], host["x"] + host["w"] - lw + 1))
            y = int(rng.integers(host["y"], host["y"] + host["h"] - lh + 1))
            if host["visible"][y:y + lh, x:x + lw][label["mask"]].all():  # fully inside what is visible of the host
                break
        else:
            return None
        labels.append({**label, "x": x, "y": y, "host": int(host_index)})
    return placed, labels


def _feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """Paste alpha that fades in from the mask's edge toward the inside. Zero everywhere outside the mask."""
    if radius <= 0:
        return mask.astype(np.float32)
    pad = math.ceil(3 * radius)
    padded = Image.fromarray(np.pad(mask, pad).astype(np.uint8) * 255)
    blurred = np.asarray(padded.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)[pad:-pad, pad:-pad] / 255
    return np.clip((blurred - 0.5) * 2, 0, 1) * mask


def _degrade(image: Image.Image, condition: dict) -> Image.Image:
    """Whole-image degradation. Ground truth does not change."""
    if "brightness_scale" in condition:
        scaled = np.asarray(image, dtype=np.float32) * condition["brightness_scale"]
        image = Image.fromarray(np.clip(scaled + 0.5, 0, 255).astype(np.uint8))
    if "gaussian_blur_sigma" in condition:
        image = image.filter(ImageFilter.GaussianBlur(condition["gaussian_blur_sigma"]))
    if "motion_blur_px" in condition:
        image = Image.fromarray(cv2.blur(np.asarray(image), (int(condition["motion_blur_px"]), 1)))  # sideways
    if "jpeg_quality" in condition:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(condition["jpeg_quality"]))
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return image


def _truth(mask: np.ndarray, **fields) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {**fields, "segmentation": {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")},
            "bbox": [float(v) for v in mask_utils.toBbox(rle)], "area": int(mask_utils.area(rle)), "iscrowd": 0}


def build_image(split: str, index: int, plan: Plan, pool: dict, env: Env, preview: bool) -> dict:
    cfg = env.cfg
    scene = cfg["scene"]
    entropy = [int(cfg["seed"]), SPLITS.index(split), index]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))  # this image's placement, nothing else
    image_id = f"s_{split}_{index:04d}"
    background = pool["assets"][plan.backgrounds[index]]
    canvas, shelf = env.background(background["source_path"], background["shelf_lines"])
    condition = plan.conditions[index]
    max_overlap = cfg["conditions"][condition].get("max_overlap_frac", scene["max_overlap_frac"])
    slots = ([("cereal_box", True)] * plan.damaged[index]
             + [("cereal_box", False)] * (plan.cereal[index] - plan.damaged[index])
             + [("distractor", None)] * plan.distractors[index])
    for retries in range(MAX_LAYOUTS):
        layout = _try_layout(rng, slots, plan.labels[index], plan.misaligned[index], canvas.shape[:2], shelf,
                             max_overlap, pool, env)
        if layout:
            break
    else:
        sys.exit(f"{split} image {index}: no layout fits on {background['source_path']} after {MAX_LAYOUTS} tries. "
                 "Its shelf lines may leave too little room. Quotas are never quietly missed, so nothing was written "
                 "for this image.")
    placed, labels = layout

    pixels = canvas.astype(np.float32)
    for obj in placed:
        alpha = _feather(obj["mask"], float(scene["feather_radius_px"]))[..., None]
        region = pixels[obj["y"]:obj["y"] + obj["h"], obj["x"]:obj["x"] + obj["w"]]
        region[:] = region * (1 - alpha) + np.clip(obj["rgb"].astype(np.float32) * obj["bright"], 0, 255) * alpha
    for label in labels:
        lh, lw = label["mask"].shape
        alpha = label["alpha"][..., None].astype(np.float32)
        region = pixels[label["y"]:label["y"] + lh, label["x"]:label["x"] + lw]
        region[:] = region * (1 - alpha) + label["rgb"].astype(np.float32) * alpha
    image = _degrade(Image.fromarray(np.clip(pixels + 0.5, 0, 255).astype(np.uint8)), cfg["conditions"][condition])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    annotations, negatives, outlines = [], [], []
    for k, obj in enumerate(placed):
        asset = pool["assets"][obj["asset_id"]]
        obj["instance_id"] = f"{image_id}_o{k}"
        common = dict(instance_id=obj["instance_id"], asset_id=asset["asset_id"], group_id=asset["group_id"],
                      name=asset["name"], visible_frac=round(obj["visible_frac"], 4), condition=condition)
        if obj["kind"] == "distractor":
            negatives.append(_truth(obj["visible"], kind="distractor", **common))
        else:
            annotations.append(_truth(obj["visible"], category_id=CATEGORIES["cereal_box"], **common))
            if obj["dented"]:  # a dented box is both a cereal box and a damaged cereal box: one instance_id
                annotations.append(_truth(obj["visible"], category_id=CATEGORIES["damaged_cereal_box"],
                                          dent_severity=asset.get("dent_severity") or "unrated", **common))
            else:
                negatives.append(_truth(obj["visible"], kind="intact_cereal_box", **common))
        outlines.append((obj["visible"], "damaged_cereal_box" if obj["dented"] else obj["kind"]))
    for k, label in enumerate(labels):
        host = placed[label["host"]]
        asset = pool["assets"][host["asset_id"]]
        full = np.zeros(canvas.shape[:2], dtype=bool)
        lh, lw = label["mask"].shape
        full[label["y"]:label["y"] + lh, label["x"]:label["x"] + lw] = label["mask"]
        common = dict(instance_id=f"{image_id}_l{k}", host_instance_id=host["instance_id"], asset_id=asset["asset_id"],
                      group_id=asset["group_id"], name=label["text"], visible_frac=1.0, condition=condition,
                      label_angle=round(label["angle"], 2))
        if label["misaligned"]:
            annotations.append(_truth(full, category_id=CATEGORIES["misaligned_label"], **common))
        else:
            negatives.append(_truth(full, kind="straight_label", **common))
        outlines.append((full, "misaligned_label" if label["misaligned"] else "straight_label"))

    thumb = None
    if preview:
        shown = np.asarray(image).copy()
        for mask, key in outlines:
            edge = mask & (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) == 0)
            shown[edge] = PREVIEW_COLORS[key]
        thumb = Image.fromarray(shown)
        thumb.thumbnail((400, 400), Image.Resampling.LANCZOS)
    png = buffer.getvalue()
    return {"image_id": image_id, "png": png, "md5": hashlib.md5(png).hexdigest(), "phash": str(imagehash.phash(image)),
            "width": image.width, "height": image.height, "condition": condition,
            "background_id": background["asset_id"], "asset_ids": [o["asset_id"] for o in placed],
            "groups": {(o["kind"], pool["assets"][o["asset_id"]]["group_id"]) for o in placed},
            "annotations": annotations, "negatives": negatives, "retries": retries, "entropy": entropy,
            "min_visible": min(o["visible_frac"] for o in placed), "thumb": thumb,
            "size_capped": sum(o["capped"] for o in placed)}


# ---------------------------------------------------------------- one split

def build_split(split: str, env: Env, write: bool, preview: bool) -> dict:
    """Generate a split in memory. With write, also save it. Never reads back what it wrote."""
    pool = load_pool(split)
    plan = make_plan(split, pool, env.cfg)
    out = _p("data", "synth", split)
    if write:
        (out / "images").mkdir(parents=True, exist_ok=True)
    images, annotations, negatives, manifest, provenance, thumbs = [], [], [], [], [], []
    counts = {"split": split, "images": 0, "instances": dict.fromkeys(CATEGORIES, 0),
              "negatives": {"intact_cereal_box": 0, "straight_label": 0, "distractor": 0},
              "distinct_groups_used": {}, "condition_balance": dict.fromkeys(env.cfg["conditions"], 0),
              "smallest_visible_frac": 1.0, "capped_draws": plan.capped, "layout_retries": 0,
              "placements_capped_by_max_upscale": 0, "damaged_by_dent_severity": {}}
    groups, by_id = set(), {v: k for k, v in CATEGORIES.items()}
    for index in range(int(env.cfg["images_per_split"][split])):
        made = build_image(split, index, plan, pool, env, preview)
        path = f"data/synth/{split}/images/{made['image_id']}.png"
        if write:
            _p(path).write_bytes(made["png"])
        images.append({"id": index, "name": made["image_id"], "file_name": f"images/{made['image_id']}.png",
                       "width": made["width"], "height": made["height"], "condition": made["condition"],
                       "background_id": made["background_id"]})
        for record in made["annotations"]:
            annotations.append({"id": len(annotations), "image_id": index, **record})
            counts["instances"][by_id[record["category_id"]]] += 1
            if "dent_severity" in record:
                tally = counts["damaged_by_dent_severity"]
                tally[record["dent_severity"]] = tally.get(record["dent_severity"], 0) + 1
        for record in made["negatives"]:
            negatives.append({"id": len(negatives), "image_id": index, **record})
            counts["negatives"][record["kind"]] += 1
        parents = "|".join([made["background_id"]] + made["asset_ids"])
        manifest.append({"image_id": made["image_id"], "path": path, "condition": made["condition"],
                         "background_id": made["background_id"], "asset_ids": "|".join(made["asset_ids"]),
                         "md5": made["md5"], "phash": made["phash"], "seed_entropy": json.dumps(made["entropy"]),
                         "font_sha256": env.font_sha256})
        provenance.append({"path": path, "source": "synthetic", "capture_date": "",
                           "license": "derived from own photos", "annotation_method": "paste mask from make_synth.py",
                           "lighting": made["condition"], "angle": "", "occlusion": f"{made['min_visible']:.4f}",
                           "md5": made["md5"], "phash": made["phash"], "group_id": "", "parent_asset_ids": parents})
        groups |= made["groups"] | {("background", made["background_id"])}
        counts["images"] += 1
        counts["condition_balance"][made["condition"]] += 1
        counts["smallest_visible_frac"] = round(min(counts["smallest_visible_frac"], made["min_visible"]), 4)
        counts["layout_retries"] += made["retries"]
        counts["placements_capped_by_max_upscale"] += made["size_capped"]
        thumbs.append((made["image_id"], made["condition"], made["thumb"]))

    counts["distinct_groups_used"] = {kind: sum(1 for k, _ in groups if k == kind)
                                      for kind in ("cereal_box", "distractor", "background")}
    truth = json.dumps({"info": {"split": split, "seed": env.cfg["seed"], "font_sha256": env.font_sha256},
                        "categories": [{"id": i, "name": name} for name, i in CATEGORIES.items()],
                        "images": images, "annotations": annotations, "negatives": negatives},
                       indent=1, sort_keys=True).encode("ascii")
    counts["annotations_sha256"] = hashlib.sha256(truth).hexdigest()
    counts["font_sha256"] = env.font_sha256
    if write:
        (out / "annotations.json").write_bytes(truth)
        with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(manifest)
        (out / "counts.json").write_text(json.dumps(counts, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                                         newline="\n")
        _update_provenance(split, provenance)
    return {"counts": counts, "md5": {m["image_id"]: m["md5"] for m in manifest}, "thumbs": thumbs}


def _update_provenance(split: str, rows: list[dict]) -> None:
    """Replace this split's synthetic rows. Other lines pass through as text, unparsed."""
    path = _p("data", "provenance.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not line.startswith(f"data/synth/{split}/")]
    buffer = io.StringIO()
    csv.DictWriter(buffer, fieldnames=PROVENANCE_FIELDS, lineterminator="\n").writerows(rows)
    path.write_text("\n".join(kept) + "\n" + buffer.getvalue(), encoding="utf-8", newline="\n")


def write_preview(split: str, thumbs: list) -> Path:
    columns, cell_w, cell_h = 4, 410, 430  # thumbnails fit in 400 x 400, portrait or landscape
    sheet = Image.new("RGB", (columns * cell_w, 24 + -(-len(thumbs) // columns) * cell_h), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=12)
    x = 6
    for key, color in PREVIEW_COLORS.items():
        draw.rectangle([x, 6, x + 12, 18], fill=color)
        draw.text((x + 16, 5), key, fill="black", font=font)
        x += 34 + int(draw.textlength(key, font=font))
    for i, (image_id, condition, thumb) in enumerate(thumbs):
        left, top = (i % columns) * cell_w + 5, 24 + (i // columns) * cell_h
        sheet.paste(thumb, (left, top))
        draw.text((left, top + 404), f"{image_id}  {condition}", fill="black", font=font)
    path = _p("data", "synth", split, "preview.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")
    return path


def report(split: str, counts: dict) -> None:
    total = sum(counts["instances"].values())
    if split == "test":  # counts only, and only this one line
        print(f"test: {counts['images']} images, {total} instances, locked")
        return
    print(f"{split}: {counts['images']} images, {total} instances")
    print(f"  instances: {counts['instances']}")
    print(f"  negatives: {counts['negatives']}")
    print(f"  groups used: {counts['distinct_groups_used']} | conditions: {counts['condition_balance']}")
    print(f"  smallest visible share {counts['smallest_visible_frac']} | capped draws {counts['capped_draws']} | "
          f"layout retries {counts['layout_retries']} | placed smaller by max_upscale "
          f"{counts['placements_capped_by_max_upscale']}")


def verify_split(split: str, env: Env) -> bool:
    """Regenerate in memory and compare with the md5s recorded at write time. Opens no written image."""
    out = _p("data", "synth", split)
    if not (out / "manifest.csv").exists():
        print(f"[verify] {split:<9} MISSING (no manifest)")
        return False
    recorded = {m["image_id"]: m["md5"] for m in _read_csv(out / "manifest.csv")}
    truth_sha = json.loads((out / "counts.json").read_text(encoding="utf-8"))["annotations_sha256"]
    fresh = build_split(split, env, write=False, preview=False)
    same = sum(recorded.get(image_id) == md5 for image_id, md5 in fresh["md5"].items())
    ok = same == len(recorded) == len(fresh["md5"]) and fresh["counts"]["annotations_sha256"] == truth_sha
    print(f"[verify] {split:<9} {'match' if ok else 'MISMATCH'}  {same} of {len(recorded)} images, annotations "
          f"{'match' if fresh['counts']['annotations_sha256'] == truth_sha else 'differ'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=[*SPLITS, "all"])
    parser.add_argument("--force", action="store_true", help="rebuild a split that already exists")
    parser.add_argument("--preview", action="store_true", help="contact sheet with ground-truth outlines (never test)")
    parser.add_argument("--verify", action="store_true", help="regenerate in memory and compare md5s; writes nothing")
    args = parser.parse_args()
    if args.preview and args.split == "test":
        sys.exit("--preview refuses the test split. Its images are never opened for viewing.")
    splits = list(SPLITS) if args.split == "all" else [args.split]
    env = load_env()
    print(f"[synth] label words containing a prompt word: 0 ({len(env.words)} words checked)")
    if args.verify:
        sys.exit(0 if all([verify_split(s, env) for s in splits]) else 1)

    for split in splits:
        start = time.perf_counter()
        exists = _p("data", "synth", split, "manifest.csv").exists()
        preview = args.preview and split != "test"
        if exists and not args.force:
            print(f"[synth] {split} already exists and is left alone. Use --force to rebuild it.")
            if not preview:
                continue
        built = build_split(split, env, write=not exists or args.force, preview=preview)
        if not exists or args.force:
            report(split, built["counts"])
        if preview:
            print(f"[synth] preview written to {write_preview(split, built['thumbs']).relative_to(ROOT).as_posix()}")
        print(f"[time] make_synth {split} {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
