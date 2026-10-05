"""Turn each product image into an RGBA cutout with a clean alpha mask, once.

Spec: docs/cutouts_spec.md. Runs before data.py and touches products only.
The paste mask becomes ground truth later, so nothing here may use SAM, Grounding DINO,
CLIP, transformers or torch. rembg (onnxruntime) is the only model allowed.

A file that ships with its own mask (the rendered set) is cut out with that mask.
An own photo on a plain sheet is cut out by backdrop threshold, with rembg as the fallback.

    python src/cutouts.py [--force] [--only <asset_id>]
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from data import read_sources

ROOT = Path(__file__).resolve().parents[1]

PRODUCT_KINDS = ("cereal_box", "distractor")
REPORT_FIELDS = ["asset_id", "filename", "method", "status", "reason", "fg_frac", "largest_frac",
                 "solidity", "touches_border", "width", "height"]
ALL_RULES = ("foreground share", "largest piece", "solidity", "border")
MASK_RULES = ("largest piece", "border")  # a shipped mask is exact, and its object is small in a real scene

SHEET_COLUMNS = 5
TILE_PX = 250
CAPTION_PX = 62
OUTLINE_RGB = (255, 0, 255)


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def load_config() -> dict:
    with open(_p("config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["cutouts"]
    if cfg["method"] != "backdrop_threshold":
        sys.exit(f"cutouts.method is {cfg['method']!r}; only backdrop_threshold is implemented.")
    if cfg["fallback"] not in ("rembg", "none"):
        sys.exit(f"cutouts.fallback is {cfg['fallback']!r}; use rembg or none.")
    return cfg


def asset_id_for(path: Path) -> str:
    """Same rule as data.py, so the two scripts agree without sharing state."""
    return "a_" + hashlib.md5(path.read_bytes()).hexdigest()[:10]


def load_photo(path: Path, max_long_side: int) -> Image.Image:
    # Phones store rotation in EXIF. Skipping this gives sideways cutouts.
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = max_long_side / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return image


# ---------------------------------------------------------------- masks

def threshold_mask(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """Pixels whose Lab distance (CIE76) from this photo's backdrop exceeds the threshold."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)  # float input gives true Lab units
    height, width = lab.shape[:2]
    bh = max(1, round(height * cfg["border_frac"]))
    bw = max(1, round(width * cfg["border_frac"]))
    ring = np.ones((height, width), dtype=bool)
    ring[bh:height - bh, bw:width - bw] = False
    backdrop = np.median(lab[ring], axis=0)  # this photo only, never pooled across photos
    return np.linalg.norm(lab - backdrop, axis=2) > cfg["delta_e_threshold"]


@functools.lru_cache(maxsize=1)
def _rembg_session(model_name: str):
    from rembg import new_session

    # Named explicitly: rembg 2.0.85 would otherwise pick its own default model.
    return new_session(model_name)


def rembg_mask(image: Image.Image, cfg: dict) -> np.ndarray:
    from rembg import remove

    alpha = remove(image, session=_rembg_session(cfg["rembg_model"]), only_mask=True)
    return np.asarray(alpha.convert("L")) >= 128


def keep_largest(m: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """Keep the largest piece and erode it. Returns (mask, largest piece's share before the rest was dropped)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros(m.shape, dtype=bool), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_frac = float(areas.max() / areas.sum())
    kept = (labels == 1 + int(np.argmax(areas))).astype(np.uint8)
    if cfg["erode_px"] > 0:
        kept = cv2.erode(kept, np.ones((3, 3), np.uint8), iterations=int(cfg["erode_px"]))  # trims edge fringe
    return kept.astype(bool), largest_frac


def clean_mask(raw: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """For a mask found in a photo: close, open, fill holes, then keep the largest piece and erode."""
    k = int(cfg["morph_kernel_px"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    # Flood the outside from a padded corner. Whatever the flood can't reach is product or a hole in it.
    padded = np.pad(m, 1)
    cv2.floodFill(padded, None, (0, 0), 2)
    return keep_largest(padded[1:-1, 1:-1] != 2, cfg)


def qa(mask: np.ndarray, largest_frac: float, cfg: dict, rules: tuple = ALL_RULES) -> tuple[dict, list[str]]:
    limits = cfg["qa"]
    area = int(mask.sum())
    fg_frac = area / mask.size
    solidity = 0.0
    if area:
        points = np.argwhere(mask)[:, ::-1].astype(np.int32)  # (x, y)
        hull = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
        solidity = area / int(hull.sum())
    touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())

    lo, hi = limits["fg_frac_range"]
    found = {
        "foreground share": None if lo <= fg_frac <= hi else f"foreground is {fg_frac:.2f} of the image (allowed {lo} to {hi})",
        "largest piece": None if largest_frac >= limits["min_largest_component_frac"] else
        f"largest piece is {largest_frac:.2f} of the foreground (needs {limits['min_largest_component_frac']})",
        "solidity": None if solidity >= limits["min_solidity"] else f"solidity {solidity:.2f} (needs {limits['min_solidity']})",
        "border": "mask touches the image border" if touches and not limits["allow_border_touch"] else None,
    }
    failures = ["the mask is empty"] if not area else [found[rule] for rule in rules if found[rule]]
    metrics = {"fg_frac": f"{fg_frac:.4f}", "largest_frac": f"{largest_frac:.4f}",
               "solidity": f"{solidity:.4f}", "touches_border": "yes" if touches else "no"}
    return metrics, failures


def make_cutout(rgb: np.ndarray, mask: np.ndarray, pad: int) -> Image.Image:
    """Crop to the mask's box plus pad. Alpha is 255 inside the mask and 0 outside, with no feathering."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, mask.shape[0])
    left, right = max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, mask.shape[1])
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, :3] = rgb[mask]  # surrounding pixels are dropped, so none can bleed in when edges are feathered
    rgba[mask, 3] = 255
    return Image.fromarray(rgba[top:bottom, left:right])


def _rejected(method: str, reason: str, metrics: dict | None = None) -> tuple[dict, None]:
    blank = {"fg_frac": "", "largest_frac": "", "solidity": "", "touches_border": ""}
    return {"method": method, "status": "rejected", "reason": reason, **(metrics or blank), "width": "", "height": ""}, None


def _accepted(method: str, metrics: dict, cutout: Image.Image) -> tuple[dict, Image.Image]:
    return {"method": method, "status": "ok", "reason": "", **metrics, "width": cutout.width, "height": cutout.height}, cutout


def cutout_from_mask(path: Path, mask_path: Path, cfg: dict) -> tuple[dict, Image.Image | None]:
    """A file that ships with its own mask: the mask is the outline. No thresholding and no model."""
    with Image.open(path) as original, Image.open(mask_path) as shipped:
        if ImageOps.exif_transpose(original).size != shipped.size:
            return _rejected("mask", f"mask is {shipped.size}, image is {original.size}")
        image = load_photo(path, cfg["max_long_side"])
        raw = np.asarray(shipped.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 127
    mask, largest_frac = keep_largest(raw, cfg)
    metrics, failures = qa(mask, largest_frac, cfg, rules=MASK_RULES)
    if failures:
        return _rejected("mask", "mask: " + "; ".join(failures), metrics)
    return _accepted("mask", metrics, make_cutout(np.asarray(image), mask, int(cfg["pad_px"])))


def cutout_from_photo(path: Path, cfg: dict) -> tuple[dict, Image.Image | None]:
    """An own photo on a plain sheet: threshold first, rembg only if that fails QA."""
    image = load_photo(path, cfg["max_long_side"])
    rgb = np.asarray(image)
    attempts = [("threshold", lambda: threshold_mask(rgb, cfg))]
    if cfg["fallback"] == "rembg":
        attempts.append(("rembg", lambda: rembg_mask(image, cfg)))
    reasons = []
    for method, make_raw in attempts:
        mask, largest_frac = clean_mask(make_raw(), cfg)
        metrics, failures = qa(mask, largest_frac, cfg)
        if not failures:
            return _accepted(method, metrics, make_cutout(rgb, mask, int(cfg["pad_px"])))
        reasons.append(f"{method}: " + "; ".join(failures))
    return _rejected(method, " | ".join(reasons), metrics)


def process(product: dict, cfg: dict) -> tuple[dict, Image.Image | None]:
    path = _p(product["source_path"])
    try:
        if product["mask_path"]:
            return cutout_from_mask(path, _p(product["mask_path"]), cfg)
        if product["source"] == "own":
            return cutout_from_photo(path, cfg)
        return _rejected("", "no mask ships with this file, and it is not an own photo on a plain sheet")
    except OSError as err:
        return _rejected("", f"cannot open: {err}")


# ---------------------------------------------------------------- report and review sheet

def read_report(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {row["asset_id"]: row for row in csv.DictReader(f)}


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _checkerboard(size: int, cell: int = 10) -> Image.Image:
    yy, xx = np.indices((size, size)) // cell
    shade = np.where((xx + yy) % 2 == 0, 205, 245).astype(np.uint8)
    return Image.fromarray(np.dstack([shade] * 3))


def _tile(row: dict, product: dict, font) -> Image.Image:
    tile = Image.new("RGB", (TILE_PX, TILE_PX + CAPTION_PX), "white")
    inner = TILE_PX - 16
    if row["status"] == "ok":
        cutout = Image.open(_p("data", "assets", "cutouts", f"{row['asset_id']}.png"))
        cutout.thumbnail((inner, inner), Image.Resampling.LANCZOS)
        board = _checkerboard(TILE_PX)
        offset = ((TILE_PX - cutout.width) // 2, (TILE_PX - cutout.height) // 2)
        board.paste(cutout, offset, cutout)
        alpha = (np.asarray(cutout)[..., 3] > 127).astype(np.uint8)
        edge = (alpha == 1) & (cv2.erode(alpha, np.ones((3, 3), np.uint8)) == 0)  # 1 px outline of the mask
        pixels = np.asarray(board).copy()
        pixels[offset[1]:offset[1] + cutout.height, offset[0]:offset[0] + cutout.width][edge] = OUTLINE_RGB
        tile.paste(Image.fromarray(pixels), (0, 0))
    else:
        photo = load_photo(_p(row["filename"]), inner)
        tile.paste(Image.new("RGB", (TILE_PX, TILE_PX), (90, 90, 90)), (0, 0))
        tile.paste(photo, ((TILE_PX - photo.width) // 2, (TILE_PX - photo.height) // 2))

    draw = ImageDraw.Draw(tile)
    status = f"{row['status']} ({row['method']})" if row["status"] == "ok" else "REJECTED"
    label = product.get("name") or product.get("group_name", "")
    if product.get("damaged") == "yes":
        label += " [dented]"
    lines = [row["asset_id"], status, label[:36], (row["reason"] or Path(row["filename"]).name)[:40]]
    for i, line in enumerate(lines):
        fill = (180, 0, 0) if row["status"] != "ok" and i in (1, 3) else (0, 0, 0)
        draw.text((6, TILE_PX + 2 + i * 14), line, fill=fill, font=font)
    return tile


def write_sheet(path: Path, rows: list[dict], products: list[dict]) -> None:
    font = ImageFont.load_default(size=12)
    n_rows = max(1, -(-len(rows) // SHEET_COLUMNS))
    sheet = Image.new("RGB", (SHEET_COLUMNS * TILE_PX, n_rows * (TILE_PX + CAPTION_PX)), "white")
    for i, (row, product) in enumerate(zip(rows, products)):
        sheet.paste(_tile(row, product, font),
                    ((i % SHEET_COLUMNS) * TILE_PX, (i // SHEET_COLUMNS) * (TILE_PX + CAPTION_PX)))
    sheet.save(path, format="PNG")


def _group_counts(rows: list[dict], products: list[dict]) -> tuple[int, int]:
    """(cereal-box groups with an intact and a dented cutout ok, distractor groups with a cutout ok)."""
    states: dict[str, set] = {}
    distractors = set()
    for row, product in zip(rows, products):
        if row["status"] != "ok":
            continue
        if product["kind"] == "cereal_box":
            states.setdefault(product["group_name"], set()).add(product["damaged"])
        else:
            distractors.add(product["group_name"])
    return sum(1 for s in states.values() if {"yes", "no"} <= s), len(distractors)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="redo every cutout, including ones already on disk")
    parser.add_argument("--only", metavar="ASSET_ID", help="redo this one asset and leave the rest as they are")
    args = parser.parse_args()

    start = time.perf_counter()
    cfg = load_config()
    products = [r for r in read_sources() if r["kind"] in PRODUCT_KINDS]
    cutouts_dir = _p("data", "assets", "cutouts")
    report_path = _p("data", "assets", "cutouts_report.csv")
    sheet_path = _p("data", "assets", "cutouts_sheet.png")
    cutouts_dir.mkdir(parents=True, exist_ok=True)
    previous = read_report(report_path)

    ids = [asset_id_for(_p(p["source_path"])) for p in products]
    if args.only and args.only not in ids:
        sys.exit(f"--only {args.only}: no listed product has that asset_id.")

    done: dict[str, dict] = {}
    rows, kept_products, n_reused, n_waiting = [], [], 0, 0
    for product, asset_id in zip(products, ids):
        png = cutouts_dir / f"{asset_id}.png"
        old = previous.get(asset_id)
        usable = old is not None and (old["status"] == "rejected" or png.exists())
        redo = args.force or args.only == asset_id
        if asset_id in done:  # the same bytes listed twice: one cutout, one report row per listing
            fields, note = done[asset_id], "same file as an earlier row"
        elif usable and not redo:
            fields, note = {k: old[k] for k in REPORT_FIELDS if k not in ("asset_id", "filename")}, "kept from last run"
            n_reused += 1
        elif args.only and not redo:
            n_waiting += 1
            continue
        else:
            fields, cutout = process(product, cfg)
            note = ""
            if cutout is not None:
                cutout.save(png, format="PNG")
            elif png.exists():
                note = f"(an older {png.name} is still on disk; nothing was deleted)"
        done[asset_id] = fields
        rows.append({"asset_id": asset_id, "filename": product["source_path"], **fields})
        kept_products.append(product)
        print(f"[cutout] {asset_id}  {Path(product['source_path']).name[:38]:<38} {fields['method']:<9} "
              f"{fields['status']:<8} fg {fields['fg_frac'] or '-':<6} solidity {fields['solidity'] or '-':<6} {note}")
        if fields["status"] == "rejected":
            print(f"         reason: {fields['reason']}")

    write_report(report_path, rows)
    write_sheet(sheet_path, rows, kept_products)

    n_ok = sum(r["status"] == "ok" for r in rows)
    by_method = {m: sum(r["status"] == "ok" and r["method"] == m for r in rows) for m in ("mask", "threshold", "rembg")}
    box_groups, distractor_groups = _group_counts(rows, kept_products)
    share = f"{100 * n_ok / len(rows):.0f}%" if rows else "n/a"
    print(f"[cutout] {len(rows)} product rows: {n_ok} ok ({share}), {len(rows) - n_ok} rejected, "
          f"{n_reused} kept from the last run")
    print(f"[cutout] ok by method: {by_method['mask']} shipped mask, {by_method['threshold']} threshold, "
          f"{by_method['rembg']} rembg")
    if n_waiting:
        print(f"[cutout] {n_waiting} product rows have no cutout yet. Run without --only to make them.")
    print(f"[cutout] cereal-box groups with an intact and a dented cutout: {box_groups} (gate: 9 or more after review)")
    print(f"[cutout] distractor groups with a cutout: {distractor_groups} (gate: 5 or more after review)")
    print(f"[cutout] report {report_path.relative_to(ROOT).as_posix()}, sheet {sheet_path.relative_to(ROOT).as_posix()}")
    print("[cutout] Next: look at the sheet, then fill data/raw/own/cutout_review.csv by hand "
          "(asset_id, keep yes or no, note).")
    print(f"[time] cutouts {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
