"""Inventory raw images, group near duplicates, split asset groups once, and gate split access.

Spec: docs/data_spec.md. Order: cutouts.py, then data.py, then make_synth.py.

    python src/data.py            # build; if the split already exists, verify it and write nothing
    python src/data.py --force    # rebuild everything, including the split
    python src/data.py --verify   # rebuild in memory, compare with disk, write nothing
    python src/data.py --background-sheet   # the sheet and template for the background review
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import math
import re
import sys
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import ExifTags, Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("val", "test", "showcase")
KINDS = ("cereal_box", "distractor", "background")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp"}
MIN_CAPTURE_DATE = date(2024, 1, 1)
STRATA = ("background", "damaged-box", "intact-only box", "distractor")
RENDERED_SET = "data/raw/synthetic/synthetic_retail_set"
SET_CATEGORIES = {"cereal_intact": ("cereal_box", "no"), "cereal_dented": ("cereal_box", "yes"),
                  "distractor": ("distractor", ""), "background": ("background", "")}

ASSET_FIELDS = ["asset_id", "group_id", "kind", "name", "damaged", "source", "source_path", "capture_date", "bg_clean",
                "shelf_lines", "md5", "phash", "status", "reason"]
PROVENANCE_FIELDS = ["path", "source", "capture_date", "license", "annotation_method", "lighting", "angle",
                     "occlusion", "md5", "phash", "group_id",
                     "parent_asset_ids",                           # composites only
                     "openimages_id", "generator_sha256", "seed"]  # the rendered set only
POOL_FIELDS = ["asset_id", "group_id", "kind", "split"]
LOAD_COLUMNS = ["row_type", "split", "path", "asset_id", "kind", "group_id", "md5", "phash", "parent_asset_ids",
                "condition", "name", "damaged"]


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def _csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fail(title: str, items: list[str]) -> None:
    sys.exit(f"{title}:\n  " + "\n  ".join(items))


# ---------------------------------------------------------------- step 1: the two sources

def validate_intake(rows: list[dict], own: Path) -> list[str]:
    errors, listed = [], {}
    for line_no, r in enumerate(rows, start=2):
        r["filename"] = r.get("filename", "").replace("\\", "/")
        where = f"intake.csv line {line_no} ({r['filename'] or 'no filename'})"
        kind = r.get("kind", "")
        if kind not in KINDS:
            errors.append(f"{where}: kind is {kind!r}, expected cereal_box, distractor or background")
            continue
        folder = "backgrounds/" if kind == "background" else "products/"
        if not r["filename"].startswith(folder):
            errors.append(f"{where}: a {kind} photo belongs under {folder}")
        elif not (own / r["filename"]).is_file():
            errors.append(f"{where}: file not found")
        key = r["filename"].lower()
        if key in listed:
            errors.append(f"{where}: listed twice (also on line {listed[key]})")
        listed.setdefault(key, line_no)
        if not r.get("group_name"):
            errors.append(f"{where}: group_name is empty")
        damaged, bg_clean, shelf = r.get("damaged", ""), r.get("bg_clean", ""), r.get("shelf_lines", "")
        if kind == "cereal_box" and damaged not in ("yes", "no"):
            errors.append(f"{where}: damaged must be yes or no on a cereal_box row")
        if kind != "cereal_box" and damaged:
            errors.append(f"{where}: damaged is only for cereal_box rows")
        if kind == "background" and bg_clean not in ("yes", "no"):
            errors.append(f"{where}: bg_clean must be yes or no on a background row")
        if kind != "background" and bg_clean:
            errors.append(f"{where}: bg_clean is only for background rows")
        if shelf and (kind != "background" or not all(part.isdigit() for part in shelf.split("|"))):
            errors.append(f"{where}: shelf_lines takes whole numbers separated by | and only on a background row")
    for folder in ("products", "backgrounds"):
        for path in sorted((own / folder).rglob("*")):
            rel = path.relative_to(own).as_posix()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS and rel.lower() not in listed:
                errors.append(f"{rel}: the photo is in the folder but not listed in intake.csv")
    return errors


def read_own(errors: list[str]) -> list[dict]:
    """Own photos, listed by hand in data/raw/own/intake.csv. They need an EXIF capture date."""
    own = _p("data", "raw", "own")
    if not (own / "intake.csv").exists():
        return []
    rows = _read_csv(own / "intake.csv")
    errors += validate_intake(rows, own)
    for r in rows:
        r.update(source="own", source_path="data/raw/own/" + r["filename"], license="own photo",
                 annotation_method="intake.csv by hand", openimages_id="", generator_sha256="", seed="",
                 mask_path="", box="")
    return [r for r in rows if r.get("kind") in KINDS]


def _render_seeds(stem: str) -> str:
    """The seeds the set's gen_all.py used for this file. Read off its code; the manifest does not list them."""
    box, lookalike = re.match(r"cereal_(\d+)_", stem), re.match(r"distractor_syn_(\d+)_", stem)
    if box:
        k = int(box.group(1))
        return f"texture {1000 + k}, damage {2000 + k}, composite {k}"
    if lookalike:
        j = int(lookalike.group(1))
        return f"texture {3000 + j}, composite {100 + j}"
    return ""


def read_rendered_set(errors: list[str]) -> list[dict]:
    """The rendered retail set: boxes and look-alikes drawn in Python, plus Open Images photos.

    No file here has camera EXIF, so the capture-date rule does not apply. Rendered rows carry the
    generator's sha256 and seeds, and every row carries its Open Images id and licence.
    """
    base = _p(RENDERED_SET)
    if not (base / "manifest.csv").exists():
        return []
    credits = {r["openimages_id"] for r in _read_csv(base / "ATTRIBUTION.csv")}
    generator = hashlib.sha256()
    for script in sorted((base / "generator").glob("*.py")):
        generator.update(script.name.encode() + b"\0" + script.read_bytes())

    rows, listed = [], set()
    for line_no, m in enumerate(_read_csv(base / "manifest.csv"), start=2):
        where = f"rendered set manifest line {line_no} ({m.get('file') or 'no file'})"
        if m.get("category") not in SET_CATEGORIES:
            errors.append(f"{where}: category {m.get('category')!r} is not one the pipeline knows")
            continue
        kind, damaged = SET_CATEGORIES[m["category"]]
        rendered = bool(m["mask_file"])  # only the boxes and look-alikes drawn by the generator ship a mask
        stem = Path(m["file"]).stem
        listed.add(m["file"].lower())
        if not (base / m["file"]).is_file():
            errors.append(f"{where}: file not found")
        if rendered and not (base / m["mask_file"]).is_file():
            errors.append(f"{where}: mask {m['mask_file']} not found")
        if kind == "cereal_box" and not m["pair_id"]:
            errors.append(f"{where}: a cereal box needs a pair_id, which is its group")
        if m["openimages_id"] not in credits:
            errors.append(f"{where}: no licence row in ATTRIBUTION.csv for Open Images id {m['openimages_id']!r}")
        boxed = kind == "distractor" and not rendered
        rows.append({
            "filename": m["file"], "source": "rendered" if rendered else "openimages_asset",
            "source_path": f"{RENDERED_SET}/{m['file']}", "kind": kind,
            # An intact box and its dented twin share a pair_id, so they are one group and stay in one split.
            "group_name": m["pair_id"] if kind == "cereal_box" else (m["product"] or stem),
            "name": m["product"].replace("_", " "), "damaged": damaged, "bg_clean": "", "shelf_lines": "",
            "lighting": "", "notes": m.get("damage_types", ""),
            "license": "rendered for this project on a CC BY 2.0 Open Images photo" if rendered else "CC BY 2.0",
            "annotation_method": ("mask from the set's renderer" if rendered
                                  else "rembg inside the Open Images box" if boxed else "none (background)"),
            "openimages_id": m["openimages_id"],
            "generator_sha256": generator.hexdigest() if rendered else "",
            "seed": _render_seeds(stem) if rendered else "",
            "mask_path": f"{RENDERED_SET}/{m['mask_file']}" if rendered else "",
            "box": ",".join(m[k] for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")) if boxed else "",
        })
    for path in sorted((base / "images").glob("*")):
        if path.suffix.lower() in IMAGE_EXTS and f"images/{path.name}".lower() not in listed:
            errors.append(f"{RENDERED_SET}/images/{path.name}: in the folder but not in the set's manifest.csv")
    return rows


def read_sources() -> list[dict]:
    """Every listed file from both sources, as rows with the same fields. cutouts.py reads this too."""
    errors: list[str] = []
    rows = read_own(errors) + read_rendered_set(errors)
    if errors:
        _fail("The source listings have problems", errors)
    if not rows:
        sys.exit(f"Nothing to work on. Expected data/raw/own/intake.csv, or {RENDERED_SET}/manifest.csv, or both.")
    return sorted(rows, key=lambda r: r["source_path"])


# ---------------------------------------------------------------- step 2: inventory

def _exif_capture(image: Image.Image) -> datetime | None:
    stamp = image.getexif().get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal)
    try:
        return datetime.strptime(str(stamp).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def inventory(rows: list[dict], phash_bits: int) -> list[str]:
    """Adds md5, asset_id, phash and capture_date to every row. Reads pixels for the pHash only."""
    errors = []
    for r in rows:
        path = _p(r["source_path"])
        r["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
        r["asset_id"] = "a_" + r["md5"][:10]  # same rule as cutouts.py
        try:
            with Image.open(path) as image:
                captured = _exif_capture(image)
                upright = ImageOps.exif_transpose(image)
                r["_hash"] = imagehash.phash(upright, hash_size=math.isqrt(phash_bits))
                r["_height"] = upright.height
        except OSError as err:
            errors.append(f"{r['source_path']}: cannot open ({err})")
            continue
        r["phash"] = str(r["_hash"])
        r["capture_date"] = ""
        if r["source"] != "own":
            continue  # rendered and Open Images files have no camera EXIF; their origin is recorded instead
        # A file's modified time says when it was copied, not when it was shot, so it is never a fallback.
        if captured is None:
            errors.append(f"{r['source_path']}: no EXIF DateTimeOriginal. Transfer originals by cable so EXIF survives.")
        elif captured.date() <= MIN_CAPTURE_DATE:
            errors.append(f"{r['source_path']}: capture date {captured.date()} is not after {MIN_CAPTURE_DATE}")
        else:
            r["capture_date"] = captured.isoformat()
    return errors


# ---------------------------------------------------------------- step 3: duplicates and groups

def mark_md5_duplicates(rows: list[dict], log: list[str]) -> None:
    first: dict[str, dict] = {}
    for r in rows:  # already in path order
        if r["md5"] in first:
            r["status"], r["reason"] = "excluded", f"md5 duplicate of {first[r['md5']]['asset_id']}"
            log.append(f"  excluded {r['source_path']}: same bytes as {first[r['md5']]['source_path']}")
        else:
            first[r["md5"]] = r


def assign_groups(rows: list[dict], max_hamming: int, log: list[str]) -> list[str]:
    """One group per (kind, group_name), then union groups that share a near-duplicate photo."""
    parent = {(r["kind"], r["group_name"]): (r["kind"], r["group_name"]) for r in rows}

    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key

    errors = []
    live = [r for r in rows if r["status"] == "ok"]
    for a, b in itertools.combinations(live, 2):
        distance = a["_hash"] - b["_hash"]
        ka, kb = (a["kind"], a["group_name"]), (b["kind"], b["group_name"])
        if distance > max_hamming or ka == kb:
            continue
        if a["kind"] != b["kind"]:
            errors.append(f"{a['source_path']} ({a['kind']}) and {b['source_path']} ({b['kind']}) are near "
                          f"duplicates (pHash distance {distance}) but have different kinds. Check the listing.")
            continue
        log.append(f"  warning: {a['source_path']} (group {a['group_name']}) and {b['source_path']} (group "
                   f"{b['group_name']}) are {distance} bits apart, so the two groups were merged. Review this.")
        root_a, root_b = find(ka), find(kb)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    members: dict[tuple, set] = {}
    for key in parent:
        members.setdefault(find(key), set()).add(key[1])
    for r in rows:
        root = find((r["kind"], r["group_name"]))
        r["group_id"] = r["kind"] + "__" + "+".join(sorted(members[root]))
    return errors


# ---------------------------------------------------------------- step 4: reviews and cutout status

def apply_background_review(rows: list[dict]) -> list[str]:
    """bg_clean and shelf lines for the set's backgrounds come from a file the author fills in after looking at them."""
    todo = [r for r in rows if r["kind"] == "background" and r["source"] != "own"]
    path = _p("data", "raw", "own", "background_review.csv")
    if not todo:
        return []
    if not path.exists():
        return ["data/raw/own/background_review.csv is missing. Run src/data.py --background-sheet, look at "
                "data/assets/backgrounds_sheet.png, then copy data/assets/background_review_template.csv there and "
                "fill it in by hand: file, bg_clean (yes or no), shelf_lines (optional, y pixels separated by |), note."]
    review = {r["file"].lower(): r for r in _read_csv(path)}
    errors = []
    for r in todo:
        row = review.get(Path(r["filename"]).name.lower())
        shelf = (row or {}).get("shelf_lines", "")
        if row is None:
            errors.append(f"background_review.csv has no row for {Path(r['filename']).name}")
        elif row.get("bg_clean") not in ("yes", "no"):
            errors.append(f"background_review.csv: bg_clean for {row['file']} must be yes or no")
        elif shelf and not all(part.isdigit() for part in shelf.split("|")):
            errors.append(f"background_review.csv: shelf_lines for {row['file']} takes whole numbers separated by |")
        else:
            r["bg_clean"], r["shelf_lines"] = row["bg_clean"], shelf
    return errors


def apply_cutout_status(rows: list[dict], log: list[str]) -> list[str]:
    report_path, review_path = _p("data", "assets", "cutouts_report.csv"), _p("data", "raw", "own", "cutout_review.csv")
    if not report_path.exists():
        return ["data/assets/cutouts_report.csv is missing. Run src/cutouts.py first."]
    if not review_path.exists():
        return ["data/raw/own/cutout_review.csv is missing. Look at data/assets/cutouts_sheet.png and fill it in "
                "by hand: asset_id, keep (yes or no), note."]
    report = {r["asset_id"]: r for r in _read_csv(report_path)}
    review = {r["asset_id"]: r for r in _read_csv(review_path)}
    errors = [f"cutout_review.csv: keep for {asset_id} must be yes or no"
              for asset_id, r in review.items() if r.get("keep") not in ("yes", "no")]

    for r in rows:
        if r["status"] != "ok":
            continue
        if r["kind"] == "background":
            if r["bg_clean"] == "no":
                r["status"], r["reason"] = "excluded", "bg_clean no"
            elif r["shelf_lines"] and any(int(y) >= r["_height"] for y in r["shelf_lines"].split("|")):
                errors.append(f"{r['source_path']}: a shelf line is outside the photo (height {r['_height']} px)")
        elif r["asset_id"] not in report:
            errors.append(f"{r['source_path']} ({r['asset_id']}) has no row in cutouts_report.csv. Run src/cutouts.py.")
        elif report[r["asset_id"]]["status"] != "ok":
            r["status"], r["reason"] = "excluded", "cutout rejected: " + report[r["asset_id"]]["reason"]
        elif r["asset_id"] not in review:
            errors.append(f"{r['source_path']} ({r['asset_id']}) has a cutout but no row in cutout_review.csv")
        elif review[r["asset_id"]]["keep"] == "no":
            r["status"], r["reason"] = "excluded", "cutout review: " + (review[r["asset_id"]].get("note") or "keep no")
        if r["status"] == "excluded":
            log.append(f"  excluded {r['source_path']}: {r['reason']}")

    usable = {r["group_id"] for r in rows if r["status"] == "ok"}
    for group_id in sorted({r["group_id"] for r in rows} - usable):
        log.append(f"  group {group_id} has no usable asset left and is excluded as a whole")
    return errors


def write_background_sheet() -> None:
    """Every listed background on one sheet with a pixel ruler, for the bg_clean and shelf-line review.

    The one place data.py opens background pixels for viewing. Nothing is computed from them.
    """
    rows = [r for r in read_sources() if r["kind"] == "background"]
    columns, cell_w, cell_h, font = 4, 420, 446, ImageFont.load_default(size=12)
    sheet = Image.new("RGB", (columns * cell_w, -(-len(rows) // columns) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        photo = ImageOps.exif_transpose(Image.open(_p(r["source_path"]))).convert("RGB")
        full_w, full_h = photo.size
        photo.thumbnail((380, 400), Image.Resampling.LANCZOS)
        left, top = (i % columns) * cell_w + 34, (i // columns) * cell_h + 4
        sheet.paste(photo, (left, top))
        for y in range(0, full_h, 100):  # ruler in the photo's own pixels, which is what shelf_lines uses
            at = top + round(y * photo.height / full_h)
            draw.line([left - (8 if y % 200 == 0 else 4), at, left - 1, at], fill="black")
            if y % 200 == 0:
                draw.text((left - 32, at - 6), str(y), fill="black", font=font)
        draw.text((left, top + 404), f"{Path(r['filename']).name}  {full_w}x{full_h}", fill="black", font=font)
        draw.text((left, top + 420), "bg_clean in intake.csv" if r["source"] == "own" else "needs a row in background_review.csv",
                  fill=(110, 110, 110), font=font)
    out = _p("data", "assets", "backgrounds_sheet.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")
    template = [{"file": Path(r["filename"]).name, "bg_clean": "", "shelf_lines": "", "note": ""}
                for r in rows if r["source"] != "own"]
    _p("data", "assets", "background_review_template.csv").write_bytes(
        _csv_bytes(["file", "bg_clean", "shelf_lines", "note"], template))
    print(f"[data] {len(rows)} backgrounds on data/assets/backgrounds_sheet.png. The ruler is in each photo's own pixels.")
    print("[data] Copy data/assets/background_review_template.csv to data/raw/own/background_review.csv and fill in "
          "bg_clean (yes or no). shelf_lines is optional: y pixels of surfaces a box can stand on, separated by |.")


# ---------------------------------------------------------------- step 5: split

def allocate(n_groups: int, splits_cfg: dict, one_each: bool) -> dict[str, int]:
    """Groups per split: one each first (when asked), then the rest by largest remainder."""
    shares = {s: Fraction(str(splits_cfg[s])) for s in SPLITS}  # exact, so real ties are seen as ties
    if sum(shares.values()) != 1:
        sys.exit("config.yaml: splits.val, splits.test and splits.showcase must add up to 1.")
    base = 1 if one_each else 0
    rest = n_groups - base * len(SPLITS)
    quotas = {s: shares[s] * rest for s in SPLITS}
    counts = {s: base + math.floor(quotas[s]) for s in SPLITS}
    tie_order = splits_cfg["allocation_tie_order"]
    by_remainder = sorted(SPLITS, key=lambda s: (-(quotas[s] - math.floor(quotas[s])), tie_order.index(s)))
    for s in by_remainder[:n_groups - sum(counts.values())]:
        counts[s] += 1
    return counts


def split_groups(rows: list[dict], cfg: dict) -> dict[str, str]:
    """group_id -> split, for groups with at least one usable asset."""
    strata: list[set] = [set() for _ in STRATA]
    dented = {r["group_id"] for r in rows if r["status"] == "ok" and r["damaged"] == "yes"}
    for r in rows:
        if r["status"] != "ok":
            continue
        index = {"background": 0, "distractor": 3}.get(r["kind"], 1 if r["group_id"] in dented else 2)
        strata[index].add(r["group_id"])

    short = [f"{STRATA[i]} groups: found {len(strata[i])}, need at least 3. {3 - len(strata[i])} more needed."
             for i in (0, 1, 3) if len(strata[i]) < 3]
    if short:
        _fail("Not enough groups to give every split one of each kind", short)

    assignment = {}
    for index, group_ids in enumerate(strata):
        ordered = sorted(group_ids)  # sorted first, so the order of rows in a listing cannot matter
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), index]))
        shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
        one_each = index != 2 or len(ordered) >= 3
        counts = allocate(len(ordered), cfg["splits"], one_each)
        labels = list(SPLITS) if one_each else []
        for s in SPLITS:
            labels += [s] * (counts[s] - (1 if one_each else 0))
        assignment.update(zip(shuffled, labels))
    return assignment


# ---------------------------------------------------------------- build, write, verify

def build() -> dict:
    """Everything data.py produces, in memory. Writes nothing."""
    with open(_p("config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = read_sources()
    for r in rows:
        r["status"], r["reason"] = "ok", ""

    by_source = ", ".join(f"{sum(r['source'] == s for r in rows)} {s}" for s in sorted({r["source"] for r in rows}))
    log = ["data.py log. No timestamps, so a rebuild is byte-identical.",
           f"step 1 sources: {len(rows)} rows ({by_source}); "
           + ", ".join(f"{sum(r['kind'] == k for r in rows)} {k}" for k in KINDS)]
    errors = inventory(rows, int(cfg["dedup"]["phash_bits"]))
    if errors:
        _fail("Some files fail the inventory", errors)
    n_own = sum(r["source"] == "own" for r in rows)
    own_note = f"{n_own} own photos, every one with an EXIF capture date after {MIN_CAPTURE_DATE}" if n_own else "no own photos"
    log.append(f"step 2 inventory: {len(rows)} files; {own_note}; {len(rows) - n_own} from the rendered set, "
               "which has no camera EXIF")

    log.append("step 3 duplicates and groups:")
    mark_md5_duplicates(rows, log)
    n_duplicates = sum(r["status"] == "excluded" for r in rows)
    n_before = len(log)
    errors = assign_groups(rows, int(cfg["dedup"]["phash_max_hamming"]), log)
    if errors:
        _fail("Near-duplicate photos with different kinds", errors)
    n_merges = len(log) - n_before
    log.append(f"  {n_duplicates} exact duplicates excluded, {n_merges} near-duplicate pairs across group names, "
               f"{len({r['group_id'] for r in rows})} groups")

    log.append("step 4 reviews and cutout status:")
    errors = apply_background_review(rows) + apply_cutout_status(rows, log)
    if errors:
        _fail("Reviews or cutouts are not ready", errors)
    log.append(f"  {sum(r['status'] == 'ok' for r in rows)} usable assets, "
               f"{sum(r['status'] == 'excluded' for r in rows)} excluded in total")

    assignment = split_groups(rows, cfg)
    usable = [r for r in rows if r["status"] == "ok"]
    summary, counts = [], {}
    for s in SPLITS:
        mine = [r for r in usable if assignment[r["group_id"]] == s]
        n = {kind: len({r["group_id"] for r in mine if r["kind"] == kind}) for kind in KINDS}
        n["dented"] = len({r["group_id"] for r in mine if r["damaged"] == "yes"})
        if not (n["background"] and n["dented"] and n["distractor"]):
            sys.exit(f"Split {s} is missing a background, a damaged-box group or a distractor group.")
        counts[s] = n
        summary.append(f"{s:<9} backgrounds {n['background']} | cereal-box groups {n['cereal_box']} "
                       f"({n['dented']} with a dented box) | distractor groups {n['distractor']} | assets {len(mine)}")
    by_group: dict[str, set] = {}
    for r in usable:
        by_group.setdefault(r["group_id"], set()).add(assignment[r["group_id"]])
    if any(len(s) > 1 for s in by_group.values()) or len({r["asset_id"] for r in usable}) != len(usable):
        sys.exit("Internal error: a group or an asset landed in more than one split.")
    log += ["step 5 split (groups, not images):"] + ["  " + line for line in summary]

    assets = [{**r, "damaged": {"yes": "true", "no": "false"}.get(r["damaged"], ""),
               "bg_clean": {"yes": "true", "no": "false"}.get(r["bg_clean"], "")} for r in rows]
    provenance = [{**a, "path": a["source_path"], "angle": "", "occlusion": "", "parent_asset_ids": ""} for a in assets]
    pools = sorted(({"asset_id": r["asset_id"], "group_id": r["group_id"], "kind": r["kind"],
                     "split": assignment[r["group_id"]]} for r in usable), key=lambda p: p["asset_id"])
    return {
        "files": {
            _p("data", "assets", "assets.csv"): _csv_bytes(ASSET_FIELDS, assets),
            _p("data", "provenance.csv"): _csv_bytes(PROVENANCE_FIELDS, provenance),
            _p("splits", "pools.csv"): _csv_bytes(POOL_FIELDS, pools),
        },
        "log": "\n".join(log) + "\n",
        "counts": counts,
        "cfg": cfg,
    }


def _synthetic_tail(disk: bytes, fresh: bytes) -> int | None:
    """If disk is the fresh rows followed only by composite rows, how many. Otherwise None."""
    if not disk.startswith(fresh):
        return None
    tail = list(csv.DictReader(io.StringIO(disk[len(fresh):].decode("utf-8")), fieldnames=PROVENANCE_FIELDS))
    return len(tail) if all(row["source"] == "synthetic" for row in tail) else None


def verify(built: dict) -> bool:
    """Compare a fresh in-memory rebuild with disk. Files are hashed as bytes; no row is parsed or printed."""
    all_match = True
    for path, fresh in built["files"].items():
        name = path.relative_to(ROOT).as_posix()
        if not path.exists():
            print(f"[verify] {name:<26} MISSING on disk")
            all_match = False
            continue
        disk, note = path.read_bytes(), ""
        if path.name == "provenance.csv" and disk != fresh:
            n_synthetic = _synthetic_tail(disk, fresh)  # make_synth.py appends its rows after the asset rows
            if n_synthetic is not None:
                disk, note = fresh, f" (asset rows; {n_synthetic} composite rows follow)"
        match = disk == fresh
        all_match &= match
        print(f"[verify] {name:<26} {'match' if match else 'MISMATCH'}  sha256 {_sha256(fresh)[:16]}{note}")
    return all_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="rebuild everything, including the split")
    mode.add_argument("--verify", action="store_true", help="rebuild in memory and compare with disk")
    mode.add_argument("--openimages", action="store_true", help="optional real-photo slice (not built yet)")
    mode.add_argument("--background-sheet", action="store_true", help="write the sheet for the background review")
    args = parser.parse_args()
    if args.background_sheet:
        write_background_sheet()
        return
    if args.openimages:
        sys.exit("The Open Images slice is not built yet. It is optional (build stage 9), "
                 "open_images.enabled is false in config.yaml, and it needs the fiftyone package.")

    built = build()
    pools_path = _p("splits", "pools.csv")
    if args.verify or (pools_path.exists() and not args.force):
        if not args.verify:
            print("[data] splits/pools.csv already exists. The split happens once, so nothing is rewritten.")
            print("[data] Comparing a fresh in-memory rebuild with what is on disk:")
        if verify(built):
            sys.exit(0)
        sys.exit("[data] Inputs changed since the split was made. Rebuilding needs --force, and that changes "
                 "the pools, so decide that on purpose.")

    provenance_path = _p("data", "provenance.csv")
    if provenance_path.exists() and b",synthetic," in provenance_path.read_bytes():
        print("[data] provenance.csv had composite rows from make_synth.py. They are dropped by this rebuild, "
              "so rerun make_synth.py --split all --force afterwards.")
    for path, content in built["files"].items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"[data] wrote {path.relative_to(ROOT).as_posix():<26} sha256 {_sha256(content)[:16]}")
    _p("data", "data_log.txt").write_text(built["log"], encoding="utf-8", newline="\n")
    print(built["log"], end="")

    scene = built["cfg"]["scene"]
    for split, n in built["counts"].items():
        for kind, key in (("cereal_box", "cereal_boxes_per_image"), ("distractor", "distractors_per_image")):
            if n[kind] < max(scene[key]):
                print(f"[data] note: {split} has {n[kind]} {kind} groups, so make_synth.py will cap "
                      f"{key} draws above {n[kind]}.")
    print("test pools locked")


# ---------------------------------------------------------------- the gate every other script uses

def load_split(name: str, columns: list[str] | None = None, allow_test: bool = False) -> list[dict]:
    """Rows for one split: its composites (once make_synth.py has run), then its pool assets.

    The test split raises PermissionError unless allow_test is set. Only eval.py (images and
    annotations) and check_splits.py (six metadata columns) may set it.
    """
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}. Use val, test or showcase.")
    if name == "test" and not allow_test:
        raise PermissionError("The test split is closed. Only eval.py may open it. See CLAUDE.md, Test-set rules.")
    unknown = set(columns or []) - set(LOAD_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown columns {sorted(unknown)}. Available: {LOAD_COLUMNS}")
    pools_path = _p("splits", "pools.csv")
    if not pools_path.exists():
        raise FileNotFoundError("splits/pools.csv does not exist yet. Run src/data.py first.")

    rows = []
    manifest = _p("data", "synth", name, "manifest.csv")
    if manifest.exists():
        for m in sorted(_read_csv(manifest), key=lambda m: m["image_id"]):
            parents = [m.get("background_id", "")] + m.get("asset_ids", "").split("|")
            rows.append({"row_type": "synthetic", "split": name, "path": m["path"], "asset_id": m["image_id"],
                         "kind": "synthetic", "group_id": "", "md5": m["md5"], "phash": m["phash"],
                         "parent_asset_ids": "|".join(p for p in parents if p),
                         "condition": m.get("condition", ""), "name": "", "damaged": ""})
    assets = {a["asset_id"]: a for a in _read_csv(_p("data", "assets", "assets.csv")) if a["status"] == "ok"}
    for pool in _read_csv(pools_path):
        if pool["split"] != name:
            continue
        asset = assets[pool["asset_id"]]
        rows.append({"row_type": "pool", "split": name, "path": asset["source_path"], "asset_id": pool["asset_id"],
                     "kind": pool["kind"], "group_id": pool["group_id"], "md5": asset["md5"], "phash": asset["phash"],
                     "parent_asset_ids": "", "condition": "", "name": asset["name"], "damaged": asset["damaged"]})
    if columns is not None:
        rows = [{c: row[c] for c in columns} for row in rows]
    return rows


def val_list_sha256() -> str:
    """Hash of the val list as load_split returns it. frozen.json stores it so a changed val set is noticed."""
    return _sha256(_csv_bytes(LOAD_COLUMNS, load_split("val")))


if __name__ == "__main__":
    main()
