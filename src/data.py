"""Inventory raw photos, group near duplicates, split asset groups once, and gate split access.

Spec: docs/data_spec.md. Order: cutouts.py, then data.py, then make_synth.py.

    python src/data.py            # build; if the split already exists, verify it and write nothing
    python src/data.py --force    # rebuild everything, including the split
    python src/data.py --verify   # rebuild in memory, compare with disk, write nothing
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import math
import sys
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import ExifTags, Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("val", "test", "showcase")
KINDS = ("cereal_box", "distractor", "background")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp"}
MIN_CAPTURE_DATE = date(2024, 1, 1)
STRATA = ("background", "damaged-box", "intact-only box", "distractor")

ASSET_FIELDS = ["asset_id", "group_id", "kind", "name", "damaged", "source_path", "capture_date", "bg_clean",
                "shelf_lines", "md5", "phash", "status", "reason"]
PROVENANCE_FIELDS = ["path", "source", "capture_date", "license", "annotation_method", "lighting", "angle",
                     "occlusion", "md5", "phash", "group_id", "parent_asset_ids"]  # last one: synthetic rows only
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


# ---------------------------------------------------------------- step 1: intake

def validate_intake(rows: list[dict], own: Path) -> list[str]:
    errors, listed = [], {}
    for line_no, r in enumerate(rows, start=2):
        r["filename"] = r.get("filename", "").replace("\\", "/")
        where = f"line {line_no} ({r['filename'] or 'no filename'})"
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


# ---------------------------------------------------------------- step 2: inventory

def _exif_capture(image: Image.Image) -> datetime | None:
    stamp = image.getexif().get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal)
    try:
        return datetime.strptime(str(stamp).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def inventory(rows: list[dict], own: Path, phash_bits: int) -> list[str]:
    """Adds md5, asset_id, phash and capture_date to every row. Reads pixels for the pHash only."""
    errors = []
    for r in rows:
        path = own / r["filename"]
        r["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
        r["asset_id"] = "a_" + r["md5"][:10]  # same rule as cutouts.py
        try:
            with Image.open(path) as image:
                captured = _exif_capture(image)
                upright = ImageOps.exif_transpose(image)
                r["_hash"] = imagehash.phash(upright, hash_size=math.isqrt(phash_bits))
                height = upright.height
        except OSError as err:
            errors.append(f"{r['filename']}: cannot open ({err})")
            continue
        r["phash"] = str(r["_hash"])
        # A file's modified time says when it was copied, not when it was shot, so it is never a fallback.
        if captured is None:
            errors.append(f"{r['filename']}: no EXIF DateTimeOriginal. Transfer originals by cable so EXIF survives.")
        elif captured.date() <= MIN_CAPTURE_DATE:
            errors.append(f"{r['filename']}: capture date {captured.date()} is not after {MIN_CAPTURE_DATE}")
        else:
            r["capture_date"] = captured.isoformat()
        if r.get("shelf_lines") and any(int(y) >= height for y in r["shelf_lines"].split("|")):
            errors.append(f"{r['filename']}: a shelf line is outside the upright photo (height {height} px)")
    return errors


# ---------------------------------------------------------------- step 3: duplicates and groups

def mark_md5_duplicates(rows: list[dict], log: list[str]) -> None:
    first: dict[str, dict] = {}
    for r in rows:  # already in filename order
        if r["md5"] in first:
            r["status"], r["reason"] = "excluded", f"md5 duplicate of {first[r['md5']]['asset_id']}"
            log.append(f"  excluded {r['filename']}: same bytes as {first[r['md5']]['filename']}")
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
            errors.append(f"{a['filename']} ({a['kind']}) and {b['filename']} ({b['kind']}) are near duplicates "
                          f"(pHash distance {distance}) but have different kinds. Check intake.csv.")
            continue
        log.append(f"  warning: {a['filename']} (group {a['group_name']}) and {b['filename']} (group "
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


# ---------------------------------------------------------------- step 4: cutout status

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
        elif r["asset_id"] not in report:
            errors.append(f"{r['filename']} ({r['asset_id']}) has no row in cutouts_report.csv. Run src/cutouts.py.")
        elif report[r["asset_id"]]["status"] != "ok":
            r["status"], r["reason"] = "excluded", "cutout rejected: " + report[r["asset_id"]]["reason"]
        elif r["asset_id"] not in review:
            errors.append(f"{r['filename']} ({r['asset_id']}) has a cutout but no row in cutout_review.csv")
        elif review[r["asset_id"]]["keep"] == "no":
            r["status"], r["reason"] = "excluded", "cutout review: " + (review[r["asset_id"]].get("note") or "keep no")
        if r["status"] == "excluded":
            log.append(f"  excluded {r['filename']}: {r['reason']}")

    usable = {r["group_id"] for r in rows if r["status"] == "ok"}
    for group_id in sorted({r["group_id"] for r in rows} - usable):
        log.append(f"  group {group_id} has no usable asset left and is excluded as a whole")
    return errors


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

    short = [f"{STRATA[i]} groups: found {len(strata[i])}, need at least 3. Shoot {3 - len(strata[i])} more."
             for i in (0, 1, 3) if len(strata[i]) < 3]
    if short:
        _fail("Not enough groups to give every split one of each kind", short)

    assignment = {}
    for index, group_ids in enumerate(strata):
        ordered = sorted(group_ids)  # sorted first, so the order of rows in intake.csv cannot matter
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
    own = _p("data", "raw", "own")
    if not (own / "intake.csv").exists():
        sys.exit("data/raw/own/intake.csv is missing. docs/intake_template.csv shows the columns.")
    rows = _read_csv(own / "intake.csv")
    errors = validate_intake(rows, own)
    if errors:
        _fail("intake.csv has problems", errors)
    rows.sort(key=lambda r: r["filename"])
    for r in rows:
        r["status"], r["reason"] = "ok", ""

    log = ["data.py log. No timestamps, so a rebuild is byte-identical.",
           f"step 1 intake: {len(rows)} rows, " + ", ".join(f"{sum(r['kind'] == k for r in rows)} {k}" for k in KINDS)]
    errors = inventory(rows, own, int(cfg["dedup"]["phash_bits"]))
    if errors:
        _fail("Some photos fail the inventory", errors)
    log.append(f"step 2 inventory: {len(rows)} files, every one with an EXIF capture date after {MIN_CAPTURE_DATE}")

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

    log.append("step 4 cutout status:")
    errors = apply_cutout_status(rows, log)
    if errors:
        _fail("Cutouts are not ready", errors)
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
               "bg_clean": {"yes": "true", "no": "false"}.get(r["bg_clean"], ""),
               "source_path": "data/raw/own/" + r["filename"]} for r in rows]
    provenance = [{"path": a["source_path"], "source": "own", "capture_date": a["capture_date"],
                   "license": "own photo", "annotation_method": "intake.csv by hand", "lighting": a.get("lighting", ""),
                   "angle": "", "occlusion": "", "md5": a["md5"], "phash": a["phash"], "group_id": a["group_id"],
                   "parent_asset_ids": ""}
                  for a in assets]
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
    """If disk is the fresh own rows followed only by synthetic rows, how many. Otherwise None."""
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
            n_synthetic = _synthetic_tail(disk, fresh)  # make_synth.py appends its rows after the own rows
            if n_synthetic is not None:
                disk, note = fresh, f" (own rows; {n_synthetic} synthetic rows follow)"
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
    args = parser.parse_args()
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
        print("[data] provenance.csv had synthetic rows from make_synth.py. They are dropped by this rebuild, "
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
    """Rows for one split: its synthetic images (once make_synth.py has run), then its pool assets.

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
