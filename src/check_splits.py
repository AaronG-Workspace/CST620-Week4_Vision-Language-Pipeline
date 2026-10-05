"""Audit the evaluation for leakage before anything is scored.

Spec: docs/check_splits_spec.md. The real enforcement is load_split's PermissionError; the
static scan here is a tripwire. This file opens no image, no annotations.json and nothing
under the scored test results. It reads the test split through load_split with six
metadata columns only.

    python src/check_splits.py            # all 19 checks
    python src/check_splits.py --fast     # skip the two subprocess rebuilds (checks 6 and 11)
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import yaml

import data
from data import SPLITS, load_split
from make_synth import prompt_words

ROOT = Path(__file__).resolve().parents[1]   # where the data, config and results live
CODE = ROOT                                   # where src/, tests/ and notebooks/ live

TEST_COLUMNS = ["path", "asset_id", "md5", "phash", "group_id", "parent_asset_ids"]
PROVENANCE_COLUMNS = ("path", "source", "capture_date", "license", "md5", "phash", "group_id")
MIN_CAPTURE_DATE = date(2024, 1, 1)
FORBIDDEN_IMPORTS = {"transformers", "torch", "segment_anything", "groundingdino", "open_clip",
                     "models", "pipeline"}  # the last two are this project's files that load the models
# (what is searched for, pattern, files allowed to contain it)
SCAN_RULES = [
    ("allow_test=True", re.compile(r"allow_test\s*=\s*True"), {"eval.py", "check_splits.py"}),
    ("synth/test or test_runs.log", re.compile(r"synth[/\\]+test|test_runs\.log"),
     {"make_synth.py", "data.py", "eval.py", "check_splits.py"}),
    ("results/test", re.compile(r"results[/\\]+test"), {"eval.py", "check_splits.py", "report.ipynb"}),
]
TITLES = {
    1: "own assets have a capture date after 2024-01-01",
    2: "provenance rows have a source and a license, synthetic parents exist",
    3: "every usable asset is in exactly one split",
    4: "no group in two splits",
    5: "every split has a background, a damaged-box group and a distractor group",
    6: "data.py --verify",
    7: "image counts match config.yaml",
    8: "every composite is built from its own split's pool",
    9: "no md5 in two splits",
    10: "no test image near a val or showcase image",
    11: "make_synth.py --verify",
    12: "Open Images rows sit in val or test only, none near an own asset",
    13: "the test split refuses without permission",
    14: "static scan of test access and model imports",
    15: "nothing test-related under cache/",
    16: "label words avoid prompt words",
    17: "frozen.json hashes match the current files",
    18: "prompts were written before the first val run, later changes carry a reason",
    19: "test scored at most once",
}


class Skip(Exception):
    pass


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _some(items, limit: int = 6) -> str:
    items = sorted(map(str, items))
    return ", ".join(items[:limit]) + (f" and {len(items) - limit} more" if len(items) > limit else "")


class Audit:
    def __init__(self, fast: bool):
        self.fast = fast
        with open(_p("config.yaml"), encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.limit = int(self.cfg["dedup"]["phash_max_hamming"])
        self.assets = data._read_csv(_p("data", "assets", "assets.csv"))
        self.pools = data._read_csv(_p("splits", "pools.csv"))
        with open(_p("data", "provenance.csv"), encoding="utf-8", newline="") as f:
            self.provenance = [{k: (row.get(k) or "").strip() for k in PROVENANCE_COLUMNS} for row in csv.DictReader(f)]
        # Val and showcase in full; test through the gate with six metadata columns and nothing else.
        self.rows = {"val": load_split("val"), "showcase": load_split("showcase"),
                     "test": load_split("test", columns=TEST_COLUMNS, allow_test=True)}
        self.composites = {s: [r for r in rows if r["parent_asset_ids"]] for s, rows in self.rows.items()}
        self.split_of = {p["asset_id"]: p["split"] for p in self.pools}

    def _need_composites(self):
        if not any(self.composites.values()):
            raise Skip("make_synth.py has not run yet")

    # ------------------------------------------------------------ provenance

    def check_1(self):
        own = [r for r in self.provenance if r["source"] == "own"]
        bad = []
        for r in own:
            try:
                ok = datetime.fromisoformat(r["capture_date"]).date() > MIN_CAPTURE_DATE
            except ValueError:
                ok = False
            if not ok:
                bad.append(r["path"])
        return not bad, f"{len(own)} own assets" + (f"; without a valid date: {_some(bad)}" if bad else "")

    def check_2(self):
        blank = [r["path"] for r in self.provenance if not (r["source"] and r["license"])]
        known = {a["asset_id"] for a in self.assets}
        orphans = {p for rows in self.composites.values() for r in rows for p in r["parent_asset_ids"].split("|")} - known
        detail = f"{len(self.provenance)} rows"
        if blank:
            detail += f"; no source or license: {_some(blank)}"
        if orphans:
            detail += f"; parents missing from assets.csv: {_some(orphans)}"
        return not (blank or orphans), detail

    # ------------------------------------------------------------ pools

    def check_3(self):
        usable = {a["asset_id"] for a in self.assets if a["status"] == "ok"}
        excluded_only = {a["asset_id"] for a in self.assets} - usable  # an md5 duplicate shares its twin's id
        pooled = [p["asset_id"] for p in self.pools]
        twice = {a for a in pooled if pooled.count(a) > 1}
        problems = []
        if usable - set(pooled):
            problems.append(f"usable but in no split: {_some(usable - set(pooled))}")
        if twice:
            problems.append(f"in more than one row: {_some(twice)}")
        if set(pooled) & excluded_only:
            problems.append(f"excluded but pooled: {_some(set(pooled) & excluded_only)}")
        if set(pooled) - usable - excluded_only:
            problems.append(f"pooled but not in assets.csv: {_some(set(pooled) - usable - excluded_only)}")
        return not problems, f"{len(usable)} usable assets" + "".join("; " + p for p in problems)

    def check_4(self):
        splits_of_group: dict[str, set] = {}
        for p in self.pools:
            splits_of_group.setdefault(p["group_id"], set()).add(p["split"])
        shared = [g for g, s in splits_of_group.items() if len(s) > 1]
        return not shared, f"{len(splits_of_group)} groups" + (f"; shared: {_some(shared)}" if shared else "")

    def check_5(self):
        dented = {a["group_id"] for a in self.assets if a["status"] == "ok" and a["damaged"] == "true"}
        missing = []
        for s in SPLITS:
            mine = [p for p in self.pools if p["split"] == s]
            have = {"background": any(p["kind"] == "background" for p in mine),
                    "damaged-box group": any(p["group_id"] in dented for p in mine),
                    "distractor group": any(p["kind"] == "distractor" for p in mine)}
            missing += [f"{s} has no {what}" for what, ok in have.items() if not ok]
        return not missing, "; ".join(missing)

    def _subprocess(self, *args: str):
        if self.fast:
            raise Skip("--fast")
        done = subprocess.run([sys.executable, *args], cwd=CODE, capture_output=True, text=True)
        tail = (done.stdout + done.stderr).strip().splitlines()[-1:] or [""]
        return done.returncode == 0, f"exit code {done.returncode}" + ("" if done.returncode == 0 else f": {tail[0][:140]}")

    def check_6(self):
        return self._subprocess("src/data.py", "--verify")

    # ------------------------------------------------------------ synthetic splits

    def check_7(self):
        self._need_composites()
        want = self.cfg["images_per_split"]
        got = {s: len(self.composites[s]) for s in SPLITS}
        wrong = [f"{s} has {got[s]}, config says {want[s]}" for s in SPLITS if got[s] != int(want[s])]
        return not wrong, "; ".join(wrong) or " / ".join(str(got[s]) for s in SPLITS)

    def check_8(self):
        self._need_composites()
        foreign = [f"{r['asset_id']} uses {p} ({self.split_of.get(p, 'in no pool')})"
                   for s in SPLITS for r in self.composites[s] for p in r["parent_asset_ids"].split("|")
                   if self.split_of.get(p) != s]
        return not foreign, _some(foreign, 4)

    def check_9(self):
        seen: dict[str, set] = {}
        for s, rows in self.rows.items():
            for r in rows:
                seen.setdefault(r["md5"], set()).add(s)
        shared = [f"{md5[:10]} in {'+'.join(sorted(s))}" for md5, s in seen.items() if len(s) > 1]
        return not shared, f"{len(seen)} files" + (f"; shared: {_some(shared, 4)}" if shared else "")

    def check_10(self):
        self._need_composites()
        others = [(s, r) for s in ("val", "showcase") for r in self.composites[s]]
        close = [f"{t['asset_id']} and {o['asset_id']} ({_hamming(t['phash'], o['phash'])} bits)"
                 for t in self.composites["test"] for _, o in others if _hamming(t["phash"], o["phash"]) <= self.limit]
        inside = sum(_hamming(a["phash"], b["phash"]) <= self.limit for a, b in combinations(self.composites["test"], 2))
        detail = f"near-duplicate pairs inside test: {inside} (information only)"
        return not close, (f"{_some(close, 4)}; " if close else "") + detail

    def check_11(self):
        self._need_composites()
        return self._subprocess("src/make_synth.py", "--split", "all", "--verify")

    def check_12(self):
        slice_rows = [r for r in self.provenance if r["source"] == "openimages"]
        if not slice_rows:
            return True, "no Open Images slice"
        own = [r for r in self.provenance if r["source"] == "own"]
        showcase_paths = {r["path"] for r in self.rows["showcase"]}
        misplaced = [r["path"] for r in slice_rows if r["path"] in showcase_paths]
        close = [r["path"] for r in slice_rows if any(_hamming(r["phash"], o["phash"]) <= self.limit for o in own)]
        return not (misplaced or close), (f"in showcase: {_some(misplaced)}; " if misplaced else "") + (
            f"near an own asset: {_some(close)}" if close else f"{len(slice_rows)} rows")

    # ------------------------------------------------------------ test access

    def check_13(self):
        try:
            load_split("test")
        except PermissionError:
            return True, "PermissionError raised"
        return False, "load_split('test') returned rows without permission"

    def check_14(self):
        hits = []
        sources = [(p.name, p.read_text(encoding="utf-8")) for folder in ("src", "tests")
                   for p in sorted((CODE / folder).glob("*.py"))]
        for nb in sorted((CODE / "notebooks").glob("*.ipynb")):
            cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
            sources.append((nb.name, "\n".join("".join(c["source"]) for c in cells if c.get("cell_type") == "code")))
        for name, text in sources:
            for label, pattern, allowed in SCAN_RULES:
                if name not in allowed:
                    hits += [f"{name}:{n} has {label}" for n, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
            if name in ("make_synth.py", "cutouts.py"):
                for node in ast.walk(ast.parse(text)):
                    names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                             else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                    hits += [f"{name}:{node.lineno} imports {m}" for m in names if m.split(".")[0] in FORBIDDEN_IMPORTS]
        return not hits, f"{len(sources)} files scanned" + "".join("; " + h for h in hits)

    def check_15(self):
        cache = _p("cache")
        if not cache.exists():
            return True, "no cache yet"
        test_ids = {r["asset_id"] for r in self.rows["test"]}
        bad = [p.relative_to(ROOT).as_posix() for p in cache.rglob("*")
               if "test" in p.name.lower() or any(i in p.name for i in test_ids)]
        return not bad, _some(bad, 4)

    def check_16(self):
        with open(_p("prompts", "prompts.yaml"), encoding="utf-8") as f:
            banned = prompt_words(yaml.safe_load(f))
        words = _p("data", "raw", "own", "label_words.txt").read_text(encoding="utf-8").lower().split()
        clashes = sorted({f"{w} (contains {b})" for w in words for b in banned if b in w})
        return not clashes, _some(clashes) if clashes else f"{len(words)} words against {len(banned)} prompt words"

    # ------------------------------------------------------------ selection and scoring logs

    def _frozen(self) -> dict:
        if not _p("frozen.json").exists():
            raise Skip("no frozen.json yet")
        return json.loads(_p("frozen.json").read_text(encoding="utf-8"))

    def check_17(self):
        frozen = self._frozen()
        now = {"val_list_sha256": data.val_list_sha256(), "prompts_sha256": _sha256(_p("prompts", "prompts.yaml")),
               "config_sha256": _sha256(_p("config.yaml"))}
        changed = [key.replace("_sha256", "") for key, value in now.items() if frozen.get(key) != value]
        return not changed, f"changed since freezing: {', '.join(changed)}. Rerun tune_val.py before eval.py." if changed else ""

    def check_18(self):
        log = _p("results", "val", "runs.log")
        if not log.exists():
            raise Skip("no val run yet")
        first_run = datetime.fromisoformat(re.split(r"[\s,]+", log.read_text(encoding="utf-8").strip())[0])
        problems = []
        for name, path in (("prompts.yaml", _p("prompts", "prompts.yaml")), ("config.yaml", _p("config.yaml"))):
            with open(path, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            written = doc.get("written")
            if name == "prompts.yaml" and (written is None or date.fromisoformat(str(written)) > first_run.date()):
                problems.append(f"prompts.yaml written {written}, first val run {first_run.date()}")
            for entry in doc.get("change_log") or []:
                if not isinstance(entry, dict) or "date" not in entry:
                    problems.append(f"{name} change_log entry without a date")
                elif date.fromisoformat(str(entry["date"])) > first_run.date() and not str(entry.get("reason", "")).strip():
                    problems.append(f"{name} change on {entry['date']} has no reason")
        return not problems, "; ".join(problems) or f"first val run {first_run.date()}"

    def check_19(self):
        log = _p("results", "test_runs.log")
        if not log.exists():
            raise Skip("test has not been scored")
        entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        frozen_now = _sha256(_p("frozen.json")) if _p("frozen.json").exists() else None
        problems, completed = [], 0
        for e in entries:
            if e.get("frozen_sha256") != frozen_now:
                problems.append(f"{e.get('event')} at {e.get('time')} used a different frozen.json")
            if e.get("event") == "started" and completed and len(str(e.get("reason", ""))) < 10:
                problems.append(f"run started at {e.get('time')} after a completed run, with no --rescore reason")
            completed += e.get("event") == "completed"
        return not problems, "; ".join(problems) or f"{completed} completed runs"


def run(fast: bool) -> int:
    lines, counts = [], {"PASS": 0, "FAIL": 0, "SKIP": 0}
    try:
        audit = Audit(fast)
        setup_error = None
    except (FileNotFoundError, KeyError) as err:
        audit, setup_error = None, f"cannot start: {err}. Run src/data.py first."
    for number, title in TITLES.items():
        try:
            if audit is None:
                # Only the check that needs no data can still run.
                ok, detail = Audit.check_13(None) if number == 13 else (False, setup_error)
            else:
                ok, detail = getattr(audit, f"check_{number}")()
            status = "PASS" if ok else "FAIL"
        except Skip as why:
            status, detail = "SKIP", str(why)
        except Exception as err:  # noqa: BLE001 - one broken input must not hide the other checks
            status, detail = "FAIL", f"{type(err).__name__}: {err}"
        counts[status] += 1
        lines.append(f"[{status}] {number:02d} {title}" + (f": {detail}" if detail else ""))
    lines.append(f"{len(TITLES)} checks: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP")
    print("\n".join(lines))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
    out = _p("results", "check_splits.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = f"check_splits.py {datetime.now().isoformat(timespec='seconds')} git {commit or 'unknown'}" + (" --fast" if fast else "")
    out.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8", newline="\n")
    return 1 if counts["FAIL"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="skip the two subprocess rebuilds (checks 6 and 11)")
    sys.exit(run(parser.parse_args().fast))


if __name__ == "__main__":
    main()
