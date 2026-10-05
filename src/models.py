"""Load Grounding DINO, SAM and CLIP once, pin CPU threads, and fingerprint the setup.

The only file that calls from_pretrained. Spec: docs/models_spec.md.

    python src/models.py --check
    python src/models.py --timing data/timing/<photo>.jpg [--threads N]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
PROMPTS_PATH = ROOT / "prompts" / "prompts.yaml"
TIMINGS_PATH = ROOT / "results" / "timings.csv"

# Set by --timing --threads before the first load. Everything else uses config.yaml.
_threads_override: int | None = None


@dataclasses.dataclass(frozen=True)
class Models:
    gd_processor: Any
    gd: Any
    sam_processor: Any
    sam: Any
    clip_processor: Any
    clip: Any
    num_threads: int
    sam_multimask_output: bool
    clip_square_pad: bool


def load_models_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


@functools.lru_cache(maxsize=1)
def get_models() -> Models:
    """Load all three models on CPU in float32 eval mode. A second call returns the same object."""
    # transformers is imported here, not at module top, so a caller can set
    # HF_HUB_OFFLINE first. huggingface_hub reads that variable once, at import.
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        CLIPModel,
        CLIPProcessor,
        SamModel,
        SamProcessor,
    )
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()  # its bars use block characters, and console output is ASCII only

    cfg = load_models_config()
    num_threads = _threads_override or int(cfg["num_threads"])
    torch.set_num_threads(num_threads)

    def load(label, processor_cls, model_cls, model_id):
        start = time.perf_counter()
        processor = processor_cls.from_pretrained(model_id)
        model = model_cls.from_pretrained(model_id).to(device="cpu", dtype=torch.float32).eval()
        print(f"[load] {label:<14} {model_id:<36} {time.perf_counter() - start:6.1f} s")
        return processor, model

    gd_processor, gd = load("grounding_dino", AutoProcessor, AutoModelForZeroShotObjectDetection, cfg["grounding_dino"])
    sam_processor, sam = load("sam", SamProcessor, SamModel, cfg["sam"])
    clip_processor, clip = load("clip", CLIPProcessor, CLIPModel, cfg["clip"])
    return Models(
        gd_processor=gd_processor,
        gd=gd,
        sam_processor=sam_processor,
        sam=sam,
        clip_processor=clip_processor,
        clip=clip,
        num_threads=num_threads,
        sam_multimask_output=bool(cfg["sam_multimask_output"]),
        clip_square_pad=bool(cfg["clip_square_pad"]),
    )


def _commit_hash(model: Any, model_id: str) -> str:
    """Resolved commit of the cached weights.

    transformers 5.18 leaves config._commit_hash as None, so the usual answer is
    the snapshot folder name in the local cache.
    """
    from huggingface_hub import try_to_load_from_cache

    commit = getattr(model.config, "_commit_hash", None)
    if commit:
        return commit
    cached = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(cached, str):
        raise RuntimeError(f"{model_id} is not in the local Hugging Face cache")
    return Path(cached).parent.name


def _size(size: Any) -> dict:
    return {k: v for k, v in dataclasses.asdict(size).items() if v is not None}


def fingerprint() -> dict:
    """What scored the images: weights, library versions and settings.

    tune_val.py stores this in frozen.json and eval.py refuses to run if it differs.
    """
    import transformers

    cfg = load_models_config()
    m = get_models()
    return {
        "models": {
            name: {"id": cfg[name], "commit": _commit_hash(model, cfg[name])}
            for name, model in (("grounding_dino", m.gd), ("sam", m.sam), ("clip", m.clip))
        },
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "num_threads": torch.get_num_threads(),
        "sam_multimask_output": m.sam_multimask_output,
        "clip_square_pad": m.clip_square_pad,
        "grounding_dino_resize": _size(m.gd_processor.image_processor.size),
        "clip_image_size": _size(m.clip_processor.image_processor.crop_size),
        # The PIL and torchvision backends resize differently, so the backend is part of the setup.
        "image_processors": {
            "grounding_dino": type(m.gd_processor.image_processor).__name__,
            "sam": type(m.sam_processor.image_processor).__name__,
            "clip": type(m.clip_processor.image_processor).__name__,
        },
    }


def fingerprint_sha256(fp: dict | None = None) -> str:
    text = json.dumps(fp if fp is not None else fingerprint(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _median_seconds(fn, runs: int = 3) -> float:
    seconds = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        seconds.append(time.perf_counter() - start)
    return statistics.median(seconds)


def time_stages(image: Image.Image, wording: str) -> dict[str, float]:
    """Median seconds per stage on one image: R1 in the spec. Reads and writes no files."""
    m = get_models()
    width, height = image.size
    # Five fixed boxes spread across the frame. Timing depends on their count, not on what they contain.
    fracs = [(0.05, 0.10, 0.35, 0.60), (0.30, 0.20, 0.60, 0.80), (0.55, 0.10, 0.90, 0.55),
             (0.10, 0.50, 0.45, 0.95), (0.50, 0.45, 0.95, 0.95)]
    boxes = [[x1 * width, y1 * height, x2 * width, y2 * height] for x1, y1, x2, y2 in fracs]
    crops = [image.crop(tuple(int(v) for v in box)) for box in boxes]

    with torch.inference_mode():
        def dino():
            inputs = m.gd_processor(images=image, text=[[wording]], return_tensors="pt")
            m.gd(**inputs)

        def sam_encode():
            inputs = m.sam_processor(images=image, return_tensors="pt")
            return m.sam.get_image_embeddings(inputs["pixel_values"])

        embedding = sam_encode()

        def sam_decode():
            inputs = m.sam_processor(images=image, input_boxes=[boxes], return_tensors="pt")
            out = m.sam(image_embeddings=embedding, input_boxes=inputs["input_boxes"],
                        multimask_output=m.sam_multimask_output)
            m.sam_processor.image_processor.post_process_masks(
                out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])

        def clip_crops():
            inputs = m.clip_processor(images=crops, return_tensors="pt")
            m.clip.get_image_features(pixel_values=inputs["pixel_values"])

        return {
            "dino_one_wording": _median_seconds(dino),
            "sam_image_encode": _median_seconds(sam_encode),
            "sam_decode_5_boxes": _median_seconds(sam_decode),
            "clip_5_crops": _median_seconds(clip_crops),
        }


def projected_seconds_per_image(stages: dict[str, float], dino_passes: int) -> float:
    return (dino_passes * stages["dino_one_wording"] + stages["sam_image_encode"]
            + stages["sam_decode_5_boxes"] + stages["clip_5_crops"])


def run_timing(photo: Path, out_path: Path = TIMINGS_PATH) -> None:
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        categories = yaml.safe_load(f)["grounding_dino"]["categories"]
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    test_images = int(cfg["images_per_split"]["test"])
    long_side = int(cfg["scene"]["canvas_long_side"])
    wording = next(iter(categories.values()))["wordings"][0]
    dino_passes = sum(len(c["wordings"]) for c in categories.values())

    image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    # Time at the size of the synthetic images. Resized in memory; the file is left alone.
    scale = long_side / max(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    print(f"[time] {photo.name} resized to {image.width}x{image.height} for timing")
    stages = time_stages(image, wording)
    fp = fingerprint()
    threads = fp["num_threads"]
    for stage, seconds in stages.items():
        print(f"[time] {stage:<20} {seconds:7.2f} s  (median of 3, {threads} threads)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    # Append, so runs at different thread counts sit side by side.
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if is_new:
            writer.writerow(["stage", "median_seconds", "num_threads", "photo", "fingerprint_sha256", "fingerprint"])
        fp_json = json.dumps(fp, sort_keys=True, separators=(",", ":"))
        for stage, seconds in stages.items():
            writer.writerow([stage, f"{seconds:.4f}", threads, photo.name, fingerprint_sha256(fp), fp_json])

    per_image = projected_seconds_per_image(stages, dino_passes)
    print(f"[time] projected per test image: {per_image:.1f} s "
          f"({dino_passes} DINO passes, 1 SAM encode, 5 masks, 5 CLIP crops)")
    print(f"[time] projected for {test_images} test images: {per_image * test_images / 60:.1f} min")
    print(f"[time] appended to {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")


def run_check() -> None:
    fp = fingerprint()
    print(json.dumps(fp, indent=2, sort_keys=True))
    print(f"[check] fingerprint sha256 {fingerprint_sha256(fp)}")
    print(f"[check] second get_models() call reloads anything: {'yes' if get_models() is not get_models() else 'no'}")
    print("[check] all three models loaded with HF_HUB_OFFLINE=1")


def main() -> None:
    global _threads_override
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="load all three offline and print the fingerprint")
    mode.add_argument("--timing", type=Path, metavar="PHOTO", help="time each stage on a photo in data/timing/")
    parser.add_argument("--threads", type=int, help="with --timing only: try a thread count other than config.yaml's")
    args = parser.parse_args()

    if args.threads is not None and args.timing is None:
        parser.error("--threads is for --timing only. Scored runs take num_threads from config.yaml.")

    if args.check:
        os.environ["HF_HUB_OFFLINE"] = "1"
        run_check()
        return

    photo = args.timing.resolve()
    if photo.parent != (ROOT / "data" / "timing").resolve():
        sys.exit("The timing photo must sit in data/timing/, outside every split and asset pool.")
    _threads_override = args.threads
    run_timing(photo)


if __name__ == "__main__":
    main()
