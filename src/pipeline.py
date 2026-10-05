"""Grounding DINO proposes boxes for a text prompt, SAM masks each box, CLIP checks each crop.

Spec: docs/pipeline_spec.md. tune_val.py, eval.py, probe.py and demo.py all call these same
functions. This file takes PIL images and settings and returns results. It opens no files,
knows nothing about splits and prints nothing.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch
from PIL import Image, ImageOps

import metrics


@dataclasses.dataclass
class Detection:
    box: list[float]                  # x1, y1, x2, y2 in pixels
    score: float                      # Grounding DINO box score
    query_index: int
    wording: str
    word_scores: dict[str, float]     # each word of the wording -> its score for this box
    other_token_score: float          # best score on [CLS], [SEP] or "."
    mask: np.ndarray | None = None    # bool, image height x width
    sam_iou_pred: float | None = None  # SAM's own quality estimate
    p_positive: float | None = None   # verifier, after temperature
    margin: float | None = None       # p_positive minus the largest negative
    kept: bool | None = None          # passed the margin; None if no verifier


def prepare(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


# ---------------------------------------------------------------- Grounding DINO

def _caption(wording: str) -> str:
    """What the processor makes of text=[[wording]]: lowercased, with a trailing period."""
    assert "." not in wording, f"a wording may not contain a period: {wording!r}"
    return wording.strip().lower() + "."


def _token_words(models, wording: str, input_ids) -> list[str | None]:
    """For each token position, the word it belongs to, or None for [CLS], [SEP] and the period."""
    caption = _caption(wording)
    encoded = models.gd_processor.tokenizer(caption)
    assert list(encoded["input_ids"]) == [int(i) for i in input_ids], "tokenizer and processor disagree on the caption"
    words = []
    for word_id in encoded.word_ids():
        text = None
        if word_id is not None:
            span = encoded.word_to_chars(word_id)
            text = caption[span.start:span.end]
        words.append(None if text in (None, ".") else text)
    return words


def forward_dino(models, image: Image.Image, wording: str) -> dict:
    """One Grounding DINO pass. Raw outputs for every query, so a threshold can be swept later without rerunning."""
    inputs = models.gd_processor(images=image, text=[[_caption(wording)[:-1]]], return_tensors="pt")
    with torch.inference_mode():
        out = models.gd(**inputs)
    return {"logits": out.logits[0], "pred_boxes": out.pred_boxes[0], "input_ids": inputs["input_ids"][0],
            "query_index": torch.arange(out.logits.shape[1]), "_inputs": inputs, "_outputs": out}


def select(models, raw: dict, size: tuple[int, int], wording: str, threshold: float) -> list[Detection]:
    """The library's selection rule, kept here so query indices and per-word scores survive.

    Copied from the installed post_process_grounded_object_detection: sigmoid, box score = max over
    text positions, keep when score > threshold (strict), boxes from centre format scaled to the image.
    """
    width, height = size
    probs = torch.sigmoid(raw["logits"].float())
    scores = probs.max(dim=-1).values
    cx, cy, w, h = raw["pred_boxes"].float().unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * torch.tensor(
        [width, height, width, height], dtype=torch.float32)
    words = _token_words(models, wording, raw["input_ids"])
    detections = []
    for q in torch.nonzero(scores > threshold).flatten().tolist():
        word_scores: dict[str, float] = {}
        other = 0.0
        for position, word in enumerate(words):
            p = float(probs[q, position])
            if word is None:
                other = max(other, p)
            else:
                word_scores[word] = max(word_scores.get(word, 0.0), p)  # a split word takes its best piece
        detections.append(Detection(box=[float(v) for v in boxes[q]], score=float(scores[q]),
                                    query_index=int(raw["query_index"][q]), wording=wording,
                                    word_scores=word_scores, other_token_score=other))
    return detections


def detect(models, image: Image.Image, wording: str, threshold: float, text_threshold: float, return_raw: bool = False):
    """Boxes for one wording. text_threshold only shapes the library's label text, so it selects nothing here."""
    raw = forward_dino(models, image, wording)
    detections = select(models, raw, image.size, wording, threshold)
    if return_raw:
        return detections, {k: v for k, v in raw.items() if not k.startswith("_")}
    return detections


def selftest(models, images: list[Image.Image], wording: str, threshold: float, text_threshold: float = 0.25) -> float:
    """select() must give the library's boxes and scores, in the same order. Returns the largest difference."""
    worst = 0.0
    for image in images:
        raw = forward_dino(models, image, wording)
        ours = select(models, raw, image.size, wording, threshold)
        theirs = models.gd_processor.post_process_grounded_object_detection(
            raw["_outputs"], raw["_inputs"]["input_ids"], threshold=threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1]])[0]
        assert len(ours) == len(theirs["scores"]), f"{len(ours)} boxes here, {len(theirs['scores'])} from the library"
        for mine, score, box in zip(ours, theirs["scores"], theirs["boxes"]):
            worst = max(worst, abs(mine.score - float(score)), *(abs(a - float(b)) for a, b in zip(mine.box, box)))
    assert worst <= 1e-4, f"selection differs from the library by {worst}"
    return worst


# ---------------------------------------------------------------- SAM

def sam_embed(models, image: Image.Image) -> torch.Tensor:
    """The image encoder, once per image. Callers keep the result and pass it to segment()."""
    inputs = models.sam_processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        return models.sam.get_image_embeddings(inputs["pixel_values"])


def segment(models, image: Image.Image, embedding: torch.Tensor, boxes: list[list[float]]):
    """Masks for every box of one image from its stored embedding. Never re-encodes the image."""
    if not boxes:
        return [], []
    inputs = models.sam_processor(images=image, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt")
    with torch.inference_mode():
        out = models.sam(image_embeddings=embedding, input_boxes=inputs["input_boxes"],
                         multimask_output=models.sam_multimask_output)
        masks = models.sam_processor.image_processor.post_process_masks(
            out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
    return [m[0].numpy().astype(bool) for m in masks], [float(v) for v in out.iou_scores[0, :, 0]]


# ---------------------------------------------------------------- CLIP verifier

def text_bank(models, phrases: list[str], templates: list[str]) -> torch.Tensor:
    """One row per phrase: the mean of its template embeddings, each normalized, then normalized again."""
    rows = []
    for phrase in phrases:
        inputs = models.clip_processor(text=[t.format(phrase) for t in templates], return_tensors="pt", padding=True)
        with torch.inference_mode():
            features = models.clip.get_text_features(**inputs).pooler_output  # the projected embedding
        features = features / features.norm(dim=-1, keepdim=True)
        mean = features.mean(dim=0)
        rows.append(mean / mean.norm())
    return torch.stack(rows)


def clip_fill(models) -> tuple[int, int, int]:
    """Padding colour: CLIP's own image_mean, a model constant and not a dataset statistic."""
    return tuple(round(255 * m) for m in models.clip_processor.image_processor.image_mean)


def crop_for_clip(image: Image.Image, box, pad_frac: float, square_pad: bool, fill) -> Image.Image:
    x1, y1, x2, y2 = box
    px, py = pad_frac * (x2 - x1), pad_frac * (y2 - y1)
    left, top = max(0, int(x1 - px)), max(0, int(y1 - py))
    right, bottom = min(image.width, int(x2 + px) + 1), min(image.height, int(y2 + py) + 1)
    crop = image.crop((left, top, max(right, left + 1), max(bottom, top + 1)))
    if not square_pad:
        return crop
    # CLIP's processor centre-crops. Unpadded, a tall cereal box would lose its top and bottom.
    side = max(crop.size)
    square = Image.new("RGB", (side, side), fill)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def embed_crops(models, image: Image.Image, boxes, pad_frac: float) -> torch.Tensor:
    """Normalized CLIP embeddings, one row per box. Callers cache these."""
    if not len(boxes):
        return torch.zeros((0, models.clip.config.projection_dim))
    fill = clip_fill(models)
    crops = [crop_for_clip(image, box, pad_frac, models.clip_square_pad, fill) for box in boxes]
    inputs = models.clip_processor(images=crops, return_tensors="pt")
    with torch.inference_mode():
        features = models.clip.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


def verifier_scores(models, crop_embeddings: torch.Tensor, bank: torch.Tensor, n_positive: int, temperature: float):
    """(p_positive, margin, n_ties) per crop. Softmax over the contrast phrases; raw cosine is never a confidence."""
    if not len(crop_embeddings):
        return np.zeros(0), np.zeros(0), 0
    with torch.inference_mode():
        logits = models.clip.logit_scale.exp() * crop_embeddings @ bank.T
        probs = torch.softmax(logits / temperature, dim=-1).numpy()
    p_positive = probs[:, :n_positive].sum(axis=1)
    margin = p_positive - probs[:, n_positive:].max(axis=1)
    return p_positive, margin, sum(metrics.count_exact_ties(row) for row in probs)


def contrast_bank(models, contrast: dict, templates: list[str]) -> tuple[torch.Tensor, int]:
    """Text bank for one category's contrast, positives first. Returns (bank, number of positive phrases)."""
    return text_bank(models, list(contrast["positive"]) + list(contrast["negative"]), templates), len(contrast["positive"])


def verify(models, image: Image.Image, detections: list[Detection], contrast: dict | None, templates: list[str],
           temperature: float, margin: float, pad_frac: float, bank=None) -> int:
    """Fills p_positive, margin and kept on each detection. With no contrast, every field stays None."""
    if contrast is None or not detections:
        return 0
    bank, n_positive = bank or contrast_bank(models, contrast, templates)
    embeddings = embed_crops(models, image, [d.box for d in detections], pad_frac)
    p_positive, margins, n_ties = verifier_scores(models, embeddings, bank, n_positive, temperature)
    for d, p, m in zip(detections, p_positive, margins):
        d.p_positive, d.margin, d.kept = float(p), float(m), bool(m >= margin)
    return n_ties


# ---------------------------------------------------------------- ensemble and the demo path

def ensemble(detections_by_wording: dict[str, list[Detection]], nms_iou: float, rng: np.random.Generator):
    """Union of every wording's boxes for one category, then NMS. Returns (kept detections, tie count)."""
    pooled = [d for detections in detections_by_wording.values() for d in detections]
    kept, n_ties = metrics.nms([d.box for d in pooled], [d.score for d in pooled], nms_iou, rng)
    return [pooled[i] for i in kept], n_ties


def category_of(prompt: str, prompts_cfg: dict) -> str | None:
    """The evaluated category whose wordings include this prompt, or None for anything else."""
    wanted = prompt.strip().lower()
    for category, entry in prompts_cfg["grounding_dino"]["categories"].items():
        if wanted in (w.lower() for w in entry["wordings"]):
            return category
    return None


def run(models, image: Image.Image, prompt: str, frozen: dict, prompts_cfg: dict, sam_embedding=None) -> dict:
    """The demo path for one free-text prompt on a prepared image, at the frozen settings."""
    timing = {}
    start = time.perf_counter()
    detections = detect(models, image, prompt, frozen["dino"]["threshold"], frozen["dino"]["text_threshold"])
    timing["dino"] = time.perf_counter() - start

    start = time.perf_counter()
    if sam_embedding is None:
        sam_embedding = sam_embed(models, image)
    timing["sam_encode"] = time.perf_counter() - start
    start = time.perf_counter()
    masks, ious = segment(models, image, sam_embedding, [d.box for d in detections])
    for d, mask, iou in zip(detections, masks, ious):
        d.mask, d.sam_iou_pred = mask, iou
    timing["sam_decode"] = time.perf_counter() - start

    start = time.perf_counter()
    category = category_of(prompt, prompts_cfg)
    verifier_cfg = prompts_cfg["clip_verifier"]
    contrast = verifier_cfg["contrasts"].get(category) if category else None
    verify(models, image, detections, contrast, verifier_cfg["templates"], frozen["clip"]["temperature"],
           frozen["clip"]["margin"], frozen["clip"]["crop_pad_frac"])
    timing["clip"] = time.perf_counter() - start
    return {"detections": detections, "category": category, "verifier": "on" if contrast else "not configured",
            "timing": timing, "sam_embedding": sam_embedding}
