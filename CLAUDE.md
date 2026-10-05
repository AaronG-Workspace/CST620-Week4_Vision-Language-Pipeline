# CLAUDE.md: CST 620 Week 4, open-vocabulary retail inspection (RetailEdge Automation)

## What this project is
A proof of concept for Priya Nair, product manager: type a product or defect description, get boxes and masks back, with no retraining per SKU. It is evaluated zero-shot on a held-out set, to a production-readiness standard.
- Grounding DINO proposes boxes from the text prompt.
- SAM draws a mask inside each box. SAM never sees the text.
- CLIP verifies each crop against a contrast pair (for example damaged vs undamaged) and abstains when the top-two margin is small.

Prompt categories (at least 3): cereal_box, damaged_cereal_box, misaligned_label.
Deliverables: src/demo.py, notebooks/report.ipynb (executed top to bottom), results/, and a written evaluation results table.
When summarizing results in this session, say what it means for Priya first, then the number.

## Environment
- Windows, PowerShell, VS Code terminal, CPU only. Use the project .venv.
- Models, no substitutions: IDEA-Research/grounding-dino-tiny, facebook/sam-vit-base, openai/clip-vit-base-patch32. Ids, thread count and model options live in config.yaml under models. Only models.py loads them.
- Versions are pinned in requirements.txt. Before using any processor or post-processing method, print its signature from the installed version with inspect.signature. Do not write these calls from memory.
- Models load from the local Hugging Face cache (HF_HOME). Recorded runs set HF_HUB_OFFLINE=1. Nothing downloads during eval or the demo.
- Console output is ASCII only: no sigma, plus-minus, arrows or emoji in print(). Write files with encoding="utf-8".
- Shell commands are PowerShell syntax.

## File roles
| File | Job | Test access |
|---|---|---|
| src/data.py | reads both sources, inventory, EXIF dates for own photos, MD5 and pHash, grouping, splits asset pools once, optional Open Images pull, provides load_split() | writes test pool rows, its main() never reads them back |
| src/cutouts.py | makes RGBA product cutouts once (the mask a rendered file ships with, rembg inside the annotated box for the set's real distractors, or backdrop threshold with rembg fallback, never SAM) | none |
| src/make_synth.py | builds composites for each split from that split's pools only, writes exact masks as COCO | writes test images and GT, never reads them back |
| src/check_splits.py | verifies pools, splits, provenance and test access | test lists: path, asset_id, md5, phash, group_id, parent_asset_ids only |
| src/models.py | loads the three models once, sets threads | none |
| src/pipeline.py | DINO, then SAM, then CLIP verify, for one image and one prompt | none (receives images from the caller) |
| src/metrics.py | matching, IoU, P/R@0.5, mIoU, Wilson CI, ECE, ties | none |
| src/tune_val.py | every selection, on val, writes frozen.json | none |
| src/demo.py | prompt in, overlay PNG and JSON out | showcase split and data/showcase_live/ only |
| src/probe.py | reproduces hallucinations: prompt ablation and per-word scores | showcase split and data/showcase_live/ only |
| src/eval.py | scores test once with frozen.json | the only file allowed |
| tests/test_metrics.py | unit tests with hand-worked values from docs/metrics_spec.md | none, uses small hand-made fixtures |
| notebooks/report.ipynb | tables and figures | none, reads results/ only |

## Data and splits
- data/raw is read-only. Never modify, move or re-save files there.
- Hand-entered files live in data/raw/own (intake.csv, cutout_review.csv, background_review.csv, label_words.txt). Code reads them and never writes them. Every script's spec is in docs/, and the build order with its gates is in docs/build_stages.md.
- Two sources feed the asset pools, and data.py reads both:
  - Own photos, listed in data/raw/own/intake.csv (source "own"): shelf backgrounds and products on a plain sheet, including boxes photographed before and after denting. They need an EXIF capture_date after 2024-01-01. check_splits.py asserts this.
  - The rendered set in data/raw/synthetic/synthetic_retail_set, listed in its own manifest.csv. Boxes and look-alikes drawn by its generator are source "rendered" and carry the generator's sha256 and seeds. Its Open Images photos (backgrounds and real distractors) are source "openimages_asset" and carry their licence and image id. No file in the set has camera EXIF, so the date rule does not apply to it.
- The same physical product photographed twice is one group. A box photographed intact and then dented is one group. In the rendered set, an intact box and its dented twin share a pair_id, and that pair is one group.
- data.py splits pools by group, once, seed 42, into val, test and showcase. Every background, cutout and Open Images file belongs to exactly one split. Rebuilding needs --force and must be byte-identical.
- make_synth.py builds each split only from that split's pools, following docs/make_synth_spec.md. Its parameters (scale, rotation, feather radius, objects per image) are fixed in config.yaml before any results and never changed because of test results.
- Ground truth for composites is the paste mask make_synth.py writes. Open Images ground truth is its own masks. Never create, refine or "clean up" ground truth with SAM, Grounding DINO, CLIP or model-assisted labeling.
- Cutout masks come from the mask a rendered file ships with (its renderer's own output), a plain-backdrop threshold, or rembg, never SAM. SAM-shaped edges would flatter SAM's mask IoU.
- No dataset-level statistic feeds a model, threshold or metric (no dataset mean or std, no fitted normalization). Processors keep their fixed normalization. Per-image processing inside cutouts.py and make_synth.py (one photo's backdrop color, resizing) is fine as long as nothing is pooled across images.

## Test-set rules
- eval.py is the only file that may load test images or test annotations.
- load_split("test") raises PermissionError unless called with allow_test=True. eval.py passes it to load test images and annotations. check_splits.py passes it only with the six metadata columns (path, asset_id, md5, phash, group_id, parent_asset_ids). No other file passes it.
- No tuning, caching, plotting, demo, probe or notebook code opens the test split. cache/ holds val artifacts only.
- If a task seems to need test data, stop and ask me.

## Selection happens on val only
- Chosen on val: the Grounding DINO threshold, the headline wording or ensemble per category, the CLIP verifier margin, calibration temperature. text_threshold is fixed in config.yaml, because with one wording per call it only changes the label text, not which boxes are kept.
- Grids, the objective, the selection order and tie-breaks are fixed in config.yaml under tuning, before any val results.
- tune_val.py writes frozen.json with the chosen values, the sha256 of the val list, prompts.yaml and config.yaml, and a timestamp. eval.py refuses to run if any of those hashes no longer match.
- prompts/prompts.yaml holds 3 wordings per category (up to 5) in operator vocabulary, written before any val results. Any later change is logged in the file with a reason. Never copy wording from dataset captions or label files.
- eval.py is pre-registered: it runs every wording in prompts.yaml on test in one run, reports per-wording metrics and the spread, and marks the frozen headline choice. It never sweeps, tunes or picks a best wording from test.

## Score the test set once
- eval.py appends every run to results/test_runs.log (timestamp, git commit, frozen.json hash). If a completed run exists, it refuses unless given --rescore "<reason>".
- eval.py reads the test list once per run and keeps it in memory.
- eval.py prints only progress counters until every output is written. It writes to results/test_tmp and moves that folder to results/test at the end, so a crash shows no numbers.
- eval.py --dry-run runs the full code path on 3 val images and never touches test. Rehearsals use only --dry-run.
- eval.py saves per-box, per-word Grounding DINO scores and an overlay PNG for every failure to results/test/failures/, so failures can be explained from saved outputs without re-reading test.
- Never change frozen.json, prompts, config.yaml or pipeline code because of a test result.

## Hallucination definitions
- Confident false positive: a box above the frozen threshold with no true-positive match (mask IoU below 0.5) for the queried category.
- Absent-attribute false positive: a damaged_cereal_box box on an intact cereal box, or a misaligned_label box on a straight label, meaning box IoU of 0.5 or more with that negative.
- Absent-attribute rate = negatives hit by at least one such box above the frozen threshold, divided by all negatives of that kind. It is a proportion (never above 100%) and gets a Wilson interval. The raw count of such boxes is reported beside it.
- probe.py reproduces a case on showcase images or a photo in data/showcase_live/ (taken the day of recording, in no split, never scored): same image, same seed, prompt ablation (cereal box / damaged cereal box / undamaged cereal box), per-word scores, and the CLIP verifier on and off.

## Seeds
- seed 42. Each random source gets its own seeded generator (torch.Generator, numpy default_rng) so one consumer cannot shift another.

## CPU budget
- torch.inference_mode() for all inference. torch.set_num_threads is set once in models.py.
- SAM: compute each image embedding once (get_image_embeddings) and pass image_embeddings for all of that image's boxes. Never re-encode per box.
- CLIP: cache crop image embeddings and prompt text embeddings.
- Grounding DINO on val: one pass per category and wording, caching raw outputs (logits, pred_boxes, input_ids) so the threshold can be swept without rerunning the model.
- Print one timing line per stage.

## Scoring rules
<!-- metrics.py and pipeline.py already follow these. If Claude Code writes or edits scoring code on camera, these are the rules to cite when you push back. -->
- Raw CLIP cosine is not a probability. Verifier confidence = softmax over the contrast pair with the model's logit_scale. Keep a box only if the margin clears the val-chosen value. Log the margin for every box.
- Check for exact score ties before any argmax, match or NMS. Log the tie count. Break ties with the seeded generator, never by list position.
- Stage attribution blames the stage where an error started. A false positive's stage is Grounding DINO (it proposed the box), or SAM when the box sits on the right object but the mask fails. When the verifier kept a false positive, the row also carries verifier_also_failed. CLIP is the stage only for a miss where the verifier rejected a correct box. The stage table also counts false positives the verifier caught. Full rules are in docs/metrics_spec.md. Also run SAM on ground-truth boxes (oracle boxes) to isolate SAM quality.
- Precision@0.5, recall@0.5 and mIoU are reported together. Precision and recall get Wilson 95% intervals. F1 and mIoU get image-level bootstrap intervals. Every row carries its instance count, image count and number of distinct product groups. mIoU covers matched instances only, so it never appears without recall.
- Which IoU: box IoU answers where Grounding DINO pointed and what CLIP saw (false positive types, stage attribution, verifier labels for temperature). Mask IoU answers whether the final output is right (true positive matching, mIoU). Mixing them would blame one stage for another's error.
- ECE uses 10 fixed equal-width bins on [0, 1] and comes with a reliability diagram.

## Ask before
- changing splits, groups, pools or provenance
- changing config.yaml or prompts.yaml
- adding a model or a package
- deleting anything
