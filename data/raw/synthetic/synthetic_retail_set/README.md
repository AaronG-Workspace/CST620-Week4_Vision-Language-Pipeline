# Synthetic retail inspection set (CST 620, Week 4)

45 images: 10 cereal boxes photographed intact and dented (20), 9 distractors, 16 clean backgrounds.
Every row, with its ground truth, is in `manifest.csv`.

## What is synthetic and what is real
- **Cereal boxes (20) and 3 look-alike distractors (tea, tissue, foil boxes):** invented brands rendered
  as 3D textured boxes in Python and composited onto real Open Images photos of tables, shelves and benches.
  Exact masks are in `masks/`, and bounding boxes come from those masks.
- **6 real distractors and 16 backgrounds:** unmodified Open Images photos (CC BY 2.0). Distractor
  boxes come from Open Images human annotations, so there are no masks for them. Credits are in `ATTRIBUTION.csv`.
- No file has camera EXIF. These do not belong in `data/raw/own/`.

## Pairs
Each intact/dented pair shares one background, pose and placement. The only change is the damage.
A near-duplicate check may flag pairs, so group them by `pair_id` instead of dropping one.

## Damage
Dents are real geometry changes (front creases, crushed corners, edge dents, crushed top edges), plus
cracked-ink and scuff marks. `damage_severity` is measured, not hand-assigned:
`pct_box_changed` = share of box pixels whose summed RGB difference from the intact image is > 30.
Subtle < 6%, moderate 6 to 10%, severe >= 10%. Result: 3 subtle, 3 moderate, 4 severe.
Most dents sit inside the outline, so the masks for intact and dented boxes are identical.

## Caveats for the evaluation table
- The boxes are clean renders, and the damage is simulated. Report results on this set as synthetic.
- The look-alike boxes and the real medicine box are printed boxes that are not cereal. They are the most likely
  confident false positives for "cereal box".
- Backgrounds were chosen because Open Images has no annotated objects on them besides furniture. They are not
  verified empty at the pixel level (a few have small items on shelves).
- `background_oi_14.jpg` has a hard cast shadow. If your intake rejects it, 15 backgrounds remain.

## Provenance
Generated in a Claude chat session on 2026-10-04. The scripts are in `generator/`; their paths are hard-coded
to that session (`gen_all.py` renders, `package.py` builds the manifest). Needs numpy, opencv-python, pillow, pandas.
