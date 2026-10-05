# Results table (numbers only)

- "Held out" means the boxes and labels are newly rendered, while the backgrounds and real distractors are public Open Images photos with possible pretraining overlap, and rendered boxes differ in style from the real photo objects.
- The rendered cereal boxes print "Cereal" and the look-alikes print their own product names, so the models may partly read the text.

Headline setup: {'cereal_box': 'cereal carton', 'damaged_cereal_box': 'damaged cereal box', 'misaligned_label': 'crooked sticker'}, threshold 0.35, verifier margin 0.0.

- cereal_box: when it flags one, it is right 86% of the time (12 of 14), and it finds 100% of the real ones (12 of 12). F1 0.92.
- damaged_cereal_box: when it flags one, it is right 50% of the time (1 of 2), and it finds 20% of the real ones (1 of 5). F1 0.29.
- misaligned_label: when it flags one, it is right 0% of the time (0 of 7), and it finds 0% of the real ones (0 of 3). F1 0.00.
- Macro F1 0.403 with the verifier, 0.354 without.
- damaged_cereal_box on an intact cereal box, verifier off: 7 of 7 hit (7 boxes).
- damaged_cereal_box on an intact cereal box, verifier on: 1 of 7 hit (1 boxes).
- misaligned_label on a straight label, verifier off: 0 of 5 hit (0 boxes).
- misaligned_label on a straight label, verifier on: 0 of 5 hit (0 boxes).
