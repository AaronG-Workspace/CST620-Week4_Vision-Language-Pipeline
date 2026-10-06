# Results table (numbers only)

- "Held out" means the boxes and labels are newly rendered, while the backgrounds and real distractors are public Open Images photos with possible pretraining overlap, and rendered boxes differ in style from the real photo objects.
- The rendered cereal boxes print "Cereal" and the look-alikes print their own product names, so the models may partly read the text.

Headline setup: {'cereal_box': 'cereal carton', 'damaged_cereal_box': 'damaged cereal box', 'misaligned_label': 'crooked sticker'}, threshold 0.35, verifier margin 0.0.

- cereal_box: when it flags one, it is right 99% of the time (96 of 97), and it finds 99% of the real ones (96 of 97). F1 0.99.
- damaged_cereal_box: when it flags one, it is right 46% of the time (11 of 24), and it finds 28% of the real ones (11 of 39). F1 0.35.
- misaligned_label: when it flags one, it is right 0% of the time (0 of 20), and it finds 0% of the real ones (0 of 38). F1 0.00.
- Macro F1 0.446 with the verifier, 0.462 without.
- damaged_cereal_box on an intact cereal box, verifier off: 56 of 58 hit (56 boxes).
- damaged_cereal_box on an intact cereal box, verifier on: 13 of 58 hit (13 boxes).
- misaligned_label on a straight label, verifier off: 0 of 38 hit (0 boxes).
- misaligned_label on a straight label, verifier on: 0 of 38 hit (0 boxes).
