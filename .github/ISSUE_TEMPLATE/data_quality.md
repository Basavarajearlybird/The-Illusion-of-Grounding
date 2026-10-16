---
name: Data quality issue
about: A specific benchmark item looks wrong
labels: data-quality
---

**variant_id**:

**Which field is wrong**
- [ ] `question` is ambiguous or unanswerable from the image
- [ ] `answer_visual` does not match the image
- [ ] `answer_textual` (injected falsehood) equals the truth / is implausible
- [ ] the degraded image is corrupt or mis-cropped
- [ ] the counterfactual `twin_id` is not an exact flip
- [ ] other

**What you observe vs. expect**

**Note**: some "wrongness" is intentional — `answer_textual` is a deliberately
injected falsehood, and degraded images are deliberately hard. Please confirm the
issue is *unintended* error (e.g. an ambiguous question that round-trip validation
should have caught) rather than the designed stressor.
