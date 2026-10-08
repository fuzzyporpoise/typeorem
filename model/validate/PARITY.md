# Phase 0 parity report: does VGG16 reproduce fontjoy's vectors?

Task TSK-003. Goal: prove the embed stage reproduces fontjoy's neighbor structure
using their own artifacts (the `224/` PNGs and `vectors-200.tsv`), before
building the full-catalog corpus.

## Setup

- Input: fontjoy's 1,883 `224/` PNGs (224x224 RGBA), row-aligned to
  `metadata.tsv` by the numeric filename prefix.
- Backbone: torchvision VGG16 ImageNet weights, `features` (through `pool5`),
  flatten to 25088, matching fontjoy's projector `tensorShape [1883, 25088]`.
- Reduction: PCA to 200 (sklearn, `random_state=0`).
- Compare: nearest-neighbor overlap@k and pairwise cosine-similarity Pearson r
  against fontjoy's `vectors-200.tsv`. Random overlap@1 is about 0.0005.

## Preprocessing sweep (VGG16)

| preprocess | ovl@1 | ovl@5 | ovl@10 | ovl@20 | pearson |
| --- | --- | --- | --- | --- | --- |
| caffe (BGR, mean-subtract) | 0.518 | 0.527 | 0.542 | 0.567 | 0.790 |
| rgb mean-subtract | 0.518 | 0.527 | 0.542 | 0.567 | 0.790 |
| imagenet (mean/std) | 0.542 | 0.555 | 0.572 | 0.599 | 0.848 |
| raw [0,255] | 0.505 | 0.502 | 0.510 | 0.535 | 0.720 |
| raw [0,1] | 0.553 | 0.557 | 0.582 | 0.612 | 0.872 |

L2-normalizing the 25088 features before PCA changes nothing material.

## Backbone comparison (PCA 200)

| backbone | raw dim | ovl@1 | ovl@5 | ovl@10 | ovl@20 | pearson |
| --- | --- | --- | --- | --- | --- | --- |
| vgg16_pool5 (raw [0,1]) | 25088 | 0.553 | 0.557 | 0.582 | 0.612 | 0.872 |
| dinov2_vits14 (pooled) | 384 | 0.335 | 0.353 | 0.373 | 0.405 | 0.747 |

## Finding

VGG16 `pool5` with [0,1] preprocessing reproduces fontjoy's space structurally
(similarity Pearson 0.87; for some faces the nearest neighbor matches exactly,
for example Roboto regular gives Yantramanav/Heebo/Padauk in both spaces) but not
identically (about 58% of top-10 neighbors shared). The original backbone is
therefore VGG16-family, but the exact weights and preprocessing are not
recoverable from the published artifacts. The residual gap is in the feature map,
not the reduction: a rotation of the same features would leave neighbor structure
unchanged, so a differing PCA basis cannot explain it. The published vectors are
also a small, stale 2017 subset.

Consequence for the shipped corpus: bit-for-bit parity with the 2017 vectors is
neither achievable nor necessary. The pipeline needs a coherent space, and VGG16
`pool5` stays the baseline because it is the closest reproduction and the original
was VGG-style. DINOv2 is measurably further from fontjoy's space.

CLIP was not run: its value is a quality comparison, and the selection model is
not yet complete (TSK-009 adds body-legibility weighting and joint 3-slot
optimization), so there is no quality signal to compare on yet.

## Reproduce

    model/.venv/bin/python model/validate/preproc_sweep.py
    model/.venv/bin/python model/validate/challengers.py
