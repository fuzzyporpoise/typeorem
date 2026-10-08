---
license: apache-2.0
library_name: onnxruntime
tags:
  - image-feature-extraction
  - dinov2
---

# Typeorem backbone (DINOv2 ViT-S/14, fp16 ONNX)

The frozen image encoder behind the custom-font feature on
[typeorem.dev](https://typeorem.dev): DINOv2 ViT-S/14 (timm
`vit_small_patch14_dinov2.lvd142m`) exported to ONNX, fp16, with ImageNet
normalization baked into the graph.

- Input `pixels`: uint8 RGB, shape `[batch, 3, 224, 224]`, values 0..255.
- Output `embedding`: float32 pooled features, shape `[batch, 384]`.

The browser renders a font's glyph grid to 224x224, runs this graph via
onnxruntime-web, then projects the 384-dim output into the catalog's 200-dim
space (`site/data/pca.bin`). The graph is produced by `model/export_onnx.py
--fp16` and published by `model/publish_model.py` in the parent repository,
https://gitlab.com/fuzzyporpoise/typeorem.

## License and attribution

This artifact is a derivative of **DINOv2** (Meta AI), distributed under
Apache-2.0; the upstream weights and license govern. See
https://github.com/facebookresearch/dinov2 and
https://huggingface.co/facebook/dinov2-small. The export tooling and the rest of
the Typeorem project are MIT.
