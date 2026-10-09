"""Export the shipped DINOv2 ViT-S/14 backbone to ONNX for in-browser use.

The custom-font path (TSK-010) needs the same frozen backbone the corpus was
built with, running client-side. `embed.py` defines it: timm
`vit_small_patch14_dinov2.lvd142m`, `num_classes=0`, 224 px, ImageNet
normalization, pooled 384-dim output. This script freezes exactly that into an
ONNX graph whose input is uint8 RGB pixels in 0..255, so the browser never has
to reimplement the normalization.

    model/.venv/bin/python model/export_onnx.py --fp16

Writes an fp32 graph and (with --fp16) the fp16 graph that actually ships: fp16
is half the size (43.5 MB vs 86.7 MB) at identical fidelity (embedding cosine
1.0000, projected vs shipped 0.9996-0.9998). int8 dynamic quantization is NOT
an option here: it drops the embedding to cosine 0.86 and the projected vector
to about 0.4. A parity check runs on each build: ONNX vs the live torch model.

The fp16 graph is not committed; publish it with model/publish_model.py, which
records the URL and hash in site/model.json (the TSK-010 hosting decision).
Publishing is manual and rare: only a backbone swap triggers it, i.e. a change to
the corpus backbone/preprocess, to embed.py's extractor (backbone, weights, resize
policy, normalization) or input size, or to this script's export recipe (opset,
dynamic axes, fp16).
"""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import paths

# The graph lands in the site checkout's site/models/ (gitignored) so the site
# can serve it locally, or in this repo's out/ when standing alone (CI).
DEFAULT_OUT_NAME = "dinov2_vits14.onnx"
FP16_NAME = "dinov2_vits14.fp16.onnx"


def default_out():
    return (paths.site_models() if paths.site_present() else BASE / "out") / DEFAULT_OUT_NAME

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TIMM_NAME = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE = 224
EMBED_DIM = 384


class PixelDino(nn.Module):
    """DINOv2 with ImageNet normalization baked in, taking uint8 pixels."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):  # uint8 [B, 3, 224, 224], 0..255
        x = pixels.float() / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return self.model(x)


def build(pretrained=True):
    import timm
    model = timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=0, img_size=IMG_SIZE)
    return PixelDino(model).eval()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(out_path, dynamic_batch=True, opset=17):
    wrap = build()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
    dynamic_axes = {"pixels": {0: "b"}, "embedding": {0: "b"}} if dynamic_batch else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap, dummy, str(out_path),
        input_names=["pixels"], output_names=["embedding"],
        opset_version=opset, dynamic_axes=dynamic_axes, dynamo=False,
    )
    report(out_path)
    return wrap


def to_fp16(src_path, dst_path):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model = onnx.load(str(src_path))
    onnx.save(convert_float_to_float16(model, keep_io_types=True), str(dst_path))
    report(dst_path)
    return dst_path


def report(path):
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB sha256 {digest(path)[:16]}")


def parity(wrap, out_path, n=2, seed=0):
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    pixels = (rng.random((n, 3, IMG_SIZE, IMG_SIZE)) * 255).astype(np.uint8)
    with torch.no_grad():
        ref = wrap(torch.from_numpy(pixels)).numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixels": pixels})[0]
    cos = float(np.dot(got[0], ref[0]) / (np.linalg.norm(got[0]) * np.linalg.norm(ref[0])))
    print(f"  parity {out_path.name}: shape {got.shape}, max abs diff {np.abs(got - ref).max():.2e}, cosine {cos:.7f}")
    assert got.shape == (n, EMBED_DIM) and cos > 0.9999, "ONNX export diverged from torch"
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=default_out())
    ap.add_argument("--fp16", action="store_true", help="also write the fp16 graph that ships")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrap = export(args.out)
    parity(wrap, args.out)
    if args.fp16:
        f16 = args.out.with_name(FP16_NAME)
        to_fp16(args.out, f16)
        parity(wrap, f16)


if __name__ == "__main__":
    main()
