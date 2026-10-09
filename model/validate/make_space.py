import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import paths
from embed import DINOv2, VGG16Pool5

CACHE = BASE / ".cache"
OUT = BASE / "out" / "spaces"
DIM = 200


def slug(family, weight, style):
    return re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-") + f"-{weight}-{style}"


def catalog_rows():
    """The shipped catalog's instance order plus the fit mask: the index contract
    every challenger space must satisfy so benchmark.mjs SPACE_BIN overrides stay
    aligned. The mask excludes house faces (source != google): the PCA is never
    refit on them, they are projected through the Google-fitted basis."""
    paths.require_site("challenger spaces are built against the shipped catalog")
    catalog = json.loads((paths.site_data() / "catalog.json").read_text())
    ids = [slug(i["family"], i["weight"], i["style"]) for i in catalog["instances"]]
    fit = [i["source"] == "google" for i in catalog["instances"]]
    return ids, fit


def feats_for(embedder, ids):
    """Raw backbone features in catalog order, cached under .cache (renders are
    cached too, so a rebuild never re-renders)."""
    key = CACHE / f"feats-{embedder.name}"
    manifest = key.with_suffix(".ids.json")
    cache = key.with_suffix(".npy")
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == ids:
        return np.load(cache)
    imgs = [np.asarray(Image.open(CACHE / "224" / f"{i}.png").convert("RGB")) for i in ids]
    print(f"  embedding {len(imgs)} with {embedder.name}")
    feats = np.concatenate([embedder.embed_batch(imgs[s:s + 16]) for s in range(0, len(imgs), 16)])
    np.save(cache, feats)
    manifest.write_text(json.dumps(ids))
    return feats


def unit_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return x / n


def build(embedders, name, dim=DIM):
    ids, fit = catalog_rows()
    blocks = [unit_rows(feats_for(e, ids)) for e in embedders]
    feats = np.concatenate(blocks, axis=1)
    mask = np.array(fit)
    pca = PCA(n_components=dim, random_state=0).fit(feats[mask])
    S = pca.transform(feats)
    C = pca.components_
    for k in range(C.shape[0]):
        j = int(np.argmax(np.abs(C[k])))
        if C[k, j] < 0:
            C[k] *= -1
            S[:, k] *= -1
    scale = np.abs(S).max(axis=1)
    scale[scale == 0] = 1
    q = np.clip(np.round(S / scale[:, None] * 127), -127, 127).astype(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.i8.bin").write_bytes(q.tobytes())
    (OUT / f"{name}.meta.json").write_text(json.dumps({
        "dim": dim, "count": int(q.shape[0]), "dtype": "int8",
        "scale": {"mode": "perRow", "values": [round(float(x), 6) for x in scale]},
    }))
    print(f"  wrote {name} {q.shape} (catalog order, {len(ids)} instances, PCA fit on {int(mask.sum())} google rows)")


def embedder_for(name):
    if name == "dinov2":
        return DINOv2()
    if name == "vgg16":
        return VGG16Pool5(mode="raw01")
    if name == "fontclip":
        from embed import FontCLIP
        return FontCLIP()
    raise SystemExit(f"unknown embedder {name}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "dinov2"):
        build([DINOv2()], "dinov2")
    if which in ("all", "vgg16"):
        build([VGG16Pool5(mode="raw01")], "vgg16")
    if which in ("all", "fontclip"):
        build([embedder_for("fontclip")], "fontclip")
    if which in ("all", "dual"):
        build([DINOv2(), embedder_for("fontclip")], "dinov2_fontclip")
