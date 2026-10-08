import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
SITE = BASE.parent / "site" / "data"

meta = json.loads((SITE / "vectors.meta.json").read_text())
raw = np.frombuffer((SITE / "vectors.i8.bin").read_bytes(), dtype=np.int8).reshape(meta["count"], meta["dim"])
V = raw.astype(np.float32) * np.array(meta["scale"]["values"], dtype=np.float32)[:, None]
catalog = json.loads((SITE / "catalog.json").read_text())["instances"]
families = np.array([c["family"] for c in catalog])
eligible = np.array([c["eligible"] for c in catalog])


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


U = unit(V)


def contrast(a, b):
    p = a * b
    return np.maximum(p, 0).sum(-1) * np.maximum(-p, 0).sum(-1)


def top(seed, k=6):
    scores = contrast(U[seed], U)
    scores = np.where(families == families[seed], -1.0, scores)
    scores = np.where(eligible, scores, -1.0)
    order = np.argsort(-scores)[:k]
    return [(catalog[i]["label"], catalog[i]["category"]) for i in order]


by_label = {c["label"]: i for i, c in enumerate(catalog)}
print(f"corpus: {len(catalog)} instances, dim {meta['dim']}")
for label in ("Montserrat 400", "Playfair Display 400", "Lora 400", "Roboto 400", "Oswald 400", "Lobster 400"):
    idx = by_label[label]
    cats = {c for _, c in top(idx, 6)}
    print(f"\n{label} [{catalog[idx]['category']}] -> {top(idx)}")
print(f"\neligible: {sum(1 for c in catalog if c['eligible'])}/{len(catalog)}")
print(f"categories: { {k: sum(1 for c in catalog if c['category']==k) for k in sorted({c['category'] for c in catalog})} }")
