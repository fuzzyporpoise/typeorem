"""Backbone-space diagnostics, after the FontFusion embedding bake-off
(arXiv:2606.06066, sec. 3): silhouette by category, leave-one-out 1-NN category
accuracy, and retrieval precision@10, all over the pairing-eligible instances
under cosine similarity.

These SCREEN candidate spaces cheaply; they measure category structure, not
pairing quality (the paper's own dual encoder trades silhouette for better
downstream behavior), so the decision metric stays downstream: benchmark.mjs
hit@k and nullmodel.mjs against the frequency null.

Run: model/.venv/bin/python model/validate/diagnostics.py [space ...]
Defaults to every space in model/out/spaces/.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import paths

OUT = BASE / "out" / "spaces"
TOPK = 10


def load_space(name):
    meta = json.loads((OUT / f"{name}.meta.json").read_text())
    q = np.frombuffer((OUT / f"{name}.i8.bin").read_bytes(), dtype=np.int8)
    q = q.reshape(meta["count"], meta["dim"]).astype(np.float64)
    scale = np.array(meta["scale"]["values"])
    x = q * scale[:, None]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return x / n


def main():
    names = sys.argv[1:] or sorted(p.name[:-7] for p in OUT.glob("*.i8.bin"))
    paths.require_site("space diagnostics read the shipped catalog")
    catalog = json.loads((paths.site_data() / "catalog.json").read_text())
    elig = np.array([bool(i["eligible"]) for i in catalog["instances"]])
    cats = np.array([i["category"] for i in catalog["instances"]])

    print(f"{'space':<18}{'count':>7}  {'silhouette':>10}  {'1NN cat acc':>11}  {'P@10 cat':>8}")
    for name in names:
        X = load_space(name)
        assert X.shape[0] == len(elig), f"{name}: {X.shape[0]} rows vs catalog {len(elig)}"
        Xe, ye = X[elig], cats[elig]
        sim = Xe @ Xe.T
        np.fill_diagonal(sim, -np.inf)
        nn = ye[np.argmax(sim, axis=1)]
        nn_acc = float(np.mean(nn == ye))
        top = np.argsort(-sim, axis=1)[:, :TOPK]
        p10 = float(np.mean([(ye[t] == ye[i]).mean() for i, t in enumerate(top)]))
        sil = float(silhouette_score(Xe, ye, metric="cosine"))
        print(f"{name:<18}{X.shape[0]:>7}  {sil:>10.3f}  {nn_acc:>11.3f}  {p10:>8.3f}")


if __name__ == "__main__":
    main()
