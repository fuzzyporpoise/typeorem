"""Ship the PCA projection model to the site so a browser can project a
custom font's embedding into the same 200-dim space as the corpus (TSK-010).

Reads the model the pipeline already saved (`model/out/pca_model.npz`:
`mean (384,)`, `components (200, 384)`) and writes `site/data/pca.bin` plus an
updated `site/data/pca.json`. No corpus rebuild: this only re-emits the model.

    model/.venv/bin/python model/export_pca.py

Format of pca.bin (float32 little-endian, row-major): the 384-float mean
followed by the 200x384 components, flattened. A custom embedding `x` projects
to `(x - mean) @ components.T`, matching sklearn's `PCA.transform`.
"""

from pathlib import Path

import numpy as np

from catalog import write_json
import paths

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
SITE = paths.site_data()


def load_ratios(pca_json):
    try:
        import json
        return json.loads(pca_json.read_text()).get("explainedVarianceRatio")
    except Exception:
        return None


def main():
    paths.require_site("the projection is emitted into the site's site/data")
    d = np.load(OUT / "pca_model.npz")
    mean = d["mean"].astype("<f4")
    components = d["components"].astype("<f4")
    (SITE / "pca.bin").write_bytes(np.concatenate([mean, components.reshape(-1)]).tobytes())
    pca_json = SITE / "pca.json"
    write_json(pca_json, {
        "v": 1,
        "dim": int(components.shape[0]),
        "nComponents": int(components.shape[0]),
        "backboneDim": int(mean.shape[0]),
        "bin": "pca.bin",
        "layout": "float32 LE row-major: mean[backboneDim] then components[dim,backboneDim]",
        "project": "(x - mean) @ components.T",
        "explainedVarianceRatio": load_ratios(pca_json),
    })
    print(f"wrote {SITE / 'pca.bin'} ({(SITE / 'pca.bin').stat().st_size / 1024:.0f} KB) and pca.json")


if __name__ == "__main__":
    main()
