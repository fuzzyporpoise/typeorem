"""Re-emit the app's projection parity fixture.

`tests/project.test.mjs` in the app checks that the browser's projection module
reproduces sklearn's `PCA.transform` for real embeddings, using a committed fixture
(real DINOv2 outputs plus the projections the saved PCA gives them). The fixture is
generated, but it is a *pin*, so it is tracked: the app's gate has to run offline.

It goes stale exactly when the PCA changes, which is a corpus rebuild (build_corpus
fits a new PCA each run) or a re-emit through export_pca.py. When that happens,
regenerate it and commit the result in the app repo:

    model/.venv/bin/python model/export_projection_fixture.py

Reads `model/out/pca_model.npz` and real renders under `model/.cache/224`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import paths  # noqa: E402
from embed import DINOv2  # noqa: E402

OUT = BASE / "out"
PNG_DIR = BASE / ".cache" / "224"
DEFAULT_CASES = 8


def pick(instances, count):
    """Evenly spaced real instances, so a fixture covers the whole corpus rather
    than one neighbourhood."""
    if len(instances) <= count:
        return instances
    step = len(instances) / count
    return [instances[int(i * step)] for i in range(count)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=DEFAULT_CASES)
    ap.add_argument("--out", type=Path, default=None,
                    help="default <app checkout>/tests/fixtures/projection.json")
    args = ap.parse_args()

    paths.require_site("the fixture lands in the app's tests/fixtures")
    target = args.out or (paths.app_root() / "tests" / "fixtures" / "projection.json")

    pca = np.load(OUT / "pca_model.npz")
    mean, components = pca["mean"], pca["components"]
    instances = json.loads((OUT / "instances.json").read_text())
    chosen = []
    for inst in pick(instances, args.cases):
        png = PNG_DIR / f"{inst['id']}.png"
        if png.is_file():
            chosen.append((inst, png))
    if not chosen:
        raise SystemExit(
            f"no renders under {PNG_DIR}; run model/build_corpus.py first (it writes them)")

    images = [np.asarray(Image.open(png).convert("RGB")) for _, png in chosen]
    embedder = DINOv2()
    print(f"embedding {len(images)} instances on {embedder.device}")
    features = np.concatenate([embedder.embed_batch(images[s:s + 4]) for s in range(0, len(images), 4)])

    cases = []
    for (inst, _), feature in zip(chosen, features):
        expected = (feature - mean) @ components.T
        cases.append({
            "id": inst["id"],
            "family": inst["family"],
            "embedding": [round(float(v), 6) for v in feature],
            "expected": [round(float(v), 6) for v in expected],
        })

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "note": ("embedding=real DINOv2 ViT-S/14 output; "
                 "expected=(x-mean)@components.T via the shipped pca_model.npz/site-data/pca.bin; "
                 "regenerate with model/export_projection_fixture.py when the PCA changes"),
        "backboneDim": int(mean.shape[0]),
        "dim": int(components.shape[0]),
        "cases": cases,
    }, indent=1) + "\n")
    print(f"wrote {target} ({len(cases)} cases, {target.stat().st_size / 1024:.0f} KB)")
    print("commit it in the app repo (it is a tracked pin, not generated output there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
