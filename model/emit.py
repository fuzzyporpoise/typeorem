import gzip
from pathlib import Path

import numpy as np

from catalog import write_json


def label_of(inst):
    suffix = " Italic" if inst["style"] == "italic" else ""
    return f"{inst['family']} {inst['weight']}{suffix}"


def emit(site_dir, instances, families, q, scale, pca, version, legibility=None, xh=None):
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    dim = q.shape[1]
    (site_dir / "vectors.i8.bin").write_bytes(q.tobytes())
    write_json(site_dir / "vectors.meta.json", {
        "v": 1, "dim": dim, "count": int(q.shape[0]), "dtype": "int8",
        "scale": {"mode": "perRow", "values": [round(float(s), 6) for s in scale]},
        "backbone": version["backbone"], "preprocess": version["preprocess"],
    })
    catalog = []
    for i, inst in enumerate(instances):
        fam = families[inst["family"]]
        catalog.append({
            "id": i, "family": inst["family"], "label": label_of(inst),
            "weight": inst["weight"], "style": inst["style"],
            "category": inst["category"], "source": "google",
            "subsets": fam["subsets"],
            "eligible": inst["pairingEligible"], "css2": inst["css2"],
            "coords": inst["coords"],
            "legibility": round(float(legibility[i]), 4) if legibility is not None else None,
            "xh": round(float(xh[i]), 4) if xh is not None and xh[i] is not None else None,
            "search": f"{inst['family']} {label_of(inst)} {inst['category']}".lower(),
        })
    write_json(site_dir / "catalog.json", {"v": 1, "count": len(catalog), "instances": catalog})
    mean = pca.mean_.astype("<f4")
    components = pca.components_.astype("<f4")
    (site_dir / "pca.bin").write_bytes(np.concatenate([mean, components.reshape(-1)]).tobytes())
    write_json(site_dir / "pca.json", {
        "v": 1, "dim": dim, "nComponents": dim, "backboneDim": int(mean.shape[0]),
        "bin": "pca.bin",
        "layout": "float32 LE row-major: mean[backboneDim] then components[dim,backboneDim]",
        "project": "(x - mean) @ components.T",
        "explainedVarianceRatio": [round(float(x), 6) for x in pca.explained_variance_ratio_],
    })
    write_json(site_dir / "corpus.version.json", version)


def report(site_dir):
    site_dir = Path(site_dir)
    total = 0
    for p in sorted(site_dir.iterdir()):
        gz = len(gzip.compress(p.read_bytes(), 9))
        total += gz
        print(f"  {p.name:22} {p.stat().st_size/1024:8.1f} KB  gz {gz/1024:7.1f} KB")
    print(f"  gz total: {total/1024/1024:.2f} MB")
    return total
