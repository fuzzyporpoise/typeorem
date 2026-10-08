"""A/B the shipped pairing metric against the two metric challengers.

Challenger 1: asymmetric role metric (heading read large, body read small, so
the partner-as-body is scored with a legibility term).
Challenger 2: interpretable multi-axis metric (share skeleton: x-height and
proportion; contrast on explicit axes: serif-ness, weight, stroke contrast).

Yardstick: model/validate/fixtures/known-pairings.json, scored the same way as
model/validate/benchmark.mjs (for heading H, how highly is partner P ranked
among all eligible instances of other families?). Random median would be about
half the pool. hit@10/50/100 and median/mean rank are the reported columns.

Decision (2026-10-07): the hybrid wins. Shipped as `roleScore` in
site/js/engine.js: normalized contrast + 0.5 * body-legibility
(challenger 1) + 0.2 * serif-split * exp(-|dxh| / 0.05) (challenger 2, gated
on the shipped x-height axis). The multi-axis term is scoped to the pairwise
partner ranking and kept out of `generate`'s joint objective, because adding
it there regressed the engine's balance (0.82 -> 0.65).

    model/.venv/bin/python model/validate/metric_ab.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from legibility import features_for, cell_bounds, O  # noqa: E402

DATA = ROOT / "site" / "data"
PNG_DIR = ROOT / "model" / ".cache" / "224"
FEAT_CACHE = ROOT / "model" / ".cache" / "render_features.json"
FIXTURES = ROOT / "model" / "validate" / "fixtures" / "known-pairings.json"


def load_corpus():
    meta = json.loads((DATA / "vectors.meta.json").read_text())
    catalog = json.loads((DATA / "catalog.json").read_text())["instances"]
    raw = np.frombuffer((DATA / "vectors.i8.bin").read_bytes(), dtype=np.int8)
    raw = raw.reshape(meta["count"], meta["dim"]).astype(np.float32)
    V = raw * np.array(meta["scale"]["values"], dtype=np.float32)[:, None]
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return catalog, V


def o_aspect(path):
    """Width/height of the 'o' ink, a proportion axis (condensed vs wide)."""
    ink = np.asarray(Image.open(path).convert("L")) < 128
    x0, y0, x1, y1 = cell_bounds(O)
    oc = ink[y0:y1, x0:x1]
    cols = np.where(oc.any(axis=0))[0]
    rows = np.where(oc.any(axis=1))[0]
    if not len(cols) or not len(rows):
        return None
    return round(float((cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1)), 4)


def build_features(catalog):
    if FEAT_CACHE.exists():
        cached = json.loads(FEAT_CACHE.read_text())
        if len(cached) == len(catalog):
            return cached
    slug = {(i["family"], i["weight"], i["style"]): i["id"]
            for i in json.loads((ROOT / "model" / "out" / "instances.json").read_text())}
    out = {}
    for c in catalog:
        s = slug.get((c["family"], c["weight"], c["style"]))
        if s is None:
            continue
        p = PNG_DIR / f"{s}.png"
        f = features_for(p)
        if f is None:
            continue
        w = o_aspect(p)
        if w is not None:
            f["o_aspect"] = w
        out[str(c["id"])] = f
    FEAT_CACHE.write_text(json.dumps(out, separators=(",", ":")))
    return out


def contrast_cols(V, hi):
    M = V * V[hi]
    return np.clip(M, 0, None).sum(axis=1) * np.clip(-M, 0, None).sum(axis=1)


def run_benchmark(catalog, score, elig, his, pis, keep=None):
    """score(hi, k) -> float, higher is a better partner. Returns metrics."""
    hits10 = hits50 = hits100 = 0
    ranks, per = [], []
    for idx, (hi, ps) in enumerate(zip(his, pis)):
        if keep is not None and idx not in keep:
            continue
        cand = [k for k in elig if catalog[k]["family"] != catalog[hi]["family"]]
        order = sorted(cand, key=lambda k: -score(hi, k))
        pos = {order[r]: r + 1 for r in range(len(order))}
        best = min((pos.get(k, len(order) + 1) for k in ps), default=len(order) + 1)
        ranks.append(best)
        per.append((catalog[hi]["family"], catalog[ps[0]]["family"], best))
        hits10 += best <= 10
        hits50 += best <= 50
        hits100 += best <= 100
    ranks = np.array(ranks)
    return {"n": len(ranks), "hit10": hits10, "hit50": hits50, "hit100": hits100,
            "median": int(np.median(ranks)), "mean": float(ranks.mean())}, per


def show(name, m):
    print(f"{name:<30}{m['hit10']:>4}/{m['n']:<4}{m['hit50']:>5}/{m['n']:<4}"
          f"{m['hit100']:>6}/{m['n']:<4}{m['median']:>9}{m['mean']:>10.0f}")


def main():
    catalog, V = load_corpus()
    feats = build_features(catalog)
    n = len(catalog)
    leg = np.array([c.get("legibility") or 0.0 for c in catalog], dtype=np.float32)
    elig = [i for i in range(n) if catalog[i]["eligible"]]

    def axis(key):
        return np.array([feats.get(str(c["id"]), {}).get(key, 0.0) for c in catalog], dtype=np.float32)

    xh, stroke, width = axis("xh"), axis("contrast"), axis("o_aspect")
    weight = np.array([c["weight"] for c in catalog], dtype=np.float32)
    serif = np.array([1.0 if c["category"] == "serif" else 0.0 for c in catalog], dtype=np.float32)

    fixtures = json.loads(FIXTURES.read_text())["pairs"]

    def rep_id(family):
        cands = [i for i in elig if catalog[i]["family"] == family]
        return next((i for i in cands if catalog[i]["weight"] == 400 and catalog[i]["style"] == "normal"), cands[0]) if cands else None

    his, pis, missing = [], [], []
    for h, p in fixtures:
        if h == p:
            continue
        hi, ps = rep_id(h), [i for i in elig if catalog[i]["family"] == p]
        if hi is None or not ps:
            missing.append(h if hi is None else p)
            continue
        his.append(hi)
        pis.append(ps)
    print(f"{len(his)} pairs evaluated ({len(fixtures)} fixtures; missing: {', '.join(sorted(set(missing))) or 'none'})\n")

    raw = {hi: contrast_cols(V, hi) for hi in his}
    cmax = {hi: float(raw[hi].max()) or 1.0 for hi in his}
    cn = lambda hi, k: raw[hi][k] / cmax[hi]

    print(f"{'metric':<30}{'hit@10':>12}{'hit@50':>12}{'hit@100':>13}{'median':>9}{'mean':>10}")
    base_fn = lambda hi, k: raw[hi][k]
    mb, base_per = run_benchmark(catalog, base_fn, elig, his, pis)
    show("0 product (shipped)", mb)

    # ---- Challenger 1: asymmetric role metric, additive and multiplicative ----
    for lam in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        m, _ = run_benchmark(catalog, lambda hi, k, lam=lam: cn(hi, k) + lam * leg[k], elig, his, pis)
        show(f"1 asym  cn + {lam}*leg(B)", m)

    for eps in (0.1, 0.3):
        m, _ = run_benchmark(catalog, lambda hi, k, eps=eps: cn(hi, k) * (eps + (1 - eps) * leg[k]), elig, his, pis)
        show(f"1 asym  cn * (leg+{eps})", m)

    # rank-blend: normalized contrast rank mixed with legibility
    for lam in (0.5, 1.0):
        m, _ = run_benchmark(catalog, lambda hi, k, lam=lam: cn(hi, k) + lam * leg[k] * (1 - cn(hi, k)), elig, his, pis)
        show(f"1 asym  cn*(1+{lam}leg)", m)

    # ---- Challenger 2: interpretable multi-axis metric ----
    def skel(hi, k, ax):
        return np.exp(-abs(xh[hi] - xh[k]) / ax)

    def axis_fn(hi, k, sk, cs, cw, cst, sx):
        s = skel(hi, k, sx) * (1.0 - 0.3 * min(1.0, abs(width[hi] - width[k]) / 0.3))
        c = cs * (serif[hi] != serif[k]) + cw * min(1.0, abs(weight[hi] - weight[k]) / 400.0) \
            + cst * min(1.0, abs(stroke[hi] - stroke[k]))
        return sk * s * c if sk else s * c

    variants = [
        ("serif only", (0, 1, 0, 0, 0.1)),
        ("weight only", (0, 0, 1, 0, 0.1)),
        ("stroke only", (0, 0, 0, 1, 0.1)),
        ("skel*serif", (1, 1, 0, 0, 0.05)),
        ("skel*serif+weight", (1, 1, 1, 0, 0.05)),
        ("skel*serif+stroke", (1, 1, 0, 1, 0.05)),
        ("skel*all", (1, 1, 1, 1, 0.05)),
    ]
    for name, a in variants:
        m, _ = run_benchmark(catalog, lambda hi, k, a=a: axis_fn(hi, k, *a), elig, his, pis)
        show(f"2 axis {name}", m)

    # ---- Hybrid: asym (legibility) + the serif contrast from challenger 2 ----
    # The serif term is gated on a shared skeleton so it only fires for
    # "differ in classification, share structure" pairs, the documented
    # failure mode. Three skeleton gates: render x-height, embedding cosine,
    # and a product of both.
    cos = {hi: V @ V[hi] for hi in his}
    cosn = lambda hi, k: (cos[hi][k] + 1) / 2
    gates = {
        "skel(xh,.05)": lambda hi, k: skel(hi, k, 0.05),
        "skel(xh,.1)": lambda hi, k: skel(hi, k, 0.1),
        "cos": cosn,
        "xh*cos": lambda hi, k: skel(hi, k, 0.1) * cosn(hi, k),
        "xh+.5cos": lambda hi, k: skel(hi, k, 0.05) + 0.5 * cosn(hi, k),
    }
    for lam in (0.5, 0.75, 1.0):
        for gname, gate in gates.items():
            for gam in (0.2, 0.5):
                m, _ = run_benchmark(catalog,
                                     lambda hi, k, lam=lam, gam=gam, gate=gate: cn(hi, k) + lam * leg[k]
                                     + gam * (serif[hi] != serif[k]) * gate(hi, k),
                                     elig, his, pis)
                show(f"3 hyb {lam}leg+{gam}ser*{gname}", m)

    # ---- Diagnostics: documented failure modes ----
    print("\nper-case rank across the candidates above")
    variants = [
        ("prod", lambda hi, k: raw[hi][k]),
        ("asym", lambda hi, k: cn(hi, k) + 1.0 * leg[k]),
        ("xh", lambda hi, k: cn(hi, k) + 0.75 * leg[k] + 0.2 * (serif[hi] != serif[k]) * skel(hi, k, 0.05)),
        ("cos", lambda hi, k: cn(hi, k) + 0.5 * leg[k] + 1.0 * (serif[hi] != serif[k]) * cosn(hi, k)),
    ]
    pers = [run_benchmark(catalog, fn, elig, his, pis)[1] for _, fn in variants]
    watch = {("Playfair Display", "Montserrat"), ("Quicksand", "Roboto Slab"),
             ("Roboto Slab", "Roboto"), ("EB Garamond", "Montserrat"), ("Abril Fatface", "Lato"),
             ("Merriweather", "Open Sans"), ("Source Serif 4", "Source Sans 3"), ("Taviraj", "Work Sans"),
             ("Roboto", "Nunito"), ("Poppins", "Open Sans"), ("Spectral", "Karla"),
             ("Nunito", "Nunito Sans"), ("Archivo Black", "Archivo"), ("Space Grotesk", "Inter")}
    print(f"  {'heading':<20}{'partner':<16}" + "".join(f"{v[0]:>7}" for v in variants))
    for row in zip(*pers):
        h, p = row[0][0], row[0][1]
        if (h, p) in watch:
            print(f"  {h:<20}{p:<16}" + "".join(f"{r[2]:>7}" for r in row) + "  *")

    print("\nhit@10 pairs per candidate:")
    for (name, _), per in zip(variants, pers):
        hits = ", ".join(f"{h}/{p}" for h, p, r in per if r <= 10)
        print(f"  {name:<6} {hits or '(none)'}")

    # ---- Robustness: does the winner hold on each half of the pairs? ----
    winner = lambda hi, k: cn(hi, k) + 0.75 * leg[k] + 0.2 * (serif[hi] != serif[k]) * skel(hi, k, 0.05)
    bfn = lambda hi, k: raw[hi][k]
    half = {0: set(range(0, len(his), 2)), 1: set(range(1, len(his), 2))}
    print("\nrobustness, odd/even halves (baseline product -> winner):")
    for hname, keep in half.items():
        mb, _ = run_benchmark(catalog, bfn, elig, his, pis, keep)
        mw, _ = run_benchmark(catalog, winner, elig, his, pis, keep)
        print(f"  half {hname} (n={mb['n']}):  prod hit@100 {mb['hit100']}/{mb['n']} median {mb['median']}"
              f"   ->   win hit@100 {mw['hit100']}/{mw['n']} median {mw['median']}")

    print("\nfeatures at weight 400 normal:")
    print(f"  {'family':<20}{'xh':>7}{'stroke':>8}{'o_aspect':>10}{'serif':>7}")
    for fam in sorted({h for h, _ in watch} | {p for _, p in watch}):
        ids = [i for i in elig if catalog[i]["family"] == fam and catalog[i]["weight"] == 400 and catalog[i]["style"] == "normal"]
        if ids:
            i = ids[0]
            print(f"  {fam:<20}{xh[i]:>7.3f}{stroke[i]:>8.3f}{width[i]:>10.3f}{serif[i]:>7.0f}")

    print("\nfeatures at weight 400 normal:")
    print(f"  {'family':<20}{'xh':>7}{'stroke':>8}{'o_aspect':>10}{'serif':>7}")
    for fam in sorted({h for h, _ in watch} | {p for _, p in watch}):
        ids = [i for i in elig if catalog[i]["family"] == fam and catalog[i]["weight"] == 400 and catalog[i]["style"] == "normal"]
        if ids:
            i = ids[0]
            print(f"  {fam:<20}{xh[i]:>7.3f}{stroke[i]:>8.3f}{width[i]:>10.3f}{serif[i]:>7.0f}")


if __name__ == "__main__":
    main()
