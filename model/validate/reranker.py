"""Learned re-ranker over the axis + embedding features.

The metric thread's next step: attack hit@10 with a learned re-ranker instead
of hand-tuned weights. This module extracts a feature vector per (heading,
candidate) pair, fits a pointwise logistic ranker, and evaluates it with
leave-one-out CV and a held-out even/odd split on the multi-partner benchmark
(fixtures/fonts-pairings.json: each heading carries every publicly endorsed
body partner), against the shipped hand metric.

    model/.venv/bin/python model/validate/reranker.py          # evaluation
    model/.venv/bin/python model/validate/reranker.py --fit    # write the model

`--fit` writes `site/data/reranker.json` (input-space weights, consumed by
site/js/reranker.js) and `model/validate/fixtures/reranker-cases.json` (sample
pairs + expected scores, the JS/Python parity fixture).

Label hygiene vs the old 40-pair set: every acceptable partner trains as a
positive (not one arbitrary pick), and negatives are sampled outside the
acceptable partners' families, so known-good instances are never treated as
negatives. Strict scoring still counts a partner family only at its canonical
400-normal instance.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
# reuse the render-feature cache + corpus loaders from the metric A/B harness
from metric_ab import load_corpus, build_features  # noqa: E402

FIXTURES = ROOT / "model" / "validate" / "fixtures" / "fonts-pairings.json"
SITE_DATA = ROOT / "site" / "data"
CASES = ROOT / "model" / "validate" / "fixtures" / "reranker-cases.json"
XH_SIGMA = 0.05
SEED = 0


class Features:
    """All pairwise features, heading-relative so they transfer across headings."""

    NAMES = ["cos", "contrast", "leg_body", "leg_head", "serif_split", "serif_skel",
             "dxh", "bodyweight", "italic", "concentration"]

    def __init__(self, catalog, V, feats):
        self.cat = catalog
        self.V = V
        self.n = len(catalog)
        self.leg = np.array([c.get("legibility") or 0.0 for c in catalog], dtype=np.float32)
        self.xh = np.array([feats.get(str(c["id"]), {}).get("xh", np.nan) for c in catalog], dtype=np.float32)
        self.weight = np.array([c["weight"] for c in catalog], dtype=np.float32)
        self.serif = np.array([1.0 if c["category"] == "serif" else 0.0 for c in catalog], dtype=np.float32)
        self.italic = np.array([1.0 if c["style"] == "italic" else 0.0 for c in catalog], dtype=np.float32)
        self._vl = V / np.linalg.norm(V, axis=1, keepdims=True)
        self.cols = {}

    def column(self, hi):
        """Feature columns over all rows for one heading (cached)."""
        if hi in self.cols:
            return self.cols[hi]
        vl, v = self._vl, self._vl[hi]
        M = vl * v
        contrast = np.clip(M, 0, None).sum(axis=1) * np.clip(-M, 0, None).sum(axis=1)
        cmax = contrast.max() or 1.0
        dxh_abs = np.abs(self.xh - self.xh[hi])
        gate = np.where(np.isnan(dxh_abs), 0.0, np.exp(-dxh_abs / XH_SIGMA))
        serif_split = (self.serif != self.serif[hi]).astype(np.float32)
        diff = np.abs(vl - v)
        conc = np.sort(diff, axis=1)[:, -5:].sum(axis=1) / (diff.sum(axis=1) + 1e-9)
        self.cols[hi] = {
            "cos": vl @ v,
            "contrast": contrast / cmax,
            "leg_body": self.leg,
            "leg_head": np.full(self.n, self.leg[hi], dtype=np.float32),
            "serif_split": serif_split,
            "serif_skel": serif_split * gate,
            "dxh": -np.nan_to_num(dxh_abs, nan=1.0),
            "bodyweight": -np.abs(np.log2(self.weight / 400.0)),
            "italic": self.italic,
            "concentration": conc,
        }
        return self.cols[hi]

    def pool(self, hi):
        """Eligible instance ids whose family differs from the heading's."""
        fam_hi = self.cat[hi]["family"]
        return np.array([i for i in range(self.n) if self.cat[i]["eligible"] and self.cat[i]["family"] != fam_hi])

    def rank_of(self, hi, ps, s):
        order = np.argsort(-s)
        rows = self.pool(hi)
        pos = {rows[order[k]]: k + 1 for k in range(len(order))}
        return min((pos.get(k, len(order) + 1) for k in ps), default=len(order) + 1)


def vectorize(F, hi, rows, names=None):
    names = names or Features.NAMES
    cols = F.column(hi)
    return np.stack([cols[n][rows] for n in names], axis=1)


def hand_score(F, hi, rows):
    """The shipped hand metric, for the reference row (no training)."""
    c = F.column(hi)
    return c["contrast"][rows] + 0.5 * c["leg_body"][rows] + 0.2 * c["serif_skel"][rows]


def build_pairs(catalog):
    """Benchmark entries as (heading_id, positive_ids, excluded_ids).

    positives = the instances to score against (all instances of each
    acceptable family for the loose rule, one canonical instance per family
    for the strict rule); excluded = instances never sampled as negatives
    (every instance of an acceptable family: known-good is not negative).
    """
    elig = [i for i in range(len(catalog)) if catalog[i]["eligible"]]

    def rep_id(f):
        cands = [i for i in elig if catalog[i]["family"] == f]
        return next((i for i in cands if catalog[i]["weight"] == 400 and catalog[i]["style"] == "normal"), cands[0]) if cands else None

    def canon_id(f):
        cands = [i for i in elig if catalog[i]["family"] == f]
        if not cands:
            return None
        return next((i for i in cands if catalog[i]["weight"] == 400 and catalog[i]["style"] == "normal"),
                    next((i for i in cands if catalog[i]["weight"] == 400), cands[0]))

    loose, canon = [], []
    for entry in json.loads(FIXTURES.read_text())["pairings"]:
        hi = rep_id(entry["heading"])
        fams = [a["family"] for a in entry["acceptable"]]
        excl = [i for i in elig if catalog[i]["family"] in set(fams)]
        ps_all = excl  # loose rule: any instance of an acceptable family
        ps_canon = [c for c in (canon_id(f) for f in fams) if c is not None]
        if hi is not None and ps_canon:
            loose.append((hi, ps_all, excl))
            canon.append((hi, ps_canon, excl))
    return elig, loose, canon


def fit_standardized(F, train_pairs, seed, names=None, neg_per=50, C=1.0):
    """Fit the logistic ranker on standardized features; return (clf, mean, std).

    Every positive instance trains as a positive row; negatives are sampled
    outside the heading's excluded ids (the acceptable partners' families).
    """
    local = np.random.default_rng(seed)
    X, y = [], []
    for hi, ps, excl in train_pairs:
        rows = F.pool(hi)
        X.append(vectorize(F, hi, np.asarray(ps), names)); y.extend([1] * len(ps))
        allowed = np.setdiff1d(rows, np.asarray(excl))
        neg = local.choice(allowed, size=min(neg_per, len(allowed)), replace=False)
        X.append(vectorize(F, hi, neg, names)); y.extend([0] * len(neg))
    X = np.vstack(X); y = np.asarray(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    return clf, mu, sd


def fit_model(F, pairs, seeds=range(8), names=None):
    """Average the ranker over negative-sampling seeds; return input-space weights."""
    ws, bs = [], []
    for s in seeds:
        clf, mu, sd = fit_standardized(F, pairs, s, names)
        coef = clf.coef_[0]
        ws.append(coef / sd)
        bs.append(float(clf.intercept_[0] - np.sum(coef * mu / sd)))
    return np.mean(ws, axis=0), float(np.mean(bs)), np.std(ws, axis=0)


def learned_score(F, weights, bias, hi, rows):
    return bias + vectorize(F, hi, rows) @ weights


def main():
    catalog, V = load_corpus()
    feats = build_features(catalog)
    F = Features(catalog, V, feats)
    _, pairs_loose, pairs_canon = build_pairs(catalog)
    print(f"{len(pairs_canon)} headings; feature count {len(F.NAMES)}\n")

    SEEDS = range(8)
    no_italic = [n for n in F.NAMES if n != "italic"]

    def report(name, ranks):
        r = np.array(ranks)
        print(f"  {name:<26}hit@10 {int((r <= 10).sum()):>2}/{len(r)}  hit@50 {int((r <= 50).sum()):>2}/{len(r)}"
              f"  hit@100 {int((r <= 100).sum()):>2}/{len(r)}  median {int(np.median(r)):>5}")

    def fit_score(train, test, seed, names):
        clf, mu, sd = fit_standardized(F, train, seed, names)
        return [F.rank_of(hi, ps, clf.decision_function((vectorize(F, hi, F.pool(hi), names) - mu) / sd))
                for hi, ps, _ in test]

    def summarize(name, make):
        runs = np.array([make(s) for s in SEEDS])
        h10 = (runs <= 10).sum(1); h50 = (runs <= 50).sum(1); h100 = (runs <= 100).sum(1)
        med = np.median(runs, axis=1)
        print(f"  {name:<26}hit@10 {h10.mean():4.1f} [{h10.min()}-{h10.max()}]  hit@50 {h50.mean():4.1f} [{h50.min()}-{h50.max()}]"
              f"  hit@100 {h100.mean():4.1f} [{h100.min()}-{h100.max()}]  median {np.median(med):.0f}")

    def loocv(seed, names, P):
        out = []
        for j in range(len(P)):
            out += fit_score([p for k, p in enumerate(P) if k != j], [P[j]], seed, names)
        return out

    def split(seed, names, P):
        ev = [p for k, p in enumerate(P) if k % 2 == 0]
        od = [p for k, p in enumerate(P) if k % 2 != 0]
        return fit_score(ev, od, seed, names) + fit_score(od, ev, seed, names)

    for P, label in ((pairs_loose, "LOOSE: any instance of an acceptable partner family"),
                     (pairs_canon, "STRICT: canonical 400-normal instance per acceptable partner")):
        print(f"\n==== {label} ====")
        print("single features / hand (no training):")
        report("hand metric", [F.rank_of(hi, ps, hand_score(F, hi, F.pool(hi))) for hi, ps, _ in P])
        for name in F.NAMES:
            report(name, [F.rank_of(hi, ps, F.column(hi)[name][F.pool(hi)]) for hi, ps, _ in P])
        print("learned, LOOCV:")
        for nm, names in (("all features", F.NAMES), ("all minus italic", no_italic)):
            summarize(nm, lambda s, names=names, P=P: loocv(s, names, P))
        print("learned, held-out even/odd split:")
        for nm, names in (("all features", F.NAMES), ("all minus italic", no_italic)):
            summarize(nm, lambda s, names=names, P=P: split(s, names, P))

    print("\nshipped input-space weights (mean over seeds, strict positives):")
    w, b, sd = fit_model(F, pairs_canon, SEEDS)
    print(f"  bias {b:+.3f}")
    for n, wi, si in sorted(zip(F.NAMES, w, sd), key=lambda t: -abs(t[1])):
        print(f"  {n:<14}{wi:+.4f}  (sd {si:.4f})")


def fit_and_write():
    catalog, V = load_corpus()
    feats = build_features(catalog)
    F = Features(catalog, V, feats)
    _, _, pairs = build_pairs(catalog)
    w, b, _ = fit_model(F, pairs)
    rng = np.random.default_rng(SEED)
    cases = []
    for hi, ps, excl in pairs:
        allowed = np.setdiff1d(F.pool(hi), np.asarray(excl))
        picks = [ps[0]] + list(rng.choice(allowed, size=2, replace=False))
        for ci in picks:
            cases.append({
                "heading": catalog[hi]["label"], "candidate": catalog[ci]["label"],
                "score": round(float(learned_score(F, w, b, hi, np.array([ci]))[0]), 6),
            })
    (SITE_DATA / "reranker.json").write_text(json.dumps({
        "v": 1, "features": Features.NAMES,
        "weights": [round(float(x), 6) for x in w], "bias": round(b, 6),
        "xhSigma": XH_SIGMA,
        "trained": {"headings": len(pairs), "positives": sum(len(ps) for _, ps, _ in pairs),
                    "negativePerHeading": 50, "seeds": 8, "date": "2026-10-07"},
        "note": "pointwise logistic re-ranker over embedding + axis features, fit on the "
                "multi-partner fonts-pairings benchmark; see .todo/metric-research.md",
    }, indent=2) + "\n")
    CASES.write_text(json.dumps({"note": "JS/Python parity cases for site/js/reranker.js", "cases": cases}, indent=1) + "\n")
    print(f"wrote site/data/reranker.json and {len(cases)} parity cases")


if __name__ == "__main__":
    if "--fit" in sys.argv:
        fit_and_write()
    else:
        main()
