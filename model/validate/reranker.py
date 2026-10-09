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

The evaluation also carries the eighth pass's lead (TSK-020): a `dsknn` feature
plus a `has_graph` indicator from the paper's training-free dual-space k-NN
(arXiv:1811.08015 sec. IV-A Eq. 1: pair with what visually similar headings
pair with, `dsknn_score`). That feature reads the fold's train pair graph, so it
is built per fold, never from the global cache.

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
import paths  # noqa: E402
# reuse the render-feature cache + corpus loaders from the metric A/B harness
from metric_ab import load_corpus, build_features  # noqa: E402

FIXTURES = ROOT / "model" / "validate" / "fixtures" / "fonts-pairings.json"
USAGE_FIXTURES = ROOT / "model" / "validate" / "fixtures" / "fontsinuse-pairings.json"
SITE_DATA = paths.site_data()
CASES = ROOT / "model" / "validate" / "fixtures" / "reranker-cases.json"
XH_SIGMA = 0.05
SEED = 0
K1 = K2 = 10


class Features:
    """All pairwise features, heading-relative so they transfer across headings.

    The first ten are pure functions of the heading and the candidate and are
    cached per heading. The last two (`dsknn`, `has_graph`) read the fold's
    train pair graph, so they are not cached in the global column cache: the
    graph (the k1 nearest train headings' partners) is memoized per fold in
    `graph()`, and `dynamic()` scores the requested rows against it."""

    NAMES = ["cos", "contrast", "leg_body", "leg_head", "serif_split", "serif_skel",
             "dxh", "bodyweight", "italic", "concentration"]
    DSKNN = ["dsknn", "has_graph"]
    ALL = NAMES + DSKNN
    DYNAMIC = frozenset(DSKNN)

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
        self._graphs = {}

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

    def graph(self, train, hi):
        """The DS-kNN pair graph for `hi` under this fold's train set, memoized.

        The graph depends only on the set of train headings, not on their order
        or the sampled negatives, so it is shared across negative-sampling seeds
        and across feature subsets (including the held-out heading's own call)."""
        key = (hi, tuple(sorted(h for h, _, _ in train)))
        g = self._graphs.get(key)
        if g is None:
            g = dsknn_graph(self.V, train, hi)
            self._graphs[key] = g
        return g

    def dynamic(self, train, hi, rows):
        """DS-kNN columns for the requested rows, from the fold's train graph."""
        cand, cw = self.graph(train, hi)
        if cand is None:
            return {"dsknn": np.zeros(len(rows), dtype=np.float32),
                    "has_graph": np.zeros(len(rows), dtype=np.float32)}
        return {"dsknn": dsknn_scores(self.V, cand, cw, rows),
                "has_graph": np.full(len(rows), 1.0, dtype=np.float32)}

    def pool(self, hi):
        """Eligible instance ids whose family differs from the heading's."""
        fam_hi = self.cat[hi]["family"]
        return np.array([i for i in range(self.n) if self.cat[i]["eligible"] and self.cat[i]["family"] != fam_hi])

    def rank_of(self, hi, ps, s):
        order = np.argsort(-s)
        rows = self.pool(hi)
        pos = {rows[order[k]]: k + 1 for k in range(len(order))}
        return min((pos.get(k, len(order) + 1) for k in ps), default=len(order) + 1)


def vectorize(F, hi, rows, names=None, train=None):
    """Stack the requested feature columns for `hi` over `rows`.

    The static columns are heading-relative and cached; the `dsknn`/`has_graph`
    columns are scored over `rows` from the fold's train graph, so they need
    `train`."""
    names = names or F.NAMES
    rows = np.asarray(rows)
    cols = F.column(hi)
    dyn = None
    out = []
    for nm in names:
        if nm in F.DYNAMIC:
            if dyn is None:
                if train is None:
                    raise ValueError("dsknn features need the fold's train pairs")
                dyn = F.dynamic(train, hi, rows)
            out.append(dyn[nm])
        else:
            out.append(cols[nm][rows])
    return np.stack(out, axis=1)


def hand_score(F, hi, rows):
    """The shipped hand metric, for the reference row (no training)."""
    c = F.column(hi)
    return c["contrast"][rows] + 0.5 * c["leg_body"][rows] + 0.2 * c["serif_skel"][rows]


def canonical_id(catalog, family):
    """Canonical instance index for a family: 400-normal, else 400, else first
    eligible. The one rule shared by the fit, the artifact, and the JS runtime."""
    cands = [i for i, c in enumerate(catalog) if c.get("eligible") and c["family"] == family]
    if not cands:
        return None
    return next((i for i in cands if catalog[i]["weight"] == 400 and catalog[i]["style"] == "normal"),
                next((i for i in cands if catalog[i]["weight"] == 400), cands[0]))


def build_pairs(catalog, fixtures=None):
    """Benchmark entries as (heading_id, positive_ids, excluded_ids).

    positives = the instances to score against (all instances of each
    acceptable family for the loose rule, one canonical instance per family
    for the strict rule); excluded = instances never sampled as negatives
    (every instance of an acceptable family: known-good is not negative).
    `fixtures` overrides the default endorsement fixture (e.g. the usage-mined
    fontsinuse-pairings.json); both fixtures share the heading/acceptable shape.
    """
    elig = [i for i in range(len(catalog)) if catalog[i]["eligible"]]

    def canon_id(f):
        return canonical_id(catalog, f)

    rep_id = canon_id  # the same 400-normal-first rule the runtime canonicalIndex uses

    loose, canon = [], []
    for entry in json.loads(Path(fixtures or FIXTURES).read_text())["pairings"]:
        hi = rep_id(entry["heading"])
        fams = [a["family"] for a in entry["acceptable"]]
        excl = [i for i in elig if catalog[i]["family"] in set(fams)]
        ps_all = excl  # loose rule: any instance of an acceptable family
        ps_canon = [c for c in (canon_id(f) for f in fams) if c is not None]
        if hi is not None and ps_canon:
            loose.append((hi, ps_all, excl))
            canon.append((hi, ps_canon, excl))
    return elig, loose, canon


def graph_pairs(catalog):
    """The shipped DS-kNN graph: the union of both fixtures, per heading family,
    partners deduped and family-sorted, self excluded, exactly as
    `make_pairgraph.build_graph` emits it. Resolving in the same order (and with
    the same canonical rule) is what keeps the fit's `dsknn` and the browser's
    `dsknnColumn` bit-for-bit comparable, so train and serve read one graph."""
    fams = {c["family"] for c in catalog if c.get("eligible")}
    headings = {}
    for fixtures in (FIXTURES, USAGE_FIXTURES):
        for entry in json.loads(Path(fixtures).read_text())["pairings"]:
            h = entry["heading"]
            if h not in fams:
                continue
            partners = headings.setdefault(h, set())
            for a in entry["acceptable"]:
                if a["family"] in fams and a["family"] != h:
                    partners.add(a["family"])
    out = []
    for h in sorted(headings):
        partners = sorted(headings[h])
        hi = canonical_id(catalog, h)
        ps = [p for p in (canonical_id(catalog, f) for f in partners) if p is not None]
        if hi is None or not ps:
            continue
        excl = [i for i, c in enumerate(catalog) if c.get("eligible") and c["family"] in set(partners)]
        out.append((hi, ps, excl))
    return out


def dsknn_graph(V, train, hi, k1=K1):
    """The DS-kNN pair graph for heading `hi`: the (candidate ids, weights) built
    from the k1 nearest train headings' partners. Returns (None, None) when no
    train heading contributes a partner (cold fold)."""
    heads = np.array([h for h, _, _ in train if h != hi])
    if len(heads) == 0:
        return None, None
    sims = V[heads] @ V[hi]
    near = heads[np.argsort(-sims, kind="stable")[:k1]]
    cand, cw = [], []
    for h in near:
        _, ps, _ = next(p for p in train if p[0] == h)
        for p in ps:
            cand.append(p)
            cw.append(float(V[h] @ V[hi]))
    if not cand:
        return None, None
    return np.array(cand), np.array(cw, dtype=np.float32)


def dsknn_scores(V, cand, cw, rows, k2=K2):
    """Eq. 1 over a prebuilt graph: score each row by the weighted similarity to
    its k2 nearest graph partners (candidate ids `cand`, weights `cw`).

    `cand` repeats an instance once per near heading, so similarities are computed
    per unique instance and gathered: BLAS gives duplicate columns 1-ULP-different
    dots otherwise, which flips the k2 tie-break away from the JS runtime's exact
    ties."""
    cand = np.asarray(cand)
    uniq, inverse = np.unique(cand, return_inverse=True)
    sim_b = (V[rows] @ V[uniq].T)[:, inverse]
    k = min(k2, sim_b.shape[1])
    order = np.argsort(-sim_b, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(sim_b, order, axis=1) * cw[order]
    return top.mean(axis=1).astype(np.float32)


def dsknn_cols(V, train, hi, rows, k1=K1, k2=K2):
    """DS-kNN score and graph-presence indicator for heading `hi` over `rows`.

    Training-free (arXiv:1811.08015 sec. IV-A Eq. 1): pair with what visually
    similar headings pair with. `train` is the fold's pair set, so the graph
    never contains a held-out heading. Cold folds score 0 with presence 0."""
    cand, cw = dsknn_graph(V, train, hi, k1)
    if cand is None:
        return np.zeros(len(rows), dtype=np.float32), 0.0
    return dsknn_scores(V, cand, cw, rows, k2), 1.0


def dsknn_score(V, train, hi, rows, k1=K1, k2=K2):
    """The DS-kNN score alone (see `dsknn_cols`)."""
    return dsknn_cols(V, train, hi, rows, k1, k2)[0]


def fit_standardized(F, train_pairs, seed, names=None, neg_per=50, C=1.0, graph=None):
    """Fit the logistic ranker on standardized features; return (clf, mean, std).

    Every positive instance trains as a positive row; negatives are sampled
    outside the heading's excluded ids (the acceptable partners' families).
    `graph` is the pair set the `dsknn` feature reads (the shipped union graph,
    minus any held-out heading); it defaults to `train_pairs` for a plain fit.
    """
    graph = train_pairs if graph is None else graph
    local = np.random.default_rng(seed)
    X, y = [], []
    for hi, ps, excl in train_pairs:
        rows = F.pool(hi)
        X.append(vectorize(F, hi, np.asarray(ps), names, graph)); y.extend([1] * len(ps))
        allowed = np.setdiff1d(rows, np.asarray(excl))
        neg = local.choice(allowed, size=min(neg_per, len(allowed)), replace=False)
        X.append(vectorize(F, hi, neg, names, graph)); y.extend([0] * len(neg))
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


def learned_score(F, weights, bias, hi, rows, names=None, train=None):
    return bias + vectorize(F, hi, rows, names, train) @ weights


def main():
    catalog, V = load_corpus()
    feats = build_features(catalog)
    F = Features(catalog, V, feats)
    GRAPH = graph_pairs(catalog)

    SEEDS = range(8)
    no_italic = [n for n in F.NAMES if n != "italic"]
    dsknn_minus_italic = [n for n in F.ALL if n != "italic"]

    def report(name, ranks):
        r = np.array(ranks)
        print(f"  {name:<26}hit@10 {int((r <= 10).sum()):>2}/{len(r)}  hit@50 {int((r <= 50).sum()):>2}/{len(r)}"
              f"  hit@100 {int((r <= 100).sum()):>2}/{len(r)}  median {int(np.median(r)):>5}")

    def fit_score(train, test, seed, names, graph):
        clf, mu, sd = fit_standardized(F, train, seed, names, graph=graph)
        return [F.rank_of(hi, ps, clf.decision_function((vectorize(F, hi, F.pool(hi), names, graph) - mu) / sd))
                for hi, ps, _ in test]

    def summarize(name, make):
        runs = np.array([make(s) for s in SEEDS])
        h10 = (runs <= 10).sum(1); h50 = (runs <= 50).sum(1); h100 = (runs <= 100).sum(1)
        med = np.median(runs, axis=1)
        print(f"  {name:<26}hit@10 {h10.mean():4.1f} [{h10.min()}-{h10.max()}]  hit@50 {h50.mean():4.1f} [{h50.min()}-{h50.max()}]"
              f"  hit@100 {h100.mean():4.1f} [{h100.min()}-{h100.max()}]  median {np.median(med):.0f}")

    def without(pairs, headings):
        return [p for p in pairs if p[0] not in headings]

    def loocv(seed, names, P):
        """LOOCV on P; the dsknn feature always reads the shipped union GRAPH,
        minus the held-out heading, so it matches what the browser will see."""
        out = []
        for j in range(len(P)):
            out += fit_score([p for k, p in enumerate(P) if k != j], [P[j]], seed, names, without(GRAPH, {P[j][0]}))
        return out

    def split(seed, names, P):
        ev = [p for k, p in enumerate(P) if k % 2 == 0]
        od = [p for k, p in enumerate(P) if k % 2 != 0]
        return (fit_score(ev, od, seed, names, without(GRAPH, {p[0] for p in od}))
                + fit_score(od, ev, seed, names, without(GRAPH, {p[0] for p in ev})))

    def cold_rate(P):
        """LOOCV: held-out headings whose fold graph is empty (cold-start fallback)."""
        return sum(1 for (hi, _, _) in P if dsknn_graph(V, without(GRAPH, {hi}), hi)[0] is None)

    def experiment(fixtures, label):
        _, pairs_loose, pairs_canon = build_pairs(catalog, fixtures)
        n = len(pairs_canon)
        print(f"\n######## {label}: {n} headings, {n} strict / {len(pairs_loose)} loose; "
              f"union graph {len(GRAPH)} headings; empty-graph (LOOCV) {cold_rate(pairs_canon)}/{n} ########")
        variants = (("all features", F.NAMES), ("all minus italic", no_italic),
                    ("all + dsknn", F.ALL), ("all + dsknn minus italic", dsknn_minus_italic))
        for P, plabel in ((pairs_loose, "LOOSE: any instance of an acceptable partner family"),
                          (pairs_canon, "STRICT: canonical 400-normal instance per acceptable partner")):
            print(f"\n==== {label} | {plabel} ====")
            print("single features / hand (no training):")
            report("hand metric", [F.rank_of(hi, ps, hand_score(F, hi, F.pool(hi))) for hi, ps, _ in P])
            for name in F.NAMES:
                report(name, [F.rank_of(hi, ps, F.column(hi)[name][F.pool(hi)]) for hi, ps, _ in P])
            report("dsknn (union graph)", [F.rank_of(hi, ps, dsknn_score(V, GRAPH, hi, F.pool(hi))) for hi, ps, _ in P])
            print("learned, LOOCV:")
            for nm, names in variants:
                summarize(nm, lambda s, names=names, P=P: loocv(s, names, P))
            print("learned, held-out even/odd split:")
            for nm, names in variants:
                summarize(nm, lambda s, names=names, P=P: split(s, names, P))

    experiment(FIXTURES, "endorsement (fonts-pairings.json)")
    experiment(USAGE_FIXTURES, "usage (fontsinuse-pairings.json)")

    print("\nshipped input-space weights (mean over seeds, union graph, strict positives):")
    w, b, sd = fit_model(F, GRAPH, SEEDS, F.ALL)
    print(f"  bias {b:+.3f}")
    for n, wi, si in sorted(zip(F.ALL, w, sd), key=lambda t: -abs(t[1])):
        print(f"  {n:<14}{wi:+.4f}  (sd {si:.4f})")


def fit_and_write():
    catalog, V = load_corpus()
    feats = build_features(catalog)
    F = Features(catalog, V, feats)
    graph = graph_pairs(catalog)
    w, b, _ = fit_model(F, graph, names=F.ALL)
    rng = np.random.default_rng(SEED)
    cases = []
    for hi, ps, excl in graph:
        allowed = np.setdiff1d(F.pool(hi), np.asarray(excl))
        picks = [ps[0]] + list(rng.choice(allowed, size=2, replace=False))
        for ci in picks:
            cases.append({
                "heading": catalog[hi]["label"], "candidate": catalog[ci]["label"],
                "score": round(float(learned_score(F, w, b, hi, np.array([ci]), F.ALL, graph)[0]), 6),
            })
    (SITE_DATA / "reranker.json").write_text(json.dumps({
        "v": 2, "features": F.ALL,
        "weights": [round(float(x), 6) for x in w], "bias": round(b, 6),
        "xhSigma": XH_SIGMA,
        "trained": {"headings": len(graph), "positives": sum(len(ps) for _, ps, _ in graph),
                    "negativePerHeading": 50, "seeds": 8, "date": "2026-10-09"},
        "note": "pointwise logistic re-ranker over embedding + axis features + the DS-kNN pair-graph "
                "feature; dsknn/has_graph are precomputed per heading from site/data/pairgraph.json, "
                "not per-pair features",
    }, indent=2) + "\n")
    CASES.write_text(json.dumps({"note": "JS/Python parity cases for site/js/reranker.js", "cases": cases}, indent=1) + "\n")
    print(f"wrote site/data/reranker.json ({len(F.ALL)} features) and {len(cases)} parity cases")


if __name__ == "__main__":
    if "--fit" in sys.argv:
        fit_and_write()
    else:
        main()
