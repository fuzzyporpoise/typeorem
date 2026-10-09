"""Build the DS-kNN pair graph the browser reads (`site/data/pairgraph.json`).

TSK-020's ship vehicle: the `dsknn` re-ranker feature reads the fold's pair
graph, which the browser cannot rebuild from the corpus, so the pipeline
precomputes it as a compact family-level artifact. Source is the committed
fixtures only (`fonts-pairings.json` + `fontsinuse-pairings.json`), never the
corpus: GF family names, no commercial names, so the artifact carries no
provenance debt. See `.todo/tsk-020-ship-design.md` in the app checkout.

    model/.venv/bin/python model/validate/make_pairgraph.py            # write the artifact
    model/.venv/bin/python model/validate/make_pairgraph.py --cases    # also write parity cases

The graph is family-level because instance ids are catalog row indices and a
corpus rebuild reshuffles them; the runtime resolves each family to its canonical
instance (weight 400 normal, else 400, else first eligible) exactly as
`reranker.canon_id` does.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import paths  # noqa: E402

FIXTURES = [
    ROOT / "model" / "validate" / "fixtures" / "fonts-pairings.json",
    ROOT / "model" / "validate" / "fixtures" / "fontsinuse-pairings.json",
]
CASES = ROOT / "model" / "validate" / "fixtures" / "pairgraph-cases.json"
K1 = K2 = 10
CASE_SEEDS = 8
CASE_CANDS = 6


def canon_id(catalog, family):
    """Canonical instance index for a family: 400-normal, else 400, else first
    eligible. Mirrors reranker.canon_id."""
    cands = [i for i, c in enumerate(catalog) if c.get("eligible") and c["family"] == family]
    if not cands:
        return None
    return next((i for i in cands if catalog[i]["weight"] == 400 and catalog[i]["style"] == "normal"),
                next((i for i in cands if catalog[i]["weight"] == 400), cands[0]))


def build_graph(catalog):
    """Union of both fixtures as {heading family: sorted partner families}."""
    fams = {c["family"] for c in catalog if c.get("eligible")}
    headings = {}
    for fx in FIXTURES:
        for entry in json.loads(fx.read_text())["pairings"]:
            h = entry["heading"]
            if h not in fams:
                continue
            partners = headings.setdefault(h, set())
            for a in entry["acceptable"]:
                if a["family"] in fams and a["family"] != h:
                    partners.add(a["family"])
    return {h: sorted(ps) for h, ps in headings.items() if ps}


def dsknn_column(catalog, V, graph, seed, rows):
    """Runtime DS-kNN column over `rows` for heading index `seed`, from the
    family-level artifact. The Python twin of the JS `dsknnColumn` (Eq. 1,
    arXiv:1811.08015 sec. IV-A): k1 nearest graph headings, then a k2-nearest
    weighted mean of their partners. Returns (scores over `rows`, has_graph)."""
    heads = []
    for entry in graph["headings"]:
        hi = canon_id(catalog, entry["family"])
        partners = [p for p in (canon_id(catalog, f) for f in entry["partners"]) if p is not None]
        if hi is not None and partners and hi != seed:
            heads.append((hi, partners))
    if not heads:
        return np.zeros(len(rows), dtype=np.float32), 0
    sims = np.array([float(V[hi] @ V[seed]) for hi, _ in heads])
    near = np.argsort(-sims, kind="stable")[:K1]
    cand, cw = [], []
    for j in near:
        for p in heads[j][1]:
            cand.append(p)
            cw.append(float(sims[j]))
    if not cand:
        return np.zeros(len(rows), dtype=np.float32), 0
    cand = np.asarray(cand)
    cw = np.asarray(cw, dtype=np.float32)
    uniq, inverse = np.unique(cand, return_inverse=True)
    sim_b = (V[rows] @ V[uniq].T)[:, inverse]  # per unique instance: exact ties, like the JS runtime
    k = min(K2, sim_b.shape[1])
    order = np.argsort(-sim_b, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(sim_b, order, axis=1) * cw[order]
    return top.mean(axis=1).astype(np.float32), 1


def write_cases(catalog, V, doc):
    """A small dsknn parity set (seed label, candidate label, score) plus a
    has_graph flag per seed, so the JS runtime can be pinned against Python."""
    seeds = [entry["family"] for entry in doc["headings"][:CASE_SEEDS]]
    cands = []
    for entry in doc["headings"]:
        for f in entry["partners"]:
            if f not in cands:
                cands.append(f)
            if len(cands) >= CASE_CANDS:
                break
        if len(cands) >= CASE_CANDS:
            break
    rows = np.array([canon_id(catalog, f) for f in cands])
    cases, flags = [], []
    for fam in seeds:
        si = canon_id(catalog, fam)
        scores, has_graph = dsknn_column(catalog, V, doc, si, rows)
        flags.append({"seed": catalog[si]["label"], "hasGraph": int(has_graph)})
        for f, s in zip(cands, scores):
            cases.append({"seed": catalog[si]["label"], "candidate": catalog[canon_id(catalog, f)]["label"],
                          "score": round(float(s), 6)})
    CASES.write_text(json.dumps({
        "note": "DS-kNN (site/js/reranker.js dsknnColumn) vs the Python twin in make_pairgraph.py",
        "graph": "pairgraph.json", "k1": K1, "k2": K2,
        "candidates": [catalog[canon_id(catalog, f)]["label"] for f in cands],
        "flags": flags, "cases": cases,
    }, indent=1) + "\n")
    print(f"wrote {CASES.name}: {len(cases)} cases over {len(seeds)} seeds")


def main():
    paths.require_site("the pair graph is built against the shipped catalog")
    catalog = json.loads((paths.site_data() / "catalog.json").read_text())["instances"]
    graph = build_graph(catalog)
    doc = {
        "v": 1, "k1": K1, "k2": K2, "generated": "2026-10-09",
        "note": "DS-kNN pair graph (arXiv:1811.08015 sec. IV-A) from the committed "
                "fixtures; GF families only. Resolve families to canonical instances at runtime.",
        "headings": [{"family": h, "partners": ps} for h, ps in sorted(graph.items())],
    }
    dest = paths.site_data() / "pairgraph.json"
    dest.write_text(json.dumps(doc, indent=1) + "\n")
    partners = sum(len(ps) for ps in graph.values())
    print(f"wrote {dest}: {len(graph)} headings, {partners} partners, {dest.stat().st_size} bytes")
    if "--cases" in sys.argv:
        from metric_ab import load_corpus
        _, V = load_corpus()
        write_cases(catalog, V, doc)


if __name__ == "__main__":
    main()
