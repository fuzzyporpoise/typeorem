# Metric validation harness

Resolves and validates the shipped pairing metric, fontjoy's "contrast
similarity", by replaying fontjoy's own reference vectors. See
`.todo/implementation-plan.md` section 5.1.

## The formula (resolved 2026-10-07)

fontjoy's readme links to `formula.png`; OCR of that image plus the method page
at `https://fontjoy.com/pairing/` give:

    contrast similarity = -N * P
    P = sum_i A_i B_i  where A_i B_i > 0
    N = sum_i A_i B_i  where A_i B_i < 0

With `N <= 0` and `P >= 0` this equals `|N| * P`, so the score is large only when
a pair is similar in some dimensions and different in others in equal measure.
This is the `product` form, shipped as `contrastSimilarity` in
`site/js/vectors.js`.

## Fixtures (`fixtures/`)

- `metadata.tsv` (committed): 1,883 rows, columns `name`, `variant`, `category`.
- `vectors-200.tsv` (gitignored, 9.6 MB): the 200-dim vectors, row-aligned to
  `metadata.tsv`.
- `reference.json`: the resolved formula, the homepage default (a hardcoded
  control, not ground truth), and the seeds for the structural check.
- `checksums.txt`: sha256 of both tsv files.

Both tsv files come from the MIT-licensed `Jack000/fontjoy` repo. Fetch and
verify with `node model/validate/fetch-fixtures.mjs`.

## What the harness reports

- The resolved formula and the shipped form.
- A control: the homepage default triple, which is hardcoded in the shipped HTML
  (`data-family` / `data-variant`), so its ranks are a sanity check and not an
  assertion. Under the shipped form it ranks Lora italic 37th for Montserrat 700,
  against 744 under `min`.
- A structural check: for several seeds, the share of top-K partners in a
  contrasting category, plus the largest single-family share (crowding).
- Top-K partners under the shipped form.

## Run

    node model/validate/fetch-fixtures.mjs
    node model/validate/replay.mjs
    node model/validate/replay.mjs --seed "Lora regular" --top 10
    node model/validate/replay.mjs --form min
    node model/validate/reranker.test.mjs

The harnesses need the site checkout (TSK-016), because they score the metric the
browser ships and read the corpus the site serves. `TYPEOREM_SITE_DIR` wins when set;
otherwise the first of `<this repo>/site`, `../site`, `../typeorem-site` that holds an
`index.html` is used, exactly as `model/paths.py` resolves it for the Python side.

The vector math itself is unit-tested in the site's `tests/vectors.test.mjs`, and the
re-ranker's JS/Python parity in `reranker.test.mjs` here.

## Metric challengers (`metric_ab.py`, `benchmark.mjs`)

`benchmark.mjs` scores the multi-partner benchmark
(`fixtures/fonts-pairings.json`, 101 headings x 505 publicly endorsed partners)
with the legacy embedding contrast (`product`) and the shipped metric
(`roleScore`: role legibility plus a gated serif/skeleton axis term), plus the
shipped metric over the body-weight guarded pool (the tool restricts body
recommendations to text weights). The match rule is **strict**: a partner
family counts only at its canonical 400-normal instance. Per heading it reports
**hit@k** (at least one acceptable partner ranks <= k), **recall@k** (fraction
of acceptable partners ranked <= k), and **nDCG@k** (binary-relevance), plus
the median best-partner rank.

`metric_ab.py` is the challenger A/B that produced the shipped metric: it scores
the original 40 pairs (`fixtures/known-pairings.json`, kept as the mining seed)
under the two candidate directions (asymmetric role metric, multi-axis metric)
and their hybrid, plus diagnostics on the documented failure cases and an
odd/even split for robustness. See `.todo/metric-research.md` for the results
tables and the decision.

`mine-pairings.mjs` rebuilds the benchmark fixture from public sources
(fontpair.co's API, listicles, pairing tools, the seed pairs); every name is
validated against the eligible families in `site/data/catalog.json`. The
fixture is committed; re-mining refreshes it from live sources.

    node model/validate/mine-pairings.mjs   # re-mine the fixture (network)
    node model/validate/benchmark.mjs
    model/.venv/bin/python model/validate/metric_ab.py

## Learned re-ranker (`reranker.py`, candidate, not shipped)

`reranker.py` is the "attack hit@10 with a learned re-ranker" experiment: a
pointwise logistic model over 10 heading-relative features, trained and
evaluated on the multi-partner benchmark (every acceptable partner is a
positive; negatives are sampled outside the acceptable partners' families) by
LOOCV and a held-out even/odd split. It beats the hand metric out-of-sample but
is not wired into the app; see `.todo/metric-research.md` for the numbers.
`--fit` writes `site/data/reranker.json`, consumed by `site/js/reranker.js` and
pinned by `tests/reranker.test.mjs`.

    model/.venv/bin/python model/validate/reranker.py          # evaluate
    model/.venv/bin/python model/validate/reranker.py --fit     # refit + export
