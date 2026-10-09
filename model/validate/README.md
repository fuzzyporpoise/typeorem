# Metric validation harness

Resolves and validates the shipped pairing metric, fontjoy's "contrast
similarity", by replaying fontjoy's own reference vectors.

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

The harnesses need the app checkout (TSK-016), because they score the metric the
browser ships and read the corpus the app serves. `TYPEOREM_SITE_DIR` wins when set
(the app checkout root, the directory that holds `site/index.html`); otherwise the
first of `<this repo>`, `../`, and `../typeorem-site` that holds one is used, exactly
as `model/paths.py` resolves it for the Python side.

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
odd/even split for robustness; run it to reproduce the results tables and the
decision.

`mine-pairings.mjs` rebuilds the benchmark fixture from public sources
(fontpair.co's API, listicles, pairing tools, the seed pairs); every name is
validated against the eligible families in `site/data/catalog.json`. The
fixture is committed; re-mining refreshes it from live sources.

    node model/validate/mine-pairings.mjs   # re-mine the fixture (network)
    node model/validate/benchmark.mjs
    model/.venv/bin/python model/validate/metric_ab.py

## Usage-mined pairs and null models (`mine-fontsinuse.mjs`, `nullmodel.mjs`)

`mine-fontsinuse.mjs` builds a second, usage-based fixture
(`fixtures/fontsinuse-pairings.json`) from the supplementary data of Choi &
Hyun, "Typeface network and the principle of font pairing" (Sci Rep 14:30820,
2024; use cases collected from fontsinuse.com): pairs observed in real design
work where both faces resolve to the shipped catalog. Pairs are unordered, so
each is emitted in both directions, with `uses` and per-medium counts for
slicing. A small documented alias map substitutes open revivals or
metric-compatible faces for commercial names (Futura -> Jost, Arial -> Arimo,
...); contested lookalikes (Helvetica, Univers, Gotham) are not aliased. The
raw CSV is not committed (the article is CC BY-NC-ND); the generator
re-downloads it, or reads `FONTSINUSE_CSV=<path>`.

`benchmark.mjs` scores any fixture via the `FIXTURE` env var. `nullmodel.mjs`
runs the paper's null-model test: real pairs must outscore frequency-conserving
random pairs (the primary baseline, since usage data is popularity-skewed) and
uniform random pairs (secondary). Slices de-skew the usage data: `refined`
keeps repeated, high-lift, non-kin pairs (deliberate choices, not popularity
residue); `endorsed` keeps pairs the curated fixture also endorses (usage plus
judgment, the highest-confidence slice).

This fixture is a stress test, not the yardstick: the curated benchmark remains
that. Current status on the shipped space: against the frequency null the
metric shows no separation on the all, Web, digital, or refined slices
(|z| < 0.1); only the 36-pair endorsed slice separates, and mostly against the
uniform null (z 0.47, AUC 0.62). Real-world usage co-occurrence, even
lift-filtered, is not what the metric tracks; curated endorsement is.

    node model/validate/mine-fontsinuse.mjs   # rebuild the usage fixture (network)
    FIXTURE=fontsinuse-pairings.json node model/validate/benchmark.mjs
    node model/validate/nullmodel.mjs

## Backbone challengers (`make_space.py`, `diagnostics.py`)

`make_space.py <dinov2|vgg16|fontclip|dual>` builds a challenger space aligned
to the shipped catalog's instance order (the index contract the `SPACE_BIN` /
`SPACE_META` / `SPACE_NAME` overrides in benchmark.mjs and nullmodel.mjs rely
on), caching raw backbone features under `.cache`. The PCA is fitted on
google-source rows only; house faces are projected through the fitted basis,
matching the pipeline invariant (never refit on house faces). `fontclip` is
FontCLIP (Tatsukawa et al. 2024, MIT; visual tower of the official ViT-B/32
checkpoint, cached at `.cache/fontclip/model.pt`); `dual` is the FontFusion-style
(arXiv:2606.06066) concat of unit-normalized dinov2 + fontclip blocks, PCA
refit to 200.

A caution on reading benchmark deltas across fits: with identical features and
instance order, refitting the PCA on a different instance snapshot moved the
curated benchmark by several hit@50 points (18 vs 23) while leaving the null
model essentially flat, so differences of that size are basis sensitivity, not
evidence of a better space.

`diagnostics.py` screens spaces with the paper's cluster metrics (silhouette by
category, 1-NN category accuracy, retrieval P@10 over eligible instances).

Result of the 2026-10-09 bake-off (curated benchmark, shipped metric + body
guard; endorsed-slice null model vs uniform):

| space           | 1NN cat | P@10  | hit@50 | median rank | endorsed z / AUC |
|-----------------|---------|-------|--------|-------------|------------------|
| dinov2 (refit)  | 0.864   | 0.775 | 23/101 | 276         | 0.51 / 0.63      |
| vgg16           | 0.906   | 0.816 | 11/101 | 362         | 0.22 / 0.59      |
| fontclip        | 0.831   | 0.774 | 7/101  | 749         | -0.04 / 0.50     |
| dinov2+fontclip | 0.864   | 0.784 | 1/101  | 588         | -0.12 / 0.48     |

The dual-encoder result does not transfer to pairing: the shipped metric's
contrast form (|N| * P over per-dimension signed products) needs each dimension
to carry one opinion, and mixing two geometries per dimension washes that out;
the dual is the worst space on both downstream measures despite mid-tier
cluster metrics. FontCLIP alone, though purpose-built for font retrieval, is
uniformly high-cosine and poor contrast geometry. dinov2 stays the backbone;
vgg16 has the best category structure but weaker downstream scores. The cluster
diagnostics anti-predicted pairing quality, so they remain a screen, never the
decision.

## Learned re-ranker (`reranker.py`, candidate, not shipped)

`reranker.py` is the "attack hit@10 with a learned re-ranker" experiment: a
pointwise logistic model over 10 heading-relative features, trained and
evaluated on the multi-partner benchmark (every acceptable partner is a
positive; negatives are sampled outside the acceptable partners' families) by
LOOCV and a held-out even/odd split. It beats the hand metric out-of-sample but
is not wired into the app; rerun `reranker.py` to reproduce the numbers.
`--fit` writes `site/data/reranker.json`, consumed by `site/js/reranker.js` and
pinned by `tests/reranker.test.mjs`.

    model/.venv/bin/python model/validate/reranker.py          # evaluate
    model/.venv/bin/python model/validate/reranker.py --fit     # refit + export

`reranker.py`'s evaluation also carries the eighth pass's lead (TSK-020): two
more columns, `dsknn` and `has_graph`, from the paper's training-free dual-space
k-NN (arXiv:1811.08015 sec. IV-A Eq. 1, `dsknn_score`: pair with what visually
similar headings pair with, K1 = K2 = 10). The `dsknn` column reads the shipped
pair graph (the union of both fixtures, 191 headings), so it is built per fold,
never from the global feature cache; the graph (the k1 nearest train headings'
partners) is memoized per fold and reused across seeds and feature subsets.
`has_graph` flags an empty fold graph (the cold-start fallback: score 0), which
the model can learn to distrust; it stays 1 on both committed fixtures, so it
only matters for off-graph or custom-font headings at runtime. The run also
cross-checks the usage fixture (`fixtures/fontsinuse-pairings.json`) with the
same folds and hygiene.

Adding `dsknn` beats the shipped-feature re-ranker out-of-sample on both
fixtures, under the strict canonical rule (mean over negative-sampling seeds
0-7; reference seventh-pass rows in the research log). The loose rule regresses
(training on all-instances-of-a-partner-family fights the canonical graph), so
the strict rule, the yardstick, is what the model ships on:

| fixture / rule      | model            | hit@10 | hit@50 | hit@100 | median |
|---------------------|------------------|--------|--------|---------|--------|
| endorsement, LOOCV  | 10 features      | 24.6   | 44.6   | 59.5    | 76     |
| endorsement, LOOCV  | + dsknn          | 30.6   | 60.2   | 76.0    | 31     |
| endorsement, split  | 10 features      | 24.0   | 43.1   | 59.5    | 72     |
| endorsement, split  | + dsknn          | 29.6   | 58.0   | 73.4    | 34     |
| usage, LOOCV        | 10 features      | 15.0   | 43.5   | 60.4    | 140    |
| usage, LOOCV        | + dsknn          | 23.5   | 54.6   | 74.4    | 82     |
| usage, split        | 10 features      | 14.8   | 43.2   | 58.1    | 152    |
| usage, split        | + dsknn          | 20.9   | 52.6   | 67.5    | 116    |

### Ship vehicle (`make_pairgraph.py`, `reranker.js`)

`--fit` now fits the 12-feature model against the shipped union graph and writes
`site/data/reranker.json` (v2) plus `fixtures/reranker-cases.json`. `dsknn`
needs the pair graph at runtime, so `make_pairgraph.py` builds the family-level
`site/data/pairgraph.json` (191 headings, ~1000 partners, 28 KB) from the
committed fixtures only; instance indices never leak into the artifact, the
runtime resolves families to canonical instances. `site/js/reranker.js` computes
one heading's whole-pool column (`resolveGraph` / `dsknnColumn`) as a per-seed
pre-pass `rankLearned` folds in, since it is not a per-pair feature. `--cases`
writes `fixtures/pairgraph-cases.json`; `reranker.test.mjs` pins the JS column
and the 12-feature score against Python. `pairgraph.json` ships as a corpus
release member (`release_corpus.py`), the same channel as `reranker.json`.

    model/.venv/bin/python model/validate/make_pairgraph.py --cases   # graph + parity cases
    model/.venv/bin/python model/validate/reranker.py --fit           # refit the 12-feature model
    node model/validate/reranker.test.mjs                             # cross-repo parity
