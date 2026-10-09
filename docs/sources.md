# Where the ideas came from

Typeorem is built on a literature sweep: read a paper, take exactly one piece of
it, wire that piece into the pipeline, and measure whether it beats what is already
there. Most pieces do not survive. This page is the map, so a follower does not have
to reconstruct it from the harnesses, the commit log, and the papers.

Each entry is: the study, the one thing we took from it, where it lives in this repo,
and the verdict. The anchor is Choi and Hyun, "Typeface network and the principle of
font pairing" (Sci Rep 14:30820, 2024); the sweep around it is public as a set of
citations below.

## How a verdict is decided

Verdicts are not opinions; each one is a harness run on a committed fixture:

- the **strict canonical rule** on the multi-partner benchmark
  (`model/validate/benchmark.mjs`, fixture `fonts-pairings.json`): a partner family
  counts only at its canonical 400-normal instance;
- **LOOCV and a held-out even/odd split** for anything learned
  (`model/validate/reranker.py`), so a number is out-of-sample;
- the **frequency null model** for the shipped metric
  (`model/validate/nullmodel.mjs`): real pairs must beat frequency-matched random
  pairs before a claim means anything.

A corpus change (embedding or PCA) is a tagged release, so it is not a cheap A/B;
metric-side changes are cheap. That asymmetry decides most of what follows.

## Kept

| study | piece we took | where it lives | verdict |
| --- | --- | --- | --- |
| fontjoy (MIT) | the contrast-product metric `\|N\| * P` and the "pair a header with a body" framing | `metric_ab.py` (resolved formula + parity), and the consumer checkout's `site/js/vectors.js` | the base metric everything is compared against |
| Visual Font Pairing, arXiv:1811.08015 | **dual-space k-NN** (sec. IV-A, Eq. 1): a heading's partners from the k nearest graph headings | `reranker.py` (`dsknn_score`, `graph_pairs`), `make_pairgraph.py`, consumer `site/js/reranker.js` (`dsknnColumn`) | kept: as a re-ranker feature it lifts strict hit@10 from 24.6 to 30.6 on the endorsement fixture (LOOCV, LOOCV mean over seeds) |
| the anchor, Sci Rep 2024 | the fontsinuse **usage pairs** (supplementary CSV) as a second, non-curated fixture, plus the paper's null-model test | `mine-fontsinuse.mjs`, `fixtures/fontsinuse-pairings.json`, `nullmodel.mjs` | kept as a stress test, not the yardstick: the shipped metric shows no separation from the frequency null on usage slices |

## Measured, then closed

| study | piece we took | where it lives | verdict |
| --- | --- | --- | --- |
| Visual Font Pairing, arXiv:1811.08015 | **ASML** (sec. IV-B): the learned asymmetric bilinear metric `x'Gy - (x-y)'M(x-y)` | `asml.py` | closed: beats the hand metric but loses to the existing logistic re-ranker on identical labels |
| FontFusion, arXiv:2606.06066 | the **dual encoder** (DINOv2 + FontCLIP concatenated, PCA-refit) | `make_space.py` (`dual`), `embed.py` | closed: the contrast metric needs one opinion per dimension; mixing two geometries is the worst space on every downstream measure |
| FontCLIP, arXiv:2403.06453 | FontCLIP as a **backbone** | `embed.py` (`FontCLIP`), `make_space.py` (`fontclip`), `diagnostics.py` | closed: uniformly high-cosine, poor contrast geometry; DINOv2 stays |
| VGG16 pool5 | the **A/B reference** backbone | `embed.py` (`VGG16Pool5`), `make_space.py` (`vgg16`) | reference only: best category structure, weaker downstream pairing |
| the anchor, Sci Rep 2024 | the **NMF three-axis** decomposition (serif/sans, basic/decorative, light/bold) | compared against the PCA-200 axes in `metric_ab.py` | not adopted: the hand axes are the interpretable subset we kept (dxh, serif, weight) |

## Queued (not built yet)

| study | piece we would take | where it would go |
| --- | --- | --- |
| Microsoft Designer, arXiv:2501.10969 | the **stroke-weight ordering guard** (a subheading must be lighter than its heading) and **stratified novelty** sampling | metric / engine |
| FontBench, arXiv:2603.08497 (on FRB) | a **style-vs-content** diagnostic so a backbone swap cannot silently regress through typography blindness | `diagnostics.py` |
| Visual Font Pairing, arXiv:1811.08015 | the **PDF-mined pairing dataset** to widen the benchmark past 101 headings | `mine-pairings.mjs` |

## Deferred or not adopted

| study | piece | why it waits |
| --- | --- | --- |
| FontMART, ACM DIS 2022 | a **learning-to-rank legibility** proxy | blocked: needs reading-speed data we do not have |
| Paired-glyph Matching, arXiv:2211.10967 | a **contrastive fine-tune** objective for the embed stage | not adopted: it is a corpus release (hours + a tag), so it is not a cheap A/B |
| Impression-word embedding, arXiv:2508.18825 | a **semantic/impression axis** | deferred: add an axis only if it carries signal past the x-height gate |
| Large-scale tag retrieval, arXiv:1909.02072 | the **tag dataset** and the render-and-learn trick | deferred with the axis; the training trick is already satisfied by owning our renders |
| Font-Agent / DFD, CVPR 2025 | a large annotated **font-quality dataset** | deferred: possible legibility training data |
| DeepFont, arXiv:1507.03196 | the discriminative **font encoder** | background: the non-DINOv2 half of the FontFusion idea |
| Contextual Font Recommendations, arXiv:2306.08188 | an **intent-to-font** recommender architecture | background: product direction, not a metric piece |

## The method, in one line

Grab one piece per paper, give it a home in the harnesses, and let the strict
benchmark or the null model say yes or no. The papers that were closed are as
useful as the one that was kept: they are the ones a follower would otherwise
re-try first.
