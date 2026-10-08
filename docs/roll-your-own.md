# Roll your own

This repo is a pipeline that turns a font catalog into a font-embedding corpus, plus the
harnesses that measure how well a pairing metric built on that corpus can rank fonts. Nothing
here is specific to one catalog, one metric, or one front-end: point it at your own serving
app, or take just the corpus and build something else.

[typeorem.dev](https://typeorem.dev/) is a font-pairing tool built with this pipeline. It is
one consumer of the corpus, not a dependency: this repo neither contains it nor needs it, and
the guide below is about producing and serving your own.

Everything here is offline: there is no server, and the corpus stops at precomputed vectors.

## What you get

The pipeline's output is a small static corpus your front-end serves. Each instance (a family
at one weight and style) has:

- `vectors.i8.bin` and `vectors.meta.json`: the int8 embedding matrix with its dimension,
  count, and per-row quantization scale, plus the backbone and preprocessing that produced it.
- `catalog.json`: one row per instance, aligned to the vector rows, carrying `family`, `label`,
  `weight`, `style`, `category`, `source`, `subsets`, `eligible`, `css2`, `coords`,
  `legibility`, `xh`, and a `search` string.
- `pca.bin` and `pca.json`: the projection (mean plus components) and its layout, so a face
  added later is projected the same way the catalog was.
- `corpus.version.json`: the contract. The backbone, the preprocessing, the catalog commit, the
  instance-policy version, and the glyph-grid geometry.

That set is all a front-end needs to pair fonts: it holds the vectors, and it can project a
visitor's own font by reproducing the same grid and projection in the browser.

## What you need

| Requirement | Notes |
| :---------- | :---- |
| Python 3.11+ | the pipeline's virtualenv |
| Node 20+ | the JS harnesses (`node model/validate/*.mjs`) |
| A consumer checkout | a directory that holds a `site/index.html`; point `TYPEOREM_SITE_DIR` at it, or keep it next to this repo as `../typeorem-site` |
| git, ~6 GB disk | the Google Fonts clone is the bulk of it |
| No GPU required | CPU embedding works, it is just slower (a few hours for the full catalog) |

```bash
python3 -m venv model/.venv
model/.venv/bin/pip install -r model/requirements.txt
```

The pins are the versions the shipped corpus and the exported graph were built with; keep them
unless you intend to rebuild both.

## The consumer checkout

The pipeline does not decide where the corpus goes: it writes into a checkout you own. Every
path that crosses between the pipeline and that checkout resolves through `model/paths.py`
(Python) and `model/validate/site.mjs` (JS), in the same order:

1. `$TYPEOREM_SITE_DIR`, when set (the checkout root, the directory that holds
   `site/index.html`),
2. `<this repo>`,
3. `<this repo>/..`,
4. `<this repo>/../typeorem-site`.

The first candidate that holds a `site/index.html` wins, so a sibling checkout needs no
configuration:

```bash
export TYPEOREM_SITE_DIR=/path/to/your-front-end   # optional, wins when set
```

The contract is small. The pipeline writes the corpus into `<checkout>/site/data/`, and it
fails with a message naming the variable when no checkout is found. The harnesses import the
metric from `<checkout>/site/js/`, because they score what a browser actually runs rather than
a second copy that could drift; if you are building your own tool, that module is yours to
write.

## Where things live

| Path | What it is |
| :--- | :--------- |
| `model/catalog.py`, `instances.py` | the Google Fonts ingest and the instance policy |
| `model/render.py`, `embed.py`, `reduce.py` | the glyph grid, the frozen backbone, PCA |
| `model/legibility.py`, `emit.py` | the body-legibility proxy and the corpus writer |
| `model/build_corpus.py` | the pipeline entry point |
| `model/export_onnx.py`, `publish_model.py` | the browser backbone: export and publish |
| `model/validate/` | the metric harnesses (see its own `README.md`) |
| `model/tests/` | pipeline tests |
| `model/MODEL_CARD.md` | the model card for the published backbone |
| the consumer checkout | `site/` (your front-end), `site/data/` (the generated corpus) |

## The pipeline

Nine stages, in order. Each is a plain module you can run or import on its own.

1. **Catalog** (`model/catalog.py`). The source of truth is a clone of
   [`google/fonts`](https://github.com/google/fonts), parsed from each family's
   `METADATA.pb` by a tolerant protobuf-text parser. Cross-checked against
   `fonts.google.com/metadata/fonts` (one JSON blob, no API key).
2. **Instance policy** (`model/instances.py`). Turns families into the instances that
   are actually loadable and pair-eligible: the weight ladder times the styles the
   family serves, a Latin-pool eligibility test (so Noto Sans JP and friends are out),
   and stable ids of the form `<family-slug>-<weight>-<style>`. It also derives the
   `css2` query fragment per instance, so a front-end never guesses axis syntax.
3. **Render** (`model/render.py`). Rasterizes a 6x4 grid of 23 distinguishing glyphs
   (`ABGJQRSW aegnosrt 138 &@.,`) to a 224x224 greyscale PNG per instance, size 38,
   baseline anchored. Variable instances are instanced through `fontTools` axes first.
4. **Embed** (`model/embed.py`). The frozen backbone: **DINOv2 ViT-S/14**
   (timm `vit_small_patch14_dinov2.lvd142m`), ImageNet normalization, 384 output dims.
5. **Reduce** (`model/reduce.py`). PCA to **200 dims**, then int8 quantization with a
   per-row scale.
6. **Legibility** (`model/legibility.py`). A body-legibility proxy fitted from the
   embeddings, emitted as a per-instance `legibility` field a role term can read (null on
   the bundled house faces, where a centroid fallback covers the body term).
7. **Emit** (`model/emit.py`). Writes the corpus into `<checkout>/site/data/`:
   `vectors.i8.bin`, `vectors.meta.json` (dim, count, per-row int8 scale, backbone,
   preprocess), `catalog.json` (one row per instance, including `source`, `css2`,
   `coords`, `legibility`, `xh`), `pca.bin` + `pca.json` (the projection and its
   layout), and `corpus.version.json`, which carries the corpus contract (backbone,
   preprocess, and the `glyphGrid` geometry). It also writes `model/out/pca_model.npz`
   (the saved projection) and `model/out/render-hashes.json`.
8. **House faces** (`model/build_corpus.py --house`). Appends the checkout's self-hosted
   `local`-source faces, projected through the **saved** PCA (never refit on them). See
   below.
9. **Backbone export** (`model/export_onnx.py`, `model/publish_model.py`). Optional, and
   only needed if your front-end pairs a visitor's own font.

### Rebuild the corpus

```bash
mkdir -p model/.cache
git clone --filter=blob:none https://github.com/google/fonts model/.cache/google-fonts
curl -o model/.cache/gf_meta.json https://fonts.google.com/metadata/fonts

model/.venv/bin/python model/instances.py       # catalog + instance policy -> model/out
model/.venv/bin/python model/build_corpus.py    # render -> embed -> PCA -> emit, + house faces
```

The pipeline writes into the consumer checkout, which need not track the result: the corpus can
ship as a release (next section), so nothing generated is committed here. Keep licensed faces,
and anything derived from them, out of the tree entirely (see the data boundary below).

The clone is blobless: the working tree is checked out and font blobs are fetched on
demand, which is why the first `build_corpus.py` run spends its time on the network.
Renders land in `model/.cache/224/` (gitignored, one PNG per instance id); an instance
that fails to render is reported and dropped from the corpus rather than failing the
run. `model/.cache/` and the virtualenv are gitignored.

The catalog is pinned by commit in `corpus.version.json` (`googleFontsCommit`), so a
rebuild is reproducible; a scheduled monthly diff-and-PR job is the intended refresh
path, not a per-session step.

### Ship the corpus (cut a release)

If your app deploys from a pinned, fetched artifact rather than a tracked tree, a rebuilt
corpus reaches it through one command:

```bash
model/.venv/bin/python model/release_corpus.py --tag corpus-v2 --repo owner/name
```

It validates the generated tree, packs a deterministic tar.gz (`site/data/*` plus
`site/model.json`), publishes it as a release asset, and writes a `corpus.lock.json` at the
consumer checkout root (the tag, the archive's sha256, and the sha256 of every member).
`--repo` defaults to this project's repo; point it at your own. `typeorem.dev` fetches and
verifies the locked tag before it deploys; how your app does the same is up to you.

The checks are the point, and they run before anything is uploaded:

- the file set is exactly what a front-end serves;
- `corpus.version.json`, `vectors.meta.json`, and `model.json` agree with this
  checkout on backbone and preprocessing, and the `glyphGrid` block equals
  `render.py`'s grid;
- counts, byte sizes, and the PCA layout agree, and catalog ids are row indices;
- every catalog row's `source` is in the public allowlist: the provenance gate, run
  again at the boundary where data leaves the machine.

`--dry-run` validates only, and `--no-upload` writes the archive and the lock without
touching GitHub. Tags are immutable handles here: the command refuses a tag that
already exists, so pick a new one (`corpus-v2`) rather than moving a release. The
release is named after its tag and its single asset is `<tag>.tar.gz`, so the tag is
the only handle anyone needs: nothing downstream has to unwrap a decorated label.

### The house faces (a `local` source)

The reference app bundles Commit Mono, a face that is **not** on Google Fonts, so the ingest
never sees it. The pipeline can fold such a self-hosted face in as a `source: "local"` family,
which is the pattern to follow for any face you ship yourself: the woff2 is decompressed to a
TTF (PIL cannot read woff2; `brotli` is pinned for `fontTools`), rendered through the same
glyph grid, embedded with the frozen backbone, and projected through the saved catalog PCA.

```bash
model/.venv/bin/python model/build_corpus.py --house   # append house faces only (idempotent)
```

It reads the face from `<checkout>/site/fonts/CommitMono-VF.woff2` and skips the append with a
message when that file is absent, so a build with no bundled face is fine. Never refit the PCA
on locally added faces: a refit shifts every shipped vector. Rows like these carry
`css2: null` so the front-end skips the CDN and uses the self-hosted `@font-face`, and no
legibility score so the centroid fallback covers the body term.

### Publish the browser backbone (only when it changes)

A front-end can pair a visitor's own font by reproducing this pipeline in the browser:
rasterize the same glyph grid, embed with the exported backbone, and project through the saved
PCA. The graph is 43.5 MB, so it is not committed anywhere: it lives on Hugging Face and a
`model.json` pointer records where.

```bash
model/.venv/bin/python model/export_onnx.py --fp16    # -> <checkout>/site/models/
model/.venv/bin/python model/publish_model.py         # upload + write the checkout's model.json
```

`.github/workflows/publish-model.yml` runs exactly those two commands on a published
release (or on demand) with the `HF_TOKEN` secret, and prints the pointer when no site
checkout is present. Publishing is manual and rare: republish only when the corpus
backbone or preprocess changes, or the export recipe does (a backbone swap). fp16 is
the shipped build: measured identical to fp32 (embedding cosine 1.0000) at half the
size. int8 is not an option; it wrecks the embedding (cosine 0.86). A front-end test that
compares `model.json` against `corpus.version.json` is the tripwire: it fails while the two
disagree, which is exactly what a backbone swap looks like before the republish.

## Validate before you trust a change

The harnesses live in `model/validate/` and are documented in
`model/validate/README.md`. They read the metric and the corpus from the consumer
checkout, because that is what ships.

```bash
node model/validate/benchmark.mjs        # hit@k / recall@k / nDCG@k on the mined pairing fixture
node model/validate/mine-pairings.mjs    # re-mine that fixture from public sources (network)
node model/validate/evaluate.mjs         # engine: legible-body rate, balance, spread
node model/validate/reranker.test.mjs    # JS/Python re-ranker parity + the endorsed-partner check
model/.venv/bin/python model/validate/reranker.py   # the learned re-ranker candidate
```

`model/render.py` plus the saved PCA are the ground truth for the browser path. The
grid geometry is enforced through the shipped corpus instead of through a second copy
of the constants: the pipeline writes `corpus.version.json.glyphGrid` and a
front-end's rasterizer asserts its copy matches that contract, so the numeric coupling lives
in one committed file. For a local font, run it through the pipeline and compare the
200-dim vector and its top partners against what the browser reports.

## Change the metric

The reference app ships `roleScore` in its `site/js/engine.js`: embedding contrast plus a
role-legibility term plus a gated serif/skeleton axis term, with its tuning constants and the
3-slot joint objective behind `generate` all in that one file. The metric is part of what
ships, so it lives in the front-end, and this repo owns the evidence that it is any good. The
harnesses import the metric module directly, so a change is measurable without a browser. Any
metric change must move `benchmark.mjs` and must not regress `evaluate.mjs`.

## The data boundary (read this before shipping data)

Every row in the corpus's `catalog.json` carries an explicit `source`: `google` (the ingested
OFL catalog) or `local` (faces you bundle yourself). Keep that allowlist explicit and check it
in your own app, the same way `release_corpus.py` re-checks every row before it uploads.

The rule behind it: **a vector derived from a commercially licensed font must never reach a
published corpus**, because a published corpus is a redistributed artifact. Keep licensed
faces, and anything derived from them, out of the repo entirely; adding a row is a deliberate
act, and the boundary check will catch it if you get it wrong.

## License and attribution

- Code and corpus: MIT (`LICENSE`).
- Method: credited to [fontjoy](https://github.com/Jack000/fontjoy) (MIT).
- Browser backbone: a DINOv2 ViT-S/14 derivative, Apache-2.0 under the upstream
  weights; that artifact is the Hugging Face model repo plus `model/MODEL_CARD.md`, not
  the MIT repo.
- Google Fonts: OFL, loaded at runtime from the CSS2 CDN (specimens) and baked into the
  corpus as vectors.
- A bundled house face, [Commit Mono](https://commitmono.com)
  (`site/fonts/CommitMono-VF.woff2`): SIL OFL 1.1. Check the license of any face you
  self-host the same way.

## Deploying

Deployment belongs to your front-end, not to this repo. This repo has no deploy step; its CI
runs the pipeline tests and the backbone publish.
