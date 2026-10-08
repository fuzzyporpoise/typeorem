# Roll your own

Typeorem is two pieces:

- **the science** (this repo): the corpus pipeline and the validation harnesses,
  the part that is an expansion of where [fontjoy](https://github.com/Jack000/fontjoy)
  started, and
- **the site** (a separate checkout, `fuzzyporpoise/typeorem` on GitLab): the static
  app that serves the hero specimen and the 3-slot pairing tool at
  [typeorem.dev](https://typeorem.dev/), and the home of the deployed page.

The science produces the corpus the site serves; the site never writes back. Both
halves are needed to run the full validation, because the harnesses score the metric
the browser actually ships.

Everything here is offline: there is no server, and the hosted site stops at
precomputed vectors.

## What you need

| Requirement | Notes |
| :---------- | :---- |
| Python 3.11+ | the pipeline's virtualenv |
| Node 20+ | the JS harnesses (`node model/validate/*.mjs`) |
| An app checkout | `git clone git@gitlab.com:fuzzyporpoise/typeorem.git ../typeorem-site` next to this repo, or point `TYPEOREM_SITE_DIR` at it |
| git, ~6 GB disk | the Google Fonts clone is the bulk of it |
| No GPU required | CPU embedding works, it is just slower (a few hours for the full catalog) |

```bash
python3 -m venv model/.venv
model/.venv/bin/pip install -r model/requirements.txt
```

The pins are the versions the shipped corpus and the exported graph were built with;
keep them unless you intend to rebuild both.

### The two checkouts

Every path that crosses between the pieces resolves through `model/paths.py` (Python)
and `model/validate/site.mjs` (JS), in the same order:

1. `$TYPEOREM_SITE_DIR`, when set (the app checkout root, the directory that holds
   `site/index.html`),
2. `<this repo>`,
3. `<this repo>/..`,
4. `<this repo>/../typeorem-site`.

The first candidate that holds a `site/index.html` wins, so a sibling clone needs no
configuration:

```bash
export TYPEOREM_SITE_DIR=/path/to/typeorem   # optional, wins when set
```

The pipeline writes the corpus into `$TYPEOREM_SITE_DIR/site/data/`, reads the house
face from `$TYPEOREM_SITE_DIR/site/fonts/`, and the harnesses import the metric from
`$TYPEOREM_SITE_DIR/site/js/`. Anything that needs the site fails with a message that
names the variable when it is missing.

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
| the site checkout | `site/` (the app), `site/data/` (the shipped corpus), `tests/` (its gate) |

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
   `css2` query fragment per instance, so the site never guesses axis syntax.
3. **Render** (`model/render.py`). Rasterizes a 6x4 grid of 23 distinguishing glyphs
   (`ABGJQRSW aegnosrt 138 &@.,`) to a 224x224 greyscale PNG per instance, size 38,
   baseline anchored. Variable instances are instanced through `fontTools` axes first.
4. **Embed** (`model/embed.py`). The frozen backbone: **DINOv2 ViT-S/14**
   (timm `vit_small_patch14_dinov2.lvd142m`), ImageNet normalization, 384 output dims.
5. **Reduce** (`model/reduce.py`). PCA to **200 dims**, then int8 quantization with a
   per-row scale.
6. **Legibility** (`model/legibility.py`). A body-legibility proxy fitted from the
   embeddings, shipped as a per-instance `legibility` field the engine's role term
   reads (null on the self-hosted house faces, where the engine's centroid fallback
   covers the body term).
7. **Emit** (`model/emit.py`). Writes `site/data/` in the site checkout:
   `vectors.i8.bin`, `vectors.meta.json` (dim, count, per-row int8 scale, backbone,
   preprocess), `catalog.json` (one row per instance, including `source`, `css2`,
   `coords`, `legibility`, `xh`), `pca.bin` + `pca.json` (the projection and its
   layout), and `corpus.version.json`, which carries the corpus contract the site's
   gate reads (backbone, preprocess, and the `glyphGrid` geometry). It also writes
   `model/out/pca_model.npz` (the saved projection) and `model/out/render-hashes.json`.
8. **House faces** (`model/build_corpus.py --house`). Appends the site's self-hosted
   `local`-source faces, projected through the **saved** PCA (never refit on them).
9. **Backbone export** (`model/export_onnx.py`, `model/publish_model.py`). Optional,
   only for the in-browser custom-font path.

### Rebuild the corpus

```bash
mkdir -p model/.cache
git clone --filter=blob:none https://github.com/google/fonts model/.cache/google-fonts
curl -o model/.cache/gf_meta.json https://fonts.google.com/metadata/fonts

model/.venv/bin/python model/instances.py       # catalog + instance policy -> model/out
model/.venv/bin/python model/build_corpus.py    # render -> embed -> PCA -> emit, + house faces
```

The pipeline writes into the app checkout, which does not track the result: the
corpus ships as a release (next section), and the app's own gate re-checks whatever
it fetched (`tests/provenance.test.mjs` is the tripwire for the data boundary below).

The clone is blobless: the working tree is checked out and font blobs are fetched on
demand, which is why the first `build_corpus.py` run spends its time on the network.
Renders land in `model/.cache/224/` (gitignored, one PNG per instance id); an instance
that fails to render is reported and dropped from the corpus rather than failing the
run. `model/.cache/` and the virtualenv are gitignored.

The catalog is pinned by commit in the app's `corpus.version.json`
(`googleFontsCommit`), so a rebuild is reproducible; a scheduled monthly diff-and-PR
job is the intended refresh path, not a per-session step.

### Ship the corpus (cut a release)

The app deploys whatever release it pins, and nothing generated is tracked there, so a
rebuilt corpus reaches the site through one command:

```bash
model/.venv/bin/python model/release_corpus.py --tag corpus-v2
```

It validates the generated tree, packs a deterministic tar.gz (`site/data/*` plus
`site/model.json`), publishes it as a release asset on this repo, and writes the app's
`corpus.lock.json` (the tag, the archive's sha256, and the sha256 of every member).
Commit the lock in the app repo and deploy there; `npm run corpus` fetches and verifies
it locally, and the app's CI does the same before the gate and before publishing Pages.

The checks are the point, and they run before anything is uploaded:

- the file set is exactly what the app serves;
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

Commit Mono is the fuzzyporpoise house face and is **not** on Google Fonts, so it is
not in the ingested catalog. It ships self-hosted in the site's `site/fonts/`, and the
corpus appends it as a `source: "local"` family: the woff2 is decompressed to a TTF
(PIL cannot read woff2; `brotli` is pinned for `fontTools`), rendered through the same
glyph grid, embedded with the frozen backbone, and projected through the saved catalog
PCA.

```bash
model/.venv/bin/python model/build_corpus.py --house   # append house faces only (idempotent)
```

Never refit the PCA on locally added faces: a refit shifts every shipped vector. Rows
like these carry `css2: null` so the site skips the CDN and uses the self-hosted
`@font-face`, and no legibility score so the engine's centroid fallback covers the
body term.

### Publish the browser backbone (only when it changes)

The site pairs a visitor's own font by reproducing this pipeline in the browser
(`site/js/glyphgrid.js` -> `embed.js` -> `project.js`). The graph is 43.5 MB, so it is
not committed anywhere: it lives on Hugging Face and the site's `site/model.json`
records the pointer.

```bash
model/.venv/bin/python model/export_onnx.py --fp16    # -> $TYPEOREM_SITE_DIR/site/models/
model/.venv/bin/python model/publish_model.py         # upload + write the site's model.json
```

`.github/workflows/publish-model.yml` runs exactly those two commands on a published
release (or on demand) with the `HF_TOKEN` secret, and prints the pointer when no site
checkout is present. Publishing is manual and rare: republish only when the corpus
backbone or preprocess changes, or the export recipe does (a backbone swap). fp16 is
the shipped build: measured identical to fp32 (embedding cosine 1.0000) at half the
size. int8 is not an option; it wrecks the embedding (cosine 0.86). The site's
`tests/model.test.mjs` is the tripwire: it fails while `site/model.json` and
`corpus.version.json` disagree, which is exactly what a backbone swap looks like
before the republish.

## Validate before you trust a change

The harnesses live in `model/validate/` and are documented in
`model/validate/README.md`. They read the metric and the corpus from the site
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
of the constants: the pipeline writes `corpus.version.json.glyphGrid` and the site's
gate (`tests/glyphgrid.test.mjs`) asserts its rasterizer matches that contract, so the
numeric coupling lives in one committed file. For a local font, run it through the
pipeline and compare the 200-dim vector and its top partners against what the browser
reports.

## Change the metric

The shipped metric is `roleScore` in the site's `site/js/engine.js` (embedding contrast
plus a role-legibility term plus a gated serif/skeleton axis term). The metric is part
of what ships, so it lives in the site repo; this repo owns the evidence that it is any
good. Its tuning constants (`ROLE_LAMBDA`, `SERIF_WEIGHT`, `XH_GATE_SIGMA`) and the
3-slot joint objective behind `generate` are all in that one file, and the harnesses
here import it directly, so a change is measurable without a browser. Any metric change
must move `benchmark.mjs` and must not regress `evaluate.mjs`. The research log,
including the challenger A/B and the re-ranker, is in the site repo's
`.todo/metric-research.md`.

## The public/private boundary (read this before shipping data)

Every row in the site's `site/data/catalog.json` carries an explicit `source`: `google`
(the ingested OFL catalog) or `local` (the bundled house faces). The site's
`tests/provenance.test.mjs` asserts that allowlist and cross-checks the `local` count
against `corpus.version.json.houseFaces`, and it runs inside `npm test`, which is what
the site's CI runs on every push.

That gate is the tripwire for one rule: **a vector derived from a commercially licensed
font must never reach `site/data`**, because a published corpus is a redistributed
artifact. Keep licensed faces, and anything derived from them, out of both repos
entirely; adding a row is a deliberate act that the gate will catch if you get it wrong.

## License and attribution

- Code and corpus: MIT (`LICENSE`).
- Method: credited to [fontjoy](https://github.com/Jack000/fontjoy) (MIT).
- Browser backbone: a DINOv2 ViT-S/14 derivative, Apache-2.0 under the upstream
  weights; that artifact is the Hugging Face model repo plus `model/MODEL_CARD.md`, not
  the MIT repo.
- Google Fonts: OFL, loaded at runtime from the CSS2 CDN (specimens) and baked into the
  corpus as vectors.
- The bundled house face, [Commit Mono](https://commitmono.com)
  (`site/fonts/CommitMono-VF.woff2`): SIL OFL 1.1.

## Deploying

Deployment belongs to the site repo: its `.gitlab-ci.yml` runs the gate on every push
and, on the default branch, copies `site/` into `public/` for GitLab Pages. This repo
has no deploy step; its CI runs the pipeline tests and the backbone publish.
