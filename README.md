# Typeorem

The corpus pipeline and the metric harnesses behind
[typeorem.dev](https://typeorem.dev/), a font-pairing tool. This repo turns the Google Fonts
catalog into a 200-dim font-embedding corpus and measures whether the pairing metric built on
top of it holds up. It is an expansion of where
[fontjoy](https://github.com/Jack000/fontjoy) (MIT) started.

```
Google Fonts  ->  glyph grid  ->  DINOv2 ViT-S/14  ->  PCA (200d)  ->  corpus
```

The corpus is a handful of static files: an int8 vector per font instance, a catalog that
names them, the PCA projection that produced them, and a version file that pins the contract.
[`docs/roll-your-own.md`](docs/roll-your-own.md) covers building one and serving it from your
own front-end.

## Why this exists

Fontjoy paired fonts with a cosine split over 2017 features. This project asks whether a
modern frozen vision backbone, a corpus of every pairing-eligible Google Fonts instance, and a
role-aware objective do better, and whether the answer can be checked rather than asserted.
The harnesses here are the check: a multi-partner benchmark mined from public pairing sources,
an engine evaluation, a challenger A/B, and a learned re-ranker.

## Layout

```
model/
  catalog.py          google/fonts METADATA.pb parser
  instances.py        the instance policy (weight ladder, Latin-pool eligibility, css2)
  render.py           the 6x4 / 23-glyph grid rasterizer
  embed.py            the frozen backbone (DINOv2 ViT-S/14, ImageNet preprocessing)
  reduce.py           PCA to 200 dims + int8 quantization
  legibility.py       the body-legibility proxy
  emit.py             writes the corpus (vectors, catalog, PCA, version)
  build_corpus.py     the pipeline entry point (+ the house-face append)
  export_onnx.py      the browser graph (--fp16 for the shipped build)
  publish_model.py    publishes it to Hugging Face, writes model.json
  paths.py            where the consumer checkout is (TYPEOREM_SITE_DIR)
  release_corpus.py   validates the corpus and publishes it as a tagged release
  MODEL_CARD.md       the model card for the published backbone
  tests/              pipeline tests
  validate/           the metric harnesses (README.md in there)
docs/
  roll-your-own.md    the full guide: rebuild, validate, publish, extend
```

## Quickstart

```bash
git clone git@github.com:fuzzyporpoise/typeorem.git
cd typeorem
python3 -m venv model/.venv
model/.venv/bin/pip install -r model/requirements.txt
```

The pipeline writes its corpus into a front-end checkout: a directory that holds a
`site/index.html`, which it checks for before it writes. Point `TYPEOREM_SITE_DIR` at yours,
or drop one next to this repo as `../typeorem-site`:

```bash
export TYPEOREM_SITE_DIR=/path/to/your-front-end
```

Read `docs/roll-your-own.md` for the full path: the catalog clone, the rebuild, the house
faces, shipping the corpus as a release, the backbone publish, and the validation. Two things
to know up front:

- The corpus is output, not source: nothing generated is tracked here. `emit.py` writes it
  into `$TYPEOREM_SITE_DIR/site/data/`, and `model/release_corpus.py` can pack and validate it
  as a tagged release. `TYPEOREM_SITE_DIR` is the only path that crosses between the pipeline
  and the front-end.
- The harnesses score the metric **your browser ships**, so they import `site/js/` from the
  consumer checkout rather than keeping a copy that could drift.

## Tests

```bash
model/.venv/bin/python model/tests/test_catalog.py             # catalog parse + instance policy
model/.venv/bin/python model/tests/test_release.py             # the release boundary: the corpus checks + the archive
model/.venv/bin/python model/tests/test_corpus_determinism.py  # render determinism (needs a pipeline run)
node model/validate/reranker.test.mjs                          # JS/Python re-ranker parity (needs a consumer checkout)
```

`.github/workflows/gate.yml` runs the standalone tests on every push and pull request;
`publish-model.yml` runs the backbone export and the Hugging Face upload on a release
(or on demand), with the `HF_TOKEN` secret. Cutting a corpus release is a local command
(`model/release_corpus.py`): it needs the Google Fonts clone and the renders, so it
belongs on the machine that built the corpus, not on a runner.

## The model

The shipped backbone is on Hugging Face as
[`fuzzyporpoise/typeorem`](https://huggingface.co/fuzzyporpoise/typeorem): DINOv2
ViT-S/14 exported to ONNX, fp16, 43.5 MB, ImageNet normalization baked into the graph.
It is a DINOv2 derivative, so that artifact is Apache-2.0 under the upstream weights;
see `model/MODEL_CARD.md`. Everything else here is MIT.

## License

Code and corpus: MIT (`LICENSE`). Method: credited to fontjoy (MIT). Google Fonts: OFL. The
bundled house face, [Commit Mono](https://commitmono.com), is SIL OFL 1.1 and lives in the
consumer checkout's `site/fonts/` (the pipeline reads it from there).
