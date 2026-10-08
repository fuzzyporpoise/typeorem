# Typeorem: the science

The corpus pipeline and the validation harnesses behind
[typeorem.dev](https://typeorem.dev/), a font-pairing tool and a unified theory of type
design. This repo is the science: it turns the Google Fonts catalog into a 200-dim
embedding space and measures whether the pairing metric built on top of it is any good.
It is an expansion of where [fontjoy](https://github.com/Jack000/fontjoy) (MIT) started.

The site is the other half, in its own repo (`fuzzyporpoise/typeorem` on GitLab): the
static app that serves the corpus and the deployed page. The science writes the corpus;
the site serves it.

```
Google Fonts  ->  glyph grid  ->  DINOv2 ViT-S/14  ->  PCA (200d)  ->  site/data
```

## Why this exists

Fontjoy paired fonts with a cosine split over 2017 features. This project asks the next
question: can a modern frozen vision backbone, a corpus of every pairing-eligible Google
Fonts instance, and a role-aware objective beat that, and can the answer be checked
rather than asserted? The harnesses here are the check: a multi-partner benchmark mined
from public pairing sources, an engine evaluation, a challenger A/B, and a learned
re-ranker.

## Layout

```
model/
  catalog.py          google/fonts METADATA.pb parser
  instances.py        the instance policy (weight ladder, Latin-pool eligibility, css2)
  render.py           the 6x4 / 23-glyph grid rasterizer
  embed.py            the frozen backbone (DINOv2 ViT-S/14, ImageNet preprocessing)
  reduce.py           PCA to 200 dims + int8 quantization
  legibility.py       the body-legibility proxy
  emit.py             writes the site's site/data
  build_corpus.py     the pipeline entry point (+ the house-face append)
  export_onnx.py      the browser graph (--fp16 for the shipped build)
  publish_model.py    publishes it to Hugging Face, writes the site's model.json
  paths.py            where the app checkout is (TYPEOREM_SITE_DIR)
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

# The app checkout is what the pipeline writes into and the harnesses read from.
git clone git@gitlab.com:fuzzyporpoise/typeorem.git ../typeorem-site
export TYPEOREM_SITE_DIR=../typeorem-site
```

Read `docs/roll-your-own.md` for the full path: the catalog clone, the rebuild, the
house faces, the backbone publish, and the validation. Two things to know up front:

- The pipeline's output is the **site's corpus**, so a rebuild is a commit in the site
  repo. `TYPEOREM_SITE_DIR` (default `../site`) is the only path that crosses between
  the two halves.
- The harnesses score the metric **the browser ships**, so they import `site/js/` from
  the app checkout rather than keeping a copy that could drift.

## Tests

```bash
model/.venv/bin/python model/tests/test_catalog.py             # catalog parse + instance policy
model/.venv/bin/python model/tests/test_corpus_determinism.py  # render determinism (needs a pipeline run)
node model/validate/reranker.test.mjs                          # JS/Python re-ranker parity (needs the app checkout)
```

`.github/workflows/gate.yml` runs the standalone tests on every push and pull request;
`publish-model.yml` runs the backbone export and the Hugging Face upload on a release
(or on demand), with the `HF_TOKEN` secret.

## The model

The shipped backbone is on Hugging Face as
[`fuzzyporpoise/typeorem`](https://huggingface.co/fuzzyporpoise/typeorem): DINOv2
ViT-S/14 exported to ONNX, fp16, 43.5 MB, ImageNet normalization baked into the graph.
It is a DINOv2 derivative, so that artifact is Apache-2.0 under the upstream weights;
see `model/MODEL_CARD.md`. Everything else here is MIT.

## License

Code and corpus: MIT (`LICENSE`). Method: credited to fontjoy (MIT). Google Fonts: OFL.
The bundled house face, [Commit Mono](https://commitmono.com), is SIL OFL 1.1 and lives
in the site repo's `site/fonts/` (the science reads it from there).
