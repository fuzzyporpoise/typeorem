"""The release boundary is a gate, so its checks get a test.

Builds a tiny fake corpus tree and asserts the validator accepts it and rejects the
failures that matter: a licensed row, a count that disagrees with the binary, a
stale glyph grid, and a backbone that does not match this checkout.
"""

import json
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import release_corpus as rel  # noqa: E402
from emit import BACKBONE, PREPROCESS  # noqa: E402
from render import GLYPHS, COLS, ROWS, SIZE, FONT_SIZE, BASELINE_Y, GLYPH_SET_VERSION  # noqa: E402

passes = 0
fails = 0


def ok(label, cond):
    global passes, fails
    if cond:
        passes += 1
    else:
        fails += 1
        print(f"FAIL {label}")


COUNT, DIM, BACKBONE_DIM = 3, 200, 384


def write(root, rel, obj=None, raw=None):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if raw is not None else json.dumps(obj, indent=2).encode())


def fixture(root, **over):
    """A minimal corpus the validator should accept, with overrides per case."""
    scale = over.pop("scale", [0.01] * COUNT)
    count = over.pop("count", COUNT)
    instances = over.pop("instances", [
        {"id": i, "family": f"Fam{i}", "source": "google"} for i in range(count)
    ])
    version = {
        "policyVersion": 3,
        "glyphSetVersion": GLYPH_SET_VERSION,
        "glyphGrid": {
            "glyphs": "".join(GLYPHS), "cols": COLS, "rows": ROWS, "size": SIZE,
            "fontSize": FONT_SIZE, "baselineY": BASELINE_Y,
        },
        "backbone": BACKBONE,
        "preprocess": PREPROCESS,
        "count": count,
    }
    version.update(over.pop("version", {}))
    write(root, "data/vectors.i8.bin", raw=bytes(count * DIM))
    write(root, "data/vectors.meta.json", {
        "v": 1, "dim": DIM, "count": count, "dtype": "int8",
        "scale": {"mode": "perRow", "values": scale},
        "backbone": BACKBONE, "preprocess": PREPROCESS,
    })
    write(root, "data/catalog.json", {"v": 1, "count": count, "instances": instances})
    write(root, "data/pca.bin", raw=bytes((BACKBONE_DIM + DIM * BACKBONE_DIM) * 4))
    write(root, "data/pca.json", {"v": 1, "dim": DIM, "backboneDim": BACKBONE_DIM, "bin": "pca.bin"})
    write(root, "data/corpus.version.json", version)
    write(root, "data/reranker.json", {"v": 1, "features": [], "weights": [], "bias": 0.0})
    write(root, "model.json", {
        "v": 1, "backbone": BACKBONE, "preprocess": PREPROCESS, "dim": BACKBONE_DIM,
        "file": "dinov2_vits14.fp16.onnx", "bytes": 1, "sha256": "0" * 64,
        "repo": "fuzzyporpoise/typeorem", "revision": "main", "url": "https://example.invalid/m.onnx",
    })


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    # The happy path.
    fixture(root)
    summary, failures = rel.validate(root)
    ok("a well-formed corpus validates", failures == [] and summary["count"] == COUNT)
    ok("the summary carries what the notes need",
       summary["backbone"] == BACKBONE and summary["googleFontsCommit"] is None)

    # A licensed row must never leave the machine.
    fixture(root, instances=[{"id": 0, "family": "Fam0", "source": "google"},
                             {"id": 1, "family": "Grilli", "source": "commercial"},
                             {"id": 2, "family": "Fam2", "source": "local"}])
    _, failures = rel.validate(root)
    ok("a non-public source is rejected",
       any("non-public source" in f for f in failures))

    # The house-face count has to agree with the rows.
    fixture(root, version={"houseFaces": {"family": "Commit Mono", "count": 6}})
    _, failures = rel.validate(root)
    ok("a house count that disagrees with the rows is rejected",
       any("houseFaces.count" in f for f in failures))

    # The binary has to match the declared count.
    fixture(root, count=COUNT)
    (root / "data/vectors.i8.bin").write_bytes(bytes(COUNT * DIM + 1))
    _, failures = rel.validate(root)
    ok("a binary that disagrees with count*dim is rejected",
       any("vectors.i8.bin is" in f for f in failures))

    # A stale grid contract would ship a rasterizer that no longer matches.
    fixture(root, version={"glyphGrid": {"glyphs": "ABC", "cols": 1, "rows": 1, "size": 1,
                                         "fontSize": 1, "baselineY": 1}})
    _, failures = rel.validate(root)
    ok("a stale glyph grid is rejected", any("glyphGrid" in f for f in failures))

    # A corpus built by a different backbone must not be released.
    fixture(root, version={"backbone": "vgg16_pool5"})
    _, failures = rel.validate(root)
    ok("a backbone that disagrees with this checkout is rejected",
       any("this checkout emits" in f for f in failures))

    # A missing file is caught before anything is packed.
    fixture(root)
    (root / "data/pca.bin").unlink()
    _, failures = rel.validate(root)
    ok("a missing file is rejected", any("missing" in f for f in failures))

    # The archive is deterministic and its member hashes match the files.
    fixture(root)
    first, members = rel.build_archive(root)
    second, _ = rel.build_archive(root)
    ok("the archive is byte-identical across builds", first == second)
    ok("the members are the corpus plus the pointer",
       set(members) == {f"site/{f}" for f in rel.CORPUS_FILES + [rel.POINTER]})
    ok("member hashes match the files",
       members["site/data/pca.json"]["sha256"] == rel.sha256_bytes((root / "data/pca.json").read_bytes()))

print(f"{passes} passed, {fails} failed")
sys.exit(1 if fails else 0)
