"""A4: a private run writes only under the private root.

The writer must leave both checkouts byte-identical. This drives the whole
orchestration (scan -> collect -> render -> embed -> emit) with a stubbed
embedder, over a fake app checkout and a fake science tree, and asserts every
file that changed or appeared is inside the private root. It needs no torch, so
it runs in the standalone gate.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import private_corpus as pc  # noqa: E402

passes = 0
fails = 0


def ok(label, cond):
    global passes, fails
    if cond:
        passes += 1
    else:
        fails += 1
        print(f"FAIL {label}")


def snapshot(root):
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def make_font(path, family, subfamily="Regular", weight=400):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    pen = TTGlyphPen(None)
    pen.moveTo((0, 0)); pen.lineTo((500, 700)); pen.lineTo((500, 0)); pen.closePath()
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A"])
    fb.setupCharacterMap({65: "A"})
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph(), "A": pen.glyph()})
    fb.setupHorizontalMetrics({".notdef": (1000, 0), "A": (1000, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({
        "familyName": family, "styleName": subfamily,
        "typographicFamily": family, "typographicSubfamily": subfamily,
        "fullName": f"{family} {subfamily}", "psName": f"{family}-{subfamily}",
    })
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWeightClass=weight)
    fb.setupPost()
    fb.setupMaxp()
    fb.save(str(path))


class StubEmbedder:
    """No torch: a fixed 384-dim feature per render, the frozen backbone's shape."""
    device = "cpu"

    def embed_batch(self, images):
        return np.zeros((len(images), 384), dtype=np.float32)


work = Path(tempfile.mkdtemp())
app = work / "app"
(app / "site" / "data").mkdir(parents=True)
(app / "site" / "index.html").write_text("<html></html>")
(app / "site" / "data" / "pca.bin").write_bytes(b"\x00" * ((384 + 200 * 384) * 4))
(app / "site" / "js").mkdir()
(app / "site" / "js" / "engine.js").write_text("export function f() {}\n")
science = work / "science"
(science / "model").mkdir(parents=True)
(science / "model" / "build_corpus.py").write_text("# pipeline\n")

private = work / "private"
(private / "fonts").mkdir(parents=True)
make_font(private / "fonts" / "Alpha-Regular.otf", "Alpha", "Regular", 400)
make_font(private / "fonts" / "Beta-Bold.otf", "Beta", "Bold", 700)

os.environ["TYPEOREM_SITE_DIR"] = str(app)
pc.load_pca = lambda: (np.zeros(384, dtype=np.float32),
                       np.zeros((200, 384), dtype=np.float32), "0" * 64, "test")

app_before, sci_before, work_before = snapshot(app), snapshot(science), snapshot(work)
catalog = pc.write_private_corpus(private, embedder=StubEmbedder())
app_after, sci_after, work_after = snapshot(app), snapshot(science), snapshot(work)

ok("the run produced the private rows", len(catalog) == 2)
ok("the app tree is byte-identical", app_before == app_after)
ok("the science tree is byte-identical", sci_before == sci_after)

allowed = (pc.CORPUS_DIRNAME + "/", pc.CACHE_DIRNAME + "/")
prefix = private.relative_to(work).as_posix()
changed_outside = []
for rel, digest in work_after.items():
    inside = rel.startswith(prefix + "/")
    sub = rel[len(prefix) + 1:] if inside else None
    if inside and sub.startswith(allowed):
        continue
    if work_before.get(rel) != digest:
        changed_outside.append(rel)
ok("every changed file is inside the private root", changed_outside == [])
if changed_outside:
    print(f"  leaked: {changed_outside[:5]}")
ok("the private corpus landed where expected",
   (private / pc.CORPUS_DIRNAME / "catalog.json").is_file()
   and (private / pc.CORPUS_DIRNAME / "vectors.i8.bin").is_file())

os.environ.pop("TYPEOREM_SITE_DIR", None)
print(f"{passes} passed, {fails} failed")
sys.exit(1 if fails else 0)
