"""The private corpus path is a boundary, so its pieces get a test.

It checks the guards that keep licensed vectors off the public tree (the site
refusal, the PCA-space identity, the `private` source), the font scan and
grouping, and the emitted schema. The render/embed step (torch) is not exercised
here; the projection and quantization are pure numpy and are.
"""

import hashlib
import json
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


def make_font(path, family, subfamily="Regular", weight=400, fixed=False):
    """A minimal but real TTF with the tables the metadata reader looks at."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    pen = TTGlyphPen(None)
    pen.moveTo((0, 0)); pen.lineTo((500, 700)); pen.lineTo((500, 0)); pen.closePath()
    notdef = TTGlyphPen(None).glyph()
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A"])
    fb.setupCharacterMap({65: "A"})
    fb.setupGlyf({".notdef": notdef, "A": pen.glyph()})
    fb.setupHorizontalMetrics({".notdef": (1000, 0), "A": (1000, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({
        "familyName": family, "styleName": subfamily,
        "typographicFamily": family, "typographicSubfamily": subfamily,
        "fullName": f"{family} {subfamily}",
        "psName": f"{family.replace(' ', '')}-{subfamily.replace(' ', '')}",
    })
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWeightClass=weight)
    fb.setupPost(isFixedPitch=1 if fixed else 0)
    fb.setupMaxp()
    fb.save(str(path))


tmp = Path(tempfile.mkdtemp())
root = tmp / "private"
(root / "fonts" / "Agipo").mkdir(parents=True)
make_font(root / "fonts" / "Agipo" / "Agipo-Regular.otf", "Agipo", "Regular", 400)
make_font(root / "fonts" / "Agipo" / "Agipo-Bold.otf", "Agipo", "Bold", 700)
make_font(root / "fonts" / "Agipo" / "Agipo-Bold-copy.otf", "Agipo", "Bold", 700)
make_font(root / "fonts" / "Fixo.ttf", "Fixo", "Regular", 400, fixed=True)
(root / "fonts" / "README.txt").write_text("not a font")
(root / "corpus").mkdir()
(root / "corpus" / "vectors.i8.bin").write_bytes(b"\x00")

# --- scan: only real font files, never the output or the cache -------------
fonts_root, files = pc.scan_fonts(root)
ok("scan finds the fonts/ subtree", fonts_root == root / "fonts")
ok("scan finds every font file", len(files) == 4)
ok("scan ignores non-fonts and never re-reads corpus/",
   all(f.suffix == ".otf" or f.name == "Fixo.ttf" for f in files))

root2 = tmp / "flat"
root2.mkdir()
make_font(root2 / "Solo.ttf", "Solo", "Regular", 400)
_, flat = pc.scan_fonts(root2)
ok("a root with no fonts/ scans the root itself", flat == [root2 / "Solo.ttf"])

# --- grouping: one row per (family, weight, style), fontUrl relative to root
instances, failures = pc.collect_instances(root, files)
ok("no file failed to parse", failures == [])
ok("grouping dedupes a repeated face", len(instances) == 3)
by_id = {i["id"]: i for i in instances}
ok("a static file yields one instance", "solo-400-normal" not in by_id and
   "agipo-400-normal" in by_id and "agipo-700-normal" in by_id)
ok("the fixed-pitch face reads as monospace", by_id["fixo-400-normal"]["category"] == "monospace")
ok("fontUrl is relative to the private root",
   by_id["agipo-400-normal"]["fontUrl"] == "fonts/Agipo/Agipo-Regular.otf")
ok("each instance keeps the file it renders from",
   by_id["agipo-400-normal"]["render_path"].endswith("Agipo-Regular.otf"))

# A web cut with a junk name record must not become a "." family; it falls back to
# its containing directory, which groups it with the desktop cuts.
from fontTools.ttLib import TTFont  # noqa: E402
junkdir = root / "fonts" / "Junk Face"
junkdir.mkdir(parents=True)
junk = junkdir / "junkweb-regular.woff"
make_font(junk, "Junk", "Regular", 400)
f = TTFont(str(junk))
for nid in (1, 2, 4, 6, 16, 17):
    for plat, enc, lang in ((1, 0, 0), (3, 1, 0x409)):
        f["name"].setName(".\x7f", nid, plat, enc, lang)
f.save(str(junk))
f.close()
ok("a junk family name falls back to the containing directory",
   pc.font_metadata(junk)["family"] == "Junk Face")
ok("a valid name is not overridden by the fallback",
   pc.font_metadata(root / "fonts" / "Agipo" / "Agipo-Regular.otf")["family"] == "Agipo")
# A decompressed woff lands in .cache/fonts; the family must come from the source
# directory, not the cache's own "fonts" folder.
junk_insts, _ = pc.collect_instances(root, [junk])
ok("a decompressed web cut keeps its source family",
   len(junk_insts) == 1 and junk_insts[0]["family"] == "Junk Face"
   and junk_insts[0]["fontUrl"] == "fonts/Junk Face/junkweb-regular.woff")

# --- a variable face walks the ladder within its wght range ----------------
var_meta = {
    "family": "Var Face", "category": "sans-serif", "variable": True,
    "axes": [{"tag": "wght", "min": 350, "max": 800, "default": 400},
             {"tag": "ital", "min": 0, "max": 1, "default": 0}],
}
insts = list(pc.font_instances(var_meta))
ok("a variable face walks the ladder inside its range",
   sorted({i["weight"] for i in insts}) == [400, 700])
ok("a variable face with an ital axis yields italic rows",
   any(i["style"] == "italic" for i in insts) and
   any(i["coords"].get("ital") == 1.0 for i in insts))

# --- the site refusal: a private corpus may not be written under site/ ------
site_tmp = tmp / "app"
(site_tmp / "site").mkdir(parents=True)
(site_tmp / "site" / "index.html").write_text("<html></html>")
os.environ["TYPEOREM_SITE_DIR"] = str(site_tmp)
try:
    pc.refuse_under_site(site_tmp / "site" / "data")
    ok("a write under site/ is refused", False)
except SystemExit:
    ok("a write under site/ is refused", True)
pc.refuse_under_site(tmp / "private" / "corpus")  # outside site/ is fine

# Even with TYPEOREM_SITE_DIR unset (or pointing elsewhere), a path inside a
# checkout is refused by walking up to the directory that holds site/index.html.
os.environ["TYPEOREM_SITE_DIR"] = str(tmp / "elsewhere")
try:
    pc.refuse_under_site(site_tmp / "private" / "corpus")
    ok("a write anywhere inside an app checkout is refused", False)
except SystemExit:
    ok("a write anywhere inside an app checkout is refused", True)
os.environ["TYPEOREM_SITE_DIR"] = str(site_tmp)

# --- emit: schema, sizes, and the private provenance -----------------------
q = np.zeros((len(instances), 200), dtype=np.int8)
q[0, 0] = 127
scale = np.ones(len(instances), dtype=np.float32)
pca = {"sha256": "ab" * 32}
pc.emit_private(root / "corpus", instances, q, scale, pca,
                "out/pca_model.npz", xh=[None] * len(instances), fonts=3)
out = root / "corpus"
catalog = json.loads((out / "catalog.json").read_text())
meta = json.loads((out / "vectors.meta.json").read_text())
version = json.loads((out / "corpus.version.json").read_text())
ok("the binary is count * dim bytes",
   (out / "vectors.i8.bin").stat().st_size == len(instances) * 200)
ok("every row is source=private", all(c["source"] == "private" for c in catalog["instances"]))
ok("ids are row indices", all(c["id"] == i for i, c in enumerate(catalog["instances"])))
ok("a private row carries css2 null and a fontUrl",
   all(c["css2"] is None and c["fontUrl"] for c in catalog["instances"]))
ok("the meta records the PCA identity",
   meta["private"]["pcaSha256"] == "ab" * 32 and meta["private"]["backbone"] == pc.BACKBONE)
ok("the version block is private", version["kind"] == "private" and
   version["sources"] == {"private": len(instances)})

# --- the projection space is identified by the canonical pca.bin bytes -----
mean = np.arange(384, dtype=np.float32)
components = np.arange(200 * 384, dtype=np.float32).reshape(200, 384)
ok("pca_bin_bytes matches the shipped layout",
   len(pc.pca_bin_bytes(mean, components)) == (384 + 200 * 384) * 4 and
   hashlib.sha256(pc.pca_bin_bytes(mean, components)).hexdigest() ==
   hashlib.sha256(pc.pca_bin_bytes(mean, components)).hexdigest())

# load_pca needs the locally built model(and is what ties a private corpus to
# the shipped space); skip when there is no pipeline run to read.
npz = BASE / "out" / "pca_model.npz"
if npz.is_file():
    d = np.load(npz)
    want = hashlib.sha256(pc.pca_bin_bytes(d["mean"], d["components"])).hexdigest()
    # A shipped pca.bin that disagrees with the saved PCA must be rejected.
    (site_tmp / "site" / "data").mkdir(parents=True)
    (site_tmp / "site" / "data" / "pca.bin").write_bytes(b"wrong")
    try:
        pc.load_pca()
        ok("a shipped pca.bin that disagrees with the saved PCA is rejected", False)
    except SystemExit:
        ok("a shipped pca.bin that disagrees with the saved PCA is rejected", True)
    # The matching bytes load and carry the same identity.
    (site_tmp / "site" / "data" / "pca.bin").write_bytes(
        pc.pca_bin_bytes(d["mean"], d["components"]))
    m2, c2, sha, source = pc.load_pca()
    ok("load_pca returns the saved components and their sha",
       c2.shape == d["components"].shape and sha == want and source.endswith("pca.bin"))
else:
    print("no model/out/pca_model.npz; skipped the load_pca checks")

# --- the location policy: never committable, never synced ------------------
ok("the default root is a machine-local cache path",
   str(pc.default_root()).endswith("typeorem") and pc.default_root().is_absolute())
gitroot = tmp / "dotfiles"
(gitroot / ".git").mkdir(parents=True)
try:
    pc.check_root_location(gitroot / "typeorem")
    ok("a private root inside a git work tree is refused", False)
except SystemExit:
    ok("a private root inside a git work tree is refused", True)
cloud = tmp / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "pairs"
try:
    pc.check_root_location(cloud)
    ok("a private root in a cloud-sync folder is refused", False)
except SystemExit:
    ok("a private root in a cloud-sync folder is refused", True)
pc.check_root_location(tmp / "plain-elsewhere")  # outside both is fine

os.environ.pop("TYPEOREM_SITE_DIR", None)
print(f"{passes} passed, {fails} failed")
sys.exit(1 if fails else 0)
