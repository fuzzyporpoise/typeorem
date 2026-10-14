"""The private corpus path (TSK-001): bring the owner's own fonts, including
commercially licensed ones, into the tool on the owner's machine only.

A private run renders the owner's faces through the same glyph grid the catalog
uses, embeds them with the frozen backbone, projects them through the SAVED PCA
((x - mean) @ components.T, never a refit), quantizes int8, and writes a corpus
into `<dir>/corpus/`. The rows carry `source: "private"`, which is not in the
public allowlist, so the release validator and the app's provenance gate both
reject them: no private vector can reach a release, `site/data`, or the deploy.

The output corpus holds only the private rows. It is a delta the app overlays on
the shipped catalog (the app repo's TSK-019): the private rows are appended after
the public ones, and each row's `fontUrl` (relative to the private root, which
the dev server mounts at `/private/`) points at the local font file.

The corpus is written under `<dir>`, which is never inside a checkout. The writer
refuses any output path under the app's `site/`, refuses a root inside a git work
tree (so a licensed font is never committable) or a cloud-sync folder (so it never
syncs off the machine), and records the identity of the PCA it projected through
(backbone, preprocess, and the sha256 of the canonical `pca.bin` bytes) so a corpus
left in an old space after a rebuild is detectable rather than silently wrong.

    model/.venv/bin/python model/build_corpus.py --private      # defaults to ~/.cache/typeorem
    TYPEOREM_PRIVATE_DIR=/some/where model/.venv/bin/python model/build_corpus.py --private

The private root holds `fonts/` (scanned recursively for .ttf/.otf/.woff/.woff2;
when there is no `fonts/`, the root itself is scanned) and gains `corpus/`. With
no directory given it defaults to `$XDG_CACHE_HOME/typeorem` (else
`~/.cache/typeorem`), which keeps the faces off a dotfiles tree and out of any
cloud sync; the owner's originals stay wherever they keep them.
"""

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import paths  # noqa: E402
from catalog import write_json  # noqa: E402
from emit import BACKBONE, PREPROCESS, label_of  # noqa: E402
from render import (GLYPHS, COLS, ROWS, SIZE, FONT_SIZE, BASELINE_Y,  # noqa: E402
                    GLYPH_SET_VERSION, render_instance)
from instances import slug  # noqa: E402

CORPUS_DIRNAME = "corpus"
FONTS_DIRNAME = "fonts"
CACHE_DIRNAME = ".cache"
FONT_EXTS = {".ttf", ".otf", ".woff", ".woff2"}
# Never scanned as font input: our own output and the decompressed-font cache.
SKIP_DIRS = {CORPUS_DIRNAME, CACHE_DIRNAME}
WGHT_LADDER = (300, 400, 700, 900)

WEIGHT_WORDS = {
    "thin": 100, "extralight": 200, "ultralight": 200, "lazer": 250,
    "light": 300, "regular": 400, "normal": 400, "book": 400, "medium": 500,
    "semibold": 600, "demibold": 600, "bold": 700,
    "extrabold": 800, "ultrabold": 800, "black": 900, "heavy": 900,
}

# Width and mono qualifiers fold into the family, the public catalog's convention
# (Archivo Narrow is its own family next to Archivo), so a width-named file does
# not collapse onto the base family's weight ladder. The file stem is the source,
# never the name records: trial cuts abbreviate their subfamilies (name 16 is
# "ZX Grotesk Trial" with name 17 "X Cm Bl It"), so only the stem spells the
# qualifier out.
QUALIFIER_WORDS = {
    "x", "mono", "compressed", "condensed", "narrow", "extended",
    "expanded", "wide",
}
QUALIFIER_STYLE_WORDS = {"italic", "oblique", "roman", "trial", "demo"}


def pca_bin_bytes(mean, components):
    """The canonical `pca.bin` layout (float32 LE: mean then components), the
    same bytes export_pca.py writes. Its sha256 identifies the projection space."""
    mean = np.asarray(mean).astype("<f4")
    components = np.asarray(components).astype("<f4")
    return np.concatenate([mean, components.reshape(-1)]).tobytes()


def load_pca():
    """The saved catalog PCA (never a refit). Returns (mean, components, sha256,
    source). The saved model is authoritative; when the shipped `site/data/pca.bin`
    is present it must hash to the same bytes, or the two are in different spaces
    and the private rows would not line up with the tool."""
    npz = BASE / "out" / "pca_model.npz"
    if not npz.is_file():
        raise SystemExit(
            f"no saved PCA at {npz}. A private run projects through the shipped "
            f"space and never refits; build the corpus first."
        )
    d = np.load(npz)
    mean, components = d["mean"], d["components"]
    sha = hashlib.sha256(pca_bin_bytes(mean, components)).hexdigest()
    source = "out/pca_model.npz"
    shipped = paths.site_data() / "pca.bin"
    if shipped.is_file():
        got = hashlib.sha256(shipped.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(
                f"{shipped} does not match out/pca_model.npz "
                f"({got[:12]} != {sha[:12]}). The shipped projection and the saved "
                f"PCA disagree; re-export the PCA before writing a private corpus."
            )
        source = str(shipped)
    return mean, components, sha, source


def refuse_under_site(out_dir):
    """A private corpus must never be able to reach the app tree, so no licensed
    vector can be released, committed, or deployed. Refuses the resolved app's
    served tree, and, independently of TYPEOREM_SITE_DIR, any path that sits
    inside an app checkout (a directory holding `site/index.html`)."""
    out = Path(out_dir).resolve()
    site = paths.site_dir().resolve()
    inside_app = out == site or site in out.parents
    if not inside_app:
        for parent in [out, *out.parents]:
            if (parent / "site" / "index.html").is_file():
                inside_app = True
                break
    if inside_app:
        raise SystemExit(
            f"refusing to write a private corpus inside the app tree: {out}\n"
            f"the private corpus must live outside the app checkout (and its "
            f"site/), so no licensed vector can be released, committed, or "
            f"deployed."
        )


# Known cloud-sync providers. A licensed font must not sync to a cloud the owner
# does not control, and a synced tree is slow to scan, so these roots are refused.
CLOUD_MARKERS = (
    "library/mobile documents", "dropbox", "google drive", "googledrive",
    "onedrive", "box sync", "creative cloud files", "icloud drive",
)


def default_root():
    """The private root when none is given: `$XDG_CACHE_HOME/typeorem` (else
    `~/.cache/typeorem`). A cache directory is machine-local and is not picked up
    by a dotfiles manager or a cloud sync, and it sits beside the caches the
    pipeline already uses (~/.cache/huggingface, ~/.cache/torch). The faces
    staged here are working copies; the owner's originals stay wherever they
    keep them."""
    base = os.environ.get("XDG_CACHE_HOME")
    base = Path(base).expanduser() if base else Path.home() / ".cache"
    return (base / "typeorem").resolve()


def check_root_location(root):
    """A licensed font must not be committable or synced, so refuse a private
    root inside a git work tree (dotfiles, either checkout) or a cloud-sync
    folder. Stage the faces under the default cache home instead."""
    root = Path(root).resolve()
    for parent in [root, *root.parents]:
        if (parent / ".git").exists():
            raise SystemExit(
                f"refusing a private root inside a git work tree: {root}\n"
                f"found {parent / '.git'}. A licensed font must never be "
                f"committable; use the machine-local default "
                f"({default_root()}) and stage the faces there."
            )
    text = str(root).lower().replace("\\", "/")
    for marker in CLOUD_MARKERS:
        if marker in text:
            raise SystemExit(
                f"refusing a private root inside a cloud-sync folder ({marker}): {root}\n"
                f"a licensed font must not sync to a cloud the owner does not "
                f"control. Stage the faces under {default_root()} instead."
            )


def resolve_root(arg=None):
    """The private root: an explicit `--private <dir>`, else TYPEOREM_PRIVATE_DIR,
    else the machine-local default (`~/.cache/typeorem`)."""
    raw = arg or os.environ.get("TYPEOREM_PRIVATE_DIR")
    if not raw:
        root = default_root()
        print(f"no private root given; using the default {root}", flush=True)
        return root
    return Path(raw).expanduser().resolve()


def scan_fonts(root):
    """Every font file under `<root>/fonts`, or under `<root>` when it has no
    `fonts/`. The corpus output and the decompressed-font cache are skipped, so a
    re-run does not feed its own artifacts back in as input."""
    fonts_root = root / FONTS_DIRNAME if (root / FONTS_DIRNAME).is_dir() else root
    found = []
    for path in sorted(fonts_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FONT_EXTS:
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        found.append(path)
    return fonts_root, found


def prepare_renderable(path, cache_dir):
    """PIL/FreeType cannot read woff/woff2, so decompress those to a TTF in the
    private cache first (fontTools decodes; brotli does the woff2 decompression).
    ttf/otf are used in place."""
    if path.suffix.lower() not in (".woff", ".woff2"):
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    ttf = cache_dir / (path.stem + ".ttf")
    if not ttf.exists() or ttf.stat().st_mtime < path.stat().st_mtime:
        font = TTFont(str(path))
        font.flavor = None
        font.save(str(ttf))
        font.close()
    return ttf


def _valid_name(value):
    """A usable name carries at least one alphanumeric. Web cuts sometimes ship a
    junk name record (an empty string, a lone '.'); those must not become a
    family."""
    if value and any(ch.isalnum() for ch in value):
        return value.strip()
    return None


def _name(font, *name_ids):
    for nid in name_ids:
        value = _valid_name(font["name"].getDebugName(nid))
        if value:
            return value
    return None


def _stem_words(text):
    """Lowercase word tokens: split camel boundaries, then any non-letter."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w.lower() for w in re.split(r"[^A-Za-z]+", text) if w]


def _qualifier(family, source_path):
    """The width/mono qualifier the file stem carries beyond the family name
    ("ZX-Grotesk-X-Compressed-Bold-Trial" under "ZX Grotesk Trial" -> "X
    Compressed"), or None. Conservative: every leftover word must be qualifier
    vocabulary (an unknown leftover means the stem is ambiguous and the base
    family stands), and a qualifier the family already names is not repeated."""
    drop = set(_stem_words(family)) | set(WEIGHT_WORDS) | QUALIFIER_STYLE_WORDS
    rest = [w for w in _stem_words(Path(source_path).stem) if w not in drop]
    if not rest or not set(rest) - {"x"}:
        return None
    if any(w not in QUALIFIER_WORDS for w in rest):
        return None
    return " ".join(w.capitalize() for w in rest)


def _family_name(font, path, fallback_path=None):
    """The face's family. Prefers the font's own name record; when that is missing
    or junk, falls back to the containing directory (the owner organizes fonts by
    family, so this still groups a family's cuts), then the file name. The fallback
    uses `fallback_path` (the original file) when the face was decompressed to a
    cache directory, so the cache's own folder name never becomes a family."""
    name = _name(font, 16, 1, 4, 6)
    if name:
        return name
    source = Path(fallback_path or path)
    if _valid_name(source.parent.name):
        return source.parent.name
    return _valid_name(source.stem) or source.stem


def _guess_category(font):
    """Best effort, from the font's own tables. The owner can edit it in the
    tool; the serif-split term reads this, so it is recorded as a guess."""
    try:
        if font["post"].isFixedPitch:
            return "monospace"
    except Exception:
        pass
    try:
        panose = font["OS/2"].panose
        family, serif = int(panose.bFamilyType), int(panose.bSerifStyle)
        if family == 2 and 2 <= serif <= 10:
            return "serif"
        if family == 2 and 11 <= serif <= 13:
            return "sans-serif"
    except Exception:
        pass
    return "sans-serif"


def font_metadata(path, fallback_path=None):
    """Family, weight, style, category, and the variable axes, read from the
    font's own tables (a dropped file carries no catalog record). `fallback_path`
    is the original file when `path` is a decompressed cache copy."""
    path = Path(path)
    font = TTFont(str(path), lazy=True)
    try:
        family = _family_name(font, path, fallback_path)
        qualifier = _qualifier(family, fallback_path or path)
        if qualifier:
            family = f"{family} {qualifier}"
        subfamily = _name(font, 17, 2) or ""
        weight = 400
        try:
            weight = int(font["OS/2"].usWeightClass)
        except Exception:
            for word, w in WEIGHT_WORDS.items():
                if re.search(rf"\b{word}\b", subfamily, re.I):
                    weight = w
                    break
        style = "normal"
        if re.search(r"italic|oblique", subfamily, re.I):
            style = "italic"
        else:
            try:
                if font["head"].macStyle & 0x2 or font["OS/2"].fsSelection & 0x1:
                    style = "italic"
            except Exception:
                pass
        axes = []
        if "fvar" in font:
            axes = [{"tag": a.axisTag, "min": float(a.minValue),
                     "max": float(a.maxValue), "default": float(a.defaultValue)}
                    for a in font["fvar"].axes]
        return {
            "family": family,
            "weight": weight,
            "style": style,
            "category": _guess_category(font),
            "variable": bool(axes),
            "axes": axes,
        }
    finally:
        font.close()


def font_instances(meta):
    """The instances a single font file contributes. A variable face walks the
    corpus weight ladder within its wght range (and its italic, when it has an
    ital axis); a static face is one instance. Ids mirror the catalog's
    <family-slug>-<weight>-<style> shape."""
    family = meta["family"]
    prefix = slug(family)
    by_tag = {a["tag"]: a for a in meta["axes"]}
    if meta["variable"] and "wght" in by_tag:
        axis = by_tag["wght"]
        weights = [w for w in WGHT_LADDER if axis["min"] <= w <= axis["max"]]
        if not weights:
            weights = [int(round(axis["default"]))]
        styles = [("normal", 0.0)]
        if "ital" in by_tag:
            styles.append(("italic", 1.0))
        for weight in weights:
            for style, ital in styles:
                coords = {"wght": weight}
                if "ital" in by_tag:
                    coords["ital"] = ital
                yield {
                    "id": f"{prefix}-{weight}-{style}",
                    "family": family, "category": meta["category"],
                    "weight": weight, "style": style,
                    "variable": True, "coords": coords,
                }
        return
    yield {
        "id": f"{prefix}-{meta['weight']}-{meta['style']}",
        "family": family, "category": meta["category"],
        "weight": meta["weight"], "style": meta["style"],
        "variable": False, "coords": {},
    }


def collect_instances(root, files):
    """Render-ready instances for the scanned files, one row per (family, weight,
    style). Each carries the file it renders from and its `fontUrl` relative to
    the private root (the app's mount). Returns (instances, failures)."""
    cache_dir = root / CACHE_DIRNAME / "fonts"
    instances, failures, seen = [], [], set()
    for path in files:
        try:
            renderable = prepare_renderable(path, cache_dir)
            meta = font_metadata(renderable, fallback_path=path)
        except Exception as e:
            failures.append((str(path.relative_to(root)), str(e)))
            continue
        try:
            font_url = path.relative_to(root).as_posix()
        except ValueError:
            font_url = path.name
        for inst in font_instances(meta):
            if inst["id"] in seen:
                continue
            seen.add(inst["id"])
            inst["render_path"] = str(renderable)
            inst["fontUrl"] = font_url
            instances.append(inst)
    instances.sort(key=lambda i: (i["family"].lower(), i["weight"], i["style"]))
    return instances, failures


def _quantize(scores):
    scale = np.abs(scores).max(axis=1)
    scale[scale == 0] = 1.0
    q = np.clip(np.round(scores / scale[:, None] * 127.0), -127, 127).astype(np.int8)
    return q, scale


def project(instances, mean, components, cache_dir, embedder=None, batch=16):
    """Render each face through the glyph grid, embed with the frozen backbone,
    and project through the saved PCA. Returns (quantized, scale, xh).

    Render and embed run a batch at a time, so memory stays bounded and the run
    prints a heartbeat (a private corpus can be thousands of faces). `embedder`
    is a test seam; production passes None and builds the frozen DINOv2."""
    if embedder is None:
        from embed import DINOv2  # lazy: keeps the module importable without torch
        embedder = DINOv2()

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"embedding {len(instances)} private faces on {getattr(embedder, 'device', 'cpu')}",
          flush=True)

    features, done, report_at = [], 0, max(25, len(instances) // 20)
    for start in range(0, len(instances), batch):
        batch_imgs = []
        for inst in instances[start:start + batch]:
            png = cache_dir / f"{inst['id']}.png"
            img = render_instance(Path(inst["render_path"]), inst["coords"])
            img.save(png)
            batch_imgs.append(np.asarray(img.convert("RGB")))
        features.append(embedder.embed_batch(batch_imgs))
        done += len(batch_imgs)
        if done >= report_at or done == len(instances):
            print(f"  {done}/{len(instances)} rendered + embedded", flush=True)
            report_at = done + max(25, len(instances) // 20)

    feats = np.concatenate(features)
    scores = (feats - mean) @ components.T
    q, scale = _quantize(scores)
    xh = width = None
    try:
        from legibility import render_axes  # lazy: pulls sklearn, pipeline-only
        axes = render_axes(instances, cache_dir)
        xh = [axes.get(i["id"], {}).get("xh") for i in instances]
        width = [axes.get(i["id"], {}).get("width") for i in instances]
    except Exception as e:  # axes are an enhancement, never a reason to fail a run
        print(f"  render axes skipped ({e})", flush=True)
    return q, scale, xh, width


def emit_private(out_dir, instances, q, scale, pca, pca_source, xh=None, width=None, fonts=0):
    """Write the delta corpus: binary + meta + catalog + a private version block,
    the fields the app's `loadCorpus` already reads plus the identity that ties
    the rows to the projection space they were written in."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = int(q.shape[1])
    scale = [round(float(s), 6) for s in scale]
    grid = {
        "glyphs": "".join(GLYPHS), "cols": COLS, "rows": ROWS, "size": SIZE,
        "fontSize": FONT_SIZE, "baselineY": BASELINE_Y,
    }
    identity = {
        "backbone": BACKBONE, "preprocess": PREPROCESS,
        "pcaSha256": pca["sha256"], "pcaSource": pca_source,
        "glyphSetVersion": GLYPH_SET_VERSION, "glyphGrid": grid,
    }

    (out_dir / "vectors.i8.bin").write_bytes(q.tobytes())
    write_json(out_dir / "vectors.meta.json", {
        "v": 1, "dim": dim, "count": int(q.shape[0]), "dtype": "int8",
        "scale": {"mode": "perRow", "values": scale},
        "backbone": BACKBONE, "preprocess": PREPROCESS,
        "private": {
            **identity, "instances": int(q.shape[0]), "fonts": fonts,
            "fontUrlBase": "",
            "note": "fontUrl is relative to the private root the app mounts; "
                    "source is not in the public allowlist, so this never ships",
        },
    })

    catalog = []
    for i, inst in enumerate(instances):
        x = None if xh is None else xh[i]
        w = None if width is None else width[i]
        catalog.append({
            "id": i, "family": inst["family"], "label": label_of(inst),
            "weight": inst["weight"], "style": inst["style"],
            "category": inst["category"], "source": "private",
            "subsets": [], "eligible": True, "css2": None,
            "coords": inst["coords"], "legibility": None,
            "xh": round(float(x), 4) if x is not None else None,
            "width": round(float(w), 4) if w is not None else None,
            "search": f"{inst['family']} {label_of(inst)} {inst['category']}".lower(),
            "fontUrl": inst["fontUrl"],
        })
    write_json(out_dir / "catalog.json",
               {"v": 1, "count": len(catalog), "instances": catalog})
    write_json(out_dir / "corpus.version.json", {
        "v": 1, "kind": "private", "count": len(catalog), "dim": dim,
        **identity,
        "sources": {"private": len(catalog)},
        "families": sorted({c["family"] for c in catalog}),
    })
    return catalog


def write_private_corpus(root, embedder=None):
    root = Path(root)
    refuse_under_site(root / CORPUS_DIRNAME)
    check_root_location(root)
    if not root.is_dir():
        raise SystemExit(
            f"private root {root} is not a directory.\n"
            f"Create it and stage your faces under {root / FONTS_DIRNAME}/:\n"
            f"    mkdir -p {root / FONTS_DIRNAME}"
        )

    _, files = scan_fonts(root)
    if not files:
        raise SystemExit(
            f"no font files under {root} (looked for "
            f"{', '.join(sorted(FONT_EXTS))} in {FONTS_DIRNAME}/ or the root)"
        )
    print(f"private root {root}: {len(files)} font files", flush=True)

    mean, components, pca_sha, pca_source = load_pca()
    print(f"projecting through the saved PCA ({pca_source}, sha {pca_sha[:12]})", flush=True)

    instances, failures = collect_instances(root, files)
    for rel, err in failures[:5]:
        print(f"    FAIL {rel}: {err}")
    if not instances:
        raise SystemExit("no usable instances: every font file failed to parse")
    families = len({i["family"] for i in instances})
    print(f"{len(instances)} instances across {families} families", flush=True)

    q, scale, xh, width = project(instances, mean, components,
                                  root / CACHE_DIRNAME / "224", embedder=embedder)
    catalog = emit_private(root / CORPUS_DIRNAME, instances, q, scale,
                           {"sha256": pca_sha}, pca_source, xh=xh, width=width, fonts=families)

    out = root / CORPUS_DIRNAME
    print(f"private corpus -> {out}")
    for name in ("vectors.i8.bin", "catalog.json", "vectors.meta.json", "corpus.version.json"):
        p = out / name
        print(f"  {name:22} {p.stat().st_size/1024:8.1f} KB")
    print(f"  {len(catalog)} rows, {families} families, source private")
    print(f"  pca sha256 {pca_sha}")
    return catalog
