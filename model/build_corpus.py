import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import paths
from embed import DINOv2
from render import (render_corpus, render_instance, GLYPH_SET_VERSION, GLYPHS,
                    COLS, ROWS, SIZE, FONT_SIZE, BASELINE_Y)
from reduce import fit_pca, quantize_int8, DIM
from emit import emit, report, label_of, BACKBONE, PREPROCESS
from legibility import fit_proxy, render_axes
from instances import POLICY_VERSION, WGHT_LADDER, slug
from catalog import google_fonts_commit, write_json

CACHE = BASE / ".cache"
OUT = BASE / "out"
# The corpus ships in the site checkout (TYPEOREM_SITE_DIR, default ../site).
SITE = paths.site_data()

# ---------------------------------------------------------------------------
# House faces (TSK-014). Commit Mono is the fuzzyporpoise house face but is not
# on Google Fonts, so it is not in the ingested catalog. It ships self-hosted in
# site/fonts, so the corpus appends it as a "local" family: rendered from the
# same woff2 the browser uses, embedded with the frozen backbone, and projected
# through the SAVED catalog PCA. Never refit the PCA on the house faces: a refit
# would shift every shipped vector, so the append reuses out/pca_model.npz as-is
# and only adds the trailing house block.
# ---------------------------------------------------------------------------
HOUSE_FAMILY = "Commit Mono"
HOUSE_WOFF2 = paths.house_font()
HOUSE_TTF = CACHE / "house" / "CommitMono-VF.ttf"
HOUSE_SUBSETS = ["latin", "latin-ext"]
HOUSE_ID_PREFIX = slug(HOUSE_FAMILY)


def fvar_range(path):
    from fontTools.ttLib import TTFont
    font = TTFont(str(path), lazy=True)
    ranges = {a.axisTag: (float(a.minValue), float(a.maxValue)) for a in font["fvar"].axes} if "fvar" in font else {}
    font.close()
    return ranges


def house_font_available():
    return HOUSE_WOFF2.is_file()


def prepare_house_font():
    """Decompress the self-hosted woff2 to a TTF, since PIL/FreeType cannot read
    woff2 (fontTools decodes it; the brotli package does the decompression)."""
    from fontTools.ttLib import TTFont
    if not house_font_available():
        raise SystemExit(
            f"house face not found at {HOUSE_WOFF2}.\n"
            f"The site checkout owns site/fonts/CommitMono-VF.woff2; clone it next to "
            f"this repo or point {paths.VAR} at it."
        )
    HOUSE_TTF.parent.mkdir(parents=True, exist_ok=True)
    if not HOUSE_TTF.exists() or HOUSE_TTF.stat().st_mtime < HOUSE_WOFF2.stat().st_mtime:
        font = TTFont(str(HOUSE_WOFF2))
        font.flavor = None
        font.save(str(HOUSE_TTF))
    return HOUSE_TTF


def house_instances():
    """The house-face ladder: the corpus weight ladder capped to the face's wght
    range, times the styles the face exposes. Ids mirror the catalog's
    <family-slug>-<weight>-<style> shape."""
    path = prepare_house_font()
    ranges = fvar_range(path)
    lo, hi = ranges.get("wght", (400.0, 400.0))
    weights = [w for w in WGHT_LADDER if lo <= w <= hi]
    styles = [("normal", 0)]
    if "ital" in ranges:
        styles.append(("italic", 1))
    out = []
    for weight in weights:
        for style, ital in styles:
            out.append({
                "id": f"{HOUSE_ID_PREFIX}-{weight}-{style}",
                "family": HOUSE_FAMILY,
                "category": "monospace",
                "weight": weight,
                "style": style,
                "variable": True,
                "file": path.name,
                "coords": {"wght": weight, "ital": ital},
                "css2": None,
                "pairingEligible": True,
                "source": "local",
            })
    return out


def project_house_faces(pca_mean, pca_components):
    """Render + embed the house faces and project them through the given PCA
    (the catalog's saved model). Returns (instances, quantized, scales, hashes)."""
    instances = house_instances()
    path = prepare_house_font()
    images, hashes = [], {}
    for inst in instances:
        png = CACHE / "224" / f"{inst['id']}.png"
        render_instance(path, inst["coords"]).save(png)
        hashes[inst["id"]] = hashlib.sha256(png.read_bytes()).hexdigest()
        images.append(np.asarray(Image.open(png).convert("RGB")))
    embedder = DINOv2()
    print(f"embedding {len(images)} house faces on {embedder.device}")
    feats = np.concatenate([embedder.embed_batch(images[s:s + 16]) for s in range(0, len(images), 16)])
    scores = (feats - pca_mean) @ pca_components.T
    q, scale = quantize_int8(scores)
    return instances, q, scale, hashes


def append_house_faces(pca=None):
    """Append the house faces to the shipped corpus, projecting them through the
    saved PCA (model/out/pca_model.npz) when `pca` is not supplied. Idempotent:
    any previously appended house rows are dropped first, so re-running does not
    duplicate them."""
    if pca is None:
        d = np.load(OUT / "pca_model.npz")
        pca_mean, pca_components = d["mean"], d["components"]
    else:
        pca_mean, pca_components = pca.mean_, pca.components_

    meta = json.loads((SITE / "vectors.meta.json").read_text())
    catalog = json.loads((SITE / "catalog.json").read_text())
    version = json.loads((SITE / "corpus.version.json").read_text())
    dim = int(meta["dim"])
    q = np.frombuffer((SITE / "vectors.i8.bin").read_bytes(), dtype=np.int8).reshape(meta["count"], dim).copy()
    scales = list(meta["scale"]["values"])

    # House rows are always the tail; drop a previous block so re-running is safe.
    first_house = next((i for i, c in enumerate(catalog["instances"]) if c.get("source") == "local"),
                       len(catalog["instances"]))
    instances = catalog["instances"][:first_house]
    q = q[:first_house]
    scales = scales[:first_house]

    hinst, hq, hscale, hashes = project_house_faces(pca_mean, pca_components)
    axes = render_axes(hinst, CACHE / "224")

    base = len(instances)
    for k, inst in enumerate(hinst):
        xh = axes.get(inst["id"], {}).get("xh")
        instances.append({
            "id": base + k,
            "family": inst["family"],
            "label": label_of(inst),
            "weight": inst["weight"],
            "style": inst["style"],
            "category": inst["category"],
            "subsets": HOUSE_SUBSETS,
            "eligible": inst["pairingEligible"],
            "css2": None,
            "coords": inst["coords"],
            "source": "local",
            "legibility": None,
            "xh": round(float(xh), 4) if xh is not None else None,
            "search": f"{inst['family']} {label_of(inst)} {inst['category']}".lower(),
        })

    q = np.concatenate([q, hq], axis=0)
    (SITE / "vectors.i8.bin").write_bytes(q.tobytes())
    meta["count"] = int(q.shape[0])
    meta["scale"]["values"] = [round(float(s), 6) for s in scales] + [round(float(s), 6) for s in hscale]
    write_json(SITE / "vectors.meta.json", meta)
    write_json(SITE / "catalog.json", {"v": catalog["v"], "count": len(instances), "instances": instances})

    version["count"] = len(instances)
    version["houseFaces"] = {
        "family": HOUSE_FAMILY,
        "source": "site/fonts/CommitMono-VF.woff2",
        "count": len(hinst),
        "weights": sorted({i["weight"] for i in hinst}),
        "styles": sorted({i["style"] for i in hinst}),
    }
    write_json(SITE / "corpus.version.json", version)

    # Keep the per-instance render hashes (the determinism test reads these) in
    # sync with the appended rows.
    rh = OUT / "render-hashes.json"
    hashes_all = {k: v for k, v in json.loads(rh.read_text()).items() if not k.startswith(HOUSE_ID_PREFIX + "-")}
    hashes_all.update(hashes)
    write_json(rh, hashes_all)
    print(f"appended {len(hinst)} house faces: {', '.join(label_of(i) for i in hinst)}")


def load_inputs():
    instances = json.loads((OUT / "instances.json").read_text())
    families = {f["family"]: f for f in json.loads((OUT / "catalog.raw.json").read_text())}
    return instances, families, {k: v["dir"] for k, v in families.items()}


def embed_all(instances, png_dir):
    images = [np.asarray(Image.open(png_dir / f"{i['id']}.png").convert("RGB")) for i in instances]
    emb = DINOv2()
    print(f"embedding {len(images)} on {emb.device}")
    return np.concatenate([emb.embed_batch(images[s:s + 16]) for s in range(0, len(images), 16)])


def main():
    paths.require_site("the corpus is emitted into the site's site/data")
    instances, families, dirs = load_inputs()
    print(f"{len(instances)} instances")
    t0 = time.time()
    hashes, failures = render_corpus(instances, dirs, CACHE / "google-fonts", CACHE / "224")
    print(f"rendered {len(hashes)} in {time.time() - t0:.0f}s, {len(failures)} failed")
    for iid, err in failures[:5]:
        print(f"    FAIL {iid}: {err}")
    instances = [i for i in instances if i["id"] in hashes]

    t0 = time.time()
    feats = embed_all(instances, CACHE / "224")
    print(f"embedded in {time.time() - t0:.0f}s")

    pca, scores = fit_pca(feats)
    q, scale = quantize_int8(scores)
    legibility = fit_proxy([i["family"] for i in instances], [i["category"] for i in instances], scores)
    print("legibility proxy fit")
    axes = render_axes(instances, CACHE / "224")
    xh = [axes.get(i["id"], {}).get("xh") for i in instances]
    print(f"render axes: xh for {sum(v is not None for v in xh)}/{len(xh)} instances")
    emit(SITE, instances, families, q, scale, pca, {
        "policyVersion": POLICY_VERSION, "glyphSetVersion": GLYPH_SET_VERSION,
        # The glyph grid travels with the corpus as its geometry contract: the
        # browser's custom-font rasterizer reproduces this grid, and the site's
        # gate asserts its copy matches these numbers (tests/glyphgrid.test.mjs).
        "glyphGrid": {
            "glyphs": "".join(GLYPHS), "cols": COLS, "rows": ROWS, "size": SIZE,
            "fontSize": FONT_SIZE, "baselineY": BASELINE_Y,
        },
        "backbone": BACKBONE, "preprocess": PREPROCESS, "pcaSeed": 0,
        "googleFontsCommit": google_fonts_commit(CACHE / "google-fonts"),
        "count": int(q.shape[0]), "dim": DIM,
    }, legibility, xh)
    np.savez(OUT / "pca_model.npz", mean=pca.mean_, components=pca.components_)
    write_json(OUT / "render-hashes.json", hashes)
    if house_font_available():
        append_house_faces(pca)
    else:
        print(f"no house face at {HOUSE_WOFF2}; skipped the house append")
    print("site/data:")
    report(SITE)


if __name__ == "__main__":
    if "--house" in sys.argv[1:]:
        paths.require_site("the house append writes into the site's site/data")
        append_house_faces()
    else:
        main()
