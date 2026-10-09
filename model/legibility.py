import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

import paths

BASE = Path(__file__).resolve().parent
PNG_DIR = BASE / ".cache" / "224"
OUT = BASE / "out"
SITE_DATA = paths.site_data()

BODY_FAMILIES = [
    "Inter", "Roboto", "Open Sans", "Lato", "Noto Sans", "Source Sans 3", "Work Sans",
    "IBM Plex Sans", "Nunito Sans", "Mulish", "Karla", "Manrope", "Public Sans", "DM Sans",
    "Rubik", "Barlow", "Fira Sans", "Cabin", "Hind", "Roboto Slab", "Lora", "Merriweather",
    "PT Serif", "Source Serif 4", "Libre Baskerville", "Bitter", "Crimson Text", "Gelasio",
    "Noto Serif", "Domine", "Zilla Slab", "Alegreya", "Cardo", "EB Garamond", "Spectral",
    "Literata", "Newsreader", "Arvo", "Bree Serif",
]
NEGATIVE_CATEGORIES = {"display", "handwriting"}

COLS, ROWS, SIZE, BASELINE_IN_CELL = 6, 4, 224, 44
A, E, N, O = 0, 9, 11, 12  # glyph indices in render.GLYPHS


def cell_bounds(idx):
    col, row = idx % COLS, idx // COLS
    x0 = int(round(col * SIZE / COLS))
    x1 = int(round((col + 1) * SIZE / COLS))
    y0, y1 = row * (SIZE // ROWS), (row + 1) * (SIZE // ROWS)
    return x0, y0, x1, y1


def runs(mask):
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur); cur = 0
    if cur:
        out.append(cur)
    return out


def features_for(path):
    if not path.exists():
        return None
    ink = np.asarray(Image.open(path).convert("L")) < 128
    x0, y0, x1, y1 = cell_bounds(A)
    rows = np.where(ink[y0:y1, x0:x1].any(axis=1))[0]
    if len(rows) == 0:
        return None
    cap_height = (y0 + BASELINE_IN_CELL) - (y0 + rows.min())

    x0, y0, x1, y1 = cell_bounds(N)
    rows = np.where(ink[y0:y1, x0:x1].any(axis=1))[0]
    if len(rows) == 0 or cap_height <= 0:
        return None
    xh = ((y0 + BASELINE_IN_CELL) - (y0 + rows.min())) / cap_height

    x0, y0, x1, y1 = cell_bounds(O)
    oc = ink[y0:y1, x0:x1]
    orows, ocols = np.where(oc.any(axis=1))[0], np.where(oc.any(axis=0))[0]
    contrast = 0.0
    if len(orows) and len(ocols):
        o_top, o_bot = y0 + orows.min(), y0 + orows.max()
        o_left, o_right = x0 + ocols.min(), x0 + ocols.max()
        widths = sorted(runs(ink[(o_top + o_bot) // 2, o_left:o_right + 1]))[-2:]
        widths += sorted(runs(ink[o_top:o_bot + 1, (o_left + o_right) // 2]))[-2:]
        if widths:
            thick, thin = max(widths), min(widths)
            contrast = (thick - thin) / thick if thick else 0.0

    x0, y0, x1, y1 = cell_bounds(E)
    ec = ink[y0:y1, x0:x1]
    erows = np.where(ec.any(axis=1))[0]
    aperture = 0.0
    if len(erows):
        e_top, e_bot = erows.min(), erows.max()
        right = [int(np.where(ec[r])[0].max()) if ec[r].any() else -1 for r in range(e_top, e_bot + 1)]
        recess = sum(1 for v in right if v >= 0 and v < max(right) - 2)
        aperture = recess / (e_bot - e_top + 1)

    return {"xh": round(float(xh), 4), "contrast": round(float(contrast), 4), "aperture": round(float(aperture), 4)}


def normalize(values):
    order = np.argsort(np.argsort(values))
    return (order / max(len(values) - 1, 1)).tolist()


def render_axes(instances, png_dir=PNG_DIR):
    """Per-instance render axes for the shipped multi-axis pairing metric.

    Only x-height is shipped today: it is the skeleton gate the serif-contrast
    term fires on. Returns {instance id: {"xh":..}}.
    """
    out = {}
    for inst in instances:
        f = features_for(png_dir / f"{inst['id']}.png")
        if f is not None:
            out[inst["id"]] = {"xh": f["xh"]}
    return out


def load_vectors():
    meta = json.loads((SITE_DATA / "vectors.meta.json").read_text())
    catalog = json.loads((SITE_DATA / "catalog.json").read_text())["instances"]
    raw = np.frombuffer((SITE_DATA / "vectors.i8.bin").read_bytes(), dtype=np.int8)
    raw = raw.reshape(meta["count"], meta["dim"]).astype(np.float32)
    V = raw * np.array(meta["scale"]["values"], dtype=np.float32)[:, None]
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return catalog, V


def fit_proxy(families, categories, vectors):
    pos = set(BODY_FAMILIES)
    y, rows = [], []
    for k, (fam, cat) in enumerate(zip(families, categories)):
        if fam in pos:
            y.append(1); rows.append(k)
        elif cat in NEGATIVE_CATEGORIES:
            y.append(0); rows.append(k)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(vectors[rows], y)
    return clf.predict_proba(vectors)[:, 1]


def main():
    instances = json.loads((OUT / "instances.json").read_text())
    render = {}
    for inst in instances:
        f = features_for(PNG_DIR / f"{inst['id']}.png")
        if f:
            render[inst["id"]] = f
    ids = list(render)
    xh = normalize([render[i]["xh"] for i in ids])
    ap = normalize([render[i]["aperture"] for i in ids])
    co = normalize([render[i]["contrast"] for i in ids])
    for k, i in enumerate(ids):
        render[i]["render"] = round((xh[k] + ap[k] + (1 - co[k])) / 3, 4)

    catalog, V = load_vectors()
    pos = set(BODY_FAMILIES)
    y, rows = [], []
    for c in catalog:
        if not c["eligible"]:
            continue
        if c["family"] in pos:
            y.append(1); rows.append(c["id"])
        elif c["category"] in NEGATIVE_CATEGORIES:
            y.append(0); rows.append(c["id"])
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(V[rows], y)
    proba = clf.predict_proba(V)[:, 1]

    out = {c["id"]: {"proxy": round(float(proba[c["id"]]), 4)} for c in catalog}
    (OUT / "legibility.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"proxy fit on {len(y)} labels ({int(sum(y))} body); wrote {len(out)} scores")

    desc = {(i["family"], i["weight"], i["style"]): i["id"] for i in instances}
    cat_of = {c["id"]: c["category"] for c in catalog}
    for key in ("render", "proxy"):
        cats = {}
        for c in catalog:
            if key == "render":
                rid = desc.get((c["family"], c["weight"], c["style"]))
                val = render.get(rid, {}).get("render") if rid else None
            else:
                val = out[c["id"]]["proxy"]
            if val is not None:
                cats.setdefault(cat_of[c["id"]], []).append(val)
        print(f"{key:7} by category: " + "  ".join(f"{c} {np.mean(cats[c]):.2f}" for c in sorted(cats)))
    print("named families (proxy):")
    for fam in ("Inter", "Roboto", "Lora", "Merriweather", "Playfair Display", "Comic Neue", "Meie Script", "Megrim", "Luckiest Guy"):
        xs = [out[c["id"]]["proxy"] for c in catalog if c["family"] == fam]
        if xs:
            print(f"  {fam:18} {np.mean(xs):.3f} (n={len(xs)})")


if __name__ == "__main__":
    main()
