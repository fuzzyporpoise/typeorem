"""Style-vs-content diagnostic, after FontBench and FRB (arXiv:2603.08497).

An encoder can be "typography blind": it reads the text content and ignores the
font, so two fonts showing the same word look closer than one font showing two
words (a Stroop-style confound). FRB and FontBench exist to catch exactly that
before a backbone swap silently regresses the embedding.

This renders several TEXTS in each of several FONTS (the corpus holds one render
per instance, so this pass renders its own), embeds them with the backbone
`make_space.py` uses, and asks which grouping the space reflects:

  same-font sim   mean cosine over (same font, different text)  -> style
  same-text sim   mean cosine over (same text, different font)  -> content
  style bias      same-font minus same-text; > 0 means style-tracked
  NN composition  for every render, is its nearest neighbour same-font (style)
                  or same-text (content), against the chance rates

A style-tracking space has a positive bias and same-font nearest neighbours; a
typography-blind space flips both. The screen guards a backbone swap (the
paired-glyph fine-tune, sweep direction 1): run it on each candidate and keep the
one that is style-tracked. It reads the shipped catalog for the eligible faces
and the backbone cache for the font files, and never touches `site/data`.

    model/.venv/bin/python model/validate/style_content.py
    model/.venv/bin/python model/validate/style_content.py --backbone vgg16
    model/.venv/bin/python model/validate/style_content.py --fonts 16 --json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import paths  # noqa: E402
from render import SIZE  # noqa: E402

CACHE = BASE / ".cache"
OUT = BASE / "out"
FONT_ROOT = CACHE / "google-fonts"

# A curated, category-spanning set so the diagnostic reads the way a designer
# would: serif / sans / mono / display / handwriting, canonical 400-normal.
FAMILIES = [
    "Playfair Display", "Lora", "Merriweather",
    "Montserrat", "Roboto", "Inter", "Oswald",
    "Roboto Mono", "Pacifico", "Lobster", "Abril Fatface", "Bebas Neue",
]
# Content variation: four different strings, differing in length and character
# mix, each rendered large and centered.
TEXTS = ["Hamburgefonstiv", "Typeorem", "quixotic", "0123456789"]


def load_fonts(want):
    """Resolve each family to its canonical 400-normal file, from the cached
    clone. Returns [(family, path)] for the families that resolve, and the
    families that did not (missing from the catalog or the clone)."""
    instances = json.loads((OUT / "instances.json").read_text())
    families = {f["family"]: f for f in json.loads((OUT / "catalog.raw.json").read_text())}
    by_fam = {}
    for inst in instances:
        if inst["weight"] != 400 or inst["style"] != "normal" or not inst.get("pairingEligible"):
            continue
        by_fam.setdefault(inst["family"], inst)
    found, missing = [], []
    for fam in want:
        inst = by_fam.get(fam)
        meta = families.get(fam)
        path = FONT_ROOT / meta["dir"] / inst["file"] if inst and meta else None
        if path and path.is_file():
            found.append((fam, path))
        else:
            missing.append(fam)
    return found, missing


def render_string(path, text, margin=16):
    """One text in one font, centered on the shipped grid canvas (grayscale ->
    RGB), scaled down until it fits, so every render is the same 224x224 the
    corpus uses."""
    draw_font = None
    for px in range(96, 8, -4):
        font = ImageFont.truetype(str(path), px)
        x0, y0, x1, y1 = font.getbbox(text)
        if x1 - x0 <= SIZE - 2 * margin and y1 - y0 <= SIZE - 2 * margin:
            draw_font = font
            break
    font = draw_font or ImageFont.truetype(str(path), 12)
    img = Image.new("L", (SIZE, SIZE), 255)
    ImageDraw.Draw(img).text((SIZE / 2, SIZE / 2), text, font=font, anchor="mm", fill=0)
    return np.asarray(img.convert("RGB"))


def embed(backbone, images):
    if backbone == "vgg16":
        from embed import VGG16Pool5 as Encoder
    else:
        from embed import DINOv2 as Encoder
    enc = Encoder()
    print(f"embedding {len(images)} renders with {enc.name} on {enc.device}")
    return np.concatenate([enc.embed_batch(images[s:s + 16]) for s in range(0, len(images), 16)])


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return x / n


def analyse(E):
    """E is (F, T, D) unit-normalized. Returns the style/content statistics.

    Two views: the raw mean cosines (style = same font, different text; content =
    same text, different font), and the per-render comparison. The per-render one
    is the sharper test: for each render, its single best same-font neighbour
    (style) against its best same-text neighbour (content); the winner is which
    grouping the space actually keyed on."""
    F, T = E.shape[0], E.shape[1]
    style = [float(E[f, t] @ E[f, u]) for f in range(F) for t in range(T) for u in range(t + 1, T)]
    content = [float(E[f, t] @ E[g, t]) for t in range(T) for f in range(F) for g in range(f + 1, F)]
    flat = E.reshape(F * T, -1)
    sim = flat @ flat.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argmax(sim, axis=1)
    same_font = same_text = other = style_wins = 0
    for k in range(F * T):
        f, t = divmod(k, T)
        g, u = divmod(int(nn[k]), T)
        if g == f:
            same_font += 1
        elif u == t:
            same_text += 1
        else:
            other += 1
        best_style = max(float(E[f, t] @ E[f, u2]) for u2 in range(T) if u2 != t)
        best_content = max(float(E[f, t] @ E[g2, t]) for g2 in range(F) if g2 != f)
        if best_style > best_content:
            style_wins += 1
    n = F * T
    return {
        "fonts": F, "texts": T,
        "style_sim": float(np.mean(style)),
        "content_sim": float(np.mean(content)),
        "style_bias": float(np.mean(style) - np.mean(content)),
        "style_win_rate": style_wins / n,
        "same_font_nn": same_font / n,
        "same_text_nn": same_text / n,
        "other_nn": other / n,
        "chance_font_nn": (T - 1) / (n - 1),
        "chance_text_nn": (F - 1) / (n - 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "vgg16"])
    ap.add_argument("--fonts", type=int, default=len(FAMILIES), help="cap the family count")
    ap.add_argument("--json", action="store_true", help="emit the statistics as JSON")
    args = ap.parse_args()

    paths.require_site("the diagnostic reads the shipped catalog for the eligible faces")
    found, missing = load_fonts(FAMILIES[: args.fonts])
    if len(found) < 4:
        print(f"need at least 4 resolvable families, found {len(found)}: {[f for f, _ in found]}")
        print("is the Google Fonts clone (model/.cache/google-fonts) present?")
        sys.exit(2)

    labels, images = [], []
    for fam, path in found:
        for text in TEXTS:
            labels.append((fam, text))
            images.append(render_string(path, text))
    E = unit(embed(args.backbone, images)).reshape(len(found), len(TEXTS), -1)
    m = analyse(E)
    m["backbone"] = args.backbone
    m["missing"] = missing

    if args.json:
        print(json.dumps(m, indent=1))
        return
    print(f"\nstyle-vs-content diagnostic, backbone = {args.backbone}")
    print(f"{m['fonts']} fonts x {m['texts']} texts = {m['fonts'] * m['texts']} renders"
          + (f"  (skipped {', '.join(missing)})" if missing else ""))
    print(f"\n  same-font sim (style)    {m['style_sim']:+.3f}")
    print(f"  same-text sim (content)  {m['content_sim']:+.3f}")
    print(f"  style bias               {m['style_bias']:+.3f}   (> 0 = style-tracked)")
    print(f"  style wins per render    {100 * m['style_win_rate']:.0f}%   "
          f"(best same-font neighbour beats best same-text)")
    print(f"\n  nearest neighbour        same-font {100 * m['same_font_nn']:.0f}%   "
          f"same-text {100 * m['same_text_nn']:.0f}%   other {100 * m['other_nn']:.0f}%")
    print(f"  chance                   same-font {100 * m['chance_font_nn']:.0f}%   "
          f"same-text {100 * m['chance_text_nn']:.0f}%")
    verdict = "style-tracked" if m["style_win_rate"] > 0.5 and m["style_bias"] > 0 \
        else "content-dominant (style-weak)"
    print(f"\n  verdict: {verdict}")


if __name__ == "__main__":
    main()
