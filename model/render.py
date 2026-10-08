import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import fontTools.ttLib as ttLib

GLYPH_SET_VERSION = 1
GLYPHS = list("ABGJQRSW" + "aegnosrt" + "138" + "&@.,")
COLS, ROWS = 6, 4
SIZE = 224
FONT_SIZE = 38
BASELINE_Y = 44

_axis_cache = {}


def axis_order(path):
    if path not in _axis_cache:
        f = ttLib.TTFont(str(path), lazy=True)
        _axis_cache[path] = [(a.axisTag, a.defaultValue) for a in f["fvar"].axes] if "fvar" in f else []
        f.close()
    return _axis_cache[path]


def cell_center(index):
    col, row = index % COLS, index // COLS
    cw, ch = SIZE / COLS, SIZE / ROWS
    return col * cw + cw / 2, row * ch + BASELINE_Y


def render_instance(path, coords):
    img = Image.new("L", (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(path), FONT_SIZE)
    if coords:
        values = [coords.get(tag, default) for tag, default in axis_order(path)]
        font.set_variation_by_axes(values)
    for i, ch in enumerate(GLYPHS):
        x, y = cell_center(i)
        draw.text((x, y), ch, font=font, anchor="ms", fill=0)
    return img


def render_corpus(instances, dirs, fonts_root, out_dir):
    fonts_root = Path(fonts_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    failures = []
    for inst in instances:
        path = fonts_root / dirs[inst["family"]] / inst["file"]
        png = out_dir / f"{inst['id']}.png"
        try:
            render_instance(path, inst["coords"]).save(png)
        except Exception as e:
            failures.append((inst["id"], str(e)))
            continue
        hashes[inst["id"]] = hashlib.sha256(png.read_bytes()).hexdigest()
    return hashes, failures
