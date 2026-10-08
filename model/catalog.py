import json
import re
import sys
from pathlib import Path

CATEGORY_MAP = {
    "SANS_SERIF": "sans-serif",
    "SERIF": "serif",
    "DISPLAY": "display",
    "HANDWRITING": "handwriting",
    "MONOSPACE": "monospace",
}

TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|[{}:]|[^\s{}:]+')


def _strip_line_comments(text):
    return re.sub(r"(?m)^\s*#.*$", "", text)


def _value(tok):
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1]
    try:
        return float(tok) if ("." in tok or "e" in tok.lower()) else int(tok)
    except ValueError:
        return tok


def _add(d, key, val):
    if key in d:
        if not isinstance(d[key], list):
            d[key] = [d[key]]
        d[key].append(val)
    else:
        d[key] = val


def parse_metadata(text):
    toks = TOKEN.findall(_strip_line_comments(text))
    pos = 0

    def message():
        nonlocal pos
        d = {}
        while pos < len(toks):
            t = toks[pos]
            if t == "}":
                pos += 1
                return d
            key = t
            pos += 1
            if pos < len(toks) and toks[pos] == ":":
                pos += 1
                _add(d, key, _value(toks[pos]))
                pos += 1
            elif pos < len(toks) and toks[pos] == "{":
                pos += 1
                _add(d, key, message())
        return d

    return message()


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def first(v, default=None):
    items = as_list(v)
    return items[0] if items else default


def normalize_family(raw):
    fonts = []
    for f in as_list(raw.get("fonts")):
        fonts.append({
            "filename": first(f.get("filename"), ""),
            "style": first(f.get("style"), "normal"),
            "weight": int(first(f.get("weight"), 400)),
            "post_script_name": first(f.get("post_script_name"), ""),
            "full_name": first(f.get("full_name"), ""),
        })
    axes = []
    for a in as_list(raw.get("axes")):
        axes.append({
            "tag": a.get("tag", ""),
            "min": float(a.get("min_value", 0)),
            "max": float(a.get("max_value", 0)),
        })
    categories = as_list(raw.get("category"))
    name = as_list(raw.get("name"))[0] if raw.get("name") else ""
    return {
        "family": name,
        "category": CATEGORY_MAP.get(categories[0], str(categories[0]).lower()) if categories else "",
        "categories": [CATEGORY_MAP.get(c, str(c).lower()) for c in categories],
        "subsets": as_list(raw.get("subsets")),
        "axes": axes,
        "variable": any(("[" in f["filename"]) for f in fonts),
        "fonts": fonts,
    }


def load_catalog(root):
    root = Path(root)
    families = []
    for path in sorted(root.glob("*/*/METADATA.pb")):
        fam = normalize_family(parse_metadata(path.read_text(encoding="utf-8")))
        fam["dir"] = str(path.parent.relative_to(root))
        families.append(fam)
    return families


def google_fonts_commit(root):
    head = Path(root) / ".git"
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def cross_check(families, endpoint_json_path):
    data = json.loads(Path(endpoint_json_path).read_text())
    published = {f["family"] for f in data["familyMetadataList"]}
    local = {f["family"] for f in families}
    return {
        "publishedCount": len(published),
        "localCount": len(local),
        "localNotPublished": sorted(local - published),
        "publishedNotLocal": sorted(published - local),
    }


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / ".cache" / "google-fonts"
    fams = load_catalog(root)
    print(f"{len(fams)} families, commit {google_fonts_commit(root)}", file=sys.stderr)
    for f in fams[:3]:
        print(json.dumps(f, ensure_ascii=False))
