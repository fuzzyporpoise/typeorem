"""Dump per-instance render features for the shipped + private corpus.

The metric ships and scores only the embedding contrast, a body-legibility term,
and a gated x-height axis; the renderer also measures stroke contrast, aperture,
and 'o' aspect (proportion), which nothing scores yet. This dumps those, plus
x-height, for every row of the shipped corpus and (optionally) a private corpus,
aligned to the same order the JS harness merges them (shipped rows, then private).

    model/.venv/bin/python model/validate/features.py \
        --out ~/.cache/typeorem/features.json

The output is derived from licensed renders, so keep it under the private root
(the default), never in either checkout.
"""

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import paths  # noqa: E402
from legibility import features_for, o_aspect  # noqa: E402
from instances import slug  # noqa: E402

CACHE_224 = BASE / ".cache" / "224"


def default_private_root():
    import os
    base = os.environ.get("XDG_CACHE_HOME")
    base = Path(base).expanduser() if base else Path.home() / ".cache"
    return (base / "typeorem").resolve()


def png_for(row, private_root):
    """The render for a row. The pipeline and the private writer both name a PNG
    by <family-slug>-<weight>-<style>; only the directory differs."""
    name = f"{slug(row['family'])}-{row['weight']}-{row['style']}.png"
    if row.get("source") == "private":
        return private_root / ".cache" / "224" / name
    return CACHE_224 / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", default=None, help="private root (else the default, if present)")
    ap.add_argument("--shipped-only", action="store_true", help="skip the private corpus")
    ap.add_argument("--out", default=None, help="output JSON (default <private>/features.json)")
    args = ap.parse_args()

    paths.require_site("the feature dump reads the shipped corpus")
    site_data = paths.site_data()
    rows = json.loads((site_data / "catalog.json").read_text())["instances"]

    private_root = Path(args.private).expanduser().resolve() if args.private else None
    if not args.shipped_only and private_root is None and (default_private_root() / "corpus" / "catalog.json").is_file():
        private_root = default_private_root()
    if private_root is not None:
        private_catalog = private_root / "corpus" / "catalog.json"
        if private_catalog.is_file():
            rows = rows + json.loads(private_catalog.read_text())["instances"]

    if args.out:
        out = Path(args.out).expanduser()
    elif private_root is not None:
        out = private_root / "features.json"
    else:
        raise SystemExit("--shipped-only needs --out <path> (refusing to guess a location)")
    out.parent.mkdir(parents=True, exist_ok=True)

    features, missing = [], 0
    for k, row in enumerate(rows):
        png = png_for(row, private_root or Path("/nonexistent"))
        f = features_for(png) if png.is_file() else None
        if f is None:
            missing += 1
            features.append({"xh": row.get("xh"), "contrast": None, "aperture": None, "o_aspect": None})
            continue
        f["o_aspect"] = o_aspect(png)
        features.append(f)
        if k and k % 1000 == 0:
            print(f"  {k}/{len(rows)}", flush=True)

    out.write_text(json.dumps({"count": len(rows), "features": features}, separators=(",", ":")))
    print(f"wrote {out} ({len(rows)} rows, {missing} without a render)")


if __name__ == "__main__":
    main()
