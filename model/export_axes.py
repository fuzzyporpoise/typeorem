"""Re-emit the per-instance render axes (x-height, width) into the shipped
catalog without a corpus rebuild.

The axes come from the cached 224 px renders, so this needs the pipeline's render
step to have run (`model/.cache/224`). It rewrites `site/data/catalog.json` in
place and leaves the vectors, the PCA, and the projection space untouched, so a
rebuild that would refit the PCA (and stale the app's projection fixture) is not
required just to add an axis.

    TYPEOREM_SITE_DIR=/path/to/typeorem model/.venv/bin/python model/export_axes.py
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import paths  # noqa: E402
from legibility import render_axes  # noqa: E402
from instances import slug  # noqa: E402
from catalog import write_json  # noqa: E402

CACHE_224 = BASE / ".cache" / "224"


def main():
    paths.require_site("the render axes are written into the site's site/data")
    site = paths.site_data()
    catalog = json.loads((site / "catalog.json").read_text())
    rows = catalog["instances"]

    # render_axes keys by the render id, <family-slug>-<weight>-<style>, the same
    # formula the pipeline and the private writer use.
    ids = [f"{slug(r['family'])}-{r['weight']}-{r['style']}" for r in rows]
    axes = render_axes([{"id": i} for i in ids], CACHE_224)

    with_xh = with_width = 0
    for row, iid in zip(rows, ids):
        a = axes.get(iid)
        row["xh"] = a["xh"] if a else None
        row["width"] = a.get("width") if a else None
        with_xh += row["xh"] is not None
        with_width += row["width"] is not None
    write_json(site / "catalog.json", catalog)
    print(f"{len(rows)} rows: xh {with_xh}, width {with_width}")


if __name__ == "__main__":
    main()
