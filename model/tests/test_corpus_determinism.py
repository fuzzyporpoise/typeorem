import hashlib
import json
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import paths
from render import render_instance

passes = 0
fails = 0


def ok(label, cond):
    global passes, fails
    if cond:
        passes += 1
    else:
        fails += 1
        print(f"FAIL {label}")


instances = json.loads((BASE / "out" / "instances.json").read_text())
families = {f["family"]: f for f in json.loads((BASE / "out" / "catalog.raw.json").read_text())}
hashes = json.loads((BASE / "out" / "render-hashes.json").read_text())
root = BASE / ".cache" / "google-fonts"
site = paths.site_data()
tmp = Path(tempfile.mkdtemp())
by_id = {i["id"]: i for i in instances}

fixtures = ["montserrat-400-normal", "montserrat-900-normal", "montserrat-400-italic",
            "lato-400-normal", "playfair-display-700-italic", "noto-sans-jp-400-normal"]
fixtures += [instances[i]["id"] for i in range(0, len(instances), 400)]

for iid in fixtures:
    if iid not in by_id:
        continue
    inst = by_id[iid]
    png = tmp / f"{iid}.png"
    render_instance(root / families[inst["family"]]["dir"] / inst["file"], inst["coords"]).save(png)
    got = hashlib.sha256(png.read_bytes()).hexdigest()
    ok(f"{iid} renders deterministically", got == hashes[iid])

# The shipped-corpus checks read the site checkout; the render determinism above
# does not, so a standalone science clone still runs this test.
if paths.site_present():
    meta = json.loads((site / "vectors.meta.json").read_text())
    catalog = json.loads((site / "catalog.json").read_text())
    ok("vectors.meta count matches catalog count", meta["count"] == catalog["count"])
    ok("count matches rendered instances", meta["count"] == len(hashes))
    ok("bin size = count * dim", (site / "vectors.i8.bin").stat().st_size == meta["count"] * meta["dim"])
    ok("catalog ids are row indices", all(c["id"] == i for i, c in enumerate(catalog["instances"])))
    ok("per-row scales present", len(meta["scale"]["values"]) == meta["count"])
else:
    print(f"no site checkout at {site}; skipped the shipped-corpus checks")
print(f"{passes} passed, {fails} failed")
sys.exit(1 if fails else 0)
