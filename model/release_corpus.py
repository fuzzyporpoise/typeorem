"""Cut a corpus release: the tagged artifact the app deploys from.

The app does not track the generated corpus (TSK-016). It pins one release in
`corpus.lock.json` and fetches it when it deploys, so the pipeline's output needs a
home, and this is it: tar up the generated files, publish them as a release asset on
this repo, and write the lock into the app checkout.

    model/.venv/bin/python model/release_corpus.py --tag corpus-v1

The release is the trust boundary, so the checks run before anything is uploaded:

- the file set is exactly what the app serves (corpus + the backbone pointer);
- `corpus.version.json` agrees with this checkout's emit constants (backbone,
  preprocess, the glyph grid) and with `vectors.meta.json`;
- counts and byte sizes agree across `vectors.meta.json`, `catalog.json`,
  `corpus.version.json`, and the binary;
- every catalog row's `source` is in the public allowlist: the TSK-017 provenance
  gate, run again at the boundary where data leaves the machine.

Publishing is manual and rare, like the backbone publish: the corpus is built here
(the Google Fonts clone, the renders, and the embedding are all local work), and a
release is cut when that output should ship. Needs `GITHUB_TOKEN` with repo scope;
consumers need no token, since the repo is public.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import paths  # noqa: E402
from render import GLYPHS, COLS, ROWS, SIZE, FONT_SIZE, BASELINE_Y, GLYPH_SET_VERSION  # noqa: E402
from emit import BACKBONE, PREPROCESS  # noqa: E402

DEFAULT_REPO = "fuzzyporpoise/typeorem"
CORPUS_FILES = [
    "data/vectors.i8.bin",
    "data/vectors.meta.json",
    "data/catalog.json",
    "data/pca.bin",
    "data/pca.json",
    "data/corpus.version.json",
    "data/reranker.json",
]
POINTER = "model.json"
ALLOWED_SOURCES = {"google", "local"}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def expected_backbone():
    """The emit constants, so a release can never disagree with this checkout."""
    return BACKBONE, PREPROCESS


def validate(site_dir):
    """Check the generated tree before it is published. Returns the parsed
    metadata the release notes need."""
    failures = []
    site_dir = Path(site_dir)
    for rel in CORPUS_FILES + [POINTER]:
        if not (site_dir / rel).is_file():
            failures.append(f"missing {site_dir / rel}")
    if failures:
        return None, failures

    def read(rel):
        return json.loads((site_dir / rel).read_text())

    version = read("data/corpus.version.json")
    meta = read("data/vectors.meta.json")
    catalog = read("data/catalog.json")
    pca = read("data/pca.json")
    pointer = read(POINTER)

    backbone, preprocess = expected_backbone()
    if version.get("backbone") != backbone or version.get("preprocess") != preprocess:
        failures.append(
            f"corpus.version.json says {version.get('backbone')}/{version.get('preprocess')}, "
            f"this checkout emits {backbone}/{preprocess}")
    if meta.get("backbone") != backbone or meta.get("preprocess") != preprocess:
        failures.append(f"vectors.meta.json says {meta.get('backbone')}/{meta.get('preprocess')}")
    if pointer.get("backbone") != backbone:
        failures.append(f"model.json points at {pointer.get('backbone')}")

    grid = version.get("glyphGrid") or {}
    want_grid = {
        "glyphs": "".join(GLYPHS), "cols": COLS, "rows": ROWS, "size": SIZE,
        "fontSize": FONT_SIZE, "baselineY": BASELINE_Y,
    }
    if grid != want_grid:
        failures.append(f"glyphGrid {grid} != this checkout's render grid {want_grid}")
    if version.get("glyphSetVersion") != GLYPH_SET_VERSION:
        failures.append(f"glyphSetVersion {version.get('glyphSetVersion')} != {GLYPH_SET_VERSION}")

    dim = meta.get("dim")
    count = meta.get("count")
    if not isinstance(dim, int) or not isinstance(count, int):
        failures.append("vectors.meta.json is missing dim/count")
    else:
        if len(meta.get("scale", {}).get("values", [])) != count:
            failures.append("per-row scales do not match count")
        size = (site_dir / "data/vectors.i8.bin").stat().st_size
        if size != count * dim:
            failures.append(f"vectors.i8.bin is {size} bytes, expected {count * dim}")
        if catalog.get("count") != count or len(catalog.get("instances", [])) != count:
            failures.append("catalog.json count disagrees with vectors.meta.json")
        if version.get("count") != count:
            failures.append("corpus.version.json count disagrees with vectors.meta.json")
        if pca.get("dim") != dim:
            failures.append(f"pca.json dim {pca.get('dim')} != corpus dim {dim}")
        want_pca = (int(pca.get("backboneDim", 0)) + dim * int(pca.get("backboneDim", 0))) * 4
        got_pca = (site_dir / "data/pca.bin").stat().st_size
        if got_pca != want_pca:
            failures.append(f"pca.bin is {got_pca} bytes, expected {want_pca}")

    instances = catalog.get("instances", [])
    bad = sorted({str(i.get("source")) for i in instances} - ALLOWED_SOURCES)
    if bad:
        failures.append(f"catalog rows carry a non-public source: {', '.join(bad)}")
    if any(i.get("id") != n for n, i in enumerate(instances)):
        failures.append("catalog ids are not row indices")
    house = version.get("houseFaces") or {}
    local = sum(1 for i in instances if i.get("source") == "local")
    if "count" in house and house["count"] != local:
        failures.append(f"houseFaces.count {house['count']} != {local} local rows")

    summary = {
        "count": count, "dim": dim, "backbone": backbone, "preprocess": preprocess,
        "googleFontsCommit": version.get("googleFontsCommit"),
        "policyVersion": version.get("policyVersion"),
        "houseFaces": house.get("family"), "local": local,
    }
    return summary, failures


def build_archive(site_dir):
    """A deterministic tar.gz of the served files: fixed mtimes and owners, sorted
    members, so the same corpus always hashes the same."""
    buffer = io.BytesIO()
    members = {}
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for rel in sorted(CORPUS_FILES + [POINTER]):
            path = Path(site_dir) / rel
            data = path.read_bytes()
            members[f"site/{rel}"] = {"sha256": sha256_bytes(data), "bytes": len(data)}
            info = tarfile.TarInfo(name=f"site/{rel}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    archive = gzip.compress(buffer.getvalue(), mtime=0)
    return archive, members


def api(path, token, method="GET", body=None, raw=None, base="https://api.github.com"):
    url = path if path.startswith("http") else f"{base}{path}"
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "typeorem-release",
               "Accept": "application/vnd.github+json"}
    data, ctype = None, None
    if raw is not None:
        data, ctype = raw, "application/octet-stream"
    elif body is not None:
        data, ctype = json.dumps(body).encode(), "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    if ctype:
        request.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} failed: {e.code} {e.read().decode()[:300]}")
    return json.loads(payload) if payload else {}


def tag_exists(repo, tag, token):
    try:
        api(f"/repos/{repo}/git/ref/tags/{tag}", token)
        return True
    except SystemExit:
        return False


def publish(repo, tag, archive, asset_name, notes, token):
    if tag_exists(repo, tag, token):
        raise SystemExit(f"tag {tag} already exists on {repo}; pick a new tag "
                         f"(releases are immutable handles for the app's lock)")
    release = api(f"/repos/{repo}/releases", token, method="POST", body={
        "tag_name": tag,
        "target_commitish": "main",
        "name": f"corpus {tag}",
        "body": notes,
        "draft": False,
        "prerelease": False,
    })
    asset = api(f"https://uploads.github.com/repos/{repo}/releases/{release['id']}/assets"
                f"?name={asset_name}", token, method="POST", raw=archive)
    return release, asset


def write_lock(lock_path, repo, tag, asset_name, archive, sha, members, url):
    lock = {
        "v": 1,
        "note": "The corpus the app deploys. Fetched, never tracked: run `npm run corpus`.",
        "repo": repo,
        "tag": tag,
        "asset": asset_name,
        "url": url,
        "sha256": sha,
        "bytes": len(archive),
        "files": members,
    }
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="release tag, for example corpus-v1")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo id, owner/name")
    ap.add_argument("--asset-name", default=None, help="default <tag>.tar.gz")
    ap.add_argument("--lock", type=Path, default=None, help="default <app checkout>/corpus.lock.json")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--no-upload", action="store_true", help="write the archive and the lock only")
    ap.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    site = paths.require_site("the release bundles the app's generated corpus")
    asset_name = args.asset_name or f"{args.tag}.tar.gz"
    lock_path = args.lock or (paths.app_root() / "corpus.lock.json")

    summary, failures = validate(site)
    if failures:
        print(f"{len(failures)} problem(s) with {site}:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"corpus ok: {summary['count']} instances, dim {summary['dim']}, "
          f"backbone {summary['backbone']}/{summary['preprocess']}, "
          f"google/commit {summary['googleFontsCommit']}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    archive, members = build_archive(site)
    sha = sha256_bytes(archive)
    out = BASE / "out" / asset_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(archive)
    print(f"wrote {out} ({len(archive) / 1024:.0f} KB, sha256 {sha[:16]}...)")

    url = f"https://github.com/{args.repo}/releases/download/{args.tag}/{asset_name}"
    if not args.no_upload:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise SystemExit("GITHUB_TOKEN is required to publish (or pass --no-upload)")
        notes = args.notes or (
            f"The shipped corpus for the Typeorem app.\n\n"
            f"- {summary['count']} instances, {summary['dim']} dims, "
            f"{summary['local']} bundled house faces ({summary['houseFaces']})\n"
            f"- backbone {summary['backbone']} ({summary['preprocess']} preprocessing)\n"
            f"- policy v{summary['policyVersion']}, google/fonts {summary['googleFontsCommit']}\n"
            f"- fetch it with the app's `npm run corpus`, which verifies these hashes\n")
        release, asset = publish(args.repo, args.tag, archive, asset_name, notes, token)
        print(f"published {release['html_url']} ({asset.get('size')} bytes)")
    else:
        print(f"--no-upload: {url} was not created")

    written = write_lock(lock_path, args.repo, args.tag, asset_name, archive, sha, members, url)
    print(f"wrote {written} (pin this in the app repo)")
    print("\nnext: commit corpus.lock.json in the app repo, then deploy there "
          "(create-pages fetches this tag)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
