"""Publish the exported fp16 backbone to Hugging Face and record where it lives.

The browser fetches the backbone lazily from HF (TSK-010 hosting decision); the
repo keeps the process, not the 43.5 MB binary. This script uploads the graph
and writes site/model.json, the committed pointer the site reads (`url`,
`sha256`, `bytes`). Re-run it whenever the backbone or export recipe changes.

    model/.venv/bin/python model/publish_model.py --repo <user>/<repo>

Needs huggingface_hub and a token (hf auth login, or HF_TOKEN). The repo must
already exist; create it once with --create-repo (if the token may create repos)
or on the web. Use --no-upload to only refresh site/model.json from a local
file. The fp16 graph comes from model/export_onnx.py --fp16.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import paths

OUT = BASE / "out"
FILE = "dinov2_vits14.fp16.onnx"
DEFAULT_REPO = "fuzzyporpoise/typeorem"


def default_graph():
    """Where export_onnx.py wrote the graph: the site's site/models when a site
    checkout is present, else this repo's out/ (the standalone/CI case)."""
    return (paths.site_models() if paths.site_present() else OUT) / FILE


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def upload(repo, path, create):
    from huggingface_hub import HfApi
    api = HfApi()
    if create:
        api.create_repo(repo, repo_type="model", exist_ok=True)
    card = BASE / "MODEL_CARD.md"
    files = [(path, FILE)]
    if card.exists():
        files.append((card, "README.md"))
    for local, remote in files:
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type="model")
        except Exception as e:  # noqa: BLE001 - surface the fix, not the traceback
            raise SystemExit(
                f"upload failed: {e}\n"
                f"Create the model repo once (web, or --create-repo with a token that may "
                f"create repos), then re-run: https://huggingface.co/new"
            )
        print(f"uploaded {remote} to {repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=default_graph())
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face repo id, user/name")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--no-upload", action="store_true", help="write model.json only")
    ap.add_argument("--create-repo", action="store_true", help="create the repo first (needs create permission)")
    args = ap.parse_args()
    if not args.file.exists():
        raise SystemExit(f"missing {args.file}; run model/export_onnx.py --fp16 first")

    if not args.no_upload:
        upload(args.repo, args.file, args.create_repo)

    url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{FILE}"
    payload = {
        "v": 1, "backbone": "dinov2_vits14", "preprocess": "imagenet", "dim": 384,
        "file": FILE, "fp16": True,
        "bytes": args.file.stat().st_size, "sha256": sha256(args.file),
        "repo": args.repo, "revision": args.revision, "url": url,
    }
    # The pointer is the site's, so the handoff writes it into the site checkout.
    # Standalone (CI just published to HF), record it here instead and print it.
    target = (paths.site_dir() if paths.site_present() else OUT) / "model.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {target} (url {url})")
    if not paths.site_present():
        print("no site checkout next to this repo; the site's site/model.json was not updated")


if __name__ == "__main__":
    main()
