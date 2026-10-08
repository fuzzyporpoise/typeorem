"""Where this repo ends and the app checkout begins.

The project is two pieces (TSK-016): **the science** (this repo: the corpus
pipeline, the harnesses, `docs/roll-your-own.md`) and **the app** (a separate
checkout that serves the corpus and owns the deployed page). The pipeline writes
its corpus into the app's `site/data/`, and the harnesses read the metric the app
ships from its `site/js/`. Both directions resolve here, once:

    TYPEOREM_SITE_DIR   path to the app checkout (the directory that contains
                        `site/index.html`; the repo root, not `site/` itself).
                        When it is unset, the first of these that holds a
                        `site/index.html` wins, so checkouts that sit next to
                        each other need no configuration:
                            <this repo>             (the shared-tree layout)
                            <this repo>/..
                            <this repo>/../typeorem-site

Nothing in the science writes anything else into the app tree.
"""

import os
from pathlib import Path

BASE = Path(__file__).resolve().parent

VAR = "TYPEOREM_SITE_DIR"


def candidates():
    """The places an app checkout is looked for, in order."""
    return [BASE.parent, BASE.parent.parent, BASE.parent.parent / "typeorem-site"]


def app_root():
    """The app checkout root: the directory whose `site/` the science writes into."""
    env = os.environ.get(VAR)
    if env:
        return Path(env).expanduser().resolve()
    for candidate in candidates():
        if (candidate / "site" / "index.html").is_file():
            return candidate.resolve()
    return candidates()[0].resolve()


def site_dir():
    """The served site directory (`<app>/site`): index.html, css/, js/, data/."""
    return app_root() / "site"


def site_data():
    return site_dir() / "data"


def site_models():
    """Where the exported browser graph is written (`site/models/`, gitignored)."""
    return site_dir() / "models"


def house_font():
    """The self-hosted house face the corpus bundles (Commit Mono, OFL). It
    lives in the app's `site/fonts/`, so the science reads it from there rather
    than keeping a second copy."""
    return site_dir() / "fonts" / "CommitMono-VF.woff2"


def site_present():
    """True when an app checkout is where we expect it. The publish path uses it
    when it is there (the local handoff) and stands alone when it is not (CI,
    where only the Hugging Face upload matters)."""
    return (site_dir() / "index.html").is_file()


def require_site(purpose):
    """Fail loudly when the app checkout is missing, instead of writing into a
    directory that does not exist or silently emitting a corpus nobody reads."""
    if not site_present():
        raise SystemExit(
            f"no app checkout at {app_root()} ({purpose}).\n"
            f"Clone the app next to this repo, or point {VAR} at it:\n"
            f"    {VAR}=/path/to/typeorem model/.venv/bin/python model/build_corpus.py"
        )
    return site_dir()
