"""Where this repo ends and the site checkout begins.

The project is two pieces (TSK-016): **the science** (this repo: the corpus
pipeline, the harnesses, `docs/roll-your-own.md`) and **the site** (a separate
checkout that serves the corpus and owns the deployed page). The pipeline writes
its corpus into the site's `site/data/`, and the harnesses read the metric the
site ships from its `site/js/`. Both directions resolve here, once:

    TYPEOREM_SITE_DIR   path to the site checkout (the directory holding
                        `site/index.html`). When it is unset, the first of these
                        that exists wins, so sibling checkouts work unconfigured:
                            <this repo>/site            (the shared-tree layout)
                            <this repo>/../site
                            <this repo>/../typeorem-site

Nothing in the science writes anything else into the site tree.
"""

import os
from pathlib import Path

BASE = Path(__file__).resolve().parent

VAR = "TYPEOREM_SITE_DIR"


def candidates():
    """The places a site checkout is looked for, in order."""
    return [BASE.parent / "site", BASE.parent.parent / "site", BASE.parent.parent / "typeorem-site"]


def site_dir():
    """The site checkout root (the directory that holds `site/index.html`)."""
    env = os.environ.get(VAR)
    if env:
        return Path(env).expanduser().resolve()
    for candidate in candidates():
        if (candidate / "index.html").is_file():
            return candidate.resolve()
    return candidates()[0].resolve()


def site_data():
    return site_dir() / "data"


def site_models():
    """Where the exported browser graph is written (`site/models/`, gitignored)."""
    return site_dir() / "models"


def house_font():
    """The self-hosted house face the corpus bundles (Commit Mono, OFL). It
    lives in the site's `site/fonts/`, so the science reads it from there rather
    than keeping a second copy."""
    return site_dir() / "fonts" / "CommitMono-VF.woff2"


def site_present():
    """True when a site checkout is where we expect it. The publish path uses the
    site when it is there (the local handoff) and stands alone when it is not
    (CI, where only the Hugging Face upload matters)."""
    return (site_dir() / "index.html").is_file()


def require_site(purpose):
    """Fail loudly when the site checkout is missing, instead of writing into a
    directory that does not exist or silently emitting a corpus nobody reads."""
    site = site_dir()
    if not (site / "index.html").is_file():
        raise SystemExit(
            f"no site checkout at {site} ({purpose}).\n"
            f"Clone the site next to this repo, or point {VAR} at it:\n"
            f"    {VAR}=/path/to/typeorem-site model/.venv/bin/python model/build_corpus.py"
        )
    return site
