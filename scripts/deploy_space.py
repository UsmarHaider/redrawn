"""Publish the interface to a Hugging Face static Space.

The interface is already a static site -- one HTML file, one JavaScript file and
one JSON payload, with every correlation computed in the browser -- so the
"deployment" is those three files plus Space front-matter. No server, no
container, and the hosted demo is byte-identical to the one `make serve` runs.

    HF_TOKEN=... python scripts/deploy_space.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "redrawn" / "ui"

# colorFrom/colorTo must come from Hugging Face's fixed list; "orange" is rejected.
SPACE_README = """---
title: redrawn
emoji: 🗺️
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Redraw the district lines and the answer changes
---

# redrawn

**Does the map change the answer?** 1.9 million New York City crashes, one
question — do neighbourhoods where speeding is cited more often have more
injurious crashes? — and a different answer from every map of the city.

Between individual crashes the correlation is **+0.055**. Across census tracts
it is +0.262, across police precincts +0.469, across community districts +0.551,
across boroughs **+0.709**. Contiguous equal-size districts drawn deliberately
span **+0.07 to +0.83**. Not one crash record changes; only the boundaries move.

This page re-runs the analysis in your browser. It holds the per-tract crash
counts and the tract adjacency graph and recomputes the zone rates and the
correlation on every interaction, so nothing you see is a stored answer — and
the "draw a new map" button really does build a fresh contiguous partition of
2,310 census tracts.

Data: [NYPD Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
and Department of City Planning boundaries, via NYC Open Data.

Method, controls and the full write-up: <https://github.com/UsmarHaider/redrawn>
"""

FILES = ["index.html", "maup.js", "data.json"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", default=os.environ.get("HF_USERNAME", "usmar"))
    ap.add_argument("--repo", default=None, help="defaults to <user>/redrawn")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("set HF_TOKEN (see .env.example)", file=sys.stderr)
        return 1

    missing = [f for f in FILES if not (UI / f).exists()]
    if missing:
        print(f"missing {missing}; run `make web` first", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    repo_id = args.repo or f"{args.user}/redrawn"
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=args.private,
        exist_ok=True,
    )

    staged = ROOT / "artifacts" / "space"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "README.md").write_text(SPACE_README)
    for name in FILES:
        (staged / name).write_bytes((UI / name).read_bytes())

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(staged),
        commit_message="Publish the redrawn interface",
    )
    print(f"https://huggingface.co/spaces/{repo_id}")
    print(f"https://{repo_id.replace('/', '-')}.static.hf.space/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
