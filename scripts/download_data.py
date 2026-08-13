#!/usr/bin/env python3
"""Fetch the raw inputs: 2.3M NYC crash records and four official boundary layers.

Everything comes from NYC Open Data over the keyless Socrata API. The crash table
is paged because a single request for two million rows times out; the boundary
layers are small enough to take whole.

    python scripts/download_data.py            # skip anything already on disk
    python scripts/download_data.py --force    # re-fetch
"""

from __future__ import annotations

import argparse
import io
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from redrawn.config import BOUNDARY_SETS, CRASH_COLUMNS, CRASHES_ID, RAW, SOCRATA  # noqa: E402

PAGE = 100_000
TIMEOUT = 180


def _get(url: str, params: dict, attempts: int = 4) -> requests.Response:
    """Socrata occasionally returns a 202/503 while it warms a query up."""
    last = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT)
            if response.status_code == 200:
                return response
            last = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:  # pragma: no cover - network
            last = str(exc)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def download_crashes(force: bool = False) -> None:
    target = RAW / "crashes.csv.gz"
    if target.exists() and not force:
        print(f"crashes    already present ({target.stat().st_size / 1e6:.0f} MB)")
        return

    url = f"{SOCRATA}/{CRASHES_ID}.csv"
    frames, offset, started = [], 0, time.time()
    while True:
        params = {
            "$select": ",".join(CRASH_COLUMNS),
            # Ordering by the primary key makes the paging stable; without it
            # Socrata is free to return rows in any order and pages can overlap.
            "$order": "collision_id",
            "$limit": PAGE,
            "$offset": offset,
        }
        chunk = pd.read_csv(io.BytesIO(_get(url, params).content), low_memory=False)
        if chunk.empty:
            break
        frames.append(chunk)
        offset += PAGE
        print(f"  ...{offset:>9,} rows  ({time.time() - started:5.0f}s)", end="\r", flush=True)

    crashes = pd.concat(frames, ignore_index=True)
    crashes.to_csv(target, index=False, compression="gzip")
    print(
        f"crashes    {len(crashes):,} rows -> {target.name} "
        f"({target.stat().st_size / 1e6:.0f} MB, {time.time() - started:.0f}s)"
    )


def download_boundaries(force: bool = False) -> None:
    for name, spec in BOUNDARY_SETS.items():
        target = RAW / f"{name}.geojson"
        if target.exists() and not force:
            print(f"{name:19s} already present ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        # $limit is required: the geojson endpoint otherwise caps at 1,000 features.
        response = _get(f"{SOCRATA}/{spec['id']}.geojson", {"$limit": 50_000})
        target.write_bytes(response.content)
        print(f"{name:19s} -> {target.name} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fetch files already on disk")
    args = parser.parse_args()

    download_boundaries(force=args.force)
    download_crashes(force=args.force)


if __name__ == "__main__":
    main()
