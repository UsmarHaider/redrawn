"""Capture the README screenshot from the running interface.

Two states, because a single picture would not make the point: the official
community-district map, and the map the search drew to maximise the
correlation. Same crashes, different boundaries, different answer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

#: (output, the partition to select, page height)
SHOTS = [
    ("ui-official.png", "cdta", 1240),
    ("ui-gerrymandered.png", "max", 1240),
]


def capture(url: str, out: Path, height: int, chrome: str) -> None:
    subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=2", f"--window-size=1240,{height}",
            "--virtual-time-budget=12000", f"--screenshot={out}", url,
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )
    print(f"  {out.relative_to(DOCS.parent)} ({out.stat().st_size / 1e3:.0f} kB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--chrome", default=CHROME, help="path to a Chrome/Chromium binary")
    args = ap.parse_args()

    if not Path(args.chrome).exists():
        print(f"Chrome not found at {args.chrome}; pass --chrome", file=sys.stderr)
        return 1

    (DOCS).mkdir(parents=True, exist_ok=True)
    for name, partition, height in SHOTS:
        # The page reads ?map= on load, so each shot lands on the intended map.
        url = f"http://127.0.0.1:{args.port}/?map={partition}"
        capture(url, DOCS / name, height, args.chrome)
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
