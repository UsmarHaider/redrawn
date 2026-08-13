"""Export everything the browser needs, and serve it.

The interface re-runs the study live: it holds the per-tract crash counts and the
tract adjacency, and rebuilds zones, rates and the correlation in JavaScript as
the reader changes the map. Nothing is precomputed except the tract geometry and
the named partitions, so the number on screen is derived the same way the
number in the README is.
"""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from .config import ARTIFACTS, FIXED_K, PAIRS, PRIMARY_PAIR, REPORTS, SEED, UI
from .data import load_boundaries, load_study
from .geometry import LAT_SCALE, simplify
from .partitions import merge_ladder, official_partitions, partition_correlation

#: Douglas-Peucker tolerance in degrees. ~0.00015 deg is about 15 m, which is
#: invisible at city scale and cuts the payload by roughly 80%.
TOLERANCE = 0.00015
PRECISION = 5


def _paths(kept) -> tuple[list[str], list[float]]:
    """One SVG path per tract, already projected, simplified and rounded.

    Projection is applied here rather than in the browser: longitude is scaled by
    cos(40.7 deg) so the city is not stretched sideways, and latitude is negated
    because SVG's y axis points down. The browser then draws the strings as-is.
    """
    shapes = {s.key: s for s in load_boundaries("tract")}
    out: list[str] = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for geoid in kept["geoid"]:
        shape = shapes[str(geoid)]
        parts = []
        for ring in shape.rings:
            if ring.hole:
                continue
            coords = simplify(ring.coords, TOLERANCE)
            if len(coords) < 3:
                continue
            projected = np.column_stack([coords[:, 0] * LAT_SCALE, -coords[:, 1]])
            rounded = np.round(projected, PRECISION)
            xmin, ymin = min(xmin, rounded[:, 0].min()), min(ymin, rounded[:, 1].min())
            xmax, ymax = max(xmax, rounded[:, 0].max()), max(ymax, rounded[:, 1].max())
            body = " ".join(f"{x:g},{y:g}" for x, y in rounded)
            parts.append("M" + body + "Z")
        out.append("".join(parts))
    return out, [xmin, ymin, xmax - xmin, ymax - ymin]


def export(path=None) -> dict:
    """Write `ui/data.json`: geometry, counts, adjacency and the named maps."""
    study = load_study()
    kept = study.kept
    neighbours = study.neighbours()
    n = kept["n"].to_numpy(np.float64)

    official = official_partitions(kept)
    partitions = {
        "tract": {"labels": official["tract"], "name": "Census tracts", "official": True},
        "nta": {"labels": official["nta"], "name": "Neighborhood tabulation areas", "official": True},
        "cdta": {"labels": official["cdta"], "name": "Community districts", "official": True},
        "borough": {"labels": official["borough"], "name": "Boroughs", "official": True},
    }
    neutral = merge_ladder(neighbours, n, [FIXED_K], np.random.default_rng(SEED))[FIXED_K]
    partitions["neutral"] = {
        "labels": neutral,
        "name": f"A neutral synthetic map (K={FIXED_K})",
        "official": False,
    }
    for key, filename, label in (
        ("max", "labels_max_balanced.npy", f"Drawn to maximise r (K={FIXED_K})"),
        ("min", "labels_min_balanced.npy", f"Drawn to minimise r (K={FIXED_K})"),
    ):
        file = REPORTS / filename
        if file.exists():
            partitions[key] = {"labels": np.load(file), "name": label, "official": False}

    x_name, y_name = PAIRS[PRIMARY_PAIR]
    describe = json.loads((REPORTS / "describe.json").read_text())
    paths, viewbox = _paths(kept)
    payload = {
        "meta": {
            "crashes": int(kept["n"].sum()),
            "tracts": int(len(kept)),
            "fixed_k": FIXED_K,
            "x": x_name,
            "y": y_name,
            "pairs": {k: {"x": v[0], "y": v[1]} for k, v in PAIRS.items()},
            "pointR": {k: describe["point_level"][k]["r"] for k in PAIRS},
            "viewBox": viewbox,
            "years": study.report["years"],
        },
        "paths": paths,
        "borough": kept["borough"].tolist(),
        "name": kept["ntaname"].tolist(),
        "n": kept["n"].astype(int).tolist(),
        "counts": {
            key: kept[key].astype(int).tolist()
            for key in sorted({v for pair in PAIRS.values() for v in pair})
        },
        # Flat CSR adjacency: neighbours of tract i are adj[adjStart[i]:adjStart[i+1]].
        "adjStart": np.concatenate(
            [[0], np.cumsum([len(s) for s in neighbours])]
        ).astype(int).tolist(),
        "adj": [int(j) for s in neighbours for j in sorted(s)],
        "partitions": {
            key: {
                "name": spec["name"],
                "official": spec["official"],
                "k": int(spec["labels"].max() + 1),
                "labels": spec["labels"].astype(int).tolist(),
            }
            for key, spec in partitions.items()
        },
    }

    target = path or (UI / "data.json")
    target.write_text(json.dumps(payload, separators=(",", ":")))

    # Expected correlations for every named map, used by the JavaScript parity
    # test so the browser cannot quietly disagree with the analysis.
    expected = {
        key: {
            pair: partition_correlation(
                spec["labels"],
                n,
                kept[PAIRS[pair][0]].to_numpy(np.float64),
                kept[PAIRS[pair][1]].to_numpy(np.float64),
            )
            for pair in PAIRS
        }
        for key, spec in partitions.items()
    }
    (ARTIFACTS / "expected_r.json").write_text(json.dumps(expected, indent=2))

    size = target.stat().st_size
    print(f"  {target.name}  {size / 1e6:.2f} MB  ({len(payload['paths'])} tracts, "
          f"{len(partitions)} named maps)")
    return {"bytes": size, "partitions": list(partitions), "expected": expected}


class Scoring(BaseModel):
    """Body of POST /api/score.

    Defined at module level on purpose: this module uses `from __future__ import
    annotations`, so FastAPI sees the parameter annotation as the *string*
    "Scoring" and resolves it against module globals. A model nested inside
    `build_app` is invisible there, and the endpoint silently degrades into
    expecting a query parameter named `body`.
    """

    labels: list[int]
    pair: str = PRIMARY_PAIR
    weighted: bool = False


def build_app():
    """FastAPI app serving the interface and a re-scoring endpoint."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    study = load_study()
    kept = study.kept
    n = kept["n"].to_numpy(np.float64)

    app = FastAPI(title="redrawn", description="Does the map change the answer?")

    @app.get("/")
    def index():
        return FileResponse(UI / "index.html")

    @app.get("/data.json")
    def data():
        target = UI / "data.json"
        if not target.exists():
            raise HTTPException(503, "run `make web` to export the interface data")
        return FileResponse(target)

    @app.get("/maup.js")
    def script():
        """The analysis port the page loads. Served explicitly rather than by
        mounting the directory, so only the two intended files are reachable."""
        return FileResponse(UI / "maup.js", media_type="text/javascript")

    @app.get("/api/summary")
    def summary():
        return JSONResponse(
            {
                "crashes": int(kept["n"].sum()),
                "tracts": int(len(kept)),
                "reports": sorted(p.stem for p in REPORTS.glob("*.json")),
            }
        )

    @app.post("/api/score")
    def score(body: Scoring):
        """Score an arbitrary partition of the kept tracts, server side."""
        if body.pair not in PAIRS:
            raise HTTPException(400, f"pair must be one of {sorted(PAIRS)}")
        labels = np.asarray(body.labels, dtype=np.int64)
        if labels.shape != (len(kept),):
            raise HTTPException(400, f"labels must have length {len(kept)}")
        if labels.min() < 0:
            raise HTTPException(400, "labels must be non-negative")
        x_name, y_name = PAIRS[body.pair]
        r = partition_correlation(
            labels,
            n,
            kept[x_name].to_numpy(np.float64),
            kept[y_name].to_numpy(np.float64),
            weighted=body.weighted,
        )
        return {"r": r, "k": int(len(np.unique(labels))), "pair": body.pair}

    return app


app = None


def get_app():
    global app
    if app is None:
        app = build_app()
    return app
