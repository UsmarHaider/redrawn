"""Build the study table: crashes cleaned, geocoded and assigned to every zoning.

The expensive step -- two million point-in-polygon tests against four boundary
layers -- runs once and is cached in `artifacts/`. Everything downstream reads
the cache.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ARTIFACTS,
    BOUNDARY_SETS,
    MIN_CRASHES_PER_TRACT,
    NYC_BBOX,
    RAW,
    YEAR_MAX,
    YEAR_MIN,
)
from .geometry import Shape, adjacency, areas, assign_points, centroids, load_geojson

#: Point-level indicators. Each is a property of a single crash, so the
#: individual-level correlation between any pair of them is well defined --
#: which is the whole basis of the comparison against zone-level correlations.
INDICATORS = {
    "injury": "at least one person injured",
    "distraction": "driver inattention/distraction cited first",
    "speeding": "unsafe speed cited first",
    "vru": "a pedestrian or cyclist was injured",
    "unspecified": "no contributing factor recorded",
}

CRASH_TABLE = ARTIFACTS / "crashes.parquet"
TRACT_TABLE = ARTIFACTS / "tracts.parquet"
ZONE_TABLE = ARTIFACTS / "zones.parquet"
BUILD_REPORT = ARTIFACTS / "build.json"


def load_boundaries(name: str) -> list[Shape]:
    spec = BOUNDARY_SETS[name]
    return load_geojson(RAW / f"{name}.geojson", spec["key"])


def clean_crashes() -> tuple[pd.DataFrame, dict]:
    """Load the raw crash export and apply the study filters, counting each drop."""
    raw = pd.read_csv(RAW / "crashes.csv.gz", low_memory=False)
    steps = {"downloaded": len(raw)}

    raw["year"] = pd.to_datetime(raw["crash_date"], errors="coerce").dt.year
    raw = raw[raw["year"].between(YEAR_MIN, YEAR_MAX)]
    steps["in_window"] = len(raw)

    lon, lat = raw["longitude"], raw["latitude"]
    xmin, ymin, xmax, ymax = NYC_BBOX
    # A surprising number of rows carry (0, 0) or a null island coordinate.
    good = lon.between(xmin, xmax) & lat.between(ymin, ymax)
    raw = raw[good]
    steps["geocoded"] = len(raw)

    injured = raw["number_of_persons_injured"].fillna(0)
    factor = raw["contributing_factor_vehicle_1"].fillna("")
    frame = pd.DataFrame(
        {
            "lon": raw["longitude"].to_numpy(np.float64),
            "lat": raw["latitude"].to_numpy(np.float64),
            "year": raw["year"].to_numpy(np.int16),
            "injury": (injured > 0).to_numpy(np.int8),
            "distraction": (factor == "Driver Inattention/Distraction").to_numpy(np.int8),
            "speeding": (factor == "Unsafe Speed").to_numpy(np.int8),
            "vru": (
                (raw["number_of_pedestrians_injured"].fillna(0) > 0)
                | (raw["number_of_cyclist_injured"].fillna(0) > 0)
            ).to_numpy(np.int8),
            "unspecified": (factor.isin(["", "Unspecified"])).to_numpy(np.int8),
        }
    )
    return frame.reset_index(drop=True), steps


def build(progress: bool = True) -> dict:
    """Clean, geocode against all four layers, and write the cached tables."""
    started = time.time()
    crashes, steps = clean_crashes()
    if progress:
        print(f"crashes    {len(crashes):,} usable of {steps['downloaded']:,} downloaded")

    lon = crashes["lon"].to_numpy()
    lat = crashes["lat"].to_numpy()

    layers: dict[str, list[Shape]] = {}
    keys: dict[str, list[str]] = {}
    for name in BOUNDARY_SETS:
        shapes = load_boundaries(name)
        layers[name] = shapes
        keys[name] = [s.key for s in shapes]
        t = time.time()
        position = assign_points(shapes, lon, lat)
        crashes[name] = position.astype(np.int32)
        matched = int((position >= 0).sum())
        if progress:
            print(
                f"{name:19s} {len(shapes):>5,} zones  "
                f"{matched / len(crashes):6.2%} of crashes matched  ({time.time() - t:.0f}s)"
            )

    # Self-check: DCP publishes the tract -> NTA crosswalk in the tract layer, so
    # routing a crash through its tract must agree with testing it against the
    # NTA polygons directly. Any disagreement is a bug in the point-in-polygon
    # code or a sliver in the layers, and it is worth knowing which.
    tract_nta = np.array([s.props["nta2020"] for s in layers["tract"]])
    nta_keys = np.array(keys["nta"])
    both = (crashes["tract"] >= 0) & (crashes["nta"] >= 0)
    via_tract = tract_nta[crashes.loc[both, "tract"].to_numpy()]
    direct = nta_keys[crashes.loc[both, "nta"].to_numpy()]
    crosswalk_agreement = float((via_tract == direct).mean())
    if progress:
        print(f"crosswalk  tract->NTA agrees with direct NTA test on {crosswalk_agreement:.4%}")

    # Everything downstream is tract-based, so drop crashes outside the tract layer.
    crashes = crashes[crashes["tract"] >= 0].reset_index(drop=True)

    tracts = _tract_table(layers["tract"], crashes)
    zones = _zone_tables(layers, keys, crashes)

    crashes.to_parquet(CRASH_TABLE, index=False)
    tracts.to_parquet(TRACT_TABLE, index=False)
    zones.to_parquet(ZONE_TABLE, index=False)

    kept = tracts[tracts["keep"]]
    report = {
        "filter_steps": steps,
        "crashes_in_tracts": int(len(crashes)),
        "crosswalk_agreement": crosswalk_agreement,
        "tracts_total": int(len(tracts)),
        "tracts_kept": int(len(kept)),
        "crashes_in_kept_tracts": int(kept["n"].sum()),
        "min_crashes_per_tract": MIN_CRASHES_PER_TRACT,
        "years": [YEAR_MIN, YEAR_MAX],
        "zone_counts": {
            name: int(zones[zones["layer"] == name]["zone"].nunique()) for name in BOUNDARY_SETS
        },
        "seconds": round(time.time() - started, 1),
    }
    BUILD_REPORT.write_text(json.dumps(report, indent=2))
    return report


def _tract_table(shapes: list[Shape], crashes: pd.DataFrame) -> pd.DataFrame:
    """Per-tract crash counts plus the geometry summaries the study needs."""
    n_tracts = len(shapes)
    grouped = crashes.groupby("tract")
    table = pd.DataFrame({"tract": np.arange(n_tracts)})
    table["geoid"] = [s.key for s in shapes]
    table["borough"] = [s.props["boroname"] for s in shapes]
    table["nta2020"] = [s.props["nta2020"] for s in shapes]
    table["ntaname"] = [s.props["ntaname"] for s in shapes]
    table["cdta2020"] = [s.props["cdta2020"] for s in shapes]

    table["n"] = grouped.size().reindex(range(n_tracts), fill_value=0).to_numpy()
    for name in INDICATORS:
        table[name] = (
            grouped[name].sum().reindex(range(n_tracts), fill_value=0).to_numpy(np.int64)
        )

    cents = centroids(shapes)
    table["lon"] = cents[:, 0]
    table["lat"] = cents[:, 1]
    table["area"] = areas(shapes)
    table["keep"] = table["n"] >= MIN_CRASHES_PER_TRACT
    return table


def _zone_tables(
    layers: dict[str, list[Shape]], keys: dict[str, list[str]], crashes: pd.DataFrame
) -> pd.DataFrame:
    """Long-format counts for every official layer: one row per (layer, zone)."""
    rows = []
    for name, shapes in layers.items():
        sub = crashes[crashes[name] >= 0]
        grouped = sub.groupby(name)
        block = pd.DataFrame({"position": np.arange(len(shapes))})
        block["layer"] = name
        block["zone"] = keys[name]
        block["n"] = grouped.size().reindex(range(len(shapes)), fill_value=0).to_numpy()
        for indicator in INDICATORS:
            block[indicator] = (
                grouped[indicator].sum().reindex(range(len(shapes)), fill_value=0).to_numpy(np.int64)
            )
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


@dataclass
class Study:
    """The cached study inputs, loaded once and passed around."""

    crashes: pd.DataFrame
    tracts: pd.DataFrame
    zones: pd.DataFrame
    report: dict

    @property
    def kept(self) -> pd.DataFrame:
        """Tracts that clear the minimum-crash threshold, reindexed from zero."""
        return self.tracts[self.tracts["keep"]].reset_index(drop=True)

    def neighbours(self) -> list[set[int]]:
        """Adjacency over the kept tracts, in kept-table order."""
        shapes = load_boundaries("tract")
        keep = self.tracts["keep"].to_numpy()
        full = adjacency(shapes, bridge=False)
        remap = -np.ones(len(shapes), dtype=np.int64)
        remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
        trimmed = [
            {int(remap[j]) for j in full[i] if keep[j]}
            for i in np.flatnonzero(keep)
        ]
        _reconnect(trimmed, self.kept)
        return trimmed


def _reconnect(neighbours: list[set[int]], kept: pd.DataFrame) -> None:
    """Bridge components left over after dropping low-crash tracts.

    Dropping tracts can sever the graph in places the full map is connected, so
    the same nearest-centroid bridging used for the raw layer is reapplied here.
    """
    from .geometry import LAT_SCALE, components

    pts = np.column_stack([kept["lon"].to_numpy() * LAT_SCALE, kept["lat"].to_numpy()])
    while True:
        groups = components(neighbours)
        if len(groups) <= 1:
            return
        head = np.array(groups[0])
        rest = np.array([i for g in groups[1:] for i in g])
        d = np.linalg.norm(pts[head][:, None, :] - pts[rest][None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        a, b = int(head[i]), int(rest[j])
        neighbours[a].add(b)
        neighbours[b].add(a)


def load_study() -> Study:
    if not CRASH_TABLE.exists():
        raise FileNotFoundError("run `make build` first -- artifacts/crashes.parquet is missing")
    return Study(
        crashes=pd.read_parquet(CRASH_TABLE),
        tracts=pd.read_parquet(TRACT_TABLE),
        zones=pd.read_parquet(ZONE_TABLE),
        report=json.loads(BUILD_REPORT.read_text()),
    )
