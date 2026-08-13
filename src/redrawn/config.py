"""Paths, dataset identifiers and the handful of constants the study is built on."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "docs" / "figures"
UI = Path(__file__).resolve().parent / "ui"

for _d in (RAW, ARTIFACTS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SOCRATA = "https://data.cityofnewyork.us/resource"

#: NYPD Motor Vehicle Collisions - Crashes. One row per crash, 2012-07 onward.
CRASHES_ID = "h9gi-nx95"

#: The four real partitions of New York City used as the "official" zonings.
#: Each is published by the Department of City Planning as a MultiPolygon layer.
BOUNDARY_SETS = {
    "tract": {
        "id": "63ge-mke6",
        "key": "geoid",
        "label_field": "ctlabel",
        "name": "2020 census tracts",
    },
    "nta": {
        "id": "9nt8-h7nd",
        "key": "nta2020",
        "label_field": "ntaname",
        "name": "neighborhood tabulation areas",
    },
    "community_district": {
        "id": "5crt-au7u",
        "key": "boro_cd",
        "label_field": "boro_cd",
        "name": "community districts",
    },
    "precinct": {
        "id": "y76i-bdw7",
        "key": "precinct",
        "label_field": "precinct",
        "name": "police precincts",
    },
}

#: Columns pulled from the crash table. Everything else is dead weight at 2M rows.
CRASH_COLUMNS = [
    "collision_id",
    "crash_date",
    "latitude",
    "longitude",
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_cyclist_injured",
    "contributing_factor_vehicle_1",
]

#: A generous bounding box for the five boroughs. Anything outside is a bad geocode.
NYC_BBOX = (-74.30, 40.47, -73.68, 40.93)

#: The study window. The table starts mid-2012 and the current year is partial,
#: so both ends are trimmed to whole years.
YEAR_MIN = 2013
YEAR_MAX = 2025

#: The indicator pairs the study correlates. Both members of a pair are
#: properties of an individual crash, so the individual-level correlation is a
#: well-defined quantity -- which is the point of the whole comparison.
#:
#: `vru` is deliberately never paired with `injury`: a crash that injures a
#: pedestrian has by construction injured someone, so the two are mechanically
#: nested and their correlation would measure the definition, not the city.
PAIRS = {
    "speeding": ("speeding", "injury"),
    "unspecified": ("unspecified", "injury"),
    "distraction": ("distraction", "injury"),
}

#: The pair carried through the headline results and the web interface.
PRIMARY_PAIR = "speeding"
PRIMARY_X, PRIMARY_Y = PAIRS[PRIMARY_PAIR]

#: Zone counts used for the synthetic partition ladder (the scale effect).
SCALE_LADDER = [1200, 800, 500, 350, 250, 180, 120, 80, 50, 30, 20, 12, 8, 5]

#: Fixed zone count for the zoning-effect and gerrymander experiments. Chosen to
#: match the 71 community districts so the official map is directly comparable.
FIXED_K = 71

#: Minimum crashes for a tract to enter the study. Tracts with a handful of
#: crashes carry almost no information and dominate an unweighted correlation.
MIN_CRASHES_PER_TRACT = 30

SEED = 20260813
