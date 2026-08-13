"""Cleaning rules and indicator construction, on a hand-written crash table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from redrawn import data as data_module
from redrawn.config import NYC_BBOX, YEAR_MAX, YEAR_MIN
from redrawn.data import INDICATORS, clean_crashes


def _raw(rows: list[dict]) -> pd.DataFrame:
    columns = [
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
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns]


@pytest.fixture
def patched_raw(monkeypatch, tmp_path):
    """Point `clean_crashes` at a table we control."""

    def install(rows):
        path = tmp_path / "crashes.csv.gz"
        _raw(rows).to_csv(path, index=False, compression="gzip")
        monkeypatch.setattr(data_module, "RAW", tmp_path)
        return path

    return install


def base_row(**overrides) -> dict:
    row = {
        "collision_id": 1,
        "crash_date": "2019-05-04T00:00:00.000",
        "latitude": 40.75,
        "longitude": -73.98,
        "number_of_persons_injured": 0,
        "number_of_persons_killed": 0,
        "number_of_pedestrians_injured": 0,
        "number_of_cyclist_injured": 0,
        "contributing_factor_vehicle_1": "Unspecified",
    }
    row.update(overrides)
    return row


def test_year_window_is_applied(patched_raw):
    patched_raw(
        [
            base_row(collision_id=1, crash_date=f"{YEAR_MIN - 1}-06-01T00:00:00.000"),
            base_row(collision_id=2, crash_date=f"{YEAR_MIN}-06-01T00:00:00.000"),
            base_row(collision_id=3, crash_date=f"{YEAR_MAX}-06-01T00:00:00.000"),
            base_row(collision_id=4, crash_date=f"{YEAR_MAX + 1}-06-01T00:00:00.000"),
        ]
    )
    frame, steps = clean_crashes()
    assert steps["downloaded"] == 4
    assert steps["in_window"] == 2
    assert sorted(frame["year"].tolist()) == [YEAR_MIN, YEAR_MAX]


def test_bad_geocodes_are_dropped(patched_raw):
    xmin, ymin, xmax, ymax = NYC_BBOX
    patched_raw(
        [
            base_row(collision_id=1),                                # good
            base_row(collision_id=2, latitude=0.0, longitude=0.0),   # null island
            base_row(collision_id=3, latitude=np.nan, longitude=np.nan),
            base_row(collision_id=4, latitude=ymax + 1, longitude=xmin - 1),
        ]
    )
    frame, steps = clean_crashes()
    assert steps["geocoded"] == 1
    assert len(frame) == 1
    assert xmin <= frame["lon"].iloc[0] <= xmax
    assert ymin <= frame["lat"].iloc[0] <= ymax


def test_injury_indicator_is_any_person_injured(patched_raw):
    patched_raw(
        [
            base_row(collision_id=1, number_of_persons_injured=0),
            base_row(collision_id=2, number_of_persons_injured=1),
            base_row(collision_id=3, number_of_persons_injured=5),
            base_row(collision_id=4, number_of_persons_injured=np.nan),
        ]
    )
    frame, _ = clean_crashes()
    assert frame["injury"].tolist() == [0, 1, 1, 0]


def test_factor_indicators(patched_raw):
    patched_raw(
        [
            base_row(collision_id=1, contributing_factor_vehicle_1="Unsafe Speed"),
            base_row(
                collision_id=2, contributing_factor_vehicle_1="Driver Inattention/Distraction"
            ),
            base_row(collision_id=3, contributing_factor_vehicle_1="Unspecified"),
            base_row(collision_id=4, contributing_factor_vehicle_1=np.nan),
            base_row(collision_id=5, contributing_factor_vehicle_1="Backing Unsafely"),
        ]
    )
    frame, _ = clean_crashes()
    assert frame["speeding"].tolist() == [1, 0, 0, 0, 0]
    assert frame["distraction"].tolist() == [0, 1, 0, 0, 0]
    # A missing factor counts as unrecorded, the same as the literal "Unspecified".
    assert frame["unspecified"].tolist() == [0, 0, 1, 1, 0]


def test_vru_indicator_covers_pedestrians_and_cyclists(patched_raw):
    patched_raw(
        [
            base_row(collision_id=1),
            base_row(collision_id=2, number_of_pedestrians_injured=1),
            base_row(collision_id=3, number_of_cyclist_injured=2),
        ]
    )
    frame, _ = clean_crashes()
    assert frame["vru"].tolist() == [0, 1, 1]


def test_every_indicator_is_binary(patched_raw):
    patched_raw([base_row(collision_id=i, number_of_persons_injured=i) for i in range(6)])
    frame, _ = clean_crashes()
    for name in INDICATORS:
        assert set(frame[name].unique()) <= {0, 1}


def test_indicators_are_documented():
    """Every indicator built by the cleaner needs a human-readable meaning, since
    the interface and the report both print them."""
    frame_columns = {"injury", "distraction", "speeding", "vru", "unspecified"}
    assert set(INDICATORS) == frame_columns
    assert all(isinstance(v, str) and v for v in INDICATORS.values())
