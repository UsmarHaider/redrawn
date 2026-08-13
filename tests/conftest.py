"""Synthetic fixtures. Nothing here touches the network or the downloaded data,
so the whole suite runs offline in CI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from redrawn.geometry import Shape, load_geojson


def square(x0: float, y0: float, size: float = 1.0) -> np.ndarray:
    return np.array(
        [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]],
        dtype=np.float64,
    )


@pytest.fixture
def grid_geojson(tmp_path):
    """A 4x4 chessboard of unit squares written out as a GeoJSON FeatureCollection.

    Neighbouring cells share exact vertices, so rook adjacency must find them.
    """
    features = []
    for row in range(4):
        for col in range(4):
            features.append(
                {
                    "type": "Feature",
                    "properties": {"id": f"{row}{col}", "row": row, "col": col},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [square(col, row).tolist()],
                    },
                }
            )
    path = tmp_path / "grid.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


@pytest.fixture
def grid_shapes(grid_geojson):
    return load_geojson(grid_geojson, "id")


@pytest.fixture
def line_graph():
    """Adjacency of six zones in a row: 0-1-2-3-4-5."""
    return [
        {1},
        {0, 2},
        {1, 3},
        {2, 4},
        {3, 5},
        {4},
    ]


@pytest.fixture
def grid_graph():
    """Rook adjacency of an 8x8 grid, as an index-based adjacency list."""
    side = 8
    nbrs: list[set[int]] = [set() for _ in range(side * side)]
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if r > 0:
                nbrs[i].add(i - side)
            if r < side - 1:
                nbrs[i].add(i + side)
            if c > 0:
                nbrs[i].add(i - 1)
            if c < side - 1:
                nbrs[i].add(i + 1)
    return nbrs


@pytest.fixture
def toy_counts():
    """64 units with crash counts and two indicator counts, correlated by design."""
    rng = np.random.default_rng(7)
    n = rng.integers(60, 400, size=64).astype(np.float64)
    base = rng.random(64)
    x = np.round(n * (0.01 + 0.05 * base))
    y = np.round(n * (0.18 + 0.22 * base + 0.02 * rng.standard_normal(64)))
    y = np.clip(y, 0, n)
    return n, x, y


def hole_shape() -> Shape:
    """A 10x10 square with a 4x4 hole punched in the middle."""
    from redrawn.geometry import Ring

    outer = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=np.float64)
    inner = np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]], dtype=np.float64)
    return Shape(key="holed", rings=[Ring(outer, False), Ring(inner, True)], props={})
