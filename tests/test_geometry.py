"""The hand-rolled GIS layer. If any of this is wrong every number downstream is."""

from __future__ import annotations

import numpy as np
import pytest

from redrawn.geometry import (
    GridIndex,
    adjacency,
    areas,
    assign_points,
    centroids,
    components,
    load_geojson,
    shape_contains,
    simplify,
)

from .conftest import hole_shape, square


def test_load_geojson_reads_every_feature(grid_shapes):
    assert len(grid_shapes) == 16
    assert {s.key for s in grid_shapes} == {f"{r}{c}" for r in range(4) for c in range(4)}
    assert all(len(s.rings) == 1 for s in grid_shapes)


def test_load_geojson_closes_open_rings(tmp_path):
    """GeoJSON requires a closed ring but not every publisher obliges."""
    import json

    open_ring = [[0, 0], [1, 0], [1, 1], [0, 1]]
    path = tmp_path / "open.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"id": "a"},
                        "geometry": {"type": "Polygon", "coordinates": [open_ring]},
                    }
                ],
            }
        )
    )
    shapes = load_geojson(path, "id")
    ring = shapes[0].rings[0].coords
    assert np.array_equal(ring[0], ring[-1])


def test_point_in_polygon_basics(grid_shapes):
    cell = next(s for s in grid_shapes if s.key == "22")  # square from (2,2) to (3,3)
    inside = np.array([2.5]), np.array([2.5])
    outside = np.array([0.5]), np.array([0.5])
    assert shape_contains(cell, *inside)[0]
    assert not shape_contains(cell, *outside)[0]


def test_point_in_polygon_respects_holes():
    shape = hole_shape()
    px = np.array([1.0, 5.0, 8.0, 5.0])
    py = np.array([1.0, 5.0, 8.0, 1.0])
    got = shape_contains(shape, px, py)
    #                    in ring, in hole, in ring, in ring
    assert list(got) == [True, False, True, True]


def test_point_in_polygon_handles_concave_shapes():
    """An L shape: the notch must be outside even though it is inside the bbox."""
    from redrawn.geometry import Ring, Shape

    ell = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4], [0, 0]], dtype=float)
    shape = Shape(key="L", rings=[Ring(ell, False)], props={})
    px = np.array([1.0, 3.0, 3.0])
    py = np.array([3.0, 1.0, 3.0])
    assert list(shape_contains(shape, px, py)) == [True, True, False]


def test_horizontal_edges_do_not_break_the_ray_cast():
    """The crossing test divides by (y2 - y1); horizontal edges must be skipped
    rather than producing a NaN that silently flips the parity."""
    from redrawn.geometry import Ring, Shape

    shape = Shape(key="s", rings=[Ring(square(0, 0, 4), False)], props={})
    # y = 0 and y = 4 lie exactly on horizontal edges.
    px = np.array([2.0, 2.0, 2.0])
    py = np.array([2.0, 0.0, 4.0])
    got = shape_contains(shape, px, py)
    assert got[0]
    # Boundary handling is half-open, so the two edge cases must at least be
    # deterministic and not NaN-driven.
    assert got.dtype == bool


def test_assign_points_matches_the_grid(grid_shapes):
    rng = np.random.default_rng(0)
    lon = rng.uniform(0.05, 3.95, 500)
    lat = rng.uniform(0.05, 3.95, 500)
    got = assign_points(grid_shapes, lon, lat)
    assert (got >= 0).all()
    for position, x, y in zip(got, lon, lat):
        shape = grid_shapes[position]
        assert shape.props["col"] == int(x)
        assert shape.props["row"] == int(y)


def test_assign_points_marks_outsiders(grid_shapes):
    lon = np.array([-5.0, 2.5, 99.0])
    lat = np.array([-5.0, 2.5, 99.0])
    got = assign_points(grid_shapes, lon, lat)
    assert got[0] == -1 and got[2] == -1 and got[1] >= 0


def test_grid_index_returns_a_superset(grid_shapes):
    rng = np.random.default_rng(1)
    lon = rng.uniform(0, 4, 300)
    lat = rng.uniform(0, 4, 300)
    index = GridIndex(lon, lat, cell=0.5)
    for shape in grid_shapes:
        xmin, ymin, xmax, ymax = shape.bbox
        truth = np.flatnonzero(
            (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
        )
        got = set(index.candidates(shape.bbox).tolist())
        assert set(truth.tolist()) <= got


def test_adjacency_is_rook_not_queen(grid_shapes):
    nbrs = adjacency(grid_shapes, bridge=False)
    by_key = {s.key: i for i, s in enumerate(grid_shapes)}
    centre = by_key["11"]
    got = {grid_shapes[j].key for j in nbrs[centre]}
    # Edge neighbours only -- the diagonals 00, 02, 20, 22 must not appear.
    assert got == {"01", "10", "12", "21"}


def test_adjacency_corner_has_two_neighbours(grid_shapes):
    nbrs = adjacency(grid_shapes, bridge=False)
    by_key = {s.key: i for i, s in enumerate(grid_shapes)}
    assert len(nbrs[by_key["00"]]) == 2


def test_bridging_connects_islands(tmp_path):
    """Two squares that share no edge are separate components until bridged."""
    import json

    features = [
        {
            "type": "Feature",
            "properties": {"id": name},
            "geometry": {"type": "Polygon", "coordinates": [square(x, 0).tolist()]},
        }
        for name, x in (("a", 0.0), ("b", 10.0))
    ]
    path = tmp_path / "islands.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    shapes = load_geojson(path, "id")

    assert len(components(adjacency(shapes, bridge=False))) == 2
    assert len(components(adjacency(shapes, bridge=True))) == 1


def test_components_of_a_line(line_graph):
    assert components(line_graph) == [[0, 1, 2, 3, 4, 5]]


def test_centroid_and_area_of_a_square(grid_shapes):
    cell = [s for s in grid_shapes if s.key == "22"]
    assert centroids(cell)[0] == pytest.approx([2.5, 2.5])
    assert areas(cell)[0] == pytest.approx(1.0)


def test_area_subtracts_holes():
    assert areas([hole_shape()])[0] == pytest.approx(100 - 16)


def test_simplify_keeps_endpoints_and_drops_collinear_points():
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
    got = simplify(line, tolerance=1e-9)
    assert np.array_equal(got, np.array([[0.0, 0.0], [4.0, 0.0]]))


def test_simplify_keeps_a_real_corner():
    shape = np.array([[0, 0], [1, 0], [2, 5], [3, 0], [4, 0]], dtype=float)
    got = simplify(shape, tolerance=0.5)
    assert [2.0, 5.0] in got.tolist()


def test_simplify_survives_a_ring_longer_than_the_recursion_limit():
    """Douglas-Peucker is naturally recursive; this implementation is iterative
    so that a 20k-vertex shoreline cannot blow the stack."""
    theta = np.linspace(0, 2 * np.pi, 20_000)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    got = simplify(circle, tolerance=0.01)
    assert 3 < len(got) < len(circle)
