"""A small hand-rolled geometry layer: enough GIS to run the study, nothing more.

Deliberately free of geopandas, shapely and GDAL. The four operations the study
actually needs -- point-in-polygon, polygon adjacency, centroids and line
simplification -- are a few dozen lines of NumPy each, and writing them out keeps
the whole pipeline installable with `pip install numpy pandas` on any machine.

Coordinates are WGS84 degrees throughout. Distances are only ever used for
tie-breaking and nearest-neighbour bridging, so degrees are scaled by cos(lat) to
keep them roughly isotropic rather than being projected properly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# New York sits near 40.7 degrees north; one degree of longitude is this fraction
# of one degree of latitude in ground distance.
LAT_SCALE = float(np.cos(np.radians(40.7)))


@dataclass(frozen=True)
class Ring:
    """One closed ring of a polygon. `hole` marks interior rings."""

    coords: np.ndarray  # (n, 2) float64, first point repeated as last
    hole: bool


@dataclass
class Shape:
    """One feature: a (multi)polygon with an identifier and free-form properties."""

    key: str
    rings: list[Ring]
    props: dict

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        outer = np.concatenate([r.coords for r in self.rings if not r.hole])
        return (outer[:, 0].min(), outer[:, 1].min(), outer[:, 0].max(), outer[:, 1].max())


def load_geojson(path: Path, key_field: str) -> list[Shape]:
    """Read a GeoJSON FeatureCollection of Polygon/MultiPolygon features.

    GeoJSON says the first ring of each polygon is the exterior and the rest are
    holes, which is what we rely on -- winding order is not trusted, because the
    NYC layers are not consistent about it.
    """
    raw = json.loads(Path(path).read_text())
    shapes: list[Shape] = []
    for feature in raw["features"]:
        geom = feature.get("geometry")
        if geom is None:
            continue
        if geom["type"] == "Polygon":
            polygons = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polygons = geom["coordinates"]
        else:
            raise ValueError(f"unsupported geometry {geom['type']}")

        rings: list[Ring] = []
        for polygon in polygons:
            for index, ring in enumerate(polygon):
                coords = np.asarray(ring, dtype=np.float64)[:, :2]
                if len(coords) < 4:
                    continue
                if not np.array_equal(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[:1]])
                rings.append(Ring(coords=coords, hole=index > 0))
        if not rings:
            continue
        props = feature.get("properties", {})
        shapes.append(Shape(key=str(props[key_field]), rings=rings, props=props))
    return shapes


def _ring_contains(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Crossing-number test of many points against one ring.

    A ray is cast in the +x direction from each point and crossings with each
    edge are counted; an odd count means inside. Horizontal edges never satisfy
    the half-open y-straddle test, so they contribute nothing and the division by
    (y2 - y1) below is never used where it is degenerate.
    """
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]

    dy = y2 - y1
    straddles = (y1[None, :] > py[:, None]) != (y2[None, :] > py[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(dy != 0, (x2 - x1) / np.where(dy != 0, dy, 1.0), 0.0)
        x_at_ray = x1[None, :] + (py[:, None] - y1[None, :]) * slope[None, :]
    crossings = straddles & (px[:, None] < x_at_ray)
    return crossings.sum(axis=1) % 2 == 1


def shape_contains(shape: Shape, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Point-in-(multi)polygon: inside an odd number of exterior rings, outside holes."""
    inside = np.zeros(len(px), dtype=bool)
    for ring in shape.rings:
        if ring.hole:
            continue
        inside |= _ring_contains(px, py, ring.coords)
    for ring in shape.rings:
        if ring.hole:
            inside &= ~_ring_contains(px, py, ring.coords)
    return inside


class GridIndex:
    """Uniform-grid bucketing of points so each polygon only tests nearby ones.

    Without this, assigning 2M crashes to 2,300 tracts is a 4.6-billion-edge
    problem. With it, each polygon looks at the few thousand points sharing its
    cells, which is two orders of magnitude less work.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, cell: float = 0.005):
        self.cell = cell
        self.x0, self.y0 = float(x.min()), float(y.min())
        self.nx = max(1, int((x.max() - self.x0) / cell) + 1)
        self.ny = max(1, int((y.max() - self.y0) / cell) + 1)

        ix = np.clip(((x - self.x0) / cell).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((y - self.y0) / cell).astype(np.int64), 0, self.ny - 1)
        flat = iy * self.nx + ix

        # CSR-style buckets: order[start[c]:start[c+1]] are the points in cell c.
        self.order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=self.nx * self.ny)
        self.start = np.concatenate([[0], np.cumsum(counts)])

    def candidates(self, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Indices of every point in a cell overlapping `bbox`."""
        xmin, ymin, xmax, ymax = bbox
        ix0 = np.clip(int((xmin - self.x0) / self.cell), 0, self.nx - 1)
        ix1 = np.clip(int((xmax - self.x0) / self.cell), 0, self.nx - 1)
        iy0 = np.clip(int((ymin - self.y0) / self.cell), 0, self.ny - 1)
        iy1 = np.clip(int((ymax - self.y0) / self.cell), 0, self.ny - 1)

        blocks = []
        for iy in range(iy0, iy1 + 1):
            base = iy * self.nx
            lo, hi = self.start[base + ix0], self.start[base + ix1 + 1]
            if hi > lo:
                blocks.append(self.order[lo:hi])
        if not blocks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(blocks)


def assign_points(
    shapes: list[Shape],
    lon: np.ndarray,
    lat: np.ndarray,
    cell: float = 0.005,
    progress: bool = False,
) -> np.ndarray:
    """Assign each point to the shape containing it; -1 where no shape does.

    The layers are partitions, so a point falls in at most one shape and already
    assigned points are skipped -- which makes later polygons progressively
    cheaper.
    """
    index = GridIndex(lon, lat, cell=cell)
    out = np.full(len(lon), -1, dtype=np.int32)

    for position, shape in enumerate(shapes):
        candidates = index.candidates(shape.bbox)
        if candidates.size == 0:
            continue
        candidates = candidates[out[candidates] < 0]
        if candidates.size == 0:
            continue
        px, py = lon[candidates], lat[candidates]
        xmin, ymin, xmax, ymax = shape.bbox
        in_box = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
        candidates, px, py = candidates[in_box], px[in_box], py[in_box]
        if candidates.size == 0:
            continue
        out[candidates[shape_contains(shape, px, py)]] = position
        if progress and position % 250 == 0:
            print(f"  assigned through shape {position:>5}/{len(shapes)}", end="\r", flush=True)
    return out


def centroids(shapes: list[Shape]) -> np.ndarray:
    """Area-weighted centroid of each shape's exterior rings (holes ignored)."""
    out = np.zeros((len(shapes), 2))
    for i, shape in enumerate(shapes):
        total_area = 0.0
        acc = np.zeros(2)
        for ring in shape.rings:
            if ring.hole:
                continue
            x, y = ring.coords[:-1, 0], ring.coords[:-1, 1]
            xn, yn = np.roll(x, -1), np.roll(y, -1)
            cross = x * yn - xn * y
            area = cross.sum() / 2.0
            if abs(area) < 1e-15:
                continue
            cx = ((x + xn) * cross).sum() / (6.0 * area)
            cy = ((y + yn) * cross).sum() / (6.0 * area)
            acc += abs(area) * np.array([cx, cy])
            total_area += abs(area)
        out[i] = acc / total_area if total_area else shape.rings[0].coords[:-1].mean(axis=0)
    return out


def areas(shapes: list[Shape]) -> np.ndarray:
    """Signed-shoelace area of each shape in square degrees, holes subtracted."""
    out = np.zeros(len(shapes))
    for i, shape in enumerate(shapes):
        total = 0.0
        for ring in shape.rings:
            x, y = ring.coords[:-1, 0], ring.coords[:-1, 1]
            area = abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2.0)
            total += -area if ring.hole else area
        out[i] = total
    return out


def _edge_keys(shape: Shape, precision: int) -> set[tuple]:
    keys = set()
    for ring in shape.rings:
        rounded = np.round(ring.coords, precision)
        for a, b in zip(rounded[:-1], rounded[1:]):
            pa, pb = (a[0], a[1]), (b[0], b[1])
            if pa == pb:
                continue
            keys.add((pa, pb) if pa < pb else (pb, pa))
    return keys


def adjacency(
    shapes: list[Shape], precision: int = 6, bridge: bool = True
) -> list[set[int]]:
    """Rook contiguity: two shapes are neighbours if they share a boundary segment.

    The NYC layers are cut from one topology, so neighbouring polygons carry
    genuinely identical vertices and an exact (rounded) edge match is reliable.
    Water is the complication -- Staten Island shares no edge with anything, and
    tracts clipped to the shoreline can end up isolated. With `bridge` set, every
    connected component after the edge pass is joined to the nearest component by
    a centroid-distance link, which is what the real map does too: the boroughs
    are connected by bridges.
    """
    edge_owner: dict[tuple, list[int]] = {}
    for i, shape in enumerate(shapes):
        for key in _edge_keys(shape, precision):
            edge_owner.setdefault(key, []).append(i)

    neighbours: list[set[int]] = [set() for _ in shapes]
    for owners in edge_owner.values():
        if len(owners) < 2:
            continue
        for a in owners:
            for b in owners:
                if a != b:
                    neighbours[a].add(b)

    if bridge:
        _bridge_components(shapes, neighbours)
    return neighbours


def components(neighbours: list[set[int]]) -> list[list[int]]:
    """Connected components of an adjacency list, as lists of node indices."""
    seen = [False] * len(neighbours)
    out = []
    for start in range(len(neighbours)):
        if seen[start]:
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            group.append(node)
            for nb in neighbours[node]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        out.append(sorted(group))
    return out


def _bridge_components(shapes: list[Shape], neighbours: list[set[int]]) -> None:
    """Repeatedly join the two closest components until the graph is connected."""
    cents = centroids(shapes)
    scaled = np.column_stack([cents[:, 0] * LAT_SCALE, cents[:, 1]])

    while True:
        groups = components(neighbours)
        if len(groups) <= 1:
            return
        # Join the first component to whichever other component holds the nearest
        # polygon to any of its members.
        head = np.array(groups[0])
        rest = np.array([i for g in groups[1:] for i in g])
        d = np.linalg.norm(scaled[head][:, None, :] - scaled[rest][None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        a, b = int(head[i]), int(rest[j])
        neighbours[a].add(b)
        neighbours[b].add(a)


def simplify(coords: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker, iterative so long rings cannot blow the recursion limit.

    Used only to shrink the map for the browser; never for the analysis.
    """
    n = len(coords)
    if n <= 3:
        return coords
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        a, b = coords[lo], coords[hi]
        seg = b - a
        length = float(np.hypot(*seg))
        pts = coords[lo + 1 : hi]
        if length < 1e-12:
            dist = np.linalg.norm(pts - a, axis=1)
        else:
            # Perpendicular distance from each interior point to the chord ab.
            # The 2-D cross product is written out: np.cross on 2-vectors is
            # deprecated in NumPy 2.
            rel = pts - a
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length
        k = int(np.argmax(dist))
        if dist[k] > tolerance:
            split = lo + 1 + k
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    return coords[keep]
