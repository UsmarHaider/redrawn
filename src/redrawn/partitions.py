"""Ways of carving 2,310 census tracts into K zones, and scoring the result.

A partition is a length-`n_tracts` integer array labelling each tract with its
zone. Three families are used:

* **official** -- the real NYC layers (NTAs, community districts, boroughs),
  obtained from the crosswalks the city publishes.
* **merged** -- built by agglomerating adjacent tracts until K zones remain.
  Running this many times with random tie-breaking gives, at every K at once, a
  scale ladder (how r moves as zones coarsen) and a zoning spread (how much r
  varies between equally-legitimate maps of the same size).
* **grown** -- multi-seed region growing, a rougher and less size-balanced
  random partition used as a robustness check on the merged family.

Every partition produced here is contiguous by construction, and
`is_contiguous` is used in the tests to prove it.
"""

from __future__ import annotations

import heapq

import numpy as np

from .stats import pearson, weighted_pearson


def zone_totals(labels: np.ndarray, values: np.ndarray, k: int | None = None) -> np.ndarray:
    """Sum a per-tract count into per-zone totals."""
    if k is None:
        k = int(labels.max()) + 1
    return np.bincount(labels, weights=values, minlength=k)


def zone_rates(
    labels: np.ndarray, n: np.ndarray, x: np.ndarray, y: np.ndarray, k: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-zone crash count and the two indicator rates. Empty zones are dropped."""
    if k is None:
        k = int(labels.max()) + 1
    zn = zone_totals(labels, n, k)
    zx = zone_totals(labels, x, k)
    zy = zone_totals(labels, y, k)
    live = zn > 0
    return zn[live], zx[live] / zn[live], zy[live] / zn[live]


def partition_correlation(
    labels: np.ndarray, n: np.ndarray, x: np.ndarray, y: np.ndarray, weighted: bool = False
) -> float:
    """The number an analyst would report from this map: r across its zones."""
    zn, rx, ry = zone_rates(labels, n, x, y)
    return weighted_pearson(rx, ry, zn) if weighted else pearson(rx, ry)


def relabel(labels: np.ndarray) -> np.ndarray:
    """Compact arbitrary labels down to 0..k-1."""
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64)


def is_contiguous(labels: np.ndarray, neighbours: list[set[int]]) -> bool:
    """True when every zone forms one connected block in the adjacency graph."""
    for zone in np.unique(labels):
        members = set(np.flatnonzero(labels == zone).tolist())
        if not members:
            continue
        start = next(iter(members))
        stack, seen = [start], {start}
        while stack:
            node = stack.pop()
            for nb in neighbours[node]:
                if nb in members and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if seen != members:
            return False
    return True


def merge_ladder(
    neighbours: list[set[int]],
    weight: np.ndarray,
    ks: list[int],
    rng: np.random.Generator,
    jitter: float = 0.35,
) -> dict[int, np.ndarray]:
    """Agglomerate adjacent tracts into progressively coarser nested partitions.

    At each step the *lightest adjacent pair* of zones is merged, measured by
    combined crash count. Merging the lightest pair rather than growing from a
    single seed is what keeps the result balanced: a zone that has grown heavy
    stops being chosen until the rest of the map has caught up, so no zone runs
    away. Real administrative units are balanced too, and an unbalanced
    synthetic map would confound the comparison against them.

    `jitter` perturbs each pair's weight by a random factor, so repeated runs
    explore genuinely different maps instead of returning one deterministic
    dendrogram. Returns the partition cut at each requested K -- the cuts are
    nested, each coarser map being a merge of the finer one.
    """
    n = len(neighbours)
    parent = np.arange(n)
    zone_weight = weight.astype(np.float64).copy()
    zone_nbrs = [set(s) for s in neighbours]
    alive = np.ones(n, dtype=bool)
    # Bumped whenever a zone changes, so stale heap entries can be recognised.
    version = np.zeros(n, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    def pair_key(a: int, b: int) -> float:
        return float((zone_weight[a] + zone_weight[b]) * (1.0 + jitter * rng.random()))

    heap: list[tuple[float, int, int, int, int]] = []
    for a in range(n):
        for b in zone_nbrs[a]:
            if a < b:
                heap.append((pair_key(a, b), a, b, 0, 0))
    heapq.heapify(heap)

    wanted = sorted({k for k in ks if 1 <= k <= n}, reverse=True)
    out: dict[int, np.ndarray] = {}
    live_count = n

    def snapshot() -> np.ndarray:
        return relabel(np.array([find(i) for i in range(n)]))

    while live_count > 1 and wanted:
        while wanted and live_count == wanted[0]:
            out[wanted.pop(0)] = snapshot()
        if not wanted:
            break

        if not heap:
            # Everything left is mutually unreachable; join the two lightest.
            live = [z for z in range(n) if alive[z]]
            if len(live) < 2:
                break
            live.sort(key=lambda z: zone_weight[z])
            a, b = live[0], live[1]
        else:
            _, a, b, va, vb = heapq.heappop(heap)
            if not (alive[a] and alive[b]) or version[a] != va or version[b] != vb or a == b:
                continue

        # Merge b into a.
        parent[b] = a
        alive[b] = False
        zone_weight[a] += zone_weight[b]
        zone_nbrs[a] |= zone_nbrs[b]
        zone_nbrs[a] = {find(v) for v in zone_nbrs[a]}
        zone_nbrs[a].discard(a)
        version[a] += 1
        live_count -= 1

        for c in zone_nbrs[a]:
            if alive[c]:
                zone_nbrs[c].add(a)
                heapq.heappush(heap, (pair_key(a, c), min(a, c), max(a, c),
                                      version[min(a, c)], version[max(a, c)]))

    while wanted and live_count == wanted[0]:
        out[wanted.pop(0)] = snapshot()
    return out


def random_contiguous(
    neighbours: list[set[int]], k: int, rng: np.random.Generator
) -> np.ndarray:
    """Multi-seed region growing: K random seeds annex random unclaimed neighbours.

    Less balanced than `merge_ladder` -- some zones run away and others stay
    tiny -- which is exactly why it is kept as a contrast.
    """
    n = len(neighbours)
    labels = np.full(n, -1, dtype=np.int64)
    seeds = rng.choice(n, size=k, replace=False)
    labels[seeds] = np.arange(k)

    frontier = [set() for _ in range(k)]
    for zone, seed in enumerate(seeds):
        frontier[zone] = {j for j in neighbours[seed] if labels[j] < 0}
    active = [z for z in range(k) if frontier[z]]

    remaining = n - k
    while remaining > 0:
        if not active:
            # Unclaimed pocket unreachable from any zone: seed it into the zone of
            # a neighbouring tract, or start it as its own growth front.
            left = np.flatnonzero(labels < 0)
            if left.size == 0:
                break
            t = int(rng.choice(left))
            touching = [labels[j] for j in neighbours[t] if labels[j] >= 0]
            labels[t] = int(rng.choice(touching)) if touching else int(rng.integers(k))
            remaining -= 1
            zone = int(labels[t])
            frontier[zone] |= {j for j in neighbours[t] if labels[j] < 0}
            if frontier[zone]:
                active.append(zone)
            continue

        pos = int(rng.integers(len(active)))
        zone = active[pos]
        pool = frontier[zone]
        while pool and labels[next(iter(pool))] >= 0:
            pool.discard(next(iter(pool)))
        if not pool:
            active.pop(pos)
            continue
        t = int(rng.choice(list(pool)))
        pool.discard(t)
        if labels[t] >= 0:
            continue
        labels[t] = zone
        remaining -= 1
        frontier[zone] |= {j for j in neighbours[t] if labels[j] < 0}
        if not frontier[zone]:
            active.pop(pos)
    return labels


def official_partitions(kept) -> dict[str, np.ndarray]:
    """Tract-based versions of the real NYC layers, via the published crosswalks.

    Community districts are represented by CDTAs -- the tract-aligned version of
    the same boundaries -- because a partition of tracts is what the merged and
    gerrymandered families are, and comparing like with like matters more here
    than matching the community-district map to the metre.
    """
    return {
        "nta": relabel(kept["nta2020"].to_numpy()),
        "cdta": relabel(kept["cdta2020"].to_numpy()),
        "borough": relabel(kept["borough"].to_numpy()),
        "tract": np.arange(len(kept), dtype=np.int64),
    }
