"""How far can a correlation be pushed by redrawing the map, and nothing else?

Simulated annealing over contiguous K-zone partitions of the tracts. The data
never changes -- not one crash is added, moved or reweighted. Only the boundaries
move, and the objective is the correlation an analyst would report from the
resulting map.

The move set is a single boundary flip: take a tract that touches another zone
and hand it over. A flip is rejected unless the donor zone stays non-empty and
*stays connected*, so every state visited is a map someone could actually
propose. With `balance` set, zones must also stay within a tolerance of equal
crash counts, which is roughly what districting law demands of real maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stats import pearson


@dataclass
class SearchResult:
    labels: np.ndarray
    r: float
    direction: str
    steps: int
    accepted: int
    proposed: int
    rejected_contiguity: int
    rejected_balance: int
    start_r: float
    trace: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "direction": self.direction,
            "r": self.r,
            "start_r": self.start_r,
            "steps": self.steps,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / max(self.proposed, 1),
            "rejected_contiguity": self.rejected_contiguity,
            "rejected_balance": self.rejected_balance,
        }


def _still_connected(members: set[int], removed: int, neighbours: list[set[int]]) -> bool:
    """Does `members` minus `removed` remain one connected block?"""
    rest = members - {removed}
    if len(rest) <= 1:
        return True
    start = next(iter(rest))
    stack, seen = [start], {start}
    while stack:
        node = stack.pop()
        for nb in neighbours[node]:
            if nb in rest and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(rest)


def anneal(
    labels: np.ndarray,
    neighbours: list[set[int]],
    n: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    direction: str = "max",
    steps: int = 60_000,
    t0: float = 0.03,
    t1: float = 3e-4,
    balance: float | None = None,
    seed: int = 0,
    trace_every: int = 200,
) -> SearchResult:
    """Anneal the boundaries toward the highest (or lowest) reportable correlation.

    `balance` is the allowed fractional deviation of a zone's crash count from
    the average; None leaves zone sizes unconstrained.
    """
    if direction not in {"max", "min"}:
        raise ValueError("direction must be 'max' or 'min'")
    rng = np.random.default_rng(seed)
    sign = 1.0 if direction == "max" else -1.0

    labels = labels.astype(np.int64).copy()
    k = int(labels.max()) + 1
    n = np.asarray(n, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    zn = np.bincount(labels, weights=n, minlength=k)
    zx = np.bincount(labels, weights=x, minlength=k)
    zy = np.bincount(labels, weights=y, minlength=k)
    members: list[set[int]] = [set() for _ in range(k)]
    for tract, zone in enumerate(labels):
        members[zone].add(int(tract))

    def score() -> float:
        live = zn > 0
        if live.sum() < 3:
            return float("nan")
        return pearson(zx[live] / zn[live], zy[live] / zn[live])

    lo = hi = None
    if balance is not None:
        mean = n.sum() / k
        lo, hi = mean * (1 - balance), mean * (1 + balance)

    current = score()
    start_r = current
    best_r, best_labels = current, labels.copy()
    accepted = proposed = bad_contig = bad_balance = 0
    trace: list[float] = []

    # Tracts with at least one neighbour, sampled uniformly; a tract whose
    # neighbours all share its zone is an interior tract and the move is skipped.
    movable = np.array([i for i in range(len(labels)) if neighbours[i]], dtype=np.int64)

    for step in range(steps):
        temperature = t0 * (t1 / t0) ** (step / max(steps - 1, 1))

        tract = int(movable[rng.integers(len(movable))])
        source = int(labels[tract])
        options = [int(labels[j]) for j in neighbours[tract] if labels[j] != source]
        if not options:
            continue
        target = options[int(rng.integers(len(options)))]
        proposed += 1

        if len(members[source]) <= 1:
            continue
        if balance is not None:
            if zn[source] - n[tract] < lo or zn[target] + n[tract] > hi:
                bad_balance += 1
                continue
        if not _still_connected(members[source], tract, neighbours):
            bad_contig += 1
            continue

        zn[source] -= n[tract]; zx[source] -= x[tract]; zy[source] -= y[tract]
        zn[target] += n[tract]; zx[target] += x[tract]; zy[target] += y[tract]
        candidate = score()

        delta = sign * (candidate - current)
        if np.isfinite(candidate) and (
            delta >= 0 or rng.random() < np.exp(delta / max(temperature, 1e-12))
        ):
            labels[tract] = target
            members[source].discard(tract)
            members[target].add(tract)
            current = candidate
            accepted += 1
            if sign * current > sign * best_r:
                best_r, best_labels = current, labels.copy()
        else:
            zn[source] += n[tract]; zx[source] += x[tract]; zy[source] += y[tract]
            zn[target] -= n[tract]; zx[target] -= x[tract]; zy[target] -= y[tract]

        if step % trace_every == 0:
            trace.append(current)

    return SearchResult(
        labels=best_labels,
        r=float(best_r),
        direction=direction,
        steps=steps,
        accepted=accepted,
        proposed=proposed,
        rejected_contiguity=bad_contig,
        rejected_balance=bad_balance,
        start_r=float(start_r),
        trace=trace,
    )


def envelope(
    start: np.ndarray,
    neighbours: list[set[int]],
    n: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    steps: int = 60_000,
    balance: float | None = None,
    seed: int = 0,
    restarts: int = 2,
) -> dict:
    """Anneal in both directions and report the reachable interval of r.

    Restarts guard against a single run stalling in a local optimum; the best
    result in each direction is kept.
    """
    runs = {"max": [], "min": []}
    for direction in ("max", "min"):
        for attempt in range(restarts):
            runs[direction].append(
                anneal(
                    start,
                    neighbours,
                    n,
                    x,
                    y,
                    direction=direction,
                    steps=steps,
                    balance=balance,
                    seed=seed + 1000 * attempt + (0 if direction == "max" else 7),
                )
            )
    best_max = max(runs["max"], key=lambda r: r.r)
    best_min = min(runs["min"], key=lambda r: r.r)
    return {
        "r_min": best_min.r,
        "r_max": best_max.r,
        "width": best_max.r - best_min.r,
        "start_r": best_max.start_r,
        "balance": balance,
        "steps": steps,
        "restarts": restarts,
        "runs": [r.summary() for r in runs["max"] + runs["min"]],
        "labels_max": best_max.labels,
        "labels_min": best_min.labels,
        "trace_max": best_max.trace,
        "trace_min": best_min.trace,
    }
