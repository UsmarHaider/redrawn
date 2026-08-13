"""The search. Every state it visits must be a map somebody could actually propose."""

from __future__ import annotations

import numpy as np
import pytest

from redrawn.gerrymander import anneal, envelope
from redrawn.partitions import is_contiguous, merge_ladder, partition_correlation


@pytest.fixture
def start(grid_graph, toy_counts):
    n, _, _ = toy_counts
    return merge_ladder(grid_graph, n, [8], np.random.default_rng(0))[8]


def test_anneal_improves_the_objective(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    got = anneal(start, grid_graph, n, x, y, direction="max", steps=4000, seed=0)
    assert got.r > got.start_r


def test_anneal_can_push_downwards(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    got = anneal(start, grid_graph, n, x, y, direction="min", steps=4000, seed=0)
    assert got.r < got.start_r


def test_anneal_never_breaks_contiguity(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    for direction in ("max", "min"):
        got = anneal(start, grid_graph, n, x, y, direction=direction, steps=4000, seed=1)
        assert is_contiguous(got.labels, grid_graph)


def test_anneal_keeps_the_zone_count(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    got = anneal(start, grid_graph, n, x, y, direction="max", steps=4000, seed=2)
    assert len(np.unique(got.labels)) == len(np.unique(start))


def test_anneal_reports_the_correlation_of_the_labels_it_returns(
    grid_graph, toy_counts, start
):
    """The returned r must be the r of the returned map, not of some other state
    the search passed through."""
    n, x, y = toy_counts
    got = anneal(start, grid_graph, n, x, y, direction="max", steps=4000, seed=3)
    assert partition_correlation(got.labels, n, x, y) == pytest.approx(got.r)


def test_anneal_rejects_contiguity_breaking_moves(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    got = anneal(start, grid_graph, n, x, y, direction="max", steps=4000, seed=4)
    assert got.rejected_contiguity > 0


def test_balance_constraint_is_respected(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    tolerance = 0.30
    got = anneal(
        start, grid_graph, n, x, y, direction="max", steps=4000, balance=tolerance, seed=5
    )
    totals = np.bincount(got.labels, weights=n)
    mean = n.sum() / len(np.unique(got.labels))
    assert totals.max() <= mean * (1 + tolerance) + 1e-9
    assert got.rejected_balance > 0


def test_balance_constraint_narrows_the_envelope(grid_graph, toy_counts, start):
    """Constrained search cannot beat unconstrained search at the same budget."""
    n, x, y = toy_counts
    free = envelope(start, grid_graph, n, x, y, steps=4000, balance=None, seed=0, restarts=1)
    tied = envelope(start, grid_graph, n, x, y, steps=4000, balance=0.30, seed=0, restarts=1)
    assert tied["width"] <= free["width"]


def test_envelope_brackets_the_starting_map(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    got = envelope(start, grid_graph, n, x, y, steps=4000, seed=0, restarts=1)
    assert got["r_min"] <= got["start_r"] <= got["r_max"]
    assert got["width"] == pytest.approx(got["r_max"] - got["r_min"])


def test_more_search_never_narrows_the_envelope(grid_graph, toy_counts, start):
    """The convergence finding depends on this being monotone: a longer run
    explores a superset of the maps a shorter one could reach."""
    n, x, y = toy_counts
    short = envelope(start, grid_graph, n, x, y, steps=1500, seed=0, restarts=1)
    long = envelope(start, grid_graph, n, x, y, steps=20_000, seed=0, restarts=1)
    assert long["r_max"] >= short["r_max"] - 1e-9
    assert long["r_min"] <= short["r_min"] + 1e-9


def test_anneal_is_reproducible(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    a = anneal(start, grid_graph, n, x, y, direction="max", steps=3000, seed=11)
    b = anneal(start, grid_graph, n, x, y, direction="max", steps=3000, seed=11)
    assert a.r == pytest.approx(b.r)
    assert np.array_equal(a.labels, b.labels)


def test_direction_must_be_valid(grid_graph, toy_counts, start):
    n, x, y = toy_counts
    with pytest.raises(ValueError):
        anneal(start, grid_graph, n, x, y, direction="sideways", steps=10)


def test_noise_can_also_be_gerrymandered(grid_graph, toy_counts, start):
    """The control that keeps the headline honest: with no relationship at all
    between x and y, redistricting still manufactures a large correlation."""
    n, x, _ = toy_counts
    rng = np.random.default_rng(0)
    noise = rng.binomial(n.astype(int), 0.25).astype(float)
    got = anneal(start, grid_graph, n, x, noise, direction="max", steps=8000, seed=0)
    assert got.r > 0.5
