"""Partition construction and scoring -- and above all, contiguity."""

from __future__ import annotations

import numpy as np
import pytest

from redrawn.partitions import (
    is_contiguous,
    merge_ladder,
    partition_correlation,
    random_contiguous,
    relabel,
    zone_rates,
    zone_totals,
)
from redrawn.stats import pearson


def test_relabel_compacts_labels():
    got = relabel(np.array([7, 7, 3, 99, 3]))
    assert sorted(set(got.tolist())) == [0, 1, 2]
    # Grouping must be preserved even though the names change.
    assert got[0] == got[1] and got[2] == got[4] and got[0] != got[2]


def test_zone_totals_sums_by_label():
    labels = np.array([0, 0, 1, 1, 1])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert zone_totals(labels, values) == pytest.approx([3.0, 12.0])


def test_zone_rates_drops_empty_zones():
    labels = np.array([0, 0, 2, 2])
    n = np.array([10.0, 10.0, 5.0, 5.0])
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([2.0, 2.0, 1.0, 1.0])
    zn, rx, ry = zone_rates(labels, n, x, y)
    # Zone 1 has no members and must not appear as a zero-crash zone.
    assert len(zn) == 2
    assert zn == pytest.approx([20.0, 10.0])
    assert rx == pytest.approx([0.1, 0.5])


def test_partition_correlation_matches_a_hand_computation():
    labels = np.array([0, 0, 1, 1, 2, 3])
    n = np.array([100.0, 100.0, 50.0, 50.0, 80.0, 40.0])
    x = np.array([10.0, 20.0, 5.0, 20.0, 4.0, 9.0])
    y = np.array([30.0, 30.0, 20.0, 10.0, 26.0, 6.0])
    zn, rx, ry = zone_rates(labels, n, x, y)
    # Zone rates computed by hand from the totals above.
    assert rx == pytest.approx([30 / 200, 25 / 100, 4 / 80, 9 / 40])
    assert ry == pytest.approx([60 / 200, 30 / 100, 26 / 80, 6 / 40])
    assert partition_correlation(labels, n, x, y) == pytest.approx(pearson(rx, ry))


def test_partition_correlation_needs_three_zones():
    """Pearson is undefined on two points; the study must not report one."""
    labels = np.array([0, 0, 1, 1])
    n = np.array([100.0, 100.0, 50.0, 50.0])
    x = np.array([10.0, 20.0, 5.0, 20.0])
    y = np.array([30.0, 30.0, 20.0, 10.0])
    assert np.isnan(partition_correlation(labels, n, x, y))


def test_partition_correlation_of_one_zone_per_unit_is_the_unit_correlation():
    rng = np.random.default_rng(0)
    n = rng.integers(50, 200, 40).astype(float)
    x = rng.integers(0, 20, 40).astype(float)
    y = rng.integers(0, 60, 40).astype(float)
    labels = np.arange(40)
    assert partition_correlation(labels, n, x, y) == pytest.approx(pearson(x / n, y / n))


@pytest.mark.parametrize("k", [2, 5, 12, 30])
def test_merge_ladder_hits_every_requested_k(grid_graph, toy_counts, k):
    n, _, _ = toy_counts
    got = merge_ladder(grid_graph, n, [k], np.random.default_rng(0))
    assert k in got
    assert len(np.unique(got[k])) == k


def test_merge_ladder_is_always_contiguous(grid_graph, toy_counts):
    n, _, _ = toy_counts
    for seed in range(6):
        ladder = merge_ladder(grid_graph, n, [4, 9, 20], np.random.default_rng(seed))
        for k, labels in ladder.items():
            assert is_contiguous(labels, grid_graph), f"seed {seed}, K={k}"


def test_merge_ladder_cuts_are_nested(grid_graph, toy_counts):
    """A coarser rung must be a merge of the finer one, never a re-partition."""
    n, _, _ = toy_counts
    ladder = merge_ladder(grid_graph, n, [8, 16], np.random.default_rng(2))
    fine, coarse = ladder[16], ladder[8]
    for zone in np.unique(fine):
        members = fine == zone
        assert len(np.unique(coarse[members])) == 1


def test_merge_ladder_produces_balanced_zones(grid_graph, toy_counts):
    """Merging the lightest adjacent pair should stop any zone running away."""
    n, _, _ = toy_counts
    labels = merge_ladder(grid_graph, n, [8], np.random.default_rng(3))[8]
    totals = np.bincount(labels, weights=n)
    assert totals.max() / np.median(totals) < 3.0


def test_merge_ladder_varies_with_the_seed(grid_graph, toy_counts):
    n, _, _ = toy_counts
    a = merge_ladder(grid_graph, n, [10], np.random.default_rng(0))[10]
    b = merge_ladder(grid_graph, n, [10], np.random.default_rng(99))[10]
    assert not np.array_equal(a, b)


def test_merge_ladder_is_reproducible(grid_graph, toy_counts):
    n, _, _ = toy_counts
    a = merge_ladder(grid_graph, n, [10], np.random.default_rng(5))[10]
    b = merge_ladder(grid_graph, n, [10], np.random.default_rng(5))[10]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("k", [2, 7, 16])
def test_random_contiguous_assigns_everything_contiguously(grid_graph, k):
    for seed in range(5):
        labels = random_contiguous(grid_graph, k, np.random.default_rng(seed))
        assert (labels >= 0).all()
        assert len(np.unique(labels)) == k
        assert is_contiguous(labels, grid_graph)


def test_is_contiguous_rejects_a_split_zone(grid_graph):
    labels = np.zeros(len(grid_graph), dtype=np.int64)
    # Corner 0 and the opposite corner 63 share a label but nothing between them.
    labels[:] = 1
    labels[0] = 0
    labels[63] = 0
    assert not is_contiguous(labels, grid_graph)


def test_is_contiguous_accepts_a_solid_block(grid_graph):
    labels = np.ones(len(grid_graph), dtype=np.int64)
    labels[[0, 1, 8, 9]] = 0  # a 2x2 block in the corner
    assert is_contiguous(labels, grid_graph)


def test_aggregation_inflates_correlation(grid_graph, toy_counts):
    """The study's central claim, on a toy city where it can be checked directly:
    coarser zones report a stronger correlation than the units do."""
    n, x, y = toy_counts
    unit_r = pearson(x / n, y / n)
    coarse = merge_ladder(grid_graph, n, [4], np.random.default_rng(1))[4]
    assert partition_correlation(coarse, n, x, y) > unit_r
