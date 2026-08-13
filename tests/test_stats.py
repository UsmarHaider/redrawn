"""Correlation, spatial weights and the permutation machinery."""

from __future__ import annotations

import numpy as np
import pytest

from redrawn.stats import (
    Weights,
    correlation_p_value,
    fisher_ci,
    morans_i,
    morans_i_permutation,
    pearson,
    phi,
    smooth_surrogate,
    spatial_null_correlation,
    weighted_pearson,
    weights_matrix,
)


def test_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(200), rng.standard_normal(200)
    assert pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_pearson_of_a_perfect_line():
    x = np.arange(50, dtype=float)
    assert pearson(x, 3 * x + 7) == pytest.approx(1.0)
    assert pearson(x, -3 * x + 7) == pytest.approx(-1.0)


def test_pearson_is_nan_when_a_side_is_constant():
    assert np.isnan(pearson(np.ones(10), np.arange(10, dtype=float)))


def test_weighted_pearson_equals_pearson_under_equal_weights():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(80), rng.standard_normal(80)
    assert weighted_pearson(x, y, np.ones(80)) == pytest.approx(pearson(x, y))


def test_weighted_pearson_matches_replication():
    """Integer weights must behave exactly like repeating the rows."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 0.0, 4.0, 2.0])
    w = np.array([1.0, 3.0, 2.0, 4.0])
    expanded_x = np.repeat(x, w.astype(int))
    expanded_y = np.repeat(y, w.astype(int))
    assert weighted_pearson(x, y, w) == pytest.approx(pearson(expanded_x, expanded_y))


def test_phi_matches_the_2x2_formula():
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    b = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    expected = (n11 * n00 - n10 * n01) / np.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    assert phi(a, b) == pytest.approx(expected)


def test_fisher_ci_brackets_the_estimate():
    lo, hi = fisher_ci(0.5, 50)
    assert lo < 0.5 < hi
    wide_lo, wide_hi = fisher_ci(0.5, 10)
    assert (wide_hi - wide_lo) > (hi - lo)


def test_correlation_p_value_is_tiny_for_a_strong_relationship():
    assert correlation_p_value(0.9, 100) < 1e-20
    assert correlation_p_value(0.0, 100) == pytest.approx(1.0)


def test_weights_row_standardise(line_graph):
    w = Weights(line_graph)
    dense = w.dense()
    assert dense.sum(axis=1) == pytest.approx(np.ones(6))
    # An end node has one neighbour, so its single weight is 1.
    assert dense[0, 1] == pytest.approx(1.0)
    # An interior node splits between two.
    assert dense[1, 0] == pytest.approx(0.5)


def test_sparse_lag_matches_dense(grid_graph):
    rng = np.random.default_rng(2)
    values = rng.standard_normal(len(grid_graph))
    w = Weights(grid_graph)
    assert w.lag(values) == pytest.approx(weights_matrix(grid_graph) @ values)


def test_morans_i_sparse_matches_dense(grid_graph):
    rng = np.random.default_rng(3)
    values = rng.standard_normal(len(grid_graph))
    sparse = morans_i(values, Weights(grid_graph))
    dense = morans_i(values, weights_matrix(grid_graph))
    assert sparse == pytest.approx(dense)


def test_morans_i_is_high_for_a_smooth_field_and_low_for_noise(grid_graph):
    side = 8
    smooth = np.array([r for r in range(side) for _ in range(side)], dtype=float)
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(side * side)
    w = Weights(grid_graph)
    assert morans_i(smooth, w) > 0.6
    assert abs(morans_i(noise, w)) < 0.35


def test_morans_i_of_a_checkerboard_is_negative(grid_graph):
    side = 8
    board = np.array(
        [1.0 if (r + c) % 2 == 0 else -1.0 for r in range(side) for c in range(side)]
    )
    assert morans_i(board, Weights(grid_graph)) < -0.9


def test_morans_i_permutation_flags_structure(grid_graph):
    side = 8
    smooth = np.array([r for r in range(side) for _ in range(side)], dtype=float)
    got = morans_i_permutation(smooth, Weights(grid_graph), replicates=199, seed=0)
    assert got["p_permutation"] <= 0.01
    assert abs(got["null_mean"]) < 0.2


def test_morans_i_permutation_does_not_flag_noise(grid_graph):
    rng = np.random.default_rng(5)
    got = morans_i_permutation(
        rng.standard_normal(len(grid_graph)), Weights(grid_graph), replicates=199, seed=1
    )
    assert got["p_permutation"] > 0.05


def test_smooth_surrogate_preserves_the_value_distribution(grid_graph):
    rng = np.random.default_rng(6)
    template = rng.random(len(grid_graph))
    surrogate = smooth_surrogate(grid_graph, template, 0.4, rng)
    assert np.allclose(np.sort(surrogate), np.sort(template))


def test_smooth_surrogate_reaches_the_target_smoothness(grid_graph):
    rng = np.random.default_rng(7)
    template = rng.random(len(grid_graph))
    w = Weights(grid_graph)
    for target in (0.2, 0.5):
        surrogate = smooth_surrogate(grid_graph, template, target, rng)
        # The bisection lands on the target from above rather than overshooting.
        assert morans_i(surrogate, w) == pytest.approx(target, abs=0.12)


def test_spatial_null_is_wider_than_the_textbook_error(grid_graph):
    """The whole inference argument: clustered data makes the naive SE too small."""
    rng = np.random.default_rng(8)
    side = 8
    x = np.array([r + 0.3 * rng.standard_normal() for r in range(side) for _ in range(side)])
    y = np.array([r + 0.3 * rng.standard_normal() for r in range(side) for _ in range(side)])
    got = spatial_null_correlation(x, y, grid_graph, replicates=120, seed=0)
    assert got["null_sd"] > got["naive_sd"]
    assert got["sd_ratio"] > 1.0
