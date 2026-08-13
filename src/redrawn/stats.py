"""Correlation, spatial autocorrelation and permutation inference.

Everything here is written out rather than pulled from a library, because the
study's claims are claims *about* these estimators and it should be possible to
read exactly what was computed.
"""

from __future__ import annotations

import numpy as np


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; NaN when either side is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if denom <= 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Correlation with each zone weighted by its crash count.

    An unweighted zone correlation treats a tract with 31 crashes and one with
    9,000 as equally informative. Weighting is the obvious fix and the study
    reports both, because the two disagree in an instructive way.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    if total <= 0 or len(x) < 3:
        return float("nan")
    mx = (w * x).sum() / total
    my = (w * y).sum() / total
    cov = (w * (x - mx) * (y - my)).sum() / total
    vx = (w * (x - mx) ** 2).sum() / total
    vy = (w * (y - my) ** 2).sum() / total
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / np.sqrt(vx * vy))


def phi(a: np.ndarray, b: np.ndarray) -> float:
    """Phi coefficient of two binary vectors -- Pearson's r, named for the 2x2 case."""
    return pearson(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher z interval for a correlation. Assumes independent observations --
    which, for zone-level rates, is exactly the assumption under audit."""
    from scipy.stats import norm

    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = norm.ppf(0.5 + level / 2)
    return (float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)))


def correlation_p_value(r: float, n: int) -> float:
    """Two-sided p-value for a correlation under the usual independence model."""
    from scipy.stats import t as student_t

    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return float("nan")
    stat = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * student_t.sf(abs(stat), n - 2))


class Weights:
    """Row-standardised contiguity weights held as flat CSR arrays.

    A dense matrix over 2,310 tracts is 43 MB and makes every Moran's I a
    five-million-flop operation; the adjacency has mean degree 5, so the sparse
    form is three orders of magnitude cheaper and the permutation tests become
    affordable.
    """

    def __init__(self, neighbours: list[set[int]]):
        self.n = len(neighbours)
        indptr = [0]
        indices: list[int] = []
        for nbs in neighbours:
            indices.extend(sorted(nbs))
            indptr.append(len(indices))
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        degree = np.diff(self.indptr).astype(np.float64)
        # Row standardisation: each row sums to 1, so S0 == number of rows with
        # at least one neighbour.
        self.data = np.repeat(np.where(degree > 0, 1.0 / np.maximum(degree, 1), 0.0),
                              np.diff(self.indptr))
        self.s0 = float(self.data.sum())
        self._src = np.repeat(np.arange(self.n), np.diff(self.indptr))

    def lag(self, values: np.ndarray) -> np.ndarray:
        """The spatial lag W z: each zone's weighted mean over its neighbours."""
        contrib = self.data * np.asarray(values, dtype=np.float64)[self.indices]
        return np.bincount(self._src, weights=contrib, minlength=self.n)

    def dense(self) -> np.ndarray:
        """Materialise the matrix. Only used by the tests, on tiny graphs."""
        w = np.zeros((self.n, self.n))
        for i in range(self.n):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                w[i, self.indices[k]] = self.data[k]
        return w


def weights_matrix(neighbours: list[set[int]], row_standardise: bool = True) -> np.ndarray:
    """Dense binary contiguity weights, optionally row-standardised."""
    n = len(neighbours)
    w = np.zeros((n, n), dtype=np.float64)
    for i, nbs in enumerate(neighbours):
        for j in nbs:
            w[i, j] = 1.0
    if row_standardise:
        totals = w.sum(axis=1, keepdims=True)
        np.divide(w, totals, out=w, where=totals > 0)
    return w


def morans_i(values: np.ndarray, w: Weights | np.ndarray) -> float:
    """Moran's I: the spatial analogue of a lag-1 autocorrelation.

        I = (n / S0) * (z' W z) / (z' z)

    with z the mean-centred values and S0 the sum of all weights. Near 0 means no
    spatial pattern; positive means neighbours resemble each other. Accepts
    either a `Weights` object or a dense matrix.
    """
    z = np.asarray(values, dtype=np.float64)
    z = z - z.mean()
    denom = (z * z).sum()
    if denom <= 0:
        return float("nan")
    if isinstance(w, Weights):
        s0, lag = w.s0, w.lag(z)
    else:
        s0, lag = w.sum(), w @ z
    if s0 <= 0:
        return float("nan")
    return float((len(z) / s0) * (z @ lag) / denom)


def smooth_surrogate(
    neighbours: list[set[int]],
    template: np.ndarray,
    target_i: float,
    rng: np.random.Generator,
    max_passes: int = 400,
) -> np.ndarray:
    """A fake map with the same spatial smoothness and value distribution as `template`.

    White noise is diffused across the adjacency graph -- each pass replaces a
    zone's value with a blend of itself and its neighbours' mean -- until its
    Moran's I reaches the target. The smoothed field is then rank-matched onto
    the sorted template values, so the surrogate has the real variable's exact
    histogram and roughly its clumpiness, but no connection to anything else on
    the map.

    This is the honest null for the gerrymander: pure white noise would be
    unrealistically easy to leave alone, since a search cannot exploit clustering
    that is not there.
    """
    w = Weights(neighbours)
    field = rng.standard_normal(len(neighbours))
    previous = field
    for _ in range(max_passes):
        if morans_i(field, w) >= target_i:
            break
        previous = field
        field = 0.5 * field + 0.5 * w.lag(field)

    # One diffusion pass overshoots the target, so the last two fields bracket
    # it. Bisect the blend between them to land on the target rather than past
    # it -- an over-smoothed surrogate would inflate the null and quietly make
    # every spatial test look more conservative than it should.
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        blended = (1 - mid) * previous + mid * field
        if morans_i(blended, w) < target_i:
            lo = mid
        else:
            hi = mid
    field = (1 - hi) * previous + hi * field

    ranks = np.argsort(np.argsort(field))
    return np.sort(np.asarray(template, dtype=np.float64))[ranks]


def morans_i_permutation(
    values: np.ndarray, w: Weights | np.ndarray, replicates: int = 999, seed: int = 0
) -> dict:
    """Permutation test for Moran's I: reshuffle values across zones.

    The null is 'this set of values, sprayed onto the map at random', which is
    the right null for asking whether the *arrangement* carries information.
    """
    rng = np.random.default_rng(seed)
    observed = morans_i(values, w)
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(replicates)
    for i in range(replicates):
        draws[i] = morans_i(rng.permutation(values), w)
    # Add-one correction: the observed value is one of the possible arrangements.
    p = (1 + int((np.abs(draws) >= abs(observed)).sum())) / (replicates + 1)
    return {
        "i": observed,
        "p_permutation": float(p),
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)),
        "replicates": replicates,
    }


def spatial_null_correlation(
    x: np.ndarray,
    y: np.ndarray,
    neighbours: list[set[int]],
    replicates: int = 499,
    seed: int = 0,
) -> dict:
    """Correlation between two zone-level variables, tested against a spatial null.

    The textbook p-value for a correlation assumes the zones are independent
    draws. They are not: neighbouring zones share traffic, streets and the
    reporting habits of the same precinct. Here `y` is compared against surrogate
    maps carrying its exact value distribution and roughly its spatial
    smoothness, but no relationship to `x`. The spread of correlations under that
    null is the honest yardstick, and it is much wider than the textbook one.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = Weights(neighbours)
    observed = pearson(x, y)
    target_i = morans_i(y, w)

    draws = np.empty(replicates)
    surrogate_i = np.empty(replicates)
    for i in range(replicates):
        surrogate = smooth_surrogate(neighbours, y, target_i, rng)
        draws[i] = pearson(x, surrogate)
        surrogate_i[i] = morans_i(surrogate, w)

    p = (1 + int((np.abs(draws) >= abs(observed)).sum())) / (replicates + 1)
    naive_sd = float(1.0 / np.sqrt(max(len(x) - 3, 1)))
    null_sd = float(draws.std(ddof=1))
    return {
        "r": observed,
        "n_zones": int(len(x)),
        "p_naive": correlation_p_value(observed, len(x)),
        "p_spatial": float(p),
        "null_sd": null_sd,
        "naive_sd": naive_sd,
        "sd_ratio": null_sd / naive_sd if naive_sd else float("nan"),
        "observed_morans_i": float(target_i),
        "surrogate_morans_i": float(surrogate_i.mean()),
        "replicates": replicates,
    }
