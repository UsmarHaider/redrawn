"""The five experiments. Each writes one JSON into `reports/`.

    describe      what the data looks like, and the individual-level correlations
    scale         how r moves as zones get coarser
    zoning        how much r varies between equally-legitimate maps of one size
    gerrymander   how far r can be pushed by redrawing alone, against two nulls
    inference     how wrong the textbook standard error is on this map
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import FIXED_K, PAIRS, PRIMARY_PAIR, REPORTS, SCALE_LADDER, SEED
from .data import INDICATORS, load_study
from .gerrymander import envelope
from .partitions import (
    is_contiguous,
    merge_ladder,
    official_partitions,
    partition_correlation,
    random_contiguous,
)
from .stats import (
    Weights,
    correlation_p_value,
    fisher_ci,
    morans_i,
    morans_i_permutation,
    pearson,
    smooth_surrogate,
    spatial_null_correlation,
    weighted_pearson,
)


def _write(name: str, payload: dict) -> dict:
    (REPORTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))
    return payload


def _counts(kept: pd.DataFrame, pair: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_name, y_name = PAIRS[pair]
    return (
        kept["n"].to_numpy(np.float64),
        kept[x_name].to_numpy(np.float64),
        kept[y_name].to_numpy(np.float64),
    )


# --------------------------------------------------------------------------- #
# 1. describe
# --------------------------------------------------------------------------- #


def describe() -> dict:
    study = load_study()
    crashes, kept = study.crashes, study.kept

    rates = {name: float(crashes[name].mean()) for name in INDICATORS}
    point = {}
    for pair, (x_name, y_name) in PAIRS.items():
        x = crashes[x_name].to_numpy(np.float64)
        y = crashes[y_name].to_numpy(np.float64)
        r = pearson(x, y)
        # The 2x2 table behind the phi coefficient, so the reader can check it.
        both = int(((x > 0) & (y > 0)).sum())
        point[pair] = {
            "x": x_name,
            "y": y_name,
            "r": r,
            "n": int(len(crashes)),
            "p": correlation_p_value(r, len(crashes)),
            "rate_y_given_x": float(y[x > 0].mean()),
            "rate_y_given_not_x": float(y[x == 0].mean()),
            "both": both,
        }

    payload = {
        "build": study.report,
        "indicator_rates": rates,
        "indicator_meanings": INDICATORS,
        "point_level": point,
        "tracts_kept": int(len(kept)),
        "crashes_per_tract": {
            "min": int(kept["n"].min()),
            "median": float(kept["n"].median()),
            "max": int(kept["n"].max()),
        },
        "by_borough": {
            str(b): int(v) for b, v in kept.groupby("borough")["n"].sum().items()
        },
        "by_year": {str(y): int(v) for y, v in crashes.groupby("year").size().items()},
    }
    return _write("describe", payload)


# --------------------------------------------------------------------------- #
# 2. scale  +  3. zoning  (one sweep, two readings)
# --------------------------------------------------------------------------- #


def sweep(replicates: int = 40) -> dict:
    """Build `replicates` independent contiguous ladders and score every rung.

    Reading down a column gives the scale effect; reading across a row at fixed K
    gives the zoning effect. Both come out of the same set of maps, which is the
    point -- they are two views of one phenomenon.
    """
    study = load_study()
    kept = study.kept
    neighbours = study.neighbours()
    n = kept["n"].to_numpy(np.float64)
    ks = sorted({*SCALE_LADDER, FIXED_K}, reverse=True)

    started = time.time()
    ladders = []
    for replicate in range(replicates):
        rng = np.random.default_rng(SEED + replicate)
        ladders.append(merge_ladder(neighbours, n, ks, rng))
        print(f"  ladder {replicate + 1}/{replicates}", end="\r", flush=True)

    # Region-grown maps at the fixed K, as a second and much rougher map family.
    grown = [
        random_contiguous(neighbours, FIXED_K, np.random.default_rng(SEED + 500 + i))
        for i in range(replicates)
    ]

    official = official_partitions(kept)
    out: dict = {
        "replicates": replicates,
        "fixed_k": FIXED_K,
        "ks": ks,
        "seconds": None,
        "pairs": {},
    }

    for pair in PAIRS:
        n_arr, x, y = _counts(kept, pair)
        rungs = {}
        for k in ks:
            values = np.array(
                [partition_correlation(lad[k], n_arr, x, y) for lad in ladders if k in lad]
            )
            weighted = np.array(
                [
                    partition_correlation(lad[k], n_arr, x, y, weighted=True)
                    for lad in ladders
                    if k in lad
                ]
            )
            rungs[k] = {
                "k": k,
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
                "spread": float(values.max() - values.min()),
                "weighted_median": float(np.median(weighted)),
            }

        grown_values = np.array([partition_correlation(g, n_arr, x, y) for g in grown])
        official_values = {
            name: {
                "k": int(labels.max() + 1),
                "r": partition_correlation(labels, n_arr, x, y),
                "r_weighted": partition_correlation(labels, n_arr, x, y, weighted=True),
            }
            for name, labels in official.items()
        }
        # The real layers that are not tract-nested, scored on their own polygons.
        for layer in ("community_district", "precinct", "nta"):
            z = study.zones[(study.zones["layer"] == layer) & (study.zones["n"] > 0)]
            x_name, y_name = PAIRS[pair]
            official_values[f"{layer}_direct"] = {
                "k": int(len(z)),
                "r": pearson(z[x_name] / z["n"], z[y_name] / z["n"]),
                "r_weighted": weighted_pearson(
                    z[x_name] / z["n"], z[y_name] / z["n"], z["n"]
                ),
            }

        fixed = rungs[FIXED_K]
        out["pairs"][pair] = {
            "x": PAIRS[pair][0],
            "y": PAIRS[pair][1],
            "point_r": pearson(
                study.crashes[PAIRS[pair][0]].to_numpy(np.float64),
                study.crashes[PAIRS[pair][1]].to_numpy(np.float64),
            ),
            "rungs": rungs,
            "official": official_values,
            "grown_at_fixed_k": {
                "median": float(np.median(grown_values)),
                "min": float(grown_values.min()),
                "max": float(grown_values.max()),
                "sd": float(grown_values.std(ddof=1)),
            },
            "zoning_at_fixed_k": fixed,
        }

    out["seconds"] = round(time.time() - started, 1)
    out["balance"] = _balance_report(ladders, grown, official, n)
    out["contiguity_check"] = _contiguity_report(ladders, grown, official, neighbours)
    return _write("sweep", out)


def _balance_report(ladders, grown, official, n) -> dict:
    def stats(labels):
        totals = np.bincount(labels, weights=n)
        totals = totals[totals > 0]
        return float(totals.max() / np.median(totals))

    return {
        "merged_at_fixed_k": float(np.median([stats(lad[FIXED_K]) for lad in ladders])),
        "grown_at_fixed_k": float(np.median([stats(g) for g in grown])),
        "official_cdta": stats(official["cdta"]),
        "official_nta": stats(official["nta"]),
        "note": "max/median crash count across zones; lower is more balanced",
    }


def _contiguity_report(ladders, grown, official, neighbours) -> dict:
    """Official layers contain islands, so a few of their zones are disconnected
    under tract rook-contiguity. Every synthetic map here is fully contiguous,
    which makes the synthetic family *stricter* than the real one."""

    def disconnected(labels) -> int:
        bad = 0
        for zone in np.unique(labels):
            members = set(np.flatnonzero(labels == zone).tolist())
            start = next(iter(members))
            stack, seen = [start], {start}
            while stack:
                node = stack.pop()
                for nb in neighbours[node]:
                    if nb in members and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            if len(seen) != len(members):
                bad += 1
        return bad

    return {
        "merged_all_contiguous": all(is_contiguous(lad[FIXED_K], neighbours) for lad in ladders),
        "grown_all_contiguous": all(is_contiguous(g, neighbours) for g in grown),
        "official_disconnected_zones": {
            name: disconnected(labels)
            for name, labels in official.items()
            if name != "tract"
        },
    }


# --------------------------------------------------------------------------- #
# 4. gerrymander
# --------------------------------------------------------------------------- #


def gerrymander(steps: int = 800_000, restarts: int = 2, pair: str = PRIMARY_PAIR) -> dict:
    """Push the correlation as far as contiguous redistricting allows.

    Run against three targets:

    * **real**       -- the actual injury counts.
    * **spatial null** -- a surrogate map with the real variable's exact
      distribution and comparable spatial smoothness, but no relationship to x.
      Whatever the search reaches here is manufactured, not found.
    * **white noise** -- an independent coin flip per crash. No spatial structure
      at all, which shows how much of the null envelope is spatial clumping
      rather than search.

    Each target is searched with and without an equal-size constraint on zones.
    """
    study = load_study()
    kept = study.kept
    neighbours = study.neighbours()
    n, x, y = _counts(kept, pair)
    rng = np.random.default_rng(SEED)

    start = merge_ladder(neighbours, n, [FIXED_K], rng)[FIXED_K]

    # Spatially-structured null: same histogram of injury rates, same clumpiness,
    # no link to speeding. Counts are rebuilt from the surrogate rate.
    real_rate = y / n
    w = Weights(neighbours)
    surrogate_rate = smooth_surrogate(neighbours, real_rate, morans_i(real_rate, w), rng)
    y_spatial = surrogate_rate * n

    # White-noise null: each crash independently injured at the citywide rate.
    p_injury = float(study.crashes[PAIRS[pair][1]].mean())
    y_noise = rng.binomial(n.astype(np.int64), p_injury).astype(np.float64)

    targets = {
        "real": y,
        "spatial_null": y_spatial,
        "white_noise": y_noise,
    }

    started = time.time()
    results: dict = {
        "pair": pair,
        "x": PAIRS[pair][0],
        "y": PAIRS[pair][1],
        "k": FIXED_K,
        "steps": steps,
        "restarts": restarts,
        "start_r": partition_correlation(start, n, x, y),
        "targets": {},
    }

    for target_index, (name, y_target) in enumerate(targets.items()):
        block = {}
        for label, balance in (("free", None), ("balanced", 0.30)):
            env = envelope(
                start,
                neighbours,
                n,
                x,
                y_target,
                steps=steps,
                balance=balance,
                # Deterministic per-scenario offset. `hash()` on a str is salted
                # per process, so it must never appear in a seed.
                seed=SEED + 100 * target_index + (0 if balance is None else 50),
                restarts=restarts,
            )
            block[label] = {
                "r_min": env["r_min"],
                "r_max": env["r_max"],
                "width": env["width"],
                "start_r": env["start_r"],
                "contiguous_max": is_contiguous(env["labels_max"], neighbours),
                "contiguous_min": is_contiguous(env["labels_min"], neighbours),
                "runs": env["runs"],
            }
            if name == "real":
                # Keep the extreme maps so the interface can draw them.
                np.save(REPORTS / f"labels_max_{label}.npy", env["labels_max"])
                np.save(REPORTS / f"labels_min_{label}.npy", env["labels_min"])
                block[label]["trace_max"] = env["trace_max"]
                block[label]["trace_min"] = env["trace_min"]
            print(f"  {name:13s} {label:9s} [{env['r_min']:+.3f}, {env['r_max']:+.3f}]")
        block["point_r_of_target"] = (
            results["start_r"] if name == "real" else None
        )
        results["targets"][name] = block

    np.save(REPORTS / "labels_start.npy", start)
    results["seconds"] = round(time.time() - started, 1)

    # The headline comparison: how much of the real envelope is more than a
    # search over an unrelated but equally clumpy map could have produced?
    real_free = results["targets"]["real"]["free"]
    null_free = results["targets"]["spatial_null"]["free"]
    results["excess_over_spatial_null"] = {
        "real_width": real_free["width"],
        "null_width": null_free["width"],
        "ratio": real_free["width"] / null_free["width"] if null_free["width"] else float("nan"),
        "real_max": real_free["r_max"],
        "null_max": null_free["r_max"],
    }

    # Three sources of uncertainty in one number each, so they can be compared
    # directly. The analyst reports only the first.
    official = official_partitions(kept)["cdta"]
    r_official = partition_correlation(official, n, x, y)
    ci = fisher_ci(r_official, FIXED_K)
    neutral = [
        partition_correlation(
            merge_ladder(neighbours, n, [FIXED_K], np.random.default_rng(SEED + i))[FIXED_K],
            n,
            x,
            y,
        )
        for i in range(40)
    ]
    real_balanced = results["targets"]["real"]["balanced"]
    results["uncertainty_budget"] = {
        "official_r": r_official,
        "sampling_ci": list(ci),
        "sampling_width": ci[1] - ci[0],
        "neutral_map_range": [float(min(neutral)), float(max(neutral))],
        "neutral_map_width": float(max(neutral) - min(neutral)),
        "adversarial_balanced_range": [real_balanced["r_min"], real_balanced["r_max"]],
        "adversarial_balanced_width": real_balanced["width"],
    }
    return _write("gerrymander", results)


def convergence(
    steps_list: tuple[int, ...] = (30_000, 60_000, 120_000, 240_000, 480_000, 960_000, 1_920_000),
) -> dict:
    """How far the search gets as a function of how long it is allowed to run.

    This is not a convergence check that passes -- it is the finding. The
    reachable envelope keeps widening at every budget tried, with no sign of a
    plateau, so every width reported in this repository is a lower bound on what
    redistricting can do. Any published robustness check that tries a fixed
    number of alternative maps and reports the spread is understating the
    problem by an unknown amount, and so is this one.
    """
    study = load_study()
    kept = study.kept
    neighbours = study.neighbours()
    n, x, y = _counts(kept, PRIMARY_PAIR)
    start = merge_ladder(neighbours, n, [FIXED_K], np.random.default_rng(SEED))[FIXED_K]

    rows = []
    for steps in steps_list:
        env = envelope(
            start, neighbours, n, x, y, steps=steps, balance=0.30, seed=SEED, restarts=2
        )
        rows.append({"steps": steps, "r_min": env["r_min"], "r_max": env["r_max"],
                     "width": env["width"]})
        print(f"  {steps:>7,} steps -> [{env['r_min']:+.3f}, {env['r_max']:+.3f}]")
    return _write("convergence", {"balance": 0.30, "k": FIXED_K, "rows": rows})


# --------------------------------------------------------------------------- #
# 5. inference
# --------------------------------------------------------------------------- #


def inference(replicates: int = 499) -> dict:
    """Is the textbook standard error for a zone-level correlation defensible here?"""
    study = load_study()
    kept = study.kept
    neighbours = study.neighbours()
    w = Weights(neighbours)

    started = time.time()
    out: dict = {"replicates": replicates, "level": "tract", "n_zones": int(len(kept))}

    autocorrelation = {}
    for name in INDICATORS:
        rate = kept[name].to_numpy(np.float64) / kept["n"].to_numpy(np.float64)
        autocorrelation[name] = morans_i_permutation(rate, w, replicates=replicates, seed=SEED)
    out["morans_i"] = autocorrelation

    tests = {}
    for pair, (x_name, y_name) in PAIRS.items():
        rx = kept[x_name].to_numpy(np.float64) / kept["n"].to_numpy(np.float64)
        ry = kept[y_name].to_numpy(np.float64) / kept["n"].to_numpy(np.float64)
        result = spatial_null_correlation(rx, ry, neighbours, replicates=replicates, seed=SEED)
        result["fisher_ci"] = fisher_ci(result["r"], len(rx))
        tests[pair] = result
        print(f"  {pair:12s} r={result['r']:+.3f}  naive sd {result['naive_sd']:.4f} "
              f"vs spatial sd {result['null_sd']:.4f}  ({result['sd_ratio']:.1f}x)")
    out["correlation_tests"] = tests
    out["seconds"] = round(time.time() - started, 1)
    return _write("inference", out)
