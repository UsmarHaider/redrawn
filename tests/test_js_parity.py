"""The browser must agree with Python.

The page recomputes correlations live rather than displaying stored answers, so
`ui/maup.js` is a second implementation of the analysis and can drift from the
first. These tests run that exact file under Node against fixtures generated
here, and require the two to agree to floating-point tolerance.

Skipped when Node is unavailable; CI installs it.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from redrawn.config import UI
from redrawn.partitions import is_contiguous, merge_ladder, partition_correlation
from redrawn.stats import pearson, weighted_pearson

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

MAUP = UI / "maup.js"


def run_node(script: str, payload: dict, tmp_path) -> dict:
    """Execute a snippet with maup.js loaded and `INPUT` bound to `payload`."""
    data_file = tmp_path / "input.json"
    data_file.write_text(json.dumps(payload))
    runner = tmp_path / "run.js"
    runner.write_text(
        "const MAUP = require({maup});\n"
        "const INPUT = require({data});\n"
        "{body}\n".format(
            maup=json.dumps(str(MAUP)), data=json.dumps(str(data_file)), body=script
        )
    )
    completed = subprocess.run(
        [NODE, str(runner)], capture_output=True, text=True, timeout=120
    )
    if completed.returncode != 0:
        raise AssertionError(f"node failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


@pytest.fixture
def city(grid_graph, toy_counts):
    n, x, y = toy_counts
    adj_start = [0]
    adj: list[int] = []
    for nbs in grid_graph:
        adj.extend(sorted(nbs))
        adj_start.append(len(adj))
    return {
        "n": n.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "adjStart": adj_start,
        "adj": adj,
        "graph": grid_graph,
        "arrays": (n, x, y),
    }


def test_maup_js_exists():
    assert MAUP.exists(), "the interface's analysis port is missing"


def test_pearson_parity(city, tmp_path):
    got = run_node(
        "console.log(JSON.stringify({r: MAUP.pearson(INPUT.a, INPUT.b)}))",
        {"a": city["x"], "b": city["y"]},
        tmp_path,
    )
    expected = pearson(np.array(city["x"]), np.array(city["y"]))
    assert got["r"] == pytest.approx(expected, abs=1e-12)


def test_weighted_pearson_parity(city, tmp_path):
    got = run_node(
        "console.log(JSON.stringify({r: MAUP.weightedPearson(INPUT.a, INPUT.b, INPUT.w)}))",
        {"a": city["x"], "b": city["y"], "w": city["n"]},
        tmp_path,
    )
    expected = weighted_pearson(
        np.array(city["x"]), np.array(city["y"]), np.array(city["n"])
    )
    assert got["r"] == pytest.approx(expected, abs=1e-12)


def test_partition_correlation_parity_over_many_maps(city, tmp_path):
    """The number the page puts on screen, checked against Python for 12 maps."""
    n, x, y = city["arrays"]
    maps, expected = [], []
    for seed in range(12):
        k = 4 + seed
        labels = merge_ladder(city["graph"], n, [k], np.random.default_rng(seed))[k]
        maps.append(labels.tolist())
        expected.append(partition_correlation(labels, n, x, y))

    got = run_node(
        """
        const out = INPUT.maps.map(function (labels) {
          return {
            r: MAUP.partitionCorrelation(labels, INPUT.n, INPUT.x, INPUT.y, false),
            rw: MAUP.partitionCorrelation(labels, INPUT.n, INPUT.x, INPUT.y, true),
            k: MAUP.zoneRates(labels, INPUT.n, INPUT.x, INPUT.y).k
          };
        });
        console.log(JSON.stringify(out));
        """,
        {"maps": maps, "n": city["n"], "x": city["x"], "y": city["y"]},
        tmp_path,
    )
    for i, (row, want) in enumerate(zip(got, expected)):
        assert row["r"] == pytest.approx(want, abs=1e-12), f"map {i}"
        assert row["k"] == len(np.unique(maps[i]))

    # And the weighted variant, which the interface also displays.
    for i, row in enumerate(got):
        want_w = partition_correlation(np.array(maps[i]), n, x, y, weighted=True)
        assert row["rw"] == pytest.approx(want_w, abs=1e-12)


def test_js_merge_produces_valid_maps(city, tmp_path):
    """The port's own map generator must obey the same constraints as Python's:
    exactly K zones, every one of them connected."""
    got = run_node(
        """
        const out = [];
        for (let seed = 1; seed <= 6; seed++) {
          for (const k of [4, 9, 20]) {
            const labels = MAUP.mergeToK(INPUT.adjStart, INPUT.adj, INPUT.n, k,
                                         MAUP.mulberry32(seed));
            out.push({
              k: k,
              zones: new Set(Array.from(labels)).size,
              contiguous: MAUP.isContiguous(labels, INPUT.adjStart, INPUT.adj),
              labels: Array.from(labels)
            });
          }
        }
        console.log(JSON.stringify(out));
        """,
        {"adjStart": city["adjStart"], "adj": city["adj"], "n": city["n"]},
        tmp_path,
    )
    assert len(got) == 18
    for row in got:
        assert row["zones"] == row["k"]
        assert row["contiguous"] is True
        # Cross-check the JS contiguity claim with the Python implementation.
        assert is_contiguous(np.array(row["labels"]), city["graph"])


def test_js_grow_produces_valid_maps(city, tmp_path):
    got = run_node(
        """
        const out = [];
        for (let seed = 1; seed <= 6; seed++) {
          const labels = MAUP.growToK(INPUT.adjStart, INPUT.adj, 7, MAUP.mulberry32(seed));
          out.push({
            zones: new Set(Array.from(labels)).size,
            labels: Array.from(labels)
          });
        }
        console.log(JSON.stringify(out));
        """,
        {"adjStart": city["adjStart"], "adj": city["adj"]},
        tmp_path,
    )
    for row in got:
        assert row["zones"] == 7
        assert is_contiguous(np.array(row["labels"]), city["graph"])


def test_js_rng_is_deterministic(tmp_path):
    got = run_node(
        """
        const a = MAUP.mulberry32(42), b = MAUP.mulberry32(42);
        const x = [], y = [];
        for (let i = 0; i < 5; i++) { x.push(a()); y.push(b()); }
        console.log(JSON.stringify({x: x, y: y}));
        """,
        {},
        tmp_path,
    )
    assert got["x"] == got["y"]
    assert all(0.0 <= v < 1.0 for v in got["x"])
    assert len(set(got["x"])) == 5
