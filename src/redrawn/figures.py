"""Figures for the README.

Palette is the project's four-slot categorical theme (blue, orange, aqua,
yellow), validated for colour-vision deficiency separation and contrast against
the light chart surface; the yellow and aqua slots sit under 3:1 on this surface
so every series that uses them is directly labelled as well as coloured.

Choropleths use a single-hue sequential ramp rather than the categorical slots,
because the quantity being mapped is ordered.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .config import FIGURES, FIXED_K, REPORTS
from .data import load_boundaries, load_study
from .geometry import LAT_SCALE
from .partitions import merge_ladder, official_partitions, partition_correlation, zone_rates

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

RAMP = LinearSegmentedColormap.from_list("redrawn", ["#eef3fa", "#9dc0e8", "#2a78d6", "#12355e"])


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK2,
            "axes.labelsize": 9,
            "axes.edgecolor": AXIS,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def _tidy(ax, xgrid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=xgrid)


def _report(name: str) -> dict:
    return json.loads((REPORTS / f"{name}.json").read_text())


def _save(fig, name: str) -> None:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(FIGURES.parents[1])}")


# --------------------------------------------------------------------------- #
# map helpers
# --------------------------------------------------------------------------- #


def _tract_polygons(kept):
    """Exterior rings of the kept tracts, in kept-table order, as a PolyCollection
    input. Longitude is scaled by cos(lat) so the city is not stretched sideways."""
    shapes = load_boundaries("tract")
    by_geoid = {s.key: s for s in shapes}
    polys, owner = [], []
    for row, geoid in enumerate(kept["geoid"]):
        shape = by_geoid.get(str(geoid))
        if shape is None:
            continue
        for ring in shape.rings:
            if ring.hole:
                continue
            coords = ring.coords.copy()
            coords[:, 0] *= LAT_SCALE
            polys.append(coords)
            owner.append(row)
    return polys, np.asarray(owner)


def _draw_map(ax, polys, owner, values, cmap=RAMP, norm=None):
    """Choropleth of the tracts, coloured by a per-tract value.

    Edges are drawn in each polygon's own fill colour (`edgecolors="face"`).
    That closes the hairline seams between tracts without painting a visible
    grid over the map -- and because tracts in the same zone get the same
    colour, the zones themselves emerge as solid blocks. Stroking the tract
    outlines instead, which is the obvious thing to try, buries the zone
    structure under 2,310 white lines and makes every map look alike.
    """
    collection = PolyCollection(
        polys,
        array=values[owner],
        cmap=cmap,
        norm=norm or Normalize(np.nanmin(values), np.nanmax(values)),
        linewidths=0.25,
        edgecolors="face",
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    return collection


# --------------------------------------------------------------------------- #
# 1. the data
# --------------------------------------------------------------------------- #


def figure_dataset() -> None:
    _style()
    study = load_study()
    describe = _report("describe")
    kept = study.kept
    polys, owner = _tract_polygons(kept)

    fig = plt.figure(figsize=(11.6, 4.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    density = kept["n"].to_numpy(float)
    collection = _draw_map(
        ax, polys, owner, density, norm=Normalize(0, np.quantile(density, 0.97))
    )
    ax.set_title(f"{int(kept['n'].sum()):,} crashes, {len(kept):,} tracts", loc="left")
    bar = fig.colorbar(collection, ax=ax, fraction=0.04, pad=-0.02, shrink=0.82)
    bar.set_label("crashes per tract", color=INK2, fontsize=8.5)
    bar.ax.tick_params(labelsize=8)
    bar.outline.set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    years = describe["by_year"]
    xs = sorted(int(y) for y in years)
    ax.bar(xs, [years[str(y)] for y in xs], color=BLUE, width=0.72)
    ax.set_title("crashes by year", loc="left")
    ax.set_xlabel("")
    ax.set_xticks([y for y in xs if y % 3 == 1])
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    _tidy(ax)

    ax = fig.add_subplot(grid[0, 2])
    rates = describe["indicator_rates"]
    names = ["injury", "unspecified", "distraction", "vru", "speeding"]
    ax.barh(names[::-1], [rates[n] * 100 for n in names[::-1]], color=ORANGE, height=0.62)
    for i, n in enumerate(names[::-1]):
        ax.text(rates[n] * 100 + 0.8, i, f"{rates[n]:.1%}", va="center", fontsize=8.5, color=INK2)
    ax.set_title("share of crashes flagged", loc="left")
    ax.set_xlim(0, 40)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    _tidy(ax, xgrid=True)
    ax.grid(axis="y", visible=False)

    _save(fig, "dataset")


# --------------------------------------------------------------------------- #
# 2. the scale effect
# --------------------------------------------------------------------------- #


def figure_scale() -> None:
    _style()
    sweep = _report("sweep")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), gridspec_kw={"wspace": 0.22})

    ax = axes[0]
    pair = sweep["pairs"]["speeding"]
    ks = sorted((int(k) for k in pair["rungs"]), reverse=True)
    med = [pair["rungs"][str(k)]["median"] for k in ks]
    # 5th-95th percentile rather than min-max: at K=5 a single unlucky draw
    # reaches -0.09 and the min-max band becomes a spike that hides the trend.
    lo = [pair["rungs"][str(k)]["q05"] for k in ks]
    hi = [pair["rungs"][str(k)]["q95"] for k in ks]

    ax.fill_between(ks, lo, hi, color=BLUE, alpha=0.16, linewidth=0,
                    label="middle 90% of 40 maps per rung")
    ax.plot(ks, med, color=BLUE, marker="o", label="median synthetic map")
    ax.axhline(pair["point_r"], color=ORANGE, ls="--", lw=1.6)
    ax.text(
        1900, pair["point_r"] + 0.035,
        f"individual crashes: r = {pair['point_r']:.3f}",
        color=ORANGE, fontsize=8.5, fontweight="semibold",
    )

    officials = [
        ("tract", "census tracts"),
        ("nta_direct", "NTAs"),
        ("precinct_direct", "police precincts"),
        ("community_district_direct", "community districts"),
        ("borough", "boroughs"),
    ]
    for key, label in officials:
        item = pair["official"][key]
        ax.plot(item["k"], item["r"], marker="D", color=INK, markersize=6, zorder=5)
        ax.annotate(
            label,
            (item["k"], item["r"]),
            textcoords="offset points",
            xytext=(6, -11),
            fontsize=8,
            color=INK,
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("number of zones (log scale, coarser to the right)")
    ax.set_ylabel("reported correlation")
    ax.set_title("Speeding vs injury: the answer depends on the zone size", loc="left")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="upper left")
    _tidy(ax)

    ax = axes[1]
    for name, colour in (("speeding", BLUE), ("unspecified", ORANGE), ("distraction", AQUA)):
        block = sweep["pairs"][name]
        ks = sorted((int(k) for k in block["rungs"]), reverse=True)
        ax.plot(
            ks,
            [block["rungs"][str(k)]["median"] for k in ks],
            marker="o",
            color=colour,
            label=f"{name} vs injury",
        )
        ax.scatter([2310], [block["point_r"]], color=colour, marker="*", s=110, zorder=5)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("number of zones (log scale, coarser to the right)")
    ax.set_ylabel("reported correlation")
    ax.set_title("Two of the three reverse sign entirely", loc="left")
    ax.legend(loc="upper left")
    _tidy(ax)
    ax.annotate(
        "stars: correlation between\nindividual crashes",
        (2310, 0.0),
        textcoords="offset points",
        xytext=(14, -34),
        fontsize=8,
        color=MUTED,
    )

    _save(fig, "scale")


# --------------------------------------------------------------------------- #
# 3. the zoning effect
# --------------------------------------------------------------------------- #


def figure_zoning() -> None:
    _style()
    study = load_study()
    kept = study.kept
    sweep = _report("sweep")
    pair = sweep["pairs"]["speeding"]
    n = kept["n"].to_numpy(float)
    x = kept["speeding"].to_numpy(float)
    y = kept["injury"].to_numpy(float)

    fig = plt.figure(figsize=(11.6, 4.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.05)

    # Two maps with the same number of zones, side by side.
    polys, owner = _tract_polygons(kept)
    official = official_partitions(kept)
    synthetic = merge_ladder(study.neighbours(), n, [FIXED_K], np.random.default_rng(3))[FIXED_K]
    for column, (labels, name) in enumerate(
        [(official["cdta"], "the official community districts"), (synthetic, "a synthetic map")]
    ):
        ax = fig.add_subplot(grid[0, column])
        _, _, ry = zone_rates(labels, n, x, y)
        _draw_map(ax, polys, owner, ry[labels], norm=Normalize(0.15, 0.40))
        r = partition_correlation(labels, n, x, y)
        ax.set_title(f"{name}\n71 zones · r = {r:+.3f}", loc="center", fontsize=9.5)

    ax = fig.add_subplot(grid[0, 2])
    rung = pair["rungs"][str(FIXED_K)]
    lo, hi = rung["min"], rung["max"]
    ax.axvspan(lo, hi, color=BLUE, alpha=0.16, linewidth=0)
    marks = [
        ("census tracts (2,310)", pair["official"]["tract"]["r"], INK),
        ("NTAs (258)", pair["official"]["nta_direct"]["r"], INK),
        ("police precincts (78)", pair["official"]["precinct_direct"]["r"], ORANGE),
        ("community districts (71)", pair["official"]["community_district_direct"]["r"], ORANGE),
        ("boroughs (5)", pair["official"]["borough"]["r"], INK),
    ]
    for i, (label, value, colour) in enumerate(marks):
        ax.plot([value], [i], marker="D", color=colour, markersize=7)
        ax.text(value + 0.014, i, f"{value:+.3f}", va="center", fontsize=8.5, color=colour)
    ax.text(
        (lo + hi) / 2, len(marks) - 0.35,
        f"40 neutral maps at K=71\nspan just {hi - lo:.2f}",
        ha="center", va="center", fontsize=8.5, color=BLUE, fontweight="semibold",
    )
    ax.set_yticks(range(len(marks)))
    ax.set_yticklabels([m[0] for m in marks])
    ax.set_xlabel("reported correlation")
    ax.set_xlim(0.18, 0.84)
    ax.set_ylim(-0.6, len(marks) + 0.15)
    ax.set_title("Every official map of NYC gives a different answer", loc="right")
    ax.yaxis.tick_right()
    _tidy(ax, xgrid=True)
    ax.grid(axis="y", visible=False)

    _save(fig, "zoning")


# --------------------------------------------------------------------------- #
# 4. the gerrymander
# --------------------------------------------------------------------------- #


def figure_gerrymander() -> None:
    _style()
    study = load_study()
    kept = study.kept
    ger = _report("gerrymander")
    n = kept["n"].to_numpy(float)
    x = kept["speeding"].to_numpy(float)
    y = kept["injury"].to_numpy(float)

    fig = plt.figure(figsize=(11.6, 7.4))
    # Two subfigures rather than one grid: the maps want to sit tight against
    # each other, the charts want room for their axis labels.
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1], hspace=0.0)

    polys, owner = _tract_polygons(kept)
    panels = [
        ("labels_min_balanced.npy", "drawn to minimise"),
        ("labels_start.npy", "drawn neutrally"),
        ("labels_max_balanced.npy", "drawn to maximise"),
    ]
    map_axes = top.subplots(1, 3, gridspec_kw={"wspace": 0.02})
    for ax, (filename, title) in zip(map_axes, panels):
        labels = np.load(REPORTS / filename)
        _, _, ry = zone_rates(labels, n, x, y)
        _draw_map(ax, polys, owner, ry[labels], norm=Normalize(0.15, 0.40))
        r = partition_correlation(labels, n, x, y)
        ax.set_title(f"{title}\nr = {r:+.3f}", loc="center", fontsize=10)

    # Envelope comparison across targets.
    chart_axes = bottom.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1], "wspace": 0.3})
    ax = chart_axes[0]
    rows = [
        ("real data", "real", BLUE),
        ("spatially matched null", "spatial_null", ORANGE),
        ("white noise", "white_noise", MUTED),
    ]
    height = 0.32
    for i, (label, key, colour) in enumerate(rows):
        for offset, mode, alpha in ((height / 2, "free", 0.35), (-height / 2, "balanced", 1.0)):
            block = ger["targets"][key][mode]
            ax.barh(
                i + offset,
                block["r_max"] - block["r_min"],
                left=block["r_min"],
                height=height,
                color=colour,
                alpha=alpha,
                linewidth=0,
            )
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(ger["start_r"], color=AQUA, lw=1.6, ls="--")
    ax.text(ger["start_r"] - 0.03, -0.62, f"neutral map ({ger['start_r']:+.2f})",
            color=AQUA, fontsize=8.5, fontweight="semibold", ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("reported correlation reachable by redrawing the map alone")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.85, len(rows) - 0.35)
    ax.set_title(
        "Pale bar: no size constraint.   Solid bar: zones within ±30% of equal size.",
        loc="left", fontsize=9,
    )
    _tidy(ax, xgrid=True)
    ax.grid(axis="y", visible=False)

    # The uncertainty budget.
    ax = chart_axes[1]
    budget = ger["uncertainty_budget"]
    bars = [
        ("sampling\nerror", budget["sampling_width"], MUTED),
        ("neutral map\nchoice", budget["neutral_map_width"], BLUE),
        ("adversarial\nredistricting", budget["adversarial_balanced_width"], ORANGE),
    ]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.02, f"{b[1]:.2f}", ha="center", fontsize=9.5, color=INK,
                fontweight="semibold")
    ax.set_ylabel("width of the interval on r")
    ax.set_ylim(0, max(b[1] for b in bars) * 1.22)
    ax.set_title("Where the uncertainty actually is", loc="left")
    ax.tick_params(axis="x", labelsize=8.5)
    _tidy(ax)

    _save(fig, "gerrymander")


# --------------------------------------------------------------------------- #
# 5. inference and convergence
# --------------------------------------------------------------------------- #


def figure_inference() -> None:
    _style()
    inference = _report("inference")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    names = ["unspecified", "distraction", "vru", "injury", "speeding"]
    values = [inference["morans_i"][n]["i"] for n in names]
    ax.barh(names, values, color=BLUE, height=0.6)
    for i, v in enumerate(values):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Moran's I across 2,310 tracts")
    ax.set_title("Everything is spatially clustered", loc="left")
    _tidy(ax, xgrid=True)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    pairs = ["speeding", "unspecified", "distraction"]
    naive = [inference["correlation_tests"][p]["naive_sd"] for p in pairs]
    spatial = [inference["correlation_tests"][p]["null_sd"] for p in pairs]
    pos = np.arange(len(pairs))
    ax.bar(pos - 0.19, naive, width=0.36, color=MUTED, label="textbook")
    ax.bar(pos + 0.19, spatial, width=0.36, color=ORANGE, label="spatial null")
    for i, p in enumerate(pairs):
        ratio = inference["correlation_tests"][p]["sd_ratio"]
        ax.text(i, max(naive[i], spatial[i]) + 0.002, f"{ratio:.1f}x", ha="center",
                fontsize=8.5, color=INK, fontweight="semibold")
    ax.set_xticks(pos)
    ax.set_xticklabels(pairs, fontsize=8.5)
    ax.set_ylabel("standard error of r")
    ax.set_title("The reported error bar is 2x too small", loc="left")
    ax.legend(loc="upper left")
    _tidy(ax)

    ax = axes[2]
    conv = _report("convergence")
    steps = [row["steps"] for row in conv["rows"]]
    ax.plot(steps, [row["r_max"] for row in conv["rows"]], marker="o", color=ORANGE, label="pushed up")
    ax.plot(steps, [row["r_min"] for row in conv["rows"]], marker="o", color=BLUE, label="pushed down")
    ax.set_xscale("log")
    ax.set_xlabel("search effort (boundary flips tried)")
    ax.set_ylabel("reachable correlation")
    ax.set_title("The envelope never closes", loc="left")
    ax.legend(loc="center right")
    _tidy(ax)

    _save(fig, "inference")


def all_figures() -> None:
    figure_dataset()
    figure_scale()
    figure_zoning()
    figure_gerrymander()
    figure_inference()
