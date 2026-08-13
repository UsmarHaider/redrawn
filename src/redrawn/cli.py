"""Command line entry point: `python -m redrawn.cli <step>`."""

from __future__ import annotations

import argparse
import json

from . import analysis, figures, web
from .config import REPORTS


def main() -> None:
    parser = argparse.ArgumentParser(prog="redrawn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="clean, geocode and cache the study tables")
    sub.add_parser("describe", help="dataset summary and individual-level correlations")

    p = sub.add_parser("sweep", help="the scale effect and the zoning effect")
    p.add_argument("--replicates", type=int, default=40)

    p = sub.add_parser("gerrymander", help="how far redistricting alone can push r")
    p.add_argument("--steps", type=int, default=800_000)
    p.add_argument("--restarts", type=int, default=2)

    sub.add_parser("convergence", help="how the envelope grows with search effort")

    p = sub.add_parser("inference", help="spatial autocorrelation and honest error bars")
    p.add_argument("--replicates", type=int, default=499)

    sub.add_parser("figures", help="render the README figures")
    sub.add_parser("web", help="export the interface payload")
    sub.add_parser("summary", help="print the headline numbers")

    args = parser.parse_args()

    if args.command == "build":
        from .data import build

        print(json.dumps(build(), indent=2))
    elif args.command == "describe":
        analysis.describe()
        print("wrote reports/describe.json")
    elif args.command == "sweep":
        analysis.sweep(replicates=args.replicates)
        print("wrote reports/sweep.json")
    elif args.command == "gerrymander":
        analysis.gerrymander(steps=args.steps, restarts=args.restarts)
        print("wrote reports/gerrymander.json")
    elif args.command == "convergence":
        analysis.convergence()
        print("wrote reports/convergence.json")
    elif args.command == "inference":
        analysis.inference(replicates=args.replicates)
        print("wrote reports/inference.json")
    elif args.command == "figures":
        figures.all_figures()
    elif args.command == "web":
        web.export()
    elif args.command == "summary":
        summary()


def summary() -> None:
    """The numbers quoted in the README, printed from the reports on disk."""
    describe = json.loads((REPORTS / "describe.json").read_text())
    sweep = json.loads((REPORTS / "sweep.json").read_text())
    ger = json.loads((REPORTS / "gerrymander.json").read_text())
    inf = json.loads((REPORTS / "inference.json").read_text())
    conv = json.loads((REPORTS / "convergence.json").read_text())

    pair = sweep["pairs"]["speeding"]
    line = "-" * 66
    print(f"\n{'redrawn -- headline numbers':^66}\n{line}")
    print(f"{'crashes':<44}{describe['build']['crashes_in_tracts']:>22,}")
    print(f"{'tracts':<44}{describe['tracts_kept']:>22,}")
    print(f"{'point-in-polygon agreement with DCP':<44}"
          f"{describe['build']['crosswalk_agreement']:>21.4%}")
    print(line)
    print("speeding vs injury, same 1.9M crashes:")
    print(f"  {'between individual crashes':<42}{pair['point_r']:>+22.3f}")
    for key, label in [
        ("tract", "census tracts (2,310)"),
        ("nta_direct", "neighborhood tabulation areas (258)"),
        ("precinct_direct", "police precincts (78)"),
        ("community_district_direct", "community districts (71)"),
        ("borough", "boroughs (5)"),
    ]:
        print(f"  {label:<42}{pair['official'][key]['r']:>+22.3f}")
    print(line)

    def span(block: dict) -> str:
        return f"{block['r_min']:+.3f} to {block['r_max']:+.3f}"

    rung = pair["rungs"][str(sweep["fixed_k"])]
    neutral_span = f"{rung['min']:+.3f} to {rung['max']:+.3f}"
    print(f"{'40 neutral maps at K=71 span':<44}{neutral_span:>22}")
    budget = ger["uncertainty_budget"]
    print(f"{'  sampling-error width (reported)':<44}{budget['sampling_width']:>22.3f}")
    print(f"{'  neutral map-choice width':<44}{budget['neutral_map_width']:>22.3f}")
    print(f"{'  adversarial, size-constrained width':<44}"
          f"{budget['adversarial_balanced_width']:>22.3f}")
    print(line)
    print(f"{'gerrymander, real data':<44}{span(ger['targets']['real']['free']):>22}")
    print(f"{'gerrymander, matched spatial null':<44}"
          f"{span(ger['targets']['spatial_null']['free']):>22}")
    last = conv["rows"][-1]
    print(f"{'envelope after 1.9M flips (still widening)':<44}{span(last):>22}")
    print(line)
    test = inf["correlation_tests"]["speeding"]
    ratio = f"{test['sd_ratio']:.1f}x too small"
    print(f"{'textbook SE vs spatial-null SE':<44}{ratio:>22}")
    print(f"{'Moran I, injury rate across tracts':<44}"
          f"{inf['morans_i']['injury']['i']:>+22.3f}")
    print(line + "\n")


if __name__ == "__main__":
    main()
