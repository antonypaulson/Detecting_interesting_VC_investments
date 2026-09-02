"""CLI: python -m vc_investments"""

from __future__ import annotations

import argparse

from vc_investments.paths import ensure_output_dir
from vc_investments.scoring import (
    load_scoring_frame,
    reference_apps,
    score_apps,
    top_scorers,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the 2019 App Store VC-candidate scoring model."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top-scoring apps to print (default: 10).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write outputs/top_scorers.csv and outputs/reference_apps.csv.",
    )
    args = parser.parse_args(argv)

    scored = score_apps(load_scoring_frame())
    leaders = top_scorers(scored, n=args.top)
    reference = reference_apps(scored)

    print("Top scoring apps (same rules as TCG_scoring_model.ipynb):")
    print(leaders.to_string(index=False))
    print("\nReference apps from the original notebook (not new picks):")
    print(reference.to_string(index=False))

    if args.save:
        out = ensure_output_dir()
        leaders.to_csv(out / "top_scorers.csv", index=False)
        reference.to_csv(out / "reference_apps.csv", index=False)
        print(f"\nWrote {out / 'top_scorers.csv'} and {out / 'reference_apps.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
