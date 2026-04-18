from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_weekly_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trump_graph",
        description="Build weekly mention network artifacts from Trump tweet CSV data.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build weekly mention-network artifacts.")
    build_parser.add_argument("--input", type=Path, required=True, help="Path to input CSV file.")
    build_parser.add_argument("--out", type=Path, required=True, help="Output directory for artifacts.")
    build_parser.add_argument(
        "--min-mention-count",
        type=int,
        default=1,
        help="Minimum per-week mention frequency required for a node.",
    )
    build_parser.add_argument(
        "--include-retweets",
        dest="include_retweets",
        action="store_true",
        default=True,
        help="Include retweets in processing (default).",
    )
    build_parser.add_argument(
        "--exclude-retweets",
        dest="include_retweets",
        action="store_false",
        help="Exclude retweets from processing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        stats = build_weekly_artifacts(
            input_csv=args.input,
            output_dir=args.out,
            min_mention_count=args.min_mention_count,
            include_retweets=args.include_retweets,
        )
        print(f"Total tweets read: {stats.total_tweets}")
        print(f"Tweets processed: {stats.processed_tweets}")
        print(f"Weeks built: {stats.weeks_built}")
        print(f"Artifacts written to: {stats.output_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
