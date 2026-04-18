from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_weekly_artifacts
from .settings import load_settings


def _build_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        prog="trump_graph",
        description="Build weekly mention network artifacts from Trump tweet CSV data.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build weekly mention-network artifacts.")
    build_parser.add_argument(
        "--input",
        type=Path,
        default=settings.build.default_input_csv,
        help="Path to input CSV file.",
    )
    build_parser.add_argument(
        "--out",
        type=Path,
        default=settings.build.default_output_dir,
        help="Output directory for artifacts.",
    )
    build_parser.add_argument(
        "--min-mention-count",
        type=int,
        default=settings.build.min_mention_count,
        help="Minimum per-week mention frequency required for a node.",
    )
    build_parser.add_argument(
        "--global-min-mentions",
        type=int,
        default=settings.build.global_min_mentions,
        help="Minimum global mention frequency required for nodes in the stable animation graph.",
    )
    build_parser.add_argument(
        "--heat-decay",
        type=float,
        default=settings.build.heat_decay,
        help="Weekly heat decay factor used by global animation payload.",
    )
    build_parser.add_argument(
        "--layout-seed",
        type=int,
        default=settings.build.layout_seed,
        help="Deterministic layout seed for global animation payload coordinates.",
    )
    build_parser.add_argument(
        "--include-retweets",
        dest="include_retweets",
        action="store_true",
        default=settings.build.include_retweets,
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
            global_min_mentions=args.global_min_mentions,
            heat_decay=args.heat_decay,
            layout_seed=args.layout_seed,
        )
        print(f"Total tweets read: {stats.total_tweets}")
        print(f"Tweets processed: {stats.processed_tweets}")
        print(f"Weeks built: {stats.weeks_built}")
        print(f"Global animation nodes: {stats.global_nodes}")
        print(f"Global animation edges: {stats.global_edges}")
        print(f"Artifacts written to: {stats.output_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
