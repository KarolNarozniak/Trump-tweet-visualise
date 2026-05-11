from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import build_weekly_artifacts
from .truth_pipeline import build_truth_artifacts
from .truth_semantics import TruthSemanticConfig
from .unified_pipeline import build_unified_artifacts
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

    truth_parser = subparsers.add_parser("build-truth", help="Build Truth Social semantic temporal graph artifacts.")
    truth_parser.add_argument(
        "--input",
        type=Path,
        default=settings.truth_build.input_path,
        help="Path to Truth Social CSV-like file with _id, owner, text columns.",
    )
    truth_parser.add_argument(
        "--out",
        type=Path,
        default=settings.truth_build.output_dir,
        help="Output directory for Truth semantic artifacts.",
    )
    truth_parser.add_argument(
        "--semantic-backend",
        choices=("hf", "deterministic"),
        default=settings.truth_build.semantic_backend,
        help="Semantic inference backend. Use deterministic for smoke tests without HF models.",
    )
    truth_parser.add_argument("--device", default=settings.truth_build.device, help="Inference device: auto, cuda, or cpu.")
    truth_parser.add_argument("--topic-model-id", default=settings.truth_build.topic_model_id)
    truth_parser.add_argument("--ner-model-id", default=settings.truth_build.ner_model_id)
    truth_parser.add_argument("--sentiment-model-id", default=settings.truth_build.sentiment_model_id)
    truth_parser.add_argument("--embedding-model-id", default=settings.truth_build.embedding_model_id)
    truth_parser.add_argument("--batch-size", type=int, default=settings.truth_build.batch_size)
    truth_parser.add_argument("--topic-threshold", type=float, default=settings.truth_build.topic_threshold)
    truth_parser.add_argument("--max-topic-labels", type=int, default=settings.truth_build.max_topic_labels)
    truth_parser.add_argument("--entity-score-threshold", type=float, default=settings.truth_build.entity_score_threshold)
    truth_parser.add_argument("--max-chunk-chars", type=int, default=settings.truth_build.max_chunk_chars)
    truth_parser.add_argument("--min-node-count", type=int, default=settings.truth_build.min_node_count)
    truth_parser.add_argument(
        "--node-types",
        default=",".join(settings.truth_build.included_node_types),
        help="Comma-separated semantic node types to include in artifacts.",
    )
    truth_parser.add_argument("--heat-decay", type=float, default=settings.truth_build.heat_decay)
    truth_parser.add_argument("--layout-seed", type=int, default=settings.truth_build.layout_seed)

    unified_parser = subparsers.add_parser(
        "build-unified",
        help="Build one combined semantic temporal graph for Twitter + Truth Social.",
    )
    unified_parser.add_argument(
        "--twitter-input",
        type=Path,
        default=settings.unified_build.twitter_input_csv,
        help="Path to Twitter archive CSV file.",
    )
    unified_parser.add_argument(
        "--truth-input",
        type=Path,
        default=settings.unified_build.truth_input_path,
        help="Path to Truth Social CSV-like file with _id, owner, text columns.",
    )
    unified_parser.add_argument(
        "--out",
        type=Path,
        default=settings.unified_build.output_dir,
        help="Output directory for unified semantic artifacts.",
    )
    unified_parser.add_argument(
        "--include-retweets",
        dest="include_retweets",
        action="store_true",
        default=settings.unified_build.include_retweets,
        help="Include Twitter retweets in unified processing (default).",
    )
    unified_parser.add_argument(
        "--exclude-retweets",
        dest="include_retweets",
        action="store_false",
        help="Exclude Twitter retweets from unified processing.",
    )
    unified_parser.add_argument(
        "--semantic-backend",
        choices=("hf", "deterministic"),
        default=settings.unified_build.semantic_backend,
        help="Semantic inference backend. Use deterministic for smoke tests without HF models.",
    )
    unified_parser.add_argument("--device", default=settings.unified_build.device, help="Inference device: auto, cuda, or cpu.")
    unified_parser.add_argument("--topic-model-id", default=settings.unified_build.topic_model_id)
    unified_parser.add_argument("--ner-model-id", default=settings.unified_build.ner_model_id)
    unified_parser.add_argument("--sentiment-model-id", default=settings.unified_build.sentiment_model_id)
    unified_parser.add_argument("--embedding-model-id", default=settings.unified_build.embedding_model_id)
    unified_parser.add_argument("--batch-size", type=int, default=settings.unified_build.batch_size)
    unified_parser.add_argument("--topic-threshold", type=float, default=settings.unified_build.topic_threshold)
    unified_parser.add_argument("--max-topic-labels", type=int, default=settings.unified_build.max_topic_labels)
    unified_parser.add_argument("--entity-score-threshold", type=float, default=settings.unified_build.entity_score_threshold)
    unified_parser.add_argument("--max-chunk-chars", type=int, default=settings.unified_build.max_chunk_chars)
    unified_parser.add_argument("--min-node-count", type=int, default=settings.unified_build.min_node_count)
    unified_parser.add_argument(
        "--node-types",
        default=",".join(settings.unified_build.included_node_types),
        help="Comma-separated semantic node types to include in artifacts.",
    )
    unified_parser.add_argument("--heat-decay", type=float, default=settings.unified_build.heat_decay)
    unified_parser.add_argument("--layout-seed", type=int, default=settings.unified_build.layout_seed)
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

    if args.command == "build-truth":
        semantic_config = TruthSemanticConfig(
            backend=args.semantic_backend,
            topic_model_id=args.topic_model_id,
            ner_model_id=args.ner_model_id,
            sentiment_model_id=args.sentiment_model_id,
            embedding_model_id=args.embedding_model_id,
            device=args.device,
            batch_size=args.batch_size,
            topic_threshold=args.topic_threshold,
            max_topic_labels=args.max_topic_labels,
            entity_score_threshold=args.entity_score_threshold,
            max_chunk_chars=args.max_chunk_chars,
        )
        stats = build_truth_artifacts(
            input_path=args.input,
            output_dir=args.out,
            semantic_config=semantic_config,
            min_node_count=args.min_node_count,
            included_node_types=tuple(node_type.strip() for node_type in args.node_types.split(",") if node_type.strip()),
            heat_decay=args.heat_decay,
            layout_seed=args.layout_seed,
        )
        print(f"Total Truth posts read: {stats.total_posts}")
        print(f"Truth posts processed: {stats.processed_posts}")
        print(f"Truth weeks built: {stats.weeks_built}")
        print(f"Truth semantic nodes: {stats.semantic_nodes}")
        print(f"Truth semantic edges: {stats.semantic_edges}")
        print(f"Embedding dim: {stats.embedding_dim}")
        print(f"Artifacts written to: {stats.output_dir}")
        return 0

    if args.command == "build-unified":
        semantic_config = TruthSemanticConfig(
            backend=args.semantic_backend,
            topic_model_id=args.topic_model_id,
            ner_model_id=args.ner_model_id,
            sentiment_model_id=args.sentiment_model_id,
            embedding_model_id=args.embedding_model_id,
            device=args.device,
            batch_size=args.batch_size,
            topic_threshold=args.topic_threshold,
            max_topic_labels=args.max_topic_labels,
            entity_score_threshold=args.entity_score_threshold,
            max_chunk_chars=args.max_chunk_chars,
        )
        stats = build_unified_artifacts(
            twitter_input_csv=args.twitter_input,
            truth_input_path=args.truth_input,
            output_dir=args.out,
            semantic_config=semantic_config,
            include_retweets=args.include_retweets,
            min_node_count=args.min_node_count,
            included_node_types=tuple(node_type.strip() for node_type in args.node_types.split(",") if node_type.strip()),
            heat_decay=args.heat_decay,
            layout_seed=args.layout_seed,
        )
        print(f"Unified Twitter posts processed: {stats.total_twitter_posts}")
        print(f"Unified Truth posts processed: {stats.total_truth_posts}")
        print(f"Unified total posts processed: {stats.processed_posts}")
        print(f"Unified weeks built: {stats.weeks_built}")
        print(f"Unified semantic nodes: {stats.semantic_nodes}")
        print(f"Unified semantic edges: {stats.semantic_edges}")
        print(f"Embedding dim: {stats.embedding_dim}")
        print(f"Artifacts written to: {stats.output_dir}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
