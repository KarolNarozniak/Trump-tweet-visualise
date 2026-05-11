from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .global_animation import DEFAULT_HEAT_DECAY, DEFAULT_LAYOUT_SEED
from .io import read_tweets_csv
from .truth_io import read_truth_posts_csv
from .truth_semantics import TruthSemanticConfig, make_truth_semantic_enricher, serialize_enriched_truth_posts
from .unified_graph import DEFAULT_UNIFIED_MIN_NODE_COUNT, build_unified_semantic_artifacts
from .unified_io import write_unified_artifacts
from .unified_preprocess import prepare_unified_posts


@dataclass(frozen=True)
class UnifiedBuildStats:
    total_twitter_posts: int
    total_truth_posts: int
    processed_posts: int
    weeks_built: int
    semantic_nodes: int
    semantic_edges: int
    embedding_dim: int
    output_dir: Path


def build_unified_artifacts(
    *,
    twitter_input_csv: Path,
    truth_input_path: Path,
    output_dir: Path,
    semantic_config: TruthSemanticConfig,
    include_retweets: bool = True,
    min_node_count: int = DEFAULT_UNIFIED_MIN_NODE_COUNT,
    included_node_types: tuple[str, ...] = ("topic", "per", "org", "loc", "hashtag", "mention"),
    heat_decay: float = DEFAULT_HEAT_DECAY,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
) -> UnifiedBuildStats:
    raw_twitter_df = read_tweets_csv(twitter_input_csv)
    raw_truth_df = read_truth_posts_csv(truth_input_path)
    combined_posts_df = prepare_unified_posts(
        raw_twitter_df,
        raw_truth_df,
        include_retweets=include_retweets,
    )

    enricher = make_truth_semantic_enricher(semantic_config)
    enriched_df, embeddings = enricher.enrich(combined_posts_df)
    if embeddings.size == 0:
        embeddings = np.empty((0, semantic_config.embedding_dim), dtype="float32")
    else:
        embeddings = embeddings.astype("float32")

    (
        week_index_df,
        weekly_summary_df,
        animation_payload,
        node_catalog_df,
        edge_catalog_df,
        temporal_deltas,
    ) = build_unified_semantic_artifacts(
        enriched_df,
        min_node_count=min_node_count,
        included_node_types=included_node_types,
        heat_decay=heat_decay,
        layout_seed=layout_seed,
    )

    embedding_index_df = pd.DataFrame(
        {
            "row_index": list(range(len(enriched_df))),
            "post_id": enriched_df["post_id"].astype(str),
            "platform_post_id": enriched_df["platform_post_id"].astype(str),
            "source_platform": enriched_df["source_platform"].astype(str),
            "created_at_utc": enriched_df["created_at_utc"].astype(str),
            "week_id": enriched_df["week_id"].astype(str),
        }
    )

    serialized_enriched_df = serialize_enriched_truth_posts(enriched_df)
    write_unified_artifacts(
        output_dir,
        week_index_df=week_index_df,
        weekly_summary_df=weekly_summary_df,
        enriched_posts_df=serialized_enriched_df,
        animation_payload=animation_payload,
        node_catalog_df=node_catalog_df,
        edge_catalog_df=edge_catalog_df,
        embeddings=embeddings,
        embedding_index_df=embedding_index_df,
        node_delta_df=temporal_deltas["node_deltas"],
        edge_delta_df=temporal_deltas["edge_deltas"],
    )

    twitter_count = int((combined_posts_df["source_platform"] == "twitter").sum())
    truth_count = int((combined_posts_df["source_platform"] == "truth").sum())
    return UnifiedBuildStats(
        total_twitter_posts=twitter_count,
        total_truth_posts=truth_count,
        processed_posts=int(len(enriched_df)),
        weeks_built=int(len(week_index_df)),
        semantic_nodes=int(len(animation_payload.get("global_nodes", []))),
        semantic_edges=int(len(animation_payload.get("global_edges", []))),
        embedding_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        output_dir=output_dir,
    )
