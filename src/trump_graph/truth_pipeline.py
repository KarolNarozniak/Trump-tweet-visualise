from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .global_animation import DEFAULT_HEAT_DECAY, DEFAULT_LAYOUT_SEED
from .truth_graph import DEFAULT_TRUTH_MIN_NODE_COUNT, TRUTH_NODE_TYPES, build_truth_semantic_artifacts
from .truth_io import read_truth_posts_csv, write_truth_artifacts
from .truth_preprocess import prepare_truth_posts
from .truth_semantics import TruthSemanticConfig, make_truth_semantic_enricher, serialize_enriched_truth_posts


@dataclass(frozen=True)
class TruthBuildStats:
    total_posts: int
    processed_posts: int
    weeks_built: int
    semantic_nodes: int
    semantic_edges: int
    embedding_dim: int
    output_dir: Path


def build_truth_artifacts(
    *,
    input_path: Path,
    output_dir: Path,
    semantic_config: TruthSemanticConfig,
    min_node_count: int = DEFAULT_TRUTH_MIN_NODE_COUNT,
    included_node_types: tuple[str, ...] = TRUTH_NODE_TYPES,
    heat_decay: float = DEFAULT_HEAT_DECAY,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
) -> TruthBuildStats:
    raw_df = read_truth_posts_csv(input_path)
    posts_df = prepare_truth_posts(raw_df)

    enricher = make_truth_semantic_enricher(semantic_config)
    enriched_df, embeddings = enricher.enrich(posts_df)
    if embeddings.size == 0:
        embeddings = np.empty((0, semantic_config.embedding_dim), dtype="float32")
    else:
        embeddings = embeddings.astype("float32")

    week_index_df, weekly_summary_df, animation_payload, node_catalog_df = build_truth_semantic_artifacts(
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
            "created_at_utc": enriched_df["created_at_utc"].astype(str),
            "week_id": enriched_df["week_id"].astype(str),
        }
    )
    serialized_enriched_df = serialize_enriched_truth_posts(enriched_df)
    write_truth_artifacts(
        output_dir,
        week_index_df=week_index_df,
        weekly_summary_df=weekly_summary_df,
        enriched_posts_df=serialized_enriched_df,
        animation_payload=animation_payload,
        node_catalog_df=node_catalog_df,
        embeddings=embeddings,
        embedding_index_df=embedding_index_df,
    )

    return TruthBuildStats(
        total_posts=int(len(raw_df)),
        processed_posts=int(len(enriched_df)),
        weeks_built=int(len(week_index_df)),
        semantic_nodes=int(len(animation_payload.get("global_nodes", []))),
        semantic_edges=int(len(animation_payload.get("global_edges", []))),
        embedding_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        output_dir=output_dir,
    )
