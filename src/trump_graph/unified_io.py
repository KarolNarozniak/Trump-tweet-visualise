from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .io import ensure_directory


def write_unified_artifacts(
    output_dir: Path,
    *,
    week_index_df: pd.DataFrame,
    weekly_summary_df: pd.DataFrame,
    enriched_posts_df: pd.DataFrame,
    animation_payload: Mapping[str, Any],
    node_catalog_df: pd.DataFrame,
    edge_catalog_df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index_df: pd.DataFrame,
    node_delta_df: pd.DataFrame,
    edge_delta_df: pd.DataFrame,
) -> None:
    ensure_directory(output_dir)
    week_index_df.to_csv(output_dir / "unified_week_index.csv", index=False)
    weekly_summary_df.to_csv(output_dir / "unified_weekly_summary.csv", index=False)
    enriched_posts_df.to_parquet(output_dir / "unified_posts_enriched.parquet", index=False)

    graph_dir = ensure_directory(output_dir / "unified_semantic_graph")
    (graph_dir / "animation_state.json").write_text(
        json.dumps(animation_payload, sort_keys=False),
        encoding="utf-8",
    )
    node_catalog_df.to_csv(graph_dir / "node_catalog.csv", index=False)
    edge_catalog_df.to_csv(graph_dir / "edge_catalog.csv", index=False)

    embeddings_dir = ensure_directory(output_dir / "unified_embeddings")
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    embedding_index_df.to_csv(embeddings_dir / "index.csv", index=False)

    training_dir = ensure_directory(output_dir / "unified_training")
    node_delta_df.to_parquet(training_dir / "temporal_node_deltas.parquet", index=False)
    edge_delta_df.to_parquet(training_dir / "temporal_edge_deltas.parquet", index=False)
