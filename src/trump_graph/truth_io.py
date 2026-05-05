from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .io import ensure_directory

TRUTH_DTYPES: dict[str, str] = {
    "_id": "string",
    "owner": "string",
    "text": "string",
}


def read_truth_posts_csv(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path, dtype=TRUTH_DTYPES)


def write_truth_artifacts(
    output_dir: Path,
    *,
    week_index_df: pd.DataFrame,
    weekly_summary_df: pd.DataFrame,
    enriched_posts_df: pd.DataFrame,
    animation_payload: Mapping[str, Any],
    node_catalog_df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index_df: pd.DataFrame,
) -> None:
    ensure_directory(output_dir)
    week_index_df.to_csv(output_dir / "truth_week_index.csv", index=False)
    weekly_summary_df.to_csv(output_dir / "truth_weekly_summary.csv", index=False)
    enriched_posts_df.to_parquet(output_dir / "truth_posts_enriched.parquet", index=False)

    graph_dir = ensure_directory(output_dir / "truth_semantic_graph")
    (graph_dir / "animation_state.json").write_text(
        json.dumps(animation_payload, sort_keys=False),
        encoding="utf-8",
    )
    node_catalog_df.to_csv(graph_dir / "node_catalog.csv", index=False)

    embeddings_dir = ensure_directory(output_dir / "truth_embeddings")
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    embedding_index_df.to_csv(embeddings_dir / "index.csv", index=False)
