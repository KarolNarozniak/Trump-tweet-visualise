from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trump_graph.global_animation import replay_week_state
from trump_graph.truth_pipeline import build_truth_artifacts
from trump_graph.truth_semantics import TruthSemanticConfig


def _deterministic_config() -> TruthSemanticConfig:
    return TruthSemanticConfig(
        backend="deterministic",
        topic_model_id="deterministic-topic",
        ner_model_id="deterministic-ner",
        sentiment_model_id="deterministic-sentiment",
        embedding_model_id="deterministic-embedding",
        device="cpu",
        batch_size=2,
        topic_threshold=0.35,
        max_topic_labels=3,
        entity_score_threshold=0.65,
        max_chunk_chars=500,
    )


def test_build_truth_artifacts_writes_expected_outputs(
    sample_truth_posts_path: Path,
    local_temp_dir: Path,
) -> None:
    output_dir = local_temp_dir / "processed_truth"

    stats = build_truth_artifacts(
        input_path=sample_truth_posts_path,
        output_dir=output_dir,
        semantic_config=_deterministic_config(),
        min_node_count=1,
        included_node_types=("topic", "per", "org", "loc", "hashtag", "mention"),
    )

    assert stats.total_posts == 4
    assert stats.processed_posts == 4
    assert stats.weeks_built >= 2
    assert stats.semantic_nodes > 0
    assert stats.embedding_dim == 384

    assert (output_dir / "truth_week_index.csv").exists()
    assert (output_dir / "truth_weekly_summary.csv").exists()
    assert (output_dir / "truth_posts_enriched.parquet").exists()
    assert (output_dir / "truth_semantic_graph" / "animation_state.json").exists()
    assert (output_dir / "truth_semantic_graph" / "node_catalog.csv").exists()
    assert (output_dir / "truth_embeddings" / "embeddings.npy").exists()
    assert (output_dir / "truth_embeddings" / "index.csv").exists()

    enriched_posts = pd.read_parquet(output_dir / "truth_posts_enriched.parquet")
    required_columns = {
        "post_id",
        "created_at_utc",
        "topic_labels",
        "entities",
        "sentiment_label",
        "semantic_nodes",
        "is_retruth",
    }
    assert required_columns.issubset(enriched_posts.columns)

    embeddings = np.load(output_dir / "truth_embeddings" / "embeddings.npy")
    assert embeddings.shape == (4, 384)

    payload = json.loads((output_dir / "truth_semantic_graph" / "animation_state.json").read_text(encoding="utf-8"))
    assert {"all", "original", "negative", "neutral", "positive"}.issubset(payload["delta_sets"].keys())

    node_ids = [str(node["id"]) for node in payload["global_nodes"]]
    edge_ids = [str(edge["id"]) for edge in payload["global_edges"]]
    _final_heat, final_edges = replay_week_state(
        node_ids=node_ids,
        edge_ids=edge_ids,
        node_week_deltas=payload["delta_sets"]["all"]["node_week_deltas"],
        edge_week_deltas=payload["delta_sets"]["all"]["edge_week_deltas"],
        heat_decay=float(payload["heat_decay"]),
        target_week_index=len(payload["weeks"]) - 1,
    )
    for edge in payload["global_edges"]:
        assert int(final_edges[edge["id"]]) == int(edge["total_co_occurrences"])
