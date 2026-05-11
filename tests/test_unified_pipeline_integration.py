from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trump_graph.global_animation import replay_week_state
from trump_graph.truth_semantics import TruthSemanticConfig
from trump_graph.unified_pipeline import build_unified_artifacts


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


def test_build_unified_artifacts_writes_expected_outputs(
    sample_tweets_csv_path: Path,
    sample_truth_posts_path: Path,
    local_temp_dir: Path,
) -> None:
    output_dir = local_temp_dir / "processed_unified"

    stats = build_unified_artifacts(
        twitter_input_csv=sample_tweets_csv_path,
        truth_input_path=sample_truth_posts_path,
        output_dir=output_dir,
        semantic_config=_deterministic_config(),
        include_retweets=True,
        min_node_count=1,
        included_node_types=("topic", "per", "org", "loc", "hashtag", "mention"),
    )

    assert stats.total_twitter_posts == 6
    assert stats.total_truth_posts == 4
    assert stats.processed_posts == 10
    assert stats.weeks_built >= 2
    assert stats.semantic_nodes > 0
    assert stats.embedding_dim == 384

    assert (output_dir / "unified_week_index.csv").exists()
    assert (output_dir / "unified_weekly_summary.csv").exists()
    assert (output_dir / "unified_posts_enriched.parquet").exists()
    assert (output_dir / "unified_semantic_graph" / "animation_state.json").exists()
    assert (output_dir / "unified_semantic_graph" / "node_catalog.csv").exists()
    assert (output_dir / "unified_semantic_graph" / "edge_catalog.csv").exists()
    assert (output_dir / "unified_embeddings" / "embeddings.npy").exists()
    assert (output_dir / "unified_embeddings" / "index.csv").exists()
    assert (output_dir / "unified_training" / "temporal_node_deltas.parquet").exists()
    assert (output_dir / "unified_training" / "temporal_edge_deltas.parquet").exists()

    enriched_posts = pd.read_parquet(output_dir / "unified_posts_enriched.parquet")
    required_columns = {
        "post_id",
        "platform_post_id",
        "source_platform",
        "created_at_utc",
        "topic_labels",
        "entities",
        "sentiment_label",
        "semantic_nodes",
    }
    assert required_columns.issubset(enriched_posts.columns)

    embeddings = np.load(output_dir / "unified_embeddings" / "embeddings.npy")
    assert embeddings.shape == (10, 384)

    payload = json.loads((output_dir / "unified_semantic_graph" / "animation_state.json").read_text(encoding="utf-8"))
    assert {"all", "twitter", "truth", "original", "negative", "neutral", "positive"}.issubset(
        payload["delta_sets"].keys()
    )

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
