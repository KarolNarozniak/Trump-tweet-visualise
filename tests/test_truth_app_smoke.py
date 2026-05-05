from __future__ import annotations

from pathlib import Path

from trump_graph.app import (
    build_truth_semantic_animation_html,
    load_truth_animation_artifacts,
    load_truth_enriched_posts,
    load_truth_node_catalog,
    load_truth_week_index,
)
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


def test_truth_app_helpers_load_and_render(sample_truth_posts_path: Path, local_temp_dir: Path) -> None:
    output_dir = local_temp_dir / "processed_truth"
    build_truth_artifacts(
        input_path=sample_truth_posts_path,
        output_dir=output_dir,
        semantic_config=_deterministic_config(),
        min_node_count=1,
    )

    week_index = load_truth_week_index(output_dir)
    assert not week_index.empty

    enriched_posts = load_truth_enriched_posts(output_dir)
    assert "semantic_nodes" in enriched_posts.columns

    node_catalog = load_truth_node_catalog(output_dir)
    assert {"node_id", "label", "node_type", "total_count"}.issubset(node_catalog.columns)

    payload = load_truth_animation_artifacts(output_dir)
    html = build_truth_semantic_animation_html(
        payload,
        included_node_types={"topic", "per", "org", "loc", "hashtag", "mention"},
        min_total_count=1,
        delta_set_name="all",
        initial_speed=2.0,
    )

    assert "truth-week-slider" in html
    assert "vis-network" in html
    assert "global_nodes" in html
