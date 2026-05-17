from __future__ import annotations

from pathlib import Path

from trump_graph.app import (
    build_forecast_comparison_animation_html,
    build_truth_semantic_animation_html,
    load_unified_animation_artifacts,
    load_unified_enriched_posts,
    load_unified_node_catalog,
    load_unified_week_index,
)
from trump_graph.forecast import build_baseline_forecast_payload
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


def test_unified_helpers_load_and_render(
    sample_tweets_csv_path: Path,
    sample_truth_posts_path: Path,
    local_temp_dir: Path,
) -> None:
    output_dir = local_temp_dir / "processed_unified"
    build_unified_artifacts(
        twitter_input_csv=sample_tweets_csv_path,
        truth_input_path=sample_truth_posts_path,
        output_dir=output_dir,
        semantic_config=_deterministic_config(),
        include_retweets=True,
        min_node_count=1,
        included_node_types=("topic", "per", "org", "loc", "hashtag", "mention"),
    )

    week_index = load_unified_week_index(output_dir)
    assert not week_index.empty

    enriched_posts = load_unified_enriched_posts(output_dir)
    assert "semantic_nodes" in enriched_posts.columns

    node_catalog = load_unified_node_catalog(output_dir)
    assert {"node_id", "label", "node_type", "total_count"}.issubset(node_catalog.columns)

    payload = load_unified_animation_artifacts(output_dir)
    html = build_truth_semantic_animation_html(
        payload=payload,
        included_node_types={"topic", "per", "org", "loc", "hashtag", "mention"},
        min_total_count=1,
        delta_set_name="all",
        initial_speed=2.0,
    )
    assert "truth-week-slider" in html
    assert "vis-network" in html
    assert "global_nodes" in html

    forecast_payload = build_baseline_forecast_payload(payload, horizon_weeks=3, lookback_weeks=2)
    forecast_html = build_truth_semantic_animation_html(
        payload=forecast_payload,
        included_node_types={"topic", "per", "org", "loc", "hashtag", "mention"},
        min_total_count=1,
        delta_set_name="all",
        initial_speed=2.0,
    )
    assert "truth-week-slider" in forecast_html

    comparison_html = build_forecast_comparison_animation_html(
        [
            {"key": "original", "title": "Original", "payload": payload, "status": "ready", "note": "Base", "metrics": {}},
            {
                "key": "baseline",
                "title": "Baseline",
                "payload": forecast_payload,
                "status": "ready",
                "note": "Forecast preview",
                "metrics": {"best_val_score": 0.1, "mae": 0.2, "rmse": 0.3},
            },
        ],
        included_node_types={"topic", "per", "org", "loc", "hashtag", "mention"},
        min_total_count=1,
        delta_set_name="all",
        initial_week_index=0,
        initial_speed=2.0,
    )
    assert "cmp-week-slider" in comparison_html
    assert "cmp-play" in comparison_html
    assert "Top active nodes by type" in comparison_html
