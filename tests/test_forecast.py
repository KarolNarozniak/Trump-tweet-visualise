from __future__ import annotations

import json
from pathlib import Path

from trump_graph.forecast import build_baseline_forecast_payload
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


def test_build_baseline_forecast_payload_extends_weeks(
    sample_truth_posts_path: Path,
    local_temp_dir: Path,
) -> None:
    output_dir = local_temp_dir / "processed_truth"
    build_truth_artifacts(
        input_path=sample_truth_posts_path,
        output_dir=output_dir,
        semantic_config=_deterministic_config(),
        min_node_count=1,
    )

    payload_path = output_dir / "truth_semantic_graph" / "animation_state.json"
    semantic_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    base_week_count = len(semantic_payload.get("weeks", []))

    forecast_payload = build_baseline_forecast_payload(
        semantic_payload,
        horizon_weeks=4,
        lookback_weeks=2,
    )

    assert len(forecast_payload["weeks"]) == base_week_count + 4
    assert len(forecast_payload["node_week_deltas"]) == base_week_count + 4
    assert len(forecast_payload["edge_week_deltas"]) == base_week_count + 4
    assert forecast_payload["graph_kind"] == "temporal_forecast"
    assert forecast_payload["forecast_backend"] == "baseline"

    future_weeks = forecast_payload["weeks"][-4:]
    assert all(bool(week.get("is_forecast")) for week in future_weeks)
