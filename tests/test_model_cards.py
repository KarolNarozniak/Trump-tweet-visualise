from __future__ import annotations

from pathlib import Path

from trump_graph.model_cards import write_forecast_model_cards
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


def test_write_forecast_model_cards_creates_cards_for_all_models(
    sample_tweets_csv_path: Path,
    sample_truth_posts_path: Path,
    local_temp_dir: Path,
) -> None:
    semantic_output = local_temp_dir / "processed_unified"
    build_unified_artifacts(
        twitter_input_csv=sample_tweets_csv_path,
        truth_input_path=sample_truth_posts_path,
        output_dir=semantic_output,
        semantic_config=_deterministic_config(),
        include_retweets=True,
        min_node_count=1,
        included_node_types=("topic", "per", "org", "loc", "hashtag", "mention"),
    )

    forecast_root = local_temp_dir / "processed_forecast"
    written_paths = write_forecast_model_cards(forecast_root, semantic_output)
    assert len(written_paths) == 4

    tgn_card = forecast_root / "tgn" / "MODEL_CARD.md"
    evolvegcn_card = forecast_root / "evolvegcn" / "MODEL_CARD.md"
    gconvgru_card = forecast_root / "gconvgru" / "MODEL_CARD.md"
    index_card = forecast_root / "MODEL_CARDS_INDEX.md"
    assert tgn_card.exists()
    assert evolvegcn_card.exists()
    assert gconvgru_card.exists()
    assert index_card.exists()

    evolvegcn_text = evolvegcn_card.read_text(encoding="utf-8")
    assert "status: **missing**" in evolvegcn_text
    assert "total_weeks_in_payload: 5" in evolvegcn_text
