from __future__ import annotations

from trump_graph.truth_semantics import filter_entities, select_topic_labels


def test_select_topic_labels_threshold_top_k_and_fallback() -> None:
    selected = select_topic_labels(
        {
            "media_communications": 0.9,
            "elections_campaigns": 0.7,
            "law_justice_crime": 0.2,
        },
        threshold=0.35,
        max_labels=1,
    )
    assert selected == [{"label": "media_communications", "score": 0.9}]

    fallback = select_topic_labels(
        {"media_communications": 0.2, "elections_campaigns": 0.1},
        threshold=0.35,
        max_labels=3,
    )
    assert fallback == [{"label": "media_communications", "score": 0.2}]


def test_filter_entities_normalizes_dedupes_and_applies_threshold() -> None:
    entities = filter_entities(
        [
            {"word": "Joe Biden", "entity_group": "PER", "score": 0.72},
            {"word": "Joe Biden", "entity_group": "PER", "score": 0.82},
            {"word": "misc text", "entity_group": "MISC", "score": 0.99},
            {"word": "CNN", "entity_group": "ORG", "score": 0.2},
        ],
        score_threshold=0.65,
    )

    assert entities == [{"text": "Joe Biden", "type": "PER", "score": 0.82}]
