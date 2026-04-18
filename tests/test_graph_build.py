from __future__ import annotations

import pytest

from trump_graph.graph_build import build_week_graph


def test_build_week_graph_weights_nodes_and_edges() -> None:
    mentions_by_tweet = [
        ["alice", "bob", "alice"],
        ["bob", "carol"],
        [],
        ["alice", "carol"],
    ]
    graph = build_week_graph(mentions_by_tweet, min_mention_count=1)

    assert graph.nodes["alice"]["weight"] == 2
    assert graph.nodes["bob"]["weight"] == 2
    assert graph.nodes["carol"]["weight"] == 2
    assert graph["alice"]["bob"]["weight"] == 1
    assert graph["bob"]["carol"]["weight"] == 1
    assert graph["alice"]["carol"]["weight"] == 1


def test_build_week_graph_applies_node_threshold() -> None:
    graph = build_week_graph([["alice", "bob"], ["alice"]], min_mention_count=2)
    assert set(graph.nodes) == {"alice"}
    assert graph.number_of_edges() == 0


def test_build_week_graph_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        build_week_graph([["alice"]], min_mention_count=0)
