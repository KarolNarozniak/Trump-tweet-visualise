from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trump_graph.global_animation import replay_week_state
from trump_graph.pipeline import build_weekly_artifacts


def test_build_weekly_artifacts_writes_expected_outputs(sample_tweets_csv_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"

    stats = build_weekly_artifacts(
        input_csv=sample_tweets_csv_path,
        output_dir=output_dir,
        min_mention_count=1,
        include_retweets=True,
        global_min_mentions=1,
    )

    assert stats.total_tweets == 6
    assert stats.processed_tweets == 6
    assert stats.weeks_built == 2
    assert stats.global_nodes == 5
    assert stats.global_edges == 4

    week_index_path = output_dir / "week_index.csv"
    summary_path = output_dir / "weekly_summary.csv"
    global_animation_path = output_dir / "global_animation" / "animation_state.json"
    assert week_index_path.exists()
    assert summary_path.exists()
    assert global_animation_path.exists()

    week_index = pd.read_csv(week_index_path)
    assert {"week_id", "week_start", "tweets_processed", "unique_mentions", "edge_count"}.issubset(week_index.columns)

    week_ids = set(week_index["week_id"].tolist())
    assert {"2020-W53", "2021-W01"} == week_ids

    for week_id in week_ids:
        week_dir = output_dir / "weeks" / week_id
        assert (week_dir / "nodes.csv").exists()
        assert (week_dir / "edges.csv").exists()
        assert (week_dir / "metrics.json").exists()

    payload = json.loads(global_animation_path.read_text(encoding="utf-8"))
    required_keys = {
        "weeks",
        "global_nodes",
        "global_edges",
        "node_week_deltas",
        "edge_week_deltas",
        "heat_decay",
        "heat_scale",
        "max_cumulative_edge",
        "hub_node_id",
        "top_label_nodes",
    }
    assert required_keys.issubset(payload.keys())

    node_ids = [str(node["id"]) for node in payload["global_nodes"]]
    edge_ids = [str(edge["id"]) for edge in payload["global_edges"]]
    _, final_edge_cumulative = replay_week_state(
        node_ids=node_ids,
        edge_ids=edge_ids,
        node_week_deltas=payload["node_week_deltas"],
        edge_week_deltas=payload["edge_week_deltas"],
        heat_decay=float(payload["heat_decay"]),
        target_week_index=len(payload["weeks"]) - 1,
    )

    for edge in payload["global_edges"]:
        assert int(final_edge_cumulative[edge["id"]]) == int(edge["total_co_mentions"])
