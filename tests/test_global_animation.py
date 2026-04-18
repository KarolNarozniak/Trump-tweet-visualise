from __future__ import annotations

from pathlib import Path

import pandas as pd

from trump_graph.global_animation import (
    edge_width_from_cumulative,
    build_global_animation_payload,
    node_size_from_total_mentions,
    replay_week_state,
)
from trump_graph.graph_build import build_week_graph
from trump_graph.io import read_tweets_csv
from trump_graph.preprocess import prepare_tweets


def _build_week_index_df(tweets_df):
    rows = []
    grouped = tweets_df.groupby("week_id", sort=False)
    for week_id, week_df in grouped:
        graph = build_week_graph(week_df["mentions"], min_mention_count=1)
        unique_mentions = len({mention for mentions in week_df["mentions"] for mention in mentions})
        rows.append(
            {
                "week_id": week_id,
                "week_start": week_df["week_start"].iloc[0].strftime("%Y-%m-%d"),
                "week_end": week_df["week_end"].iloc[0].strftime("%Y-%m-%d"),
                "tweets_processed": int(len(week_df)),
                "tweets_with_mentions": int((week_df["mentions"].map(len) > 0).sum()),
                "unique_mentions": int(unique_mentions),
                "edge_count": int(graph.number_of_edges()),
            }
        )
    return pd.DataFrame(rows).sort_values(["week_start", "week_id"], kind="mergesort").reset_index(drop=True)


def _prepared_fixture(sample_tweets_csv_path: Path):
    raw_df = read_tweets_csv(sample_tweets_csv_path)
    tweets_df = prepare_tweets(raw_df, include_retweets=True)
    week_index_df = _build_week_index_df(tweets_df)
    return tweets_df, week_index_df


def test_global_payload_filters_by_min_mentions(sample_tweets_csv_path: Path) -> None:
    tweets_df, week_index_df = _prepared_fixture(sample_tweets_csv_path)
    payload = build_global_animation_payload(
        tweets_df=tweets_df,
        week_index_df=week_index_df,
        global_min_mentions=2,
        heat_decay=0.85,
        layout_seed=11,
    )

    node_ids = {node["id"] for node in payload["global_nodes"]}
    assert node_ids == {"alice", "bob", "carol"}

    edge_ids = {edge["id"] for edge in payload["global_edges"]}
    assert edge_ids == {"alice|bob", "alice|carol", "bob|carol"}


def test_global_payload_layout_is_deterministic(sample_tweets_csv_path: Path) -> None:
    tweets_df, week_index_df = _prepared_fixture(sample_tweets_csv_path)
    payload_a = build_global_animation_payload(
        tweets_df=tweets_df,
        week_index_df=week_index_df,
        global_min_mentions=1,
        heat_decay=0.85,
        layout_seed=42,
    )
    payload_b = build_global_animation_payload(
        tweets_df=tweets_df,
        week_index_df=week_index_df,
        global_min_mentions=1,
        heat_decay=0.85,
        layout_seed=42,
    )

    pos_a = {(node["id"], node["x"], node["y"]) for node in payload_a["global_nodes"]}
    pos_b = {(node["id"], node["x"], node["y"]) for node in payload_b["global_nodes"]}
    assert pos_a == pos_b


def test_replay_state_from_deltas(sample_tweets_csv_path: Path) -> None:
    tweets_df, week_index_df = _prepared_fixture(sample_tweets_csv_path)
    payload = build_global_animation_payload(
        tweets_df=tweets_df,
        week_index_df=week_index_df,
        global_min_mentions=1,
        heat_decay=0.85,
        layout_seed=7,
    )

    node_ids = [node["id"] for node in payload["global_nodes"]]
    edge_ids = [edge["id"] for edge in payload["global_edges"]]
    node_heat, edge_cumulative = replay_week_state(
        node_ids=node_ids,
        edge_ids=edge_ids,
        node_week_deltas=payload["node_week_deltas"],
        edge_week_deltas=payload["edge_week_deltas"],
        heat_decay=payload["heat_decay"],
        target_week_index=1,
    )

    assert round(node_heat["alice"], 3) == 2.700
    assert round(node_heat["bob"], 3) == 1.700
    assert round(node_heat["carol"], 3) == 1.700
    assert round(node_heat["dave"], 3) == 1.000
    assert round(node_heat["eve"], 3) == 1.000

    assert edge_cumulative["alice|bob"] == 1
    assert edge_cumulative["alice|carol"] == 1
    assert edge_cumulative["bob|carol"] == 1
    assert edge_cumulative["alice|eve"] == 1


def test_size_and_edge_width_mappings() -> None:
    assert round(node_size_from_total_mentions(25, p99_total_mentions=100), 3) == 22.000
    assert round(edge_width_from_cumulative(25, max_cumulative_edge=100), 3) == 3.250
    assert edge_width_from_cumulative(0, max_cumulative_edge=100) == 0.0
