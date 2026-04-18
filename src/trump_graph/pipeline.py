from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .global_animation import (
    DEFAULT_GLOBAL_MIN_MENTIONS,
    DEFAULT_HEAT_DECAY,
    DEFAULT_LAYOUT_SEED,
    build_global_animation_payload,
)
from .graph_build import build_week_graph, graph_edges_to_frame, graph_nodes_to_frame
from .io import read_tweets_csv, write_global_animation_artifacts, write_index_files, write_week_artifacts
from .metrics import compute_week_metrics
from .preprocess import prepare_tweets


@dataclass(frozen=True)
class BuildStats:
    total_tweets: int
    processed_tweets: int
    weeks_built: int
    global_nodes: int
    global_edges: int
    output_dir: Path


def _default_week_index_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "week_id",
            "week_start",
            "week_end",
            "tweets_processed",
            "tweets_with_mentions",
            "unique_mentions",
            "edge_count",
            "density",
        ]
    )


def _default_weekly_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "week_id",
            "week_start",
            "week_end",
            "tweets_processed",
            "tweets_with_mentions",
            "unique_mentions",
            "edge_count",
            "density",
            "top_mention",
            "top_mention_weight",
            "top_edge_source",
            "top_edge_target",
            "top_edge_weight",
        ]
    )


def build_weekly_artifacts(
    input_csv: Path,
    output_dir: Path,
    min_mention_count: int = 1,
    include_retweets: bool = True,
    global_min_mentions: int = DEFAULT_GLOBAL_MIN_MENTIONS,
    heat_decay: float = DEFAULT_HEAT_DECAY,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
) -> BuildStats:
    if min_mention_count < 1:
        raise ValueError("min_mention_count must be >= 1")
    if global_min_mentions < 1:
        raise ValueError("global_min_mentions must be >= 1")

    raw_df = read_tweets_csv(input_csv)
    tweets_df = prepare_tweets(raw_df, include_retweets=include_retweets)

    week_index_rows: list[dict[str, Any]] = []
    weekly_summary_rows: list[dict[str, Any]] = []

    grouped = tweets_df.groupby("week_id", sort=False)
    for week_id, week_df in grouped:
        week_start = pd.Timestamp(week_df["week_start"].iloc[0])
        week_end = pd.Timestamp(week_df["week_end"].iloc[0])

        graph = build_week_graph(week_df["mentions"], min_mention_count=min_mention_count)
        nodes_df = graph_nodes_to_frame(graph)
        edges_df = graph_edges_to_frame(graph)

        metrics = compute_week_metrics(week_df, graph, nodes_df, edges_df)
        week_metrics = {
            "week_id": week_id,
            "week_start": week_start.strftime("%Y-%m-%d"),
            "week_end": week_end.strftime("%Y-%m-%d"),
            "include_retweets": include_retweets,
            "min_mention_count": int(min_mention_count),
            **metrics,
        }
        write_week_artifacts(output_dir=output_dir, week_id=week_id, nodes_df=nodes_df, edges_df=edges_df, metrics=week_metrics)

        week_index_row = {
            "week_id": week_id,
            "week_start": week_metrics["week_start"],
            "week_end": week_metrics["week_end"],
            "tweets_processed": metrics["tweets_processed"],
            "tweets_with_mentions": metrics["tweets_with_mentions"],
            "unique_mentions": metrics["unique_mentions"],
            "edge_count": metrics["edge_count"],
            "density": metrics["density"],
        }
        week_index_rows.append(week_index_row)

        top_mention = metrics["top_mentions"][0] if metrics["top_mentions"] else {"node": "", "weight": 0}
        top_edge = metrics["top_edges"][0] if metrics["top_edges"] else {"source": "", "target": "", "weight": 0}
        weekly_summary_rows.append(
            {
                **week_index_row,
                "top_mention": top_mention.get("node", ""),
                "top_mention_weight": int(top_mention.get("weight", 0)),
                "top_edge_source": top_edge.get("source", ""),
                "top_edge_target": top_edge.get("target", ""),
                "top_edge_weight": int(top_edge.get("weight", 0)),
            }
        )

    if week_index_rows:
        week_index_df = pd.DataFrame(week_index_rows).sort_values(["week_start", "week_id"], kind="mergesort")
    else:
        week_index_df = _default_week_index_frame()

    if weekly_summary_rows:
        weekly_summary_df = pd.DataFrame(weekly_summary_rows).sort_values(["week_start", "week_id"], kind="mergesort")
    else:
        weekly_summary_df = _default_weekly_summary_frame()

    week_index_df = week_index_df.reset_index(drop=True)
    weekly_summary_df = weekly_summary_df.reset_index(drop=True)
    write_index_files(output_dir=output_dir, week_index_df=week_index_df, weekly_summary_df=weekly_summary_df)

    animation_payload = build_global_animation_payload(
        tweets_df=tweets_df,
        week_index_df=week_index_df,
        global_min_mentions=global_min_mentions,
        heat_decay=heat_decay,
        layout_seed=layout_seed,
    )
    write_global_animation_artifacts(output_dir=output_dir, animation_payload=animation_payload)

    return BuildStats(
        total_tweets=int(len(raw_df)),
        processed_tweets=int(len(tweets_df)),
        weeks_built=int(len(week_index_df)),
        global_nodes=int(len(animation_payload.get("global_nodes", []))),
        global_edges=int(len(animation_payload.get("global_edges", []))),
        output_dir=output_dir,
    )
