from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


def _records_with_ints(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.head(limit).to_dict(orient="records"):
        normalized = {}
        for key, value in record.items():
            if isinstance(value, (int, float)) and float(value).is_integer():
                normalized[key] = int(value)
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def compute_week_metrics(
    week_df: pd.DataFrame,
    graph: nx.Graph,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    tweets_processed = int(len(week_df))
    tweets_with_mentions = int((week_df["mentions"].map(len) > 0).sum())
    density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0

    return {
        "tweets_processed": tweets_processed,
        "tweets_with_mentions": tweets_with_mentions,
        "unique_mentions": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "density": round(density, 6),
        "top_mentions": _records_with_ints(nodes_df, top_n),
        "top_edges": _records_with_ints(edges_df, top_n),
    }
