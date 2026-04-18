from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence

import networkx as nx
import pandas as pd


def build_week_graph(mentions_by_tweet: Iterable[Sequence[str]], min_mention_count: int = 1) -> nx.Graph:
    if min_mention_count < 1:
        raise ValueError("min_mention_count must be >= 1")

    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()

    for mentions in mentions_by_tweet:
        unique_mentions = tuple(dict.fromkeys(mentions))
        if not unique_mentions:
            continue

        for mention in unique_mentions:
            node_counts[mention] += 1

        for source, target in combinations(sorted(unique_mentions), 2):
            edge_counts[(source, target)] += 1

    graph = nx.Graph()
    kept_nodes = {node for node, count in node_counts.items() if count >= min_mention_count}

    for node in sorted(kept_nodes):
        graph.add_node(node, weight=int(node_counts[node]))

    sorted_edges = sorted(edge_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    for (source, target), weight in sorted_edges:
        if source in kept_nodes and target in kept_nodes:
            graph.add_edge(source, target, weight=int(weight))

    return graph


def graph_nodes_to_frame(graph: nx.Graph) -> pd.DataFrame:
    rows = [{"node": node, "weight": int(data.get("weight", 0))} for node, data in graph.nodes(data=True)]
    nodes_df = pd.DataFrame(rows, columns=["node", "weight"])
    if nodes_df.empty:
        return nodes_df
    return nodes_df.sort_values(["weight", "node"], ascending=[False, True], kind="mergesort").reset_index(
        drop=True
    )


def graph_edges_to_frame(graph: nx.Graph) -> pd.DataFrame:
    rows = [
        {"source": source, "target": target, "weight": int(data.get("weight", 0))}
        for source, target, data in graph.edges(data=True)
    ]
    edges_df = pd.DataFrame(rows, columns=["source", "target", "weight"])
    if edges_df.empty:
        return edges_df
    return edges_df.sort_values(
        ["weight", "source", "target"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
