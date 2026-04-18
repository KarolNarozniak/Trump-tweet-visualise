from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from pyvis.network import Network

COMMUNITY_PALETTE: tuple[str, ...] = (
    "#22D3EE",
    "#F97316",
    "#A78BFA",
    "#34D399",
    "#F43F5E",
    "#FACC15",
    "#38BDF8",
    "#FB7185",
    "#2DD4BF",
    "#C084FC",
    "#4ADE80",
    "#F59E0B",
)
DEFAULT_NODE_COLOR = "#94A3B8"


def load_week_index(processed_dir: Path) -> pd.DataFrame:
    week_index_path = processed_dir / "week_index.csv"
    if not week_index_path.exists():
        raise FileNotFoundError(f"Missing week index: {week_index_path}")

    week_index = pd.read_csv(week_index_path)
    if week_index.empty:
        return week_index

    return week_index.sort_values(["week_start", "week_id"], kind="mergesort").reset_index(drop=True)


def load_week_artifacts(processed_dir: Path, week_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    week_dir = processed_dir / "weeks" / week_id

    nodes_path = week_dir / "nodes.csv"
    edges_path = week_dir / "edges.csv"
    metrics_path = week_dir / "metrics.json"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return nodes_df, edges_df, metrics


def _node_size(weight: int, max_weight: int) -> float:
    if max_weight <= 0:
        return 10.0
    return 10.0 + 30.0 * (weight / max_weight) ** 0.5


def _edge_width(weight: int, max_weight: int) -> float:
    if max_weight <= 0:
        return 1.0
    return 1.0 + 5.0 * (weight / max_weight) ** 0.5


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(148, 163, 184, {alpha:.3f})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha:.3f})"


def _build_graph_from_frames(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    if not nodes_df.empty:
        graph.add_nodes_from(nodes_df["node"].astype(str).tolist())
    if not edges_df.empty:
        for row in edges_df.itertuples(index=False):
            graph.add_edge(str(row.source), str(row.target), weight=int(row.weight))
    return graph


def _node_community_color_map(graph: nx.Graph) -> dict[str, str]:
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_edges() == 0:
        fallback_color = COMMUNITY_PALETTE[0]
        return {str(node): fallback_color for node in graph.nodes}

    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    sorted_communities = sorted(communities, key=lambda cluster: (-len(cluster), sorted(cluster)[0]))

    color_map: dict[str, str] = {}
    for index, cluster in enumerate(sorted_communities):
        cluster_color = COMMUNITY_PALETTE[index % len(COMMUNITY_PALETTE)]
        for node in sorted(cluster):
            color_map[str(node)] = cluster_color

    for node in graph.nodes:
        color_map.setdefault(str(node), DEFAULT_NODE_COLOR)
    return color_map


def _node_color_style(base_color: str) -> dict[str, Any]:
    return {
        "background": base_color,
        "border": "#0F172A",
        "highlight": {"background": base_color, "border": "#E2E8F0"},
        "hover": {"background": base_color, "border": "#E2E8F0"},
    }


def build_pyvis_html(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_edges: int = 400,
    show_labels: bool = True,
    show_edges: bool = True,
) -> tuple[str, int, int]:
    network = Network(
        height="760px",
        width="100%",
        directed=False,
        notebook=False,
        bgcolor="#060B14",
        font_color="#E5E7EB",
    )
    network.toggle_physics(True)
    network.barnes_hut(gravity=-7000, spring_length=140, spring_strength=0.02, damping=0.9)
    node_color_map = _node_community_color_map(_build_graph_from_frames(nodes_df, edges_df))

    if not nodes_df.empty:
        max_node_weight = int(nodes_df["weight"].max())
        for row in nodes_df.itertuples(index=False):
            node_name = str(row.node)
            node_weight = int(row.weight)
            network.add_node(
                node_name,
                label=f"@{node_name}" if show_labels else "",
                value=node_weight,
                size=_node_size(node_weight, max_node_weight),
                title=f"@{node_name}<br>Mentions: {node_weight}",
                color=_node_color_style(node_color_map.get(node_name, DEFAULT_NODE_COLOR)),
            )

    total_edges = int(len(edges_df))
    displayed_edges_df = edges_df.head(max_edges).copy() if show_edges else edges_df.head(0).copy()
    displayed_edges = int(len(displayed_edges_df))

    if displayed_edges > 0:
        max_edge_weight = int(displayed_edges_df["weight"].max())
        for row in displayed_edges_df.itertuples(index=False):
            edge_weight = int(row.weight)
            source_color = node_color_map.get(str(row.source), DEFAULT_NODE_COLOR)
            network.add_edge(
                str(row.source),
                str(row.target),
                value=edge_weight,
                width=_edge_width(edge_weight, max_edge_weight),
                title=f"@{row.source} <> @{row.target}<br>Co-mentions: {edge_weight}",
                color=_hex_to_rgba(source_color, 0.34),
            )

    network.set_options(
        """
        {
          "interaction": {
            "hover": true,
            "tooltipDelay": 90,
            "navigationButtons": true,
            "keyboard": true
          },
          "nodes": {
            "borderWidth": 1.2,
            "shape": "dot",
            "font": {
              "size": 13,
              "color": "#F8FAFC"
            }
          },
          "edges": {
            "smooth": false
          },
          "physics": {
            "stabilization": {
              "iterations": 180
            }
          }
        }
        """
    )
    return network.generate_html(notebook=False), displayed_edges, total_edges
