from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

GLOBAL_HUB_NODE_ID = "realdonaldtrump"
DEFAULT_GLOBAL_MIN_MENTIONS = 8
DEFAULT_HEAT_DECAY = 0.85
DEFAULT_LAYOUT_SEED = 42
DEFAULT_NODE_SIZE_MIN = 3.0
DEFAULT_NODE_SIZE_SCALE = 15.0
DEFAULT_TOP_LABEL_COUNT = 40


def node_size_from_total_mentions(
    total_mentions: int,
    p99_total_mentions: float,
    size_min: float = DEFAULT_NODE_SIZE_MIN,
    size_scale: float = DEFAULT_NODE_SIZE_SCALE,
) -> float:
    if p99_total_mentions <= 0:
        return float(size_min)
    normalized = min(float(total_mentions) / float(p99_total_mentions), 1.0)
    return float(size_min + size_scale * (normalized**0.5))


def edge_width_from_cumulative(cumulative: int, max_cumulative_edge: int) -> float:
    if cumulative <= 0:
        return 0.0
    if max_cumulative_edge <= 0:
        return 0.9
    normalized = max(0.0, min(1.0, float(cumulative) / float(max_cumulative_edge)))
    return float(0.9 + 7.0 * (normalized**0.5))


def replay_week_state(
    node_ids: Sequence[str],
    edge_ids: Sequence[str],
    node_week_deltas: Sequence[Sequence[Sequence[Any]]],
    edge_week_deltas: Sequence[Sequence[Sequence[Any]]],
    heat_decay: float,
    target_week_index: int,
) -> tuple[dict[str, float], dict[str, int]]:
    if target_week_index < 0:
        raise ValueError("target_week_index must be >= 0")
    if target_week_index >= len(node_week_deltas):
        raise ValueError("target_week_index out of range for node_week_deltas")

    node_heat = {node_id: 0.0 for node_id in node_ids}
    edge_cumulative = {edge_id: 0 for edge_id in edge_ids}

    for week_index in range(target_week_index + 1):
        for node_id in node_ids:
            node_heat[node_id] *= heat_decay
        for node_id, delta in node_week_deltas[week_index]:
            node_heat[str(node_id)] = node_heat.get(str(node_id), 0.0) + float(delta)
        for edge_id, delta in edge_week_deltas[week_index]:
            edge_cumulative[str(edge_id)] = edge_cumulative.get(str(edge_id), 0) + int(delta)

    return node_heat, edge_cumulative


def _count_mentions_and_edges(
    mentions_by_tweet: Iterable[Sequence[str]],
) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()

    for mentions in mentions_by_tweet:
        unique_mentions = tuple(dict.fromkeys(mentions))
        if not unique_mentions:
            continue

        for mention in unique_mentions:
            node_counts[str(mention)] += 1
        for source, target in combinations(sorted(unique_mentions), 2):
            edge_counts[(str(source), str(target))] += 1

    return node_counts, edge_counts


def _normalize_layout_positions(positions: dict[str, Any]) -> dict[str, tuple[float, float]]:
    if not positions:
        return {}

    target_extent = 1000.0
    max_abs = max(
        max(abs(float(coord[0])), abs(float(coord[1])))
        for coord in positions.values()
    )
    scale = 1.0 if max_abs <= 1e-12 else target_extent / max_abs

    normalized: dict[str, tuple[float, float]] = {}
    for node_id, coord in positions.items():
        normalized_x = float(coord[0]) * scale
        normalized_y = float(coord[1]) * scale
        if not math.isfinite(normalized_x) or not math.isfinite(normalized_y):
            normalized_x = 0.0
            normalized_y = 0.0
        normalized[str(node_id)] = (round(normalized_x, 4), round(normalized_y, 4))
    return normalized


def _dense_force_layout(subgraph: nx.Graph, seed: int, iterations: int = 90) -> dict[str, tuple[float, float]]:
    node_ids = sorted(str(node_id) for node_id in subgraph.nodes())
    node_count = len(node_ids)
    if node_count == 0:
        return {}
    if node_count == 1:
        return {node_ids[0]: (0.0, 0.0)}

    adjacency = nx.to_numpy_array(subgraph, nodelist=node_ids, weight="weight", dtype="float64")
    random_state = np.random.RandomState(seed)
    positions = random_state.rand(node_count, 2).astype(adjacency.dtype)

    optimal_distance = math.sqrt(1.0 / node_count)
    temperature = max(
        float(positions[:, 0].max() - positions[:, 0].min()),
        float(positions[:, 1].max() - positions[:, 1].min()),
    ) * 0.1
    if temperature <= 0 or not math.isfinite(temperature):
        temperature = 0.1
    delta_temperature = temperature / float(iterations + 1)

    for _ in range(iterations):
        delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.clip(distance, 0.01, None, out=distance)

        displacement = np.einsum(
            "ijk,ij->ik",
            delta,
            (optimal_distance * optimal_distance / (distance * distance)) - (adjacency * distance / optimal_distance),
        )
        displacement_length = np.linalg.norm(displacement, axis=-1)
        displacement_length = np.clip(displacement_length, 0.01, None)
        delta_pos = np.einsum("ij,i->ij", displacement, temperature / displacement_length)
        positions += delta_pos

        temperature -= delta_temperature
        if (np.linalg.norm(delta_pos) / node_count) < 1e-4:
            break

    scaled_positions = nx.rescale_layout(positions, scale=1.0)
    return {
        str(node_id): (float(scaled_positions[index, 0]), float(scaled_positions[index, 1]))
        for index, node_id in enumerate(node_ids)
    }


def _component_ring_layout(graph: nx.Graph, layout_seed: int) -> dict[str, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}

    components = [sorted(component) for component in nx.connected_components(graph)]
    components.sort(key=lambda component: (-len(component), component[0]))

    positions: dict[str, tuple[float, float]] = {}
    golden_angle = 2.399963229728653
    component_scales = [max(90.0, 42.0 * math.sqrt(len(component))) for component in components]
    base_radius = (component_scales[0] + 280.0) if component_scales else 300.0

    for component_index, component_nodes in enumerate(components):
        if component_index == 0:
            center_x = 0.0
            center_y = 0.0
        else:
            angle = component_index * golden_angle
            radius = base_radius + 220.0 * math.sqrt(component_index)
            center_x = radius * math.cos(angle)
            center_y = radius * math.sin(angle)

        component_size = len(component_nodes)
        if component_size == 1:
            node_id = str(component_nodes[0])
            positions[node_id] = (center_x, center_y)
            continue

        subgraph = graph.subgraph(component_nodes)
        local_scale = component_scales[component_index]
        try:
            local_positions = _dense_force_layout(subgraph, seed=layout_seed + component_index)
        except Exception:
            local_positions = {
                str(node_id): (float(coord[0]), float(coord[1]))
                for node_id, coord in nx.circular_layout(subgraph, scale=1.0).items()
            }
        local_positions = {
            str(node_id): (float(coord[0]) * local_scale, float(coord[1]) * local_scale)
            for node_id, coord in local_positions.items()
        }

        for node_id, coord in local_positions.items():
            x = float(coord[0]) + center_x
            y = float(coord[1]) + center_y
            positions[str(node_id)] = (x, y)

    return positions


def build_global_animation_payload(
    tweets_df: pd.DataFrame,
    week_index_df: pd.DataFrame,
    global_min_mentions: int = DEFAULT_GLOBAL_MIN_MENTIONS,
    heat_decay: float = DEFAULT_HEAT_DECAY,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
) -> dict[str, Any]:
    if global_min_mentions < 1:
        raise ValueError("global_min_mentions must be >= 1")
    if not (0.0 < heat_decay < 1.0):
        raise ValueError("heat_decay must be in the open interval (0, 1)")

    node_total_counts, edge_total_counts = _count_mentions_and_edges(tweets_df["mentions"])
    kept_nodes = sorted(node for node, count in node_total_counts.items() if count >= global_min_mentions)
    kept_node_set = set(kept_nodes)

    filtered_edge_counts: Counter[tuple[str, str]] = Counter(
        {
            (source, target): count
            for (source, target), count in edge_total_counts.items()
            if source in kept_node_set and target in kept_node_set
        }
    )

    global_graph = nx.Graph()
    for node_id in kept_nodes:
        global_graph.add_node(node_id, total_mentions=int(node_total_counts[node_id]))
    for (source, target), count in filtered_edge_counts.items():
        global_graph.add_edge(source, target, weight=int(count))

    component_positions = _component_ring_layout(global_graph, layout_seed=layout_seed)
    if any(
        not math.isfinite(float(coord[0])) or not math.isfinite(float(coord[1]))
        for coord in component_positions.values()
    ):
        component_positions = {str(node_id): tuple(coord) for node_id, coord in nx.circular_layout(global_graph).items()}
    layout_positions = _normalize_layout_positions(component_positions)

    p99_mentions = float(pd.Series([node_total_counts[node] for node in kept_nodes], dtype="float64").quantile(0.99))
    if not math.isfinite(p99_mentions) or p99_mentions <= 0:
        p99_mentions = 1.0

    top_label_nodes = [
        node_id
        for node_id, _count in sorted(
            ((node_id, node_total_counts[node_id]) for node_id in kept_nodes),
            key=lambda pair: (-pair[1], pair[0]),
        )[:DEFAULT_TOP_LABEL_COUNT]
    ]

    global_nodes = [
        {
            "id": node_id,
            "total_mentions": int(node_total_counts[node_id]),
            "size": round(node_size_from_total_mentions(node_total_counts[node_id], p99_mentions), 4),
            "x": layout_positions.get(node_id, (0.0, 0.0))[0],
            "y": layout_positions.get(node_id, (0.0, 0.0))[1],
            "is_hub": node_id == GLOBAL_HUB_NODE_ID,
        }
        for node_id in kept_nodes
    ]

    global_edges = []
    edge_ids: list[str] = []
    for source, target in sorted(filtered_edge_counts):
        edge_id = f"{source}|{target}"
        edge_ids.append(edge_id)
        global_edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "total_co_mentions": int(filtered_edge_counts[(source, target)]),
            }
        )

    week_records = week_index_df[["week_id", "week_start", "week_end", "tweets_processed", "tweets_with_mentions", "unique_mentions", "edge_count"]].to_dict(orient="records")
    week_ids = [str(record["week_id"]) for record in week_records]

    grouped_weeks = {
        str(week_id): group
        for week_id, group in tweets_df.groupby("week_id", sort=False)
    }

    node_week_deltas: list[list[list[Any]]] = []
    edge_week_deltas: list[list[list[Any]]] = []
    node_peak_heat: Counter[str] = Counter()
    node_heat_running = {node_id: 0.0 for node_id in kept_nodes}

    for week_id in week_ids:
        week_df = grouped_weeks.get(week_id)
        if week_df is None:
            node_week_deltas.append([])
            edge_week_deltas.append([])
            for node_id in kept_nodes:
                node_heat_running[node_id] *= heat_decay
                if node_heat_running[node_id] > node_peak_heat[node_id]:
                    node_peak_heat[node_id] = node_heat_running[node_id]
            continue

        week_mentions = (
            [mention for mention in mentions if mention in kept_node_set]
            for mentions in week_df["mentions"]
        )
        week_node_counts, week_edge_counts = _count_mentions_and_edges(week_mentions)

        week_node_delta_entries = [[node_id, int(delta)] for node_id, delta in sorted(week_node_counts.items()) if delta > 0]
        week_edge_delta_entries = []
        for source, target in sorted(week_edge_counts):
            edge_id = f"{source}|{target}"
            delta = int(week_edge_counts[(source, target)])
            if delta > 0:
                week_edge_delta_entries.append([edge_id, delta])

        node_week_deltas.append(week_node_delta_entries)
        edge_week_deltas.append(week_edge_delta_entries)

        node_week_delta_map = {node_id: int(delta) for node_id, delta in week_node_delta_entries}
        for node_id in kept_nodes:
            next_heat = node_heat_running[node_id] * heat_decay + float(node_week_delta_map.get(node_id, 0))
            node_heat_running[node_id] = next_heat
            if next_heat > node_peak_heat[node_id]:
                node_peak_heat[node_id] = next_heat

    max_cumulative_edge = int(max((edge["total_co_mentions"] for edge in global_edges), default=0))
    heat_scale = float(max(node_peak_heat.values(), default=1.0))
    if heat_scale <= 0:
        heat_scale = 1.0

    return {
        "version": 1,
        "heat_decay": float(heat_decay),
        "global_min_mentions": int(global_min_mentions),
        "layout_seed": int(layout_seed),
        "heat_scale": round(heat_scale, 6),
        "max_cumulative_edge": max_cumulative_edge,
        "hub_node_id": GLOBAL_HUB_NODE_ID,
        "top_label_nodes": top_label_nodes,
        "weeks": week_records,
        "global_nodes": global_nodes,
        "global_edges": global_edges,
        "node_week_deltas": node_week_deltas,
        "edge_week_deltas": edge_week_deltas,
    }
