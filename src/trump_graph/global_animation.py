from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from typing import Any, Iterable, Mapping, Sequence

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

    target_extent = 1300.0
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


def _ranked_polar_layout(
    node_ids: Sequence[str],
    node_total_counts: Mapping[str, int],
    node_sizes: Mapping[str, float],
    layout_seed: int,
) -> dict[str, tuple[float, float]]:
    if not node_ids:
        return {}

    ordered_nodes = sorted(node_ids, key=lambda node_id: (-int(node_total_counts[node_id]), str(node_id)))
    if len(ordered_nodes) == 1:
        return {str(ordered_nodes[0]): (0.0, 0.0)}

    max_node_size = float(max(node_sizes.values(), default=8.0))
    base_radius = max(52.0, max_node_size * 4.0)
    radial_step = max(58.0, max_node_size * 3.2)
    min_arc_spacing = max(28.0, max_node_size * 2.9)
    clockwise_factor = -1.0

    # Deterministic phase so the same data/seed produces the same orientation.
    phase = (int(layout_seed) % 360) * (math.pi / 180.0)

    positions: dict[str, tuple[float, float]] = {str(ordered_nodes[0]): (0.0, 0.0)}
    cursor = 1
    ring_index = 1
    while cursor < len(ordered_nodes):
        radius = base_radius + (ring_index - 1) * radial_step
        circumference = 2.0 * math.pi * radius
        ring_capacity = max(8, int(circumference / min_arc_spacing))
        ring_nodes = ordered_nodes[cursor : cursor + ring_capacity]
        if not ring_nodes:
            break

        angle_step = (2.0 * math.pi) / float(len(ring_nodes))
        ring_phase = phase + (ring_index * 0.41)
        for local_index, node_id in enumerate(ring_nodes):
            angle = ring_phase + clockwise_factor * (local_index * angle_step)
            positions[str(node_id)] = (radius * math.cos(angle), radius * math.sin(angle))

        cursor += len(ring_nodes)
        ring_index += 1

    return positions


def _resolve_node_overlaps(
    positions: Mapping[str, tuple[float, float]],
    node_sizes: Mapping[str, float],
    iterations: int = 22,
) -> dict[str, tuple[float, float]]:
    if not positions:
        return {}

    node_ids = sorted(str(node_id) for node_id in positions.keys())
    coordinates = np.array([positions[node_id] for node_id in node_ids], dtype="float64")
    preferred = coordinates.copy()
    sizes = np.array([float(node_sizes.get(node_id, 6.0)) for node_id in node_ids], dtype="float64")

    def _minimum_distance(size_a: float, size_b: float) -> float:
        return 10.0 + ((size_a + size_b) * 2.4)

    for _ in range(max(1, iterations)):
        moved = False
        for left_index in range(len(node_ids) - 1):
            for right_index in range(left_index + 1, len(node_ids)):
                dx = float(coordinates[right_index, 0] - coordinates[left_index, 0])
                dy = float(coordinates[right_index, 1] - coordinates[left_index, 1])
                distance_sq = (dx * dx) + (dy * dy)
                minimum_distance = _minimum_distance(sizes[left_index], sizes[right_index])
                if distance_sq >= (minimum_distance * minimum_distance):
                    continue

                distance = math.sqrt(distance_sq) if distance_sq > 1e-12 else 0.0
                if distance <= 1e-6:
                    angle = ((left_index * 92821 + right_index * 68917) % 3600) * (2.0 * math.pi / 3600.0)
                    ux = math.cos(angle)
                    uy = math.sin(angle)
                    distance = 0.0
                else:
                    ux = dx / distance
                    uy = dy / distance

                overlap = minimum_distance - distance
                shift = overlap * 0.5
                coordinates[left_index, 0] -= ux * shift
                coordinates[left_index, 1] -= uy * shift
                coordinates[right_index, 0] += ux * shift
                coordinates[right_index, 1] += uy * shift
                moved = True

        # Keep the ranking structure stable while relieving collisions.
        coordinates = (coordinates * 0.985) + (preferred * 0.015)
        if not moved:
            break

    return {
        node_id: (round(float(coordinates[index, 0]), 4), round(float(coordinates[index, 1]), 4))
        for index, node_id in enumerate(node_ids)
    }


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

    p99_mentions = float(pd.Series([node_total_counts[node] for node in kept_nodes], dtype="float64").quantile(0.99))
    if not math.isfinite(p99_mentions) or p99_mentions <= 0:
        p99_mentions = 1.0

    node_sizes = {
        str(node_id): node_size_from_total_mentions(node_total_counts[node_id], p99_mentions)
        for node_id in kept_nodes
    }
    ranked_positions = _ranked_polar_layout(
        node_ids=kept_nodes,
        node_total_counts=node_total_counts,
        node_sizes=node_sizes,
        layout_seed=layout_seed,
    )
    layout_positions = _normalize_layout_positions(ranked_positions)
    layout_positions = _resolve_node_overlaps(layout_positions, node_sizes=node_sizes)

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
            "size": round(node_sizes[node_id], 4),
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
