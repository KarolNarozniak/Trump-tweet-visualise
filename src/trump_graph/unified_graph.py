from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .global_animation import (
    DEFAULT_HEAT_DECAY,
    DEFAULT_LAYOUT_SEED,
    _normalize_layout_positions,
    _ranked_polar_layout,
    _resolve_node_overlaps,
    node_size_from_total_mentions,
)
from .truth_graph import TRUTH_NODE_TYPES, TRUTH_SENTIMENTS, truth_node_label, truth_node_type
from .truth_semantics import json_dumps_compact, parse_json_list

DEFAULT_UNIFIED_MIN_NODE_COUNT = 8
UNIFIED_PLATFORMS = ("twitter", "truth")


@dataclass(frozen=True)
class UnifiedDeltaSet:
    node_week_deltas: list[list[list[Any]]]
    edge_week_deltas: list[list[list[Any]]]


def _semantic_nodes_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [str(node_id) for node_id in parse_json_list(value)]
    if isinstance(value, list):
        return [str(node_id) for node_id in value]
    return [str(node_id) for node_id in value] if value is not None else []


def _count_nodes_and_edges(nodes_by_post: Iterable[Sequence[str]]) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    for nodes in nodes_by_post:
        unique_nodes = tuple(dict.fromkeys(str(node_id) for node_id in nodes if str(node_id)))
        for node_id in unique_nodes:
            node_counts[node_id] += 1
        for source, target in combinations(sorted(unique_nodes), 2):
            edge_counts[(source, target)] += 1
    return node_counts, edge_counts


def _filter_node_types(nodes: Sequence[str], included_node_types: set[str]) -> list[str]:
    return [node for node in nodes if truth_node_type(node) in included_node_types]


def _build_delta_set(
    *,
    enriched_df: pd.DataFrame,
    week_ids: Sequence[str],
    kept_node_set: set[str],
    edge_id_set: set[str],
    included_node_types: set[str],
) -> UnifiedDeltaSet:
    grouped_weeks = {str(week_id): group for week_id, group in enriched_df.groupby("week_id", sort=False)}
    node_week_deltas: list[list[list[Any]]] = []
    edge_week_deltas: list[list[list[Any]]] = []

    for week_id in week_ids:
        week_df = grouped_weeks.get(str(week_id))
        if week_df is None:
            node_week_deltas.append([])
            edge_week_deltas.append([])
            continue

        nodes_by_post = [
            [
                node
                for node in _filter_node_types(_semantic_nodes_from_value(nodes_value), included_node_types)
                if node in kept_node_set
            ]
            for nodes_value in week_df["semantic_nodes"].tolist()
        ]
        node_counts, edge_counts = _count_nodes_and_edges(nodes_by_post)
        node_week_deltas.append([[node_id, int(count)] for node_id, count in sorted(node_counts.items()) if count > 0])

        week_edges: list[list[Any]] = []
        for (source, target), count in sorted(edge_counts.items()):
            edge_id = f"{source}|{target}"
            if edge_id in edge_id_set and count > 0:
                week_edges.append([edge_id, int(count)])
        edge_week_deltas.append(week_edges)

    return UnifiedDeltaSet(node_week_deltas=node_week_deltas, edge_week_deltas=edge_week_deltas)


def _max_heat_from_deltas(
    node_ids: Sequence[str],
    node_week_deltas: Sequence[Sequence[Sequence[Any]]],
    heat_decay: float,
) -> float:
    heat = {node_id: 0.0 for node_id in node_ids}
    peak = 0.0
    for week_entries in node_week_deltas:
        for node_id in node_ids:
            heat[node_id] *= heat_decay
        for node_id, delta in week_entries:
            node_key = str(node_id)
            heat[node_key] = heat.get(node_key, 0.0) + float(delta)
        peak = max(peak, max(heat.values(), default=0.0))
    return peak if peak > 0 else 1.0


def _build_week_frames(
    enriched_df: pd.DataFrame,
    *,
    kept_node_set: set[str],
    included_node_types: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    week_index_rows: list[dict[str, Any]] = []
    weekly_summary_rows: list[dict[str, Any]] = []

    for week_id, week_df in enriched_df.groupby("week_id", sort=False):
        nodes_by_post = [
            [
                node
                for node in _filter_node_types(_semantic_nodes_from_value(nodes_value), included_node_types)
                if node in kept_node_set
            ]
            for nodes_value in week_df["semantic_nodes"].tolist()
        ]
        node_counts, edge_counts = _count_nodes_and_edges(nodes_by_post)
        topic_counts = Counter({node: count for node, count in node_counts.items() if truth_node_type(node) == "topic"})
        entity_counts = Counter(
            {
                node: count
                for node, count in node_counts.items()
                if truth_node_type(node) in {"per", "org", "loc"}
            }
        )
        top_topic, top_topic_count = topic_counts.most_common(1)[0] if topic_counts else ("", 0)
        top_entity, top_entity_count = entity_counts.most_common(1)[0] if entity_counts else ("", 0)

        week_start = pd.Timestamp(week_df["week_start"].iloc[0]).strftime("%Y-%m-%d")
        week_end = pd.Timestamp(week_df["week_end"].iloc[0]).strftime("%Y-%m-%d")
        base_row = {
            "week_id": str(week_id),
            "week_start": week_start,
            "week_end": week_end,
            "posts_processed": int(len(week_df)),
            "twitter_posts": int((week_df["source_platform"] == "twitter").sum()),
            "truth_posts": int((week_df["source_platform"] == "truth").sum()),
            "reposts": int(week_df["is_repost"].sum()),
            "original_posts": int((~week_df["is_repost"]).sum()),
            "unique_nodes": int(len(node_counts)),
            "edge_count": int(len(edge_counts)),
            "negative_posts": int((week_df["sentiment_label"] == "negative").sum()),
            "neutral_posts": int((week_df["sentiment_label"] == "neutral").sum()),
            "positive_posts": int((week_df["sentiment_label"] == "positive").sum()),
        }
        week_index_rows.append(base_row)
        weekly_summary_rows.append(
            {
                **base_row,
                "top_topic": truth_node_label(top_topic) if top_topic else "",
                "top_topic_node": top_topic,
                "top_topic_count": int(top_topic_count),
                "top_entity": truth_node_label(top_entity) if top_entity else "",
                "top_entity_node": top_entity,
                "top_entity_count": int(top_entity_count),
            }
        )

    if not week_index_rows:
        empty_week_index = pd.DataFrame(
            columns=[
                "week_id",
                "week_start",
                "week_end",
                "posts_processed",
                "twitter_posts",
                "truth_posts",
                "reposts",
                "original_posts",
                "unique_nodes",
                "edge_count",
                "negative_posts",
                "neutral_posts",
                "positive_posts",
            ]
        )
        empty_weekly_summary = pd.DataFrame(
            columns=[
                "week_id",
                "week_start",
                "week_end",
                "posts_processed",
                "twitter_posts",
                "truth_posts",
                "reposts",
                "original_posts",
                "unique_nodes",
                "edge_count",
                "negative_posts",
                "neutral_posts",
                "positive_posts",
                "top_topic",
                "top_topic_node",
                "top_topic_count",
                "top_entity",
                "top_entity_node",
                "top_entity_count",
            ]
        )
        return empty_week_index, empty_weekly_summary

    week_index_df = pd.DataFrame(week_index_rows).sort_values(["week_start", "week_id"], kind="mergesort")
    weekly_summary_df = pd.DataFrame(weekly_summary_rows).sort_values(["week_start", "week_id"], kind="mergesort")
    return week_index_df.reset_index(drop=True), weekly_summary_df.reset_index(drop=True)


def _temporal_delta_frames(
    *,
    week_ids: Sequence[str],
    global_nodes: Sequence[Mapping[str, Any]],
    global_edges: Sequence[Mapping[str, Any]],
    node_week_deltas: Sequence[Sequence[Sequence[Any]]],
    edge_week_deltas: Sequence[Sequence[Sequence[Any]]],
    heat_decay: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_id_to_total = {str(node["id"]): int(node["total_count"]) for node in global_nodes}
    node_id_to_type = {str(node["id"]): str(node["node_type"]) for node in global_nodes}
    edge_id_to_meta = {
        str(edge["id"]): (
            str(edge["source"]),
            str(edge["target"]),
            int(edge["total_co_occurrences"]),
        )
        for edge in global_edges
    }

    node_heat = {node_id: 0.0 for node_id in node_id_to_total}
    node_cumulative = {node_id: 0 for node_id in node_id_to_total}
    edge_cumulative = {edge_id: 0 for edge_id in edge_id_to_meta}

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for week_index, week_id in enumerate(week_ids):
        for node_id in node_heat:
            node_heat[node_id] *= heat_decay

        week_node_deltas = {str(node_id): int(delta) for node_id, delta in node_week_deltas[week_index]}
        for node_id, delta in week_node_deltas.items():
            node_cumulative[node_id] = node_cumulative.get(node_id, 0) + delta
            node_heat[node_id] = node_heat.get(node_id, 0.0) + float(delta)
            node_rows.append(
                {
                    "week_id": str(week_id),
                    "node_id": node_id,
                    "node_type": node_id_to_type.get(node_id, ""),
                    "weekly_count": int(delta),
                    "cumulative_count": int(node_cumulative.get(node_id, 0)),
                    "heat_after_update": round(float(node_heat.get(node_id, 0.0)), 6),
                    "total_count": int(node_id_to_total.get(node_id, 0)),
                }
            )

        week_edge_deltas = {str(edge_id): int(delta) for edge_id, delta in edge_week_deltas[week_index]}
        for edge_id, delta in week_edge_deltas.items():
            source, target, total_count = edge_id_to_meta.get(edge_id, ("", "", 0))
            edge_cumulative[edge_id] = edge_cumulative.get(edge_id, 0) + delta
            edge_rows.append(
                {
                    "week_id": str(week_id),
                    "edge_id": edge_id,
                    "source": source,
                    "target": target,
                    "weekly_count": int(delta),
                    "cumulative_count": int(edge_cumulative.get(edge_id, 0)),
                    "total_count": int(total_count),
                }
            )

    node_delta_df = pd.DataFrame(
        node_rows,
        columns=[
            "week_id",
            "node_id",
            "node_type",
            "weekly_count",
            "cumulative_count",
            "heat_after_update",
            "total_count",
        ],
    )
    edge_delta_df = pd.DataFrame(
        edge_rows,
        columns=[
            "week_id",
            "edge_id",
            "source",
            "target",
            "weekly_count",
            "cumulative_count",
            "total_count",
        ],
    )
    return node_delta_df, edge_delta_df


def build_unified_semantic_artifacts(
    enriched_df: pd.DataFrame,
    *,
    min_node_count: int = DEFAULT_UNIFIED_MIN_NODE_COUNT,
    included_node_types: Iterable[str] = TRUTH_NODE_TYPES,
    heat_decay: float = DEFAULT_HEAT_DECAY,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    if min_node_count < 1:
        raise ValueError("min_node_count must be >= 1")
    if not (0.0 < heat_decay < 1.0):
        raise ValueError("heat_decay must be in the open interval (0, 1)")

    included_types = {node_type.strip().lower() for node_type in included_node_types if node_type.strip()}
    if not included_types:
        raise ValueError("included_node_types must contain at least one node type")

    nodes_by_post = [
        _filter_node_types(_semantic_nodes_from_value(nodes_value), included_types)
        for nodes_value in enriched_df["semantic_nodes"].tolist()
    ]
    node_total_counts, edge_total_counts = _count_nodes_and_edges(nodes_by_post)
    kept_nodes = sorted(node_id for node_id, count in node_total_counts.items() if count >= min_node_count)
    kept_node_set = set(kept_nodes)
    filtered_edge_counts: Counter[tuple[str, str]] = Counter(
        {
            (source, target): count
            for (source, target), count in edge_total_counts.items()
            if source in kept_node_set and target in kept_node_set
        }
    )

    week_index_df, weekly_summary_df = _build_week_frames(
        enriched_df,
        kept_node_set=kept_node_set,
        included_node_types=included_types,
    )
    week_ids = [str(week_id) for week_id in week_index_df["week_id"].tolist()]

    if kept_nodes:
        p99_counts = float(pd.Series([node_total_counts[node] for node in kept_nodes], dtype="float64").quantile(0.99))
    else:
        p99_counts = 1.0
    if p99_counts <= 0:
        p99_counts = 1.0

    node_sizes = {
        node_id: node_size_from_total_mentions(node_total_counts[node_id], p99_counts)
        for node_id in kept_nodes
    }
    ranked_positions = _ranked_polar_layout(kept_nodes, node_total_counts, node_sizes, layout_seed)
    layout_positions = _resolve_node_overlaps(_normalize_layout_positions(ranked_positions), node_sizes=node_sizes)

    global_nodes = [
        {
            "id": node_id,
            "label": truth_node_label(node_id),
            "node_type": truth_node_type(node_id),
            "total_count": int(node_total_counts[node_id]),
            "size": round(float(node_sizes[node_id]), 4),
            "x": layout_positions.get(node_id, (0.0, 0.0))[0],
            "y": layout_positions.get(node_id, (0.0, 0.0))[1],
        }
        for node_id in kept_nodes
    ]

    global_edges = []
    for source, target in sorted(filtered_edge_counts):
        edge_id = f"{source}|{target}"
        count = int(filtered_edge_counts[(source, target)])
        global_edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "total_co_occurrences": count,
            }
        )
    edge_id_set = {str(edge["id"]) for edge in global_edges}

    delta_sets: dict[str, dict[str, list[list[list[Any]]]]] = {}
    all_deltas = _build_delta_set(
        enriched_df=enriched_df,
        week_ids=week_ids,
        kept_node_set=kept_node_set,
        edge_id_set=edge_id_set,
        included_node_types=included_types,
    )
    delta_sets["all"] = {
        "node_week_deltas": all_deltas.node_week_deltas,
        "edge_week_deltas": all_deltas.edge_week_deltas,
    }

    for platform in UNIFIED_PLATFORMS:
        platform_df = enriched_df.loc[enriched_df["source_platform"] == platform].copy()
        platform_deltas = _build_delta_set(
            enriched_df=platform_df,
            week_ids=week_ids,
            kept_node_set=kept_node_set,
            edge_id_set=edge_id_set,
            included_node_types=included_types,
        )
        delta_sets[platform] = {
            "node_week_deltas": platform_deltas.node_week_deltas,
            "edge_week_deltas": platform_deltas.edge_week_deltas,
        }

    original_df = enriched_df.loc[~enriched_df["is_repost"]].copy()
    original_deltas = _build_delta_set(
        enriched_df=original_df,
        week_ids=week_ids,
        kept_node_set=kept_node_set,
        edge_id_set=edge_id_set,
        included_node_types=included_types,
    )
    delta_sets["original"] = {
        "node_week_deltas": original_deltas.node_week_deltas,
        "edge_week_deltas": original_deltas.edge_week_deltas,
    }

    for sentiment in TRUTH_SENTIMENTS:
        sentiment_df = enriched_df.loc[enriched_df["sentiment_label"] == sentiment].copy()
        sentiment_deltas = _build_delta_set(
            enriched_df=sentiment_df,
            week_ids=week_ids,
            kept_node_set=kept_node_set,
            edge_id_set=edge_id_set,
            included_node_types=included_types,
        )
        delta_sets[sentiment] = {
            "node_week_deltas": sentiment_deltas.node_week_deltas,
            "edge_week_deltas": sentiment_deltas.edge_week_deltas,
        }

    heat_scale = max(
        _max_heat_from_deltas(kept_nodes, delta_set["node_week_deltas"], heat_decay)
        for delta_set in delta_sets.values()
    ) if delta_sets else 1.0
    max_cumulative_edge = int(max((edge["total_co_occurrences"] for edge in global_edges), default=0))

    top_label_nodes = [
        node_id
        for node_id, _count in sorted(
            ((node_id, node_total_counts[node_id]) for node_id in kept_nodes),
            key=lambda pair: (-pair[1], pair[0]),
        )[:40]
    ]

    week_records = week_index_df.to_dict(orient="records")
    animation_payload = {
        "version": 1,
        "graph_kind": "unified_semantic",
        "heat_decay": float(heat_decay),
        "min_node_count": int(min_node_count),
        "layout_seed": int(layout_seed),
        "heat_scale": round(float(heat_scale), 6),
        "max_cumulative_edge": max_cumulative_edge,
        "top_label_nodes": top_label_nodes,
        "available_node_types": sorted(included_types),
        "available_delta_sets": sorted(delta_sets.keys()),
        "weeks": week_records,
        "global_nodes": global_nodes,
        "global_edges": global_edges,
        "node_week_deltas": delta_sets["all"]["node_week_deltas"],
        "edge_week_deltas": delta_sets["all"]["edge_week_deltas"],
        "delta_sets": delta_sets,
    }

    node_catalog_df = pd.DataFrame(
        [
            {
                "node_id": node["id"],
                "label": node["label"],
                "node_type": node["node_type"],
                "total_count": node["total_count"],
            }
            for node in global_nodes
        ],
        columns=["node_id", "label", "node_type", "total_count"],
    ).sort_values(["total_count", "node_id"], ascending=[False, True], kind="mergesort")

    edge_catalog_df = pd.DataFrame(
        [
            {
                "edge_id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "total_count": edge["total_co_occurrences"],
            }
            for edge in global_edges
        ],
        columns=["edge_id", "source", "target", "total_count"],
    ).sort_values(["total_count", "edge_id"], ascending=[False, True], kind="mergesort")

    top_nodes_by_week: dict[str, str] = {}
    for week_id, week_df in enriched_df.groupby("week_id", sort=False):
        week_nodes_by_post = [
            [
                node
                for node in _filter_node_types(_semantic_nodes_from_value(nodes_value), included_types)
                if node in kept_node_set
            ]
            for nodes_value in week_df["semantic_nodes"].tolist()
        ]
        node_counts, _edge_counts = _count_nodes_and_edges(week_nodes_by_post)
        top_nodes_by_week[str(week_id)] = json_dumps_compact(
            [
                {"node": node_id, "label": truth_node_label(node_id), "count": int(count)}
                for node_id, count in node_counts.most_common(12)
            ]
        )
    weekly_summary_df = weekly_summary_df.copy()
    weekly_summary_df["top_nodes_json"] = weekly_summary_df["week_id"].map(top_nodes_by_week).fillna("[]")
    for column in ["top_topic_node", "top_entity_node"]:
        weekly_summary_df[column] = weekly_summary_df[column].fillna("")

    node_delta_df, edge_delta_df = _temporal_delta_frames(
        week_ids=week_ids,
        global_nodes=global_nodes,
        global_edges=global_edges,
        node_week_deltas=delta_sets["all"]["node_week_deltas"],
        edge_week_deltas=delta_sets["all"]["edge_week_deltas"],
        heat_decay=float(heat_decay),
    )
    temporal_deltas_df = {
        "node_deltas": node_delta_df.reset_index(drop=True),
        "edge_deltas": edge_delta_df.reset_index(drop=True),
    }

    return (
        week_index_df,
        weekly_summary_df.reset_index(drop=True),
        animation_payload,
        node_catalog_df.reset_index(drop=True),
        edge_catalog_df.reset_index(drop=True),
        temporal_deltas_df,
    )
