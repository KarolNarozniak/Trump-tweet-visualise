from __future__ import annotations

from collections import Counter
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
from .truth_semantics import json_dumps_compact, parse_json_list

DEFAULT_TRUTH_MIN_NODE_COUNT = 8
DEFAULT_TRUTH_TOP_LABEL_COUNT = 40
TRUTH_NODE_TYPES = ("topic", "per", "org", "loc", "hashtag", "mention")
TRUTH_SENTIMENTS = ("negative", "neutral", "positive")


def truth_node_type(node_id: str) -> str:
    return str(node_id).split("::", 1)[0].lower()


def truth_node_label(node_id: str) -> str:
    node_type, _, value = str(node_id).partition("::")
    if node_type == "topic":
        return value.replace("_", " ").title()
    if node_type == "hashtag":
        return f"#{value}"
    if node_type == "mention":
        return f"@{value}"
    return value.title()


def _semantic_nodes_from_value(value: Any) -> list[str]:
    return [str(node_id) for node_id in parse_json_list(value)] if isinstance(value, str) else [str(node_id) for node_id in value]


def _count_nodes_and_edges(nodes_by_post: Iterable[Sequence[str]]) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    for nodes in nodes_by_post:
        unique_nodes = tuple(dict.fromkeys(str(node) for node in nodes if str(node)))
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
) -> dict[str, Any]:
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
            [node for node in _semantic_nodes_from_value(nodes_value) if node in kept_node_set]
            for nodes_value in week_df["semantic_nodes"].tolist()
        ]
        node_counts, edge_counts = _count_nodes_and_edges(nodes_by_post)
        node_week_deltas.append([[node_id, int(count)] for node_id, count in sorted(node_counts.items()) if count > 0])

        edge_entries = []
        for (source, target), count in sorted(edge_counts.items()):
            edge_id = f"{source}|{target}"
            if edge_id in edge_id_set and count > 0:
                edge_entries.append([edge_id, int(count)])
        edge_week_deltas.append(edge_entries)

    return {
        "node_week_deltas": node_week_deltas,
        "edge_week_deltas": edge_week_deltas,
    }


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


def _week_frames(enriched_df: pd.DataFrame, kept_node_set: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    week_index_rows: list[dict[str, Any]] = []
    weekly_summary_rows: list[dict[str, Any]] = []

    for week_id, week_df in enriched_df.groupby("week_id", sort=False):
        nodes_by_post = [
            [node for node in _semantic_nodes_from_value(nodes_value) if node in kept_node_set]
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
            "retruths": int(week_df["is_retruth"].sum()),
            "original_posts": int((~week_df["is_retruth"]).sum()),
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

    week_index_df = pd.DataFrame(week_index_rows).sort_values(["week_start", "week_id"], kind="mergesort")
    weekly_summary_df = pd.DataFrame(weekly_summary_rows).sort_values(["week_start", "week_id"], kind="mergesort")
    return week_index_df.reset_index(drop=True), weekly_summary_df.reset_index(drop=True)


def build_truth_semantic_artifacts(
    enriched_df: pd.DataFrame,
    *,
    min_node_count: int = DEFAULT_TRUTH_MIN_NODE_COUNT,
    included_node_types: Iterable[str] = TRUTH_NODE_TYPES,
    heat_decay: float = DEFAULT_HEAT_DECAY,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if min_node_count < 1:
        raise ValueError("min_node_count must be >= 1")
    if not (0.0 < heat_decay < 1.0):
        raise ValueError("heat_decay must be in the open interval (0, 1)")

    included_types = {node_type.strip().lower() for node_type in included_node_types}
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

    week_index_df, weekly_summary_df = _week_frames(enriched_df, kept_node_set)
    week_ids = [str(week_id) for week_id in week_index_df["week_id"].tolist()]

    p99_counts = float(pd.Series([node_total_counts[node] for node in kept_nodes], dtype="float64").quantile(0.99))
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
            "total_mentions": int(node_total_counts[node_id]),
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
                "total_co_mentions": count,
            }
        )
    edge_id_set = {str(edge["id"]) for edge in global_edges}

    delta_sets = {
        "all": _build_delta_set(
            enriched_df=enriched_df,
            week_ids=week_ids,
            kept_node_set=kept_node_set,
            edge_id_set=edge_id_set,
        )
    }
    original_df = enriched_df.loc[~enriched_df["is_retruth"]].copy()
    delta_sets["original"] = _build_delta_set(
        enriched_df=original_df,
        week_ids=week_ids,
        kept_node_set=kept_node_set,
        edge_id_set=edge_id_set,
    )

    for sentiment in TRUTH_SENTIMENTS:
        sentiment_df = enriched_df.loc[enriched_df["sentiment_label"] == sentiment].copy()
        original_sentiment_df = original_df.loc[original_df["sentiment_label"] == sentiment].copy()
        delta_sets[sentiment] = _build_delta_set(
            enriched_df=sentiment_df,
            week_ids=week_ids,
            kept_node_set=kept_node_set,
            edge_id_set=edge_id_set,
        )
        delta_sets[f"original_{sentiment}"] = _build_delta_set(
            enriched_df=original_sentiment_df,
            week_ids=week_ids,
            kept_node_set=kept_node_set,
            edge_id_set=edge_id_set,
        )

    heat_scale = max(
        _max_heat_from_deltas(kept_nodes, delta_set["node_week_deltas"], heat_decay)
        for delta_set in delta_sets.values()
    )
    max_cumulative_edge = int(max((edge["total_co_occurrences"] for edge in global_edges), default=0))
    top_label_nodes = [
        node_id
        for node_id, _count in sorted(
            ((node_id, node_total_counts[node_id]) for node_id in kept_nodes),
            key=lambda pair: (-pair[1], pair[0]),
        )[:DEFAULT_TRUTH_TOP_LABEL_COUNT]
    ]
    week_records = week_index_df.to_dict(orient="records")

    animation_payload = {
        "version": 1,
        "graph_kind": "truth_semantic",
        "heat_decay": float(heat_decay),
        "min_node_count": int(min_node_count),
        "layout_seed": int(layout_seed),
        "heat_scale": round(float(heat_scale), 6),
        "max_cumulative_edge": max_cumulative_edge,
        "top_label_nodes": top_label_nodes,
        "available_node_types": list(TRUTH_NODE_TYPES),
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

    weekly_summary_df = weekly_summary_df.copy()
    for column in ["top_topic_node", "top_entity_node"]:
        weekly_summary_df[column] = weekly_summary_df[column].fillna("")

    top_nodes_by_week: dict[str, str] = {}
    for week_id, week_df in enriched_df.groupby("week_id", sort=False):
        week_nodes_by_post = [
            [node for node in _semantic_nodes_from_value(nodes_value) if node in kept_node_set]
            for nodes_value in week_df["semantic_nodes"].tolist()
        ]
        node_counts, _edge_counts = _count_nodes_and_edges(week_nodes_by_post)
        top_nodes_by_week[str(week_id)] = json_dumps_compact(
            [
                {"node": node_id, "label": truth_node_label(node_id), "count": int(count)}
                for node_id, count in node_counts.most_common(10)
            ]
        )
    weekly_summary_df["top_nodes_json"] = weekly_summary_df["week_id"].map(top_nodes_by_week).fillna("[]")

    return week_index_df, weekly_summary_df, animation_payload, node_catalog_df.reset_index(drop=True)
