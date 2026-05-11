from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any, Iterable, Mapping


WEEK_ID_PATTERN = re.compile(r"^(?P<year>\d{4})-W(?P<week>\d{2})$")


def _parse_week_start(week_record: Mapping[str, Any]) -> date:
    week_start_raw = str(week_record.get("week_start", "")).strip()
    if week_start_raw:
        return date.fromisoformat(week_start_raw)

    week_id = str(week_record.get("week_id", "")).strip()
    match = WEEK_ID_PATTERN.match(week_id)
    if match is None:
        raise ValueError(f"Unable to infer week start from week record: {week_record}")
    year = int(match.group("year"))
    week = int(match.group("week"))
    return date.fromisocalendar(year, week, 1)


def _future_week_records(weeks: list[Mapping[str, Any]], horizon_weeks: int) -> list[dict[str, Any]]:
    if not weeks or horizon_weeks < 1:
        return []
    last_week = weeks[-1]
    start_date = _parse_week_start(last_week)
    future_records: list[dict[str, Any]] = []
    for step in range(1, horizon_weeks + 1):
        week_start = start_date + timedelta(days=7 * step)
        week_end = week_start + timedelta(days=6)
        iso = week_start.isocalendar()
        future_records.append(
            {
                "week_id": f"{iso.year}-W{iso.week:02d}",
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "is_forecast": True,
                "forecast_step": int(step),
                "posts_processed": 0,
                "reposts": 0,
                "unique_nodes": 0,
                "edge_count": 0,
            }
        )
    return future_records


def _sparse_history_map(
    week_deltas: Iterable[Iterable[Iterable[Any]]],
) -> dict[str, list[tuple[int, int]]]:
    history: dict[str, list[tuple[int, int]]] = {}
    for week_index, week_entries in enumerate(week_deltas):
        for entity_id, delta_value in week_entries:
            entity_key = str(entity_id)
            delta = int(delta_value)
            if delta <= 0:
                continue
            history.setdefault(entity_key, []).append((week_index, delta))
    return history


def _window_sum(history: list[tuple[int, int]], *, start_week: int) -> int:
    total = 0
    for week_index, delta in reversed(history):
        if week_index < start_week:
            break
        total += int(delta)
    return total


def _last_week_value(history: list[tuple[int, int]], *, last_week_index: int) -> int:
    if not history:
        return 0
    week_index, delta = history[-1]
    return int(delta) if int(week_index) == int(last_week_index) else 0


def _predict_sparse_future(
    *,
    entity_ids: list[str],
    history_map: dict[str, list[tuple[int, int]]],
    week_count: int,
    horizon_weeks: int,
    lookback_weeks: int,
    decay: float,
    floor: int,
) -> list[list[list[Any]]]:
    future_entries_by_week: list[list[list[Any]]] = [[] for _ in range(horizon_weeks)]
    lookback_start = max(0, int(week_count) - int(lookback_weeks))
    for entity_id in entity_ids:
        history = history_map.get(entity_id, [])
        avg_value = _window_sum(history, start_week=lookback_start) / float(max(1, lookback_weeks))
        last_value = float(_last_week_value(history, last_week_index=max(0, week_count - 1)))
        base_value = max(0.0, (0.65 * avg_value) + (0.35 * last_value))
        if base_value <= 0:
            continue

        for step in range(horizon_weeks):
            predicted = int(round(base_value * (decay**step)))
            if predicted >= floor:
                future_entries_by_week[step].append([entity_id, predicted])
    return future_entries_by_week


def _max_heat(
    node_ids: list[str],
    node_week_deltas: list[list[list[Any]]],
    heat_decay: float,
) -> float:
    heat = {node_id: 0.0 for node_id in node_ids}
    peak = 0.0
    for week_entries in node_week_deltas:
        for node_id in node_ids:
            heat[node_id] *= heat_decay
        for node_id, delta in week_entries:
            key = str(node_id)
            heat[key] = heat.get(key, 0.0) + float(delta)
        peak = max(peak, max(heat.values(), default=0.0))
    return peak if peak > 0 else 1.0


def build_baseline_forecast_payload(
    semantic_payload: Mapping[str, Any],
    *,
    horizon_weeks: int,
    lookback_weeks: int = 12,
    node_decay: float = 0.92,
    edge_decay: float = 0.96,
    node_floor: int = 1,
    edge_floor: int = 1,
) -> dict[str, Any]:
    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be >= 1")
    if lookback_weeks < 1:
        raise ValueError("lookback_weeks must be >= 1")
    if not (0.0 < node_decay <= 1.0):
        raise ValueError("node_decay must be in the interval (0, 1]")
    if not (0.0 < edge_decay <= 1.0):
        raise ValueError("edge_decay must be in the interval (0, 1]")

    weeks = [dict(week) for week in semantic_payload.get("weeks", [])]
    node_week_deltas = [
        [[str(node_id), int(delta)] for node_id, delta in week_entries]
        for week_entries in semantic_payload.get("node_week_deltas", [])
    ]
    edge_week_deltas = [
        [[str(edge_id), int(delta)] for edge_id, delta in week_entries]
        for week_entries in semantic_payload.get("edge_week_deltas", [])
    ]
    node_ids = [str(node.get("id")) for node in semantic_payload.get("global_nodes", [])]
    edge_ids = [str(edge.get("id")) for edge in semantic_payload.get("global_edges", [])]

    week_count = len(weeks)
    if week_count == 0:
        raise ValueError("Semantic payload contains no weeks.")

    node_history = _sparse_history_map(node_week_deltas)
    edge_history = _sparse_history_map(edge_week_deltas)
    future_node_deltas = _predict_sparse_future(
        entity_ids=node_ids,
        history_map=node_history,
        week_count=week_count,
        horizon_weeks=horizon_weeks,
        lookback_weeks=lookback_weeks,
        decay=node_decay,
        floor=max(1, int(node_floor)),
    )
    future_edge_deltas = _predict_sparse_future(
        entity_ids=edge_ids,
        history_map=edge_history,
        week_count=week_count,
        horizon_weeks=horizon_weeks,
        lookback_weeks=lookback_weeks,
        decay=edge_decay,
        floor=max(1, int(edge_floor)),
    )
    future_weeks = _future_week_records(weeks, horizon_weeks)

    combined_weeks = weeks + future_weeks
    combined_node_deltas = node_week_deltas + future_node_deltas
    combined_edge_deltas = edge_week_deltas + future_edge_deltas
    heat_decay = float(semantic_payload.get("heat_decay", 0.85))
    max_cumulative_edge = int(semantic_payload.get("max_cumulative_edge", 1))
    if max_cumulative_edge <= 0:
        max_cumulative_edge = 1
    heat_scale = _max_heat(node_ids, combined_node_deltas, heat_decay)

    forecast_payload = dict(semantic_payload)
    forecast_payload["graph_kind"] = "temporal_forecast"
    forecast_payload["weeks"] = combined_weeks
    forecast_payload["node_week_deltas"] = combined_node_deltas
    forecast_payload["edge_week_deltas"] = combined_edge_deltas
    forecast_payload["delta_sets"] = {
        "all": {
            "node_week_deltas": combined_node_deltas,
            "edge_week_deltas": combined_edge_deltas,
        }
    }
    forecast_payload["heat_scale"] = round(float(heat_scale), 6)
    forecast_payload["max_cumulative_edge"] = int(max_cumulative_edge)
    forecast_payload["forecast_horizon_weeks"] = int(horizon_weeks)
    forecast_payload["forecast_backend"] = "baseline"
    forecast_payload["forecast_source"] = "generated_from_semantic_payload"
    return forecast_payload
