from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .app import load_unified_animation_artifacts
from .forecast import build_baseline_forecast_payload, build_forecast_payload_with_future_deltas
from .io import ensure_directory


SUPPORTED_FORECAST_MODELS = ("baseline", "tgn", "evolvegcn", "gconvgru")


@dataclass(frozen=True)
class ForecastTrainingConfig:
    model_key: str
    semantic_input_dir: Path
    output_root_dir: Path
    horizon_weeks: int
    lookback_weeks: int
    validation_weeks: int
    epochs: int
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    device: str
    seed: int


@dataclass(frozen=True)
class ForecastTrainingStats:
    model_key: str
    output_dir: Path
    weeks_seen: int
    nodes_seen: int
    edges_seen: int
    horizon_weeks: int
    validation_weeks: int
    epochs: int
    train_seconds: float
    best_val_loss: float
    mae: float
    rmse: float
    device_used: str


def _load_temporal_layers() -> tuple[type[Any], type[Any]]:
    try:
        from torch_geometric_temporal.nn.recurrent import EvolveGCNH, GConvGRU
    except ImportError as error:
        raise RuntimeError(
            "torch-geometric-temporal is required for evolvegcn/gconvgru training. "
            "Install requirements-ml.txt plus PyG wheels for your Torch/CUDA combination."
        ) from error
    return EvolveGCNH, GConvGRU


@dataclass(frozen=True)
class _TemporalSeries:
    node_ids: list[str]
    edge_ids: list[str]
    week_records: list[dict[str, Any]]
    node_week_matrix: np.ndarray
    edge_week_matrix: np.ndarray
    edge_source_index: np.ndarray
    edge_target_index: np.ndarray
    edge_affinity: np.ndarray
    edge_weight_for_gnn: np.ndarray


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    requested = str(device).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device value: {device!r}")


def _matrix_from_sparse_deltas(
    entity_ids: list[str],
    week_deltas: list[list[list[Any]]],
) -> np.ndarray:
    entity_index = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    matrix = np.zeros((len(week_deltas), len(entity_ids)), dtype=np.float32)
    for week_idx, week_entries in enumerate(week_deltas):
        for entity_id_raw, delta_raw in week_entries:
            entity_id = str(entity_id_raw)
            idx = entity_index.get(entity_id)
            if idx is None:
                continue
            delta_value = float(delta_raw)
            if delta_value <= 0:
                continue
            matrix[week_idx, idx] += delta_value
    return matrix


def _build_temporal_series(payload: Mapping[str, Any]) -> _TemporalSeries:
    week_records = [dict(week) for week in payload.get("weeks", [])]
    if not week_records:
        raise ValueError("Unified semantic payload does not contain any weeks.")

    global_nodes = [dict(node) for node in payload.get("global_nodes", [])]
    global_edges = [dict(edge) for edge in payload.get("global_edges", [])]
    node_ids = [str(node.get("id")) for node in global_nodes]
    all_edge_ids = [str(edge.get("id")) for edge in global_edges]
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edge_index_map = {edge_id: idx for idx, edge_id in enumerate(all_edge_ids)}

    node_week_deltas = payload.get("node_week_deltas", [])
    edge_week_deltas = payload.get("edge_week_deltas", [])
    if len(node_week_deltas) != len(week_records) or len(edge_week_deltas) != len(week_records):
        raise ValueError("Weekly deltas are inconsistent with week records in semantic payload.")

    node_week_matrix = _matrix_from_sparse_deltas(node_ids, node_week_deltas)
    edge_week_matrix_all = _matrix_from_sparse_deltas(all_edge_ids, edge_week_deltas)

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_ids: list[str] = []
    edge_week_columns: list[np.ndarray] = []
    edge_weights: list[float] = []
    edge_affinity: list[float] = []
    node_totals = np.maximum(node_week_matrix.sum(axis=0), 1.0)

    for edge in global_edges:
        source_id = str(edge.get("source"))
        target_id = str(edge.get("target"))
        source_idx = node_index.get(source_id)
        target_idx = node_index.get(target_id)
        if source_idx is None or target_idx is None:
            continue
        edge_id = str(edge.get("id"))
        edge_idx = edge_index_map.get(edge_id)
        if edge_idx is None:
            continue
        edge_total = float(edge_week_matrix_all[:, edge_idx].sum())
        source_total = float(node_totals[source_idx])
        target_total = float(node_totals[target_idx])
        pair_scale = math.sqrt(max(1.0, source_total) * max(1.0, target_total))
        affinity = edge_total / pair_scale if pair_scale > 0 else 0.0
        edge_ids.append(edge_id)
        edge_week_columns.append(edge_week_matrix_all[:, edge_idx])
        edge_sources.append(source_idx)
        edge_targets.append(target_idx)
        edge_weights.append(float(edge.get("total_co_occurrences", edge_total)))
        edge_affinity.append(float(np.clip(affinity, 0.0, 3.0)))

    if not edge_sources:
        raise ValueError("Unified semantic payload has no usable edges for training.")

    edge_weight_array = np.asarray(edge_weights, dtype=np.float32)
    max_weight = float(np.max(edge_weight_array)) if edge_weight_array.size else 1.0
    if max_weight <= 0:
        max_weight = 1.0
    edge_week_matrix = np.stack(edge_week_columns, axis=1).astype(np.float32)

    return _TemporalSeries(
        node_ids=node_ids,
        edge_ids=edge_ids,
        week_records=week_records,
        node_week_matrix=node_week_matrix,
        edge_week_matrix=edge_week_matrix,
        edge_source_index=np.asarray(edge_sources, dtype=np.int64),
        edge_target_index=np.asarray(edge_targets, dtype=np.int64),
        edge_affinity=np.asarray(edge_affinity, dtype=np.float32),
        edge_weight_for_gnn=(edge_weight_array / max_weight).astype(np.float32),
    )


def _build_graph_tensors(series: _TemporalSeries, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.as_tensor(series.edge_source_index, dtype=torch.long, device=device)
    dst = torch.as_tensor(series.edge_target_index, dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(series.edge_weight_for_gnn, dtype=torch.float32, device=device)

    edge_index = torch.stack(
        (
            torch.cat((src, dst), dim=0),
            torch.cat((dst, src), dim=0),
        ),
        dim=0,
    )
    edge_weight_undirected = torch.cat((edge_weight, edge_weight), dim=0)

    node_count = len(series.node_ids)
    adjacency = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight_undirected,
        size=(node_count, node_count),
        device=device,
    ).coalesce()

    row_sum = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1e-6)
    inv_row_sum = 1.0 / row_sum
    normalized_values = adjacency.values() * inv_row_sum[adjacency.indices()[0]]
    normalized_adjacency = torch.sparse_coo_tensor(
        indices=adjacency.indices(),
        values=normalized_values,
        size=adjacency.size(),
        device=device,
    ).coalesce()

    return edge_index, edge_weight_undirected, normalized_adjacency


class _TemporalNodeForecaster(nn.Module):
    def reset_temporal_state(self) -> None:
        return None

    def init_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor | None:
        return None

    def forward_step(
        self,
        x_t: torch.Tensor,
        *,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError


class _TGNLikeForecaster(_TemporalNodeForecaster):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_size=2, hidden_size=hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def init_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((num_nodes, self.hidden_dim), device=device, dtype=torch.float32)

    def forward_step(
        self,
        x_t: torch.Tensor,
        *,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            raise ValueError("Hidden state is required for TGN-like forecaster.")
        neighborhood = torch.sparse.mm(adjacency, x_t.unsqueeze(-1)).squeeze(-1)
        features = torch.stack((x_t, neighborhood), dim=-1)
        next_hidden = self.gru(features, hidden)
        prediction = self.readout(next_hidden).squeeze(-1)
        return prediction, next_hidden


class _EvolveGCNForecaster(_TemporalNodeForecaster):
    def __init__(self, *, num_nodes: int, hidden_dim: int) -> None:
        super().__init__()
        evolve_cls, _ = _load_temporal_layers()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.cell = evolve_cls(num_of_nodes=num_nodes, in_channels=hidden_dim)
        self.readout = nn.Linear(hidden_dim, 1)

    def reset_temporal_state(self) -> None:
        if hasattr(self.cell, "reinitialize_weight"):
            self.cell.reinitialize_weight()

    def forward_step(
        self,
        x_t: torch.Tensor,
        *,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        x = self.input_proj(x_t.unsqueeze(-1))
        x = torch.relu(x)
        x = self.cell(x, edge_index, edge_weight)
        prediction = self.readout(x).squeeze(-1)
        return prediction, None


class _GConvGRUForecaster(_TemporalNodeForecaster):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        _, gconvgru_cls = _load_temporal_layers()
        self.cell = gconvgru_cls(in_channels=1, out_channels=hidden_dim, K=2)
        self.readout = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def init_hidden(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((num_nodes, self.hidden_dim), device=device, dtype=torch.float32)

    def forward_step(
        self,
        x_t: torch.Tensor,
        *,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state = self.cell(x_t.unsqueeze(-1), edge_index, edge_weight, H=hidden)
        prediction = self.readout(hidden_state).squeeze(-1)
        return prediction, hidden_state


def _build_model(
    *,
    model_key: str,
    num_nodes: int,
    hidden_dim: int,
    device: torch.device,
) -> _TemporalNodeForecaster:
    if model_key == "tgn":
        model = _TGNLikeForecaster(hidden_dim=hidden_dim)
    elif model_key == "evolvegcn":
        model = _EvolveGCNForecaster(num_nodes=num_nodes, hidden_dim=hidden_dim)
    elif model_key == "gconvgru":
        model = _GConvGRUForecaster(hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
    return model.to(device)


def _weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = 1.0 + (target > 0).float() * 2.5
    return torch.mean(weights * (prediction - target) ** 2)


def _run_sequence_pass(
    *,
    model: _TemporalNodeForecaster,
    x_sequence: torch.Tensor,
    train_indices: range,
    adjacency: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.reset_temporal_state()
    hidden = model.init_hidden(num_nodes=x_sequence.shape[1], device=x_sequence.device)
    losses: list[float] = []

    for week_idx in train_indices:
        x_t = x_sequence[week_idx]
        y_t = x_sequence[week_idx + 1]
        prediction, hidden = model.forward_step(
            x_t,
            adjacency=adjacency,
            edge_index=edge_index,
            edge_weight=edge_weight,
            hidden=hidden,
        )
        loss = _weighted_mse_loss(prediction, y_t)
        losses.append(float(loss.detach().cpu().item()))

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if hidden is not None:
                hidden = hidden.detach()
    return float(np.mean(losses)) if losses else float("nan")


def _predict_future_node_counts(
    *,
    model: _TemporalNodeForecaster,
    x_sequence: torch.Tensor,
    horizon_weeks: int,
    adjacency: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> np.ndarray:
    model.eval()
    model.reset_temporal_state()
    hidden = model.init_hidden(num_nodes=x_sequence.shape[1], device=x_sequence.device)

    with torch.no_grad():
        for week_idx in range(x_sequence.shape[0] - 1):
            _, hidden = model.forward_step(
                x_sequence[week_idx],
                adjacency=adjacency,
                edge_index=edge_index,
                edge_weight=edge_weight,
                hidden=hidden,
            )
        current_x = x_sequence[-1]
        future_rows: list[np.ndarray] = []
        max_log_value = float(torch.max(x_sequence).item()) + 1.75
        for _step in range(horizon_weeks):
            prediction, hidden = model.forward_step(
                current_x,
                adjacency=adjacency,
                edge_index=edge_index,
                edge_weight=edge_weight,
                hidden=hidden,
            )
            prediction = torch.clamp(prediction, min=0.0, max=max_log_value)
            future_rows.append(prediction.cpu().numpy())
            current_x = prediction

    future_log = np.vstack(future_rows).astype(np.float32)
    return np.expm1(future_log).clip(min=0.0)


def _project_edge_counts(
    *,
    future_node_counts: np.ndarray,
    source_index: np.ndarray,
    target_index: np.ndarray,
    edge_affinity: np.ndarray,
) -> np.ndarray:
    horizon_weeks = future_node_counts.shape[0]
    edge_count = source_index.shape[0]
    projected = np.zeros((horizon_weeks, edge_count), dtype=np.float32)
    for step in range(horizon_weeks):
        source_values = future_node_counts[step, source_index]
        target_values = future_node_counts[step, target_index]
        pair_values = np.sqrt(np.maximum(0.0, source_values) * np.maximum(0.0, target_values))
        projected[step] = np.maximum(0.0, pair_values * edge_affinity)
    return projected


def _dense_to_sparse_weekly(
    values: np.ndarray,
    entity_ids: list[str],
) -> list[list[list[Any]]]:
    sparse: list[list[list[Any]]] = []
    for week_values in values:
        week_entries: list[list[Any]] = []
        for idx, value in enumerate(week_values):
            count_value = int(round(float(value)))
            if count_value > 0:
                week_entries.append([entity_ids[idx], count_value])
        sparse.append(week_entries)
    return sparse


def _validation_metrics(
    *,
    model: _TemporalNodeForecaster,
    x_sequence: torch.Tensor,
    validation_indices: range,
    adjacency: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[float, float, float]:
    model.eval()
    model.reset_temporal_state()
    hidden = model.init_hidden(num_nodes=x_sequence.shape[1], device=x_sequence.device)

    mae_values: list[float] = []
    rmse_values: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for week_idx in validation_indices:
            x_t = x_sequence[week_idx]
            y_t = x_sequence[week_idx + 1]
            prediction, hidden = model.forward_step(
                x_t,
                adjacency=adjacency,
                edge_index=edge_index,
                edge_weight=edge_weight,
                hidden=hidden,
            )
            losses.append(float(_weighted_mse_loss(prediction, y_t).cpu().item()))
            pred_counts = torch.expm1(torch.clamp(prediction, min=0.0))
            true_counts = torch.expm1(y_t)
            abs_error = torch.abs(pred_counts - true_counts)
            mae_values.append(float(torch.mean(abs_error).cpu().item()))
            rmse_values.append(float(torch.sqrt(torch.mean((pred_counts - true_counts) ** 2)).cpu().item()))
            if hidden is not None:
                hidden = hidden.detach()

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    mean_mae = float(np.mean(mae_values)) if mae_values else float("nan")
    mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
    return mean_loss, mean_mae, mean_rmse


def _write_model_outputs(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    graph_dir = ensure_directory(output_dir / "forecast_graph")
    (graph_dir / "animation_state.json").write_text(
        json.dumps(payload, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def train_forecast_model(config: ForecastTrainingConfig) -> ForecastTrainingStats:
    if config.model_key not in SUPPORTED_FORECAST_MODELS:
        raise ValueError(f"Unsupported forecast model key: {config.model_key}")
    if config.horizon_weeks < 1:
        raise ValueError("horizon_weeks must be >= 1")
    if config.lookback_weeks < 1:
        raise ValueError("lookback_weeks must be >= 1")
    if config.validation_weeks < 1:
        raise ValueError("validation_weeks must be >= 1")
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.hidden_dim < 4:
        raise ValueError("hidden_dim must be >= 4")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")

    semantic_payload = load_unified_animation_artifacts(config.semantic_input_dir)
    output_dir = ensure_directory(config.output_root_dir / config.model_key)

    if config.model_key == "baseline":
        baseline_payload = build_baseline_forecast_payload(
            semantic_payload,
            horizon_weeks=config.horizon_weeks,
            lookback_weeks=config.lookback_weeks,
        )
        metrics = {
            "model_key": "baseline",
            "generated_at_utc": datetime.now(tz=UTC).isoformat(),
            "horizon_weeks": int(config.horizon_weeks),
            "lookback_weeks": int(config.lookback_weeks),
            "validation_weeks": int(config.validation_weeks),
        }
        _write_model_outputs(output_dir=output_dir, payload=baseline_payload, metrics=metrics)
        return ForecastTrainingStats(
            model_key="baseline",
            output_dir=output_dir,
            weeks_seen=int(len(semantic_payload.get("weeks", []))),
            nodes_seen=int(len(semantic_payload.get("global_nodes", []))),
            edges_seen=int(len(semantic_payload.get("global_edges", []))),
            horizon_weeks=int(config.horizon_weeks),
            validation_weeks=int(config.validation_weeks),
            epochs=0,
            train_seconds=0.0,
            best_val_loss=float("nan"),
            mae=float("nan"),
            rmse=float("nan"),
            device_used="n/a",
        )

    _set_seed(config.seed)
    device = _resolve_device(config.device)
    series = _build_temporal_series(semantic_payload)

    if len(series.week_records) < (config.validation_weeks + 3):
        raise ValueError(
            "Not enough historical weeks for training + validation. "
            f"Weeks available: {len(series.week_records)}."
        )

    edge_index, edge_weight, adjacency = _build_graph_tensors(series, device)
    x_sequence_np = np.log1p(series.node_week_matrix).astype(np.float32)
    x_sequence = torch.as_tensor(x_sequence_np, dtype=torch.float32, device=device)

    last_train_week = len(series.week_records) - config.validation_weeks - 1
    train_indices = range(0, max(1, last_train_week))
    validation_indices = range(last_train_week, len(series.week_records) - 1)

    model = _build_model(
        model_key=config.model_key,
        num_nodes=len(series.node_ids),
        hidden_dim=config.hidden_dim,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_start_ts = time.perf_counter()
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_mae = float("inf")
    best_rmse = float("inf")

    for _epoch in range(config.epochs):
        model.train()
        _run_sequence_pass(
            model=model,
            x_sequence=x_sequence,
            train_indices=train_indices,
            adjacency=adjacency,
            edge_index=edge_index,
            edge_weight=edge_weight,
            optimizer=optimizer,
        )
        val_loss, val_mae, val_rmse = _validation_metrics(
            model=model,
            x_sequence=x_sequence,
            validation_indices=validation_indices,
            adjacency=adjacency,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae = val_mae
            best_rmse = val_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    future_node_counts = _predict_future_node_counts(
        model=model,
        x_sequence=x_sequence,
        horizon_weeks=config.horizon_weeks,
        adjacency=adjacency,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    future_edge_counts = _project_edge_counts(
        future_node_counts=future_node_counts,
        source_index=series.edge_source_index,
        target_index=series.edge_target_index,
        edge_affinity=series.edge_affinity,
    )
    future_node_deltas = _dense_to_sparse_weekly(future_node_counts, series.node_ids)
    future_edge_deltas = _dense_to_sparse_weekly(future_edge_counts, series.edge_ids)

    forecast_payload = build_forecast_payload_with_future_deltas(
        semantic_payload,
        future_node_deltas=future_node_deltas,
        future_edge_deltas=future_edge_deltas,
        backend=config.model_key,
        source="trained_on_unified_semantic",
        model_meta={
            "epochs": int(config.epochs),
            "hidden_dim": int(config.hidden_dim),
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "validation_weeks": int(config.validation_weeks),
            "seed": int(config.seed),
        },
    )

    train_seconds = float(time.perf_counter() - train_start_ts)
    metrics = {
        "model_key": config.model_key,
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "device_used": str(device),
        "weeks_seen": int(len(series.week_records)),
        "nodes_seen": int(len(series.node_ids)),
        "edges_seen": int(len(series.edge_ids)),
        "horizon_weeks": int(config.horizon_weeks),
        "lookback_weeks": int(config.lookback_weeks),
        "validation_weeks": int(config.validation_weeks),
        "epochs": int(config.epochs),
        "best_val_score": float(best_val_loss),
        "mae": float(best_mae),
        "rmse": float(best_rmse),
        "train_seconds": float(round(train_seconds, 3)),
    }
    _write_model_outputs(output_dir=output_dir, payload=forecast_payload, metrics=metrics)

    return ForecastTrainingStats(
        model_key=config.model_key,
        output_dir=output_dir,
        weeks_seen=int(len(series.week_records)),
        nodes_seen=int(len(series.node_ids)),
        edges_seen=int(len(series.edge_ids)),
        horizon_weeks=int(config.horizon_weeks),
        validation_weeks=int(config.validation_weeks),
        epochs=int(config.epochs),
        train_seconds=train_seconds,
        best_val_loss=float(best_val_loss),
        mae=float(best_mae),
        rmse=float(best_rmse),
        device_used=str(device),
    )
