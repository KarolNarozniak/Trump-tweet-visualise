from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from .app import load_unified_animation_artifacts
from .forecast_registry import FORECAST_MODEL_REGISTRY
from .io import ensure_directory


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _artifact_summary(artifact_path: Path) -> dict[str, int]:
    payload = _load_json(artifact_path)
    if not payload:
        return {"weeks": 0, "nodes": 0, "edges": 0, "forecast_weeks": 0}
    weeks = payload.get("weeks", [])
    forecast_weeks = 0
    for week in weeks:
        if isinstance(week, dict) and bool(week.get("is_forecast", False)):
            forecast_weeks += 1
    return {
        "weeks": int(len(weeks)),
        "nodes": int(len(payload.get("global_nodes", []))),
        "edges": int(len(payload.get("global_edges", []))),
        "forecast_weeks": int(forecast_weeks),
    }


def _semantic_source_summary(semantic_source_dir: Path | None) -> dict[str, int]:
    if semantic_source_dir is None:
        return {"weeks": 0, "nodes": 0, "edges": 0}
    try:
        payload = load_unified_animation_artifacts(semantic_source_dir)
    except (FileNotFoundError, ValueError):
        return {"weeks": 0, "nodes": 0, "edges": 0}
    return {
        "weeks": int(len(payload.get("weeks", []))),
        "nodes": int(len(payload.get("global_nodes", []))),
        "edges": int(len(payload.get("global_edges", []))),
    }


def _yaml_front_matter(
    *,
    model_name: str,
    model_key: str,
    metrics: dict[str, Any],
) -> str:
    mae = _safe_float(metrics.get("mae"))
    rmse = _safe_float(metrics.get("rmse"))
    best_val = _safe_float(metrics.get("best_val_score"))
    lines = [
        "---",
        "license: mit",
        "library_name: pytorch",
        "tags:",
        "- temporal-graph",
        "- graph-forecasting",
        "- trump-graph",
        f"- model:{model_key}",
        "datasets:",
        "- trump-graph-unified-semantic",
    ]
    if mae is not None or rmse is not None or best_val is not None:
        lines.extend(
            [
                "model-index:",
                f"- name: {model_name}",
                "  results:",
                "  - task:",
                "      type: graph-forecasting",
                "      name: Weekly node activity forecast",
                "    dataset:",
                "      type: trump-graph-unified-semantic",
                "      name: Trump Graph Unified Semantic Timeline",
                "    metrics:",
            ]
        )
        if best_val is not None:
            lines.extend(
                [
                    "    - type: best_val_score",
                    f"      value: {best_val:.6f}",
                ]
            )
        if mae is not None:
            lines.extend(
                [
                    "    - type: mae",
                    f"      value: {mae:.6f}",
                ]
            )
        if rmse is not None:
            lines.extend(
                [
                    "    - type: rmse",
                    f"      value: {rmse:.6f}",
                ]
            )
    lines.append("---")
    return "\n".join(lines)


def _model_card_body(
    *,
    model: dict[str, str],
    status: str,
    artifact_path: Path,
    metrics_path: Path,
    summary: dict[str, int],
    metrics: dict[str, Any],
) -> str:
    generated_at_utc = datetime.now(tz=UTC).isoformat()
    best_val = _safe_float(metrics.get("best_val_score"))
    mae = _safe_float(metrics.get("mae"))
    rmse = _safe_float(metrics.get("rmse"))
    train_seconds = _safe_float(metrics.get("train_seconds"))
    epochs = metrics.get("epochs", "n/a")
    validation_weeks = metrics.get("validation_weeks", "n/a")
    device_used = metrics.get("device_used", "n/a")
    horizon_weeks = metrics.get("horizon_weeks", "n/a")

    metrics_table = [
        "| Metric | Value |",
        "|---|---:|",
        f"| best_val_score | {f'{best_val:.6f}' if best_val is not None else 'n/a'} |",
        f"| mae | {f'{mae:.6f}' if mae is not None else 'n/a'} |",
        f"| rmse | {f'{rmse:.6f}' if rmse is not None else 'n/a'} |",
        f"| train_seconds | {f'{train_seconds:.3f}' if train_seconds is not None else 'n/a'} |",
    ]

    status_section = (
        "## Status\n\n"
        f"- status: **{status}**\n"
        f"- generated_at_utc: `{generated_at_utc}`\n"
        f"- artifact_path: `{artifact_path}`\n"
        f"- metrics_path: `{metrics_path}`\n"
    )
    if status != "ready":
        status_section += (
            "\n### Missing Artifact Checklist\n\n"
            "1. Train the model with `python -m trump_graph train-forecast --model "
            f"{model['key']} --device cuda`.\n"
            "2. Confirm `forecast_graph/animation_state.json` and `metrics.json` exist.\n"
            "3. Restart Streamlit if table status is cached.\n"
        )

    return (
        f"# {model['label']} Model Card\n\n"
        "## Model Details\n\n"
        f"- model_key: `{model['key']}`\n"
        f"- model_name: {model['name']}\n"
        f"- family: {model.get('family', 'n/a')}\n"
        f"- category: {model['category']}\n"
        f"- paper: {model['paper_url']}\n"
        f"- implementation_reference: {model['impl_url']}\n\n"
        "## Intended Use\n\n"
        "This model forecasts weekly semantic graph activity (node mentions and edge co-occurrence intensity) "
        "for exploratory temporal analysis in the Trump Graph educational project.\n\n"
        "## Out-of-Scope Use\n\n"
        "- not for factual or legal claims about individuals\n"
        "- not for policy or political decision automation\n"
        "- not calibrated as a causal inference model\n\n"
        "## Training and Evaluation Data\n\n"
        "- source: unified semantic timeline produced by `build-unified`\n"
        "- time granularity: weekly snapshots\n"
        f"- total_weeks_in_payload: {summary['weeks']}\n"
        f"- nodes_in_payload: {summary['nodes']}\n"
        f"- edges_in_payload: {summary['edges']}\n"
        f"- forecast_weeks_in_payload: {summary['forecast_weeks']}\n\n"
        "## Training Procedure\n\n"
        f"- epochs: {epochs}\n"
        f"- validation_weeks: {validation_weeks}\n"
        f"- horizon_weeks: {horizon_weeks}\n"
        f"- device_used: {device_used}\n\n"
        "## Evaluation Results\n\n"
        + "\n".join(metrics_table)
        + "\n\n"
        "## Limitations, Bias, and Risks\n\n"
        "- quality depends on upstream topic/entity labeling quality\n"
        "- sparse or noisy weeks can destabilize long-horizon predictions\n"
        "- model outputs represent learned temporal correlations, not ground truth intent\n\n"
        + status_section
    )


def write_forecast_model_cards(forecast_root: Path, semantic_source_dir: Path | None = None) -> list[Path]:
    forecast_root = ensure_directory(forecast_root)
    semantic_summary = _semantic_source_summary(semantic_source_dir)
    written_paths: list[Path] = []
    index_lines = [
        "# Forecast Model Cards",
        "",
        f"Generated at: {datetime.now(tz=UTC).isoformat()}",
        "",
        "| Model | Status | Artifact | Card |",
        "|---|---|---|---|",
    ]

    for model in FORECAST_MODEL_REGISTRY:
        model_dir = ensure_directory(forecast_root / model["key"])
        artifact_path = model_dir / "forecast_graph" / "animation_state.json"
        metrics_path = model_dir / "metrics.json"
        metrics = _load_json(metrics_path)
        status = "ready" if artifact_path.exists() else "missing"
        summary = _artifact_summary(artifact_path)
        if status != "ready":
            summary["weeks"] = int(max(summary["weeks"], semantic_summary["weeks"]))
            summary["nodes"] = int(max(summary["nodes"], semantic_summary["nodes"]))
            summary["edges"] = int(max(summary["edges"], semantic_summary["edges"]))
        card_path = model_dir / "MODEL_CARD.md"

        content = (
            _yaml_front_matter(model_name=model["name"], model_key=model["key"], metrics=metrics)
            + "\n\n"
            + _model_card_body(
                model=model,
                status=status,
                artifact_path=artifact_path,
                metrics_path=metrics_path,
                summary=summary,
                metrics=metrics,
            )
        )
        card_path.write_text(content, encoding="utf-8")
        written_paths.append(card_path)
        index_lines.append(
            f"| {model['label']} | {status} | `{artifact_path}` | `{card_path}` |"
        )

    index_path = forecast_root / "MODEL_CARDS_INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    written_paths.append(index_path)
    return written_paths
