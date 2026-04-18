from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

TWEET_DTYPES: dict[str, str] = {
    "id": "string",
    "text": "string",
    "isRetweet": "string",
    "isDeleted": "string",
    "device": "string",
    "favorites": "Int64",
    "retweets": "Int64",
    "date": "string",
    "isFlagged": "string",
}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_tweets_csv(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path, dtype=TWEET_DTYPES)


def write_week_artifacts(
    output_dir: Path,
    week_id: str,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    metrics: Mapping[str, Any],
) -> None:
    week_dir = ensure_directory(output_dir / "weeks" / week_id)
    nodes_df.to_csv(week_dir / "nodes.csv", index=False)
    edges_df.to_csv(week_dir / "edges.csv", index=False)
    (week_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_index_files(
    output_dir: Path,
    week_index_df: pd.DataFrame,
    weekly_summary_df: pd.DataFrame,
) -> None:
    ensure_directory(output_dir)
    week_index_df.to_csv(output_dir / "week_index.csv", index=False)
    weekly_summary_df.to_csv(output_dir / "weekly_summary.csv", index=False)


def write_global_animation_artifacts(
    output_dir: Path,
    animation_payload: Mapping[str, Any],
) -> None:
    animation_dir = ensure_directory(output_dir / "global_animation")
    (animation_dir / "animation_state.json").write_text(
        json.dumps(animation_payload, sort_keys=False),
        encoding="utf-8",
    )
