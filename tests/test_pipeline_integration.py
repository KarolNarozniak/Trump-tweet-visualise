from __future__ import annotations

from pathlib import Path

import pandas as pd

from trump_graph.pipeline import build_weekly_artifacts


def test_build_weekly_artifacts_writes_expected_outputs(sample_tweets_csv_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"

    stats = build_weekly_artifacts(
        input_csv=sample_tweets_csv_path,
        output_dir=output_dir,
        min_mention_count=1,
        include_retweets=True,
    )

    assert stats.total_tweets == 6
    assert stats.processed_tweets == 6
    assert stats.weeks_built == 2

    week_index_path = output_dir / "week_index.csv"
    summary_path = output_dir / "weekly_summary.csv"
    assert week_index_path.exists()
    assert summary_path.exists()

    week_index = pd.read_csv(week_index_path)
    assert {"week_id", "week_start", "tweets_processed", "unique_mentions", "edge_count"}.issubset(week_index.columns)

    week_ids = set(week_index["week_id"].tolist())
    assert {"2020-W53", "2021-W01"} == week_ids

    for week_id in week_ids:
        week_dir = output_dir / "weeks" / week_id
        assert (week_dir / "nodes.csv").exists()
        assert (week_dir / "edges.csv").exists()
        assert (week_dir / "metrics.json").exists()
