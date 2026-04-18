from __future__ import annotations

from pathlib import Path

from trump_graph.app import (
    build_global_animation_html,
    load_global_animation_artifacts,
    load_week_artifacts,
    load_week_index,
)
from trump_graph.pipeline import build_weekly_artifacts


def test_app_helpers_load_and_render(sample_tweets_csv_path: Path, local_temp_dir: Path) -> None:
    output_dir = local_temp_dir / "processed"
    build_weekly_artifacts(
        input_csv=sample_tweets_csv_path,
        output_dir=output_dir,
        min_mention_count=1,
        include_retweets=True,
        global_min_mentions=1,
    )

    week_index = load_week_index(output_dir)
    assert not week_index.empty

    selected_week = str(week_index.iloc[0]["week_id"])
    nodes_df, edges_df, metrics = load_week_artifacts(output_dir, selected_week)
    assert "tweets_processed" in metrics
    assert "unique_mentions" in metrics
    assert len(nodes_df.columns) == 2
    assert len(edges_df.columns) == 3

    payload = load_global_animation_artifacts(output_dir)
    html = build_global_animation_html(
        payload=payload,
        include_hub=True,
        always_label_top_nodes=True,
        initial_week_index=0,
        initial_speed=2.0,
    )

    assert "tg-week-slider" in html
    assert "vis-network" in html
    assert "global_nodes" in html
