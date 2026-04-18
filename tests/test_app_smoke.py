from __future__ import annotations

from pathlib import Path

from trump_graph.app import build_pyvis_html, load_week_artifacts, load_week_index
from trump_graph.pipeline import build_weekly_artifacts


def test_app_helpers_load_and_render(sample_tweets_csv_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    build_weekly_artifacts(
        input_csv=sample_tweets_csv_path,
        output_dir=output_dir,
        min_mention_count=1,
        include_retweets=True,
    )

    week_index = load_week_index(output_dir)
    assert not week_index.empty

    selected_week = str(week_index.iloc[0]["week_id"])
    nodes_df, edges_df, metrics = load_week_artifacts(output_dir, selected_week)
    html, displayed_edges, total_edges = build_pyvis_html(nodes_df, edges_df, max_edges=10)

    assert "<html" in html.lower()
    assert displayed_edges <= total_edges
    assert "tweets_processed" in metrics
    assert "unique_mentions" in metrics
