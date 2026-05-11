from __future__ import annotations

import pandas as pd

from trump_graph.unified_preprocess import prepare_unified_posts


def test_prepare_unified_posts_merges_platforms_and_adds_week_columns(
    sample_tweets_csv_path,
    sample_truth_posts_path,
) -> None:
    tweets_df = pd.read_csv(sample_tweets_csv_path)
    truth_df = pd.read_csv(sample_truth_posts_path)

    combined = prepare_unified_posts(tweets_df, truth_df, include_retweets=True)
    assert set(combined["source_platform"].unique()) == {"twitter", "truth"}
    assert {"week_id", "week_start", "week_end", "mentions", "hashtags", "urls"}.issubset(combined.columns)
    assert combined["created_at_utc"].is_monotonic_increasing

    twitter_rows = combined.loc[combined["source_platform"] == "twitter"]
    assert len(twitter_rows) == 6
    assert twitter_rows["is_repost"].sum() == 1


def test_prepare_unified_posts_respects_exclude_retweets(
    sample_tweets_csv_path,
    sample_truth_posts_path,
) -> None:
    tweets_df = pd.read_csv(sample_tweets_csv_path)
    truth_df = pd.read_csv(sample_truth_posts_path)

    combined = prepare_unified_posts(tweets_df, truth_df, include_retweets=False)
    twitter_rows = combined.loc[combined["source_platform"] == "twitter"]
    assert len(twitter_rows) == 5
    assert twitter_rows["is_repost"].sum() == 0
