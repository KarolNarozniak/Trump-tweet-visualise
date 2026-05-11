from __future__ import annotations

import pandas as pd

from .preprocess import prepare_tweets
from .truth_preprocess import (
    add_truth_week_columns,
    extract_truth_hashtags,
    extract_truth_urls,
    normalize_truth_text,
    prepare_truth_posts,
)

UNIFIED_OWNER_ID = "realdonaldtrump"
UNIFIED_COLUMNS: tuple[str, ...] = (
    "post_id",
    "platform_post_id",
    "source_platform",
    "owner_id",
    "text",
    "created_at_utc",
    "is_repost",
    "mentions",
    "hashtags",
    "urls",
    "text_length",
    "iso_year",
    "iso_week",
    "week_id",
    "week_start",
    "week_end",
)


def prepare_twitter_posts_for_unified(
    raw_tweets_df: pd.DataFrame,
    *,
    include_retweets: bool,
) -> pd.DataFrame:
    tweets_df = prepare_tweets(raw_tweets_df, include_retweets=include_retweets).copy()
    tweets_df["text"] = tweets_df["text"].map(normalize_truth_text)
    tweets_df["created_at_utc"] = pd.to_datetime(tweets_df["date"], errors="coerce", utc=True)
    tweets_df = tweets_df.loc[tweets_df["created_at_utc"].notna()].copy()

    tweets_df["post_id"] = "twitter::" + tweets_df["id"].astype("string").str.strip()
    tweets_df["platform_post_id"] = tweets_df["id"].astype("string").str.strip()
    tweets_df["source_platform"] = "twitter"
    tweets_df["owner_id"] = UNIFIED_OWNER_ID
    tweets_df["is_repost"] = tweets_df["isRetweet"].astype("string").str.lower().eq("t")
    tweets_df["hashtags"] = tweets_df["text"].map(extract_truth_hashtags)
    tweets_df["urls"] = tweets_df["text"].map(extract_truth_urls)
    tweets_df["text_length"] = tweets_df["text"].str.len().astype(int)

    tweets_df = add_truth_week_columns(tweets_df)
    standardized = tweets_df.loc[:, UNIFIED_COLUMNS].copy()
    return standardized.sort_values(["created_at_utc", "post_id"], kind="mergesort").reset_index(drop=True)


def prepare_truth_posts_for_unified(raw_truth_df: pd.DataFrame) -> pd.DataFrame:
    truth_df = prepare_truth_posts(raw_truth_df).copy()
    truth_df["post_id"] = "truth::" + truth_df["post_id"].astype("string").str.strip()
    truth_df["platform_post_id"] = truth_df["post_id"].astype("string").str.removeprefix("truth::")
    truth_df["source_platform"] = "truth"
    truth_df["is_repost"] = truth_df["is_retruth"].astype(bool)
    standardized = truth_df.loc[:, UNIFIED_COLUMNS].copy()
    return standardized.sort_values(["created_at_utc", "post_id"], kind="mergesort").reset_index(drop=True)


def prepare_unified_posts(
    raw_tweets_df: pd.DataFrame,
    raw_truth_df: pd.DataFrame,
    *,
    include_retweets: bool,
) -> pd.DataFrame:
    twitter_df = prepare_twitter_posts_for_unified(raw_tweets_df, include_retweets=include_retweets)
    truth_df = prepare_truth_posts_for_unified(raw_truth_df)
    combined = pd.concat([twitter_df, truth_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["post_id"], keep="first").copy()
    combined["created_at_utc"] = pd.to_datetime(combined["created_at_utc"], errors="coerce", utc=True)
    combined = combined.loc[combined["created_at_utc"].notna()].copy()
    combined = add_truth_week_columns(combined)
    return combined.loc[:, UNIFIED_COLUMNS].sort_values(["created_at_utc", "post_id"], kind="mergesort").reset_index(
        drop=True
    )
