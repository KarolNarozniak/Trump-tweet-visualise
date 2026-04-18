from __future__ import annotations

import pandas as pd

from trump_graph.preprocess import add_week_columns, extract_mentions, prepare_tweets


def test_extract_mentions_normalizes_html_and_dedupes() -> None:
    text = "Thanks @Alice, @alice, and @BOB &amp; @Bob!"
    assert extract_mentions(text) == ["alice", "bob"]


def test_add_week_columns_handles_year_boundary() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-12-31 23:00:00",
                    "2021-01-01 12:00:00",
                    "2021-01-04 09:00:00",
                ]
            )
        }
    )

    with_week = add_week_columns(df)
    assert with_week["week_id"].tolist() == ["2020-W53", "2020-W53", "2021-W01"]


def test_prepare_tweets_filters_invalid_rows_and_retweets() -> None:
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "text": ["Hello @Alice", "", "RT @Bob", "Hello @Carol"],
            "isRetweet": ["f", "f", "t", "f"],
            "isDeleted": ["f", "f", "f", "f"],
            "device": ["x", "x", "x", "x"],
            "favorites": [1, 2, 3, 4],
            "retweets": [1, 2, 3, 4],
            "date": ["2021-01-01 00:00:00", "2021-01-01 00:00:00", "2021-01-01 00:00:00", "bad-date"],
            "isFlagged": ["f", "f", "f", "f"],
        }
    )

    prepared = prepare_tweets(df, include_retweets=False)
    assert prepared["id"].tolist() == ["1"]
    assert prepared["mentions"].tolist() == [["alice"]]
