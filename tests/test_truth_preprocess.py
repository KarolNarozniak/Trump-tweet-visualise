from __future__ import annotations

import pandas as pd

from trump_graph.truth_preprocess import (
    decode_truth_created_at_utc,
    extract_truth_hashtags,
    extract_truth_mentions,
    is_retruth_text,
    prepare_truth_posts,
)


def test_decode_truth_created_at_utc_from_snowflake_id() -> None:
    decoded = decode_truth_created_at_utc("108221541018717248")
    assert decoded == pd.Timestamp("2022-04-30 14:41:06.423000+0000")


def test_truth_extractors_do_not_truncate_long_handles() -> None:
    text = "RT @realamericasvoice and @realamericasvoice #MAGA #MAGA"
    assert extract_truth_mentions(text) == ["realamericasvoice"]
    assert extract_truth_hashtags(text) == ["maga"]
    assert is_retruth_text(text) is True


def test_prepare_truth_posts_dedupes_and_buckets_weeks() -> None:
    df = pd.DataFrame(
        {
            "_id": ["108221541018717248", "108221541018717248", "108294954209768016"],
            "owner": ["1", "1", "1"],
            "text": ["Hello @LongTruthHandle", "Duplicate", "RT Something #News"],
        }
    )

    prepared = prepare_truth_posts(df)

    assert prepared["post_id"].tolist() == ["108221541018717248", "108294954209768016"]
    assert prepared["mentions"].tolist()[0] == ["longtruthhandle"]
    assert prepared["is_retruth"].tolist() == [False, True]
    assert prepared["week_id"].tolist() == ["2022-W17", "2022-W19"]
