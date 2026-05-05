from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

TRUTH_MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_]{1,30})")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z][A-Za-z0-9_]{1,49})")
URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
RETRUTH_PATTERN = re.compile(r"^\s*(?:RT|ReTruth)\b", re.IGNORECASE)


def decode_truth_created_at_ms(post_id: int | str) -> int:
    return int(post_id) >> 16


def decode_truth_created_at_utc(post_id: int | str) -> pd.Timestamp:
    return pd.to_datetime(decode_truth_created_at_ms(post_id), unit="ms", utc=True)


def normalize_truth_text(text: Any) -> str:
    normalized = html.unescape("" if pd.isna(text) else str(text))
    return re.sub(r"\s+", " ", normalized).strip()


def is_retruth_text(text: str) -> bool:
    return bool(RETRUTH_PATTERN.search(text))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def extract_truth_mentions(text: str) -> list[str]:
    return _dedupe_preserve_order([match.group(1) for match in TRUTH_MENTION_PATTERN.finditer(text)])


def extract_truth_hashtags(text: str) -> list[str]:
    return _dedupe_preserve_order([match.group(1) for match in HASHTAG_PATTERN.finditer(text)])


def extract_truth_urls(text: str) -> list[str]:
    return _dedupe_preserve_order([match.group(0).rstrip(".,);]") for match in URL_PATTERN.finditer(text)])


def add_truth_week_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    iso_calendar = out["created_at_utc"].dt.isocalendar()
    out["iso_year"] = iso_calendar["year"].astype(int)
    out["iso_week"] = iso_calendar["week"].astype(int)
    out["week_id"] = out["iso_year"].astype(str) + "-W" + out["iso_week"].map("{:02d}".format)
    out["week_start"] = (
        out["created_at_utc"] - pd.to_timedelta(out["created_at_utc"].dt.weekday, unit="D")
    ).dt.normalize()
    out["week_end"] = out["week_start"] + pd.Timedelta(days=6)
    return out


def prepare_truth_posts(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"_id", "owner", "text"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Truth input is missing required columns: {', '.join(missing_columns)}")

    out = df.rename(columns={"_id": "post_id", "owner": "owner_id"}).copy()
    out["post_id"] = out["post_id"].astype("string").str.strip()
    out["owner_id"] = out["owner_id"].astype("string").str.strip()
    out["text"] = out["text"].map(normalize_truth_text)

    out = out.loc[out["post_id"].ne("") & out["text"].ne("")].copy()
    out = out.drop_duplicates(subset=["post_id"], keep="first").copy()

    out["created_at_utc"] = out["post_id"].map(decode_truth_created_at_utc)
    out["is_retruth"] = out["text"].map(is_retruth_text)
    out["mentions"] = out["text"].map(extract_truth_mentions)
    out["hashtags"] = out["text"].map(extract_truth_hashtags)
    out["urls"] = out["text"].map(extract_truth_urls)
    out["text_length"] = out["text"].str.len().astype(int)

    out = add_truth_week_columns(out)
    out = out.sort_values(["created_at_utc", "post_id"], kind="mergesort").reset_index(drop=True)
    return out
