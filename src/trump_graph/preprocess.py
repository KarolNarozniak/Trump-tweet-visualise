from __future__ import annotations

import html
import re

import pandas as pd

MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_]{1,15})")


def extract_mentions(text: str) -> list[str]:
    unescaped_text = html.unescape(text)
    raw_mentions = [match.group(1).lower() for match in MENTION_PATTERN.finditer(unescaped_text)]

    seen: set[str] = set()
    deduped_mentions: list[str] = []
    for mention in raw_mentions:
        if mention not in seen:
            seen.add(mention)
            deduped_mentions.append(mention)
    return deduped_mentions


def add_week_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    iso_calendar = out["date"].dt.isocalendar()

    out["iso_year"] = iso_calendar["year"].astype(int)
    out["iso_week"] = iso_calendar["week"].astype(int)
    out["week_id"] = out["iso_year"].astype(str) + "-W" + out["iso_week"].map("{:02d}".format)
    out["week_start"] = (out["date"] - pd.to_timedelta(out["date"].dt.weekday, unit="D")).dt.normalize()
    out["week_end"] = out["week_start"] + pd.Timedelta(days=6)
    return out


def _is_non_empty_text(values: pd.Series) -> pd.Series:
    return values.fillna("").str.strip().ne("")


def prepare_tweets(df: pd.DataFrame, include_retweets: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    valid_mask = out["date"].notna() & _is_non_empty_text(out["text"])
    out = out.loc[valid_mask].copy()
    out["text"] = out["text"].str.strip()

    if not include_retweets:
        out = out.loc[out["isRetweet"] != "t"].copy()

    out["mentions"] = out["text"].map(extract_mentions)
    out = add_week_columns(out)
    out = out.sort_values(["date", "id"], kind="mergesort").reset_index(drop=True)
    return out
