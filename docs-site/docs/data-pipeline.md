# Data Pipeline

## Input

Expected default input CSV:

- `tweets_01-08-2021.csv`

Configured by:

- TOML: `build.input_csv`
- env: `TG_BUILD_INPUT_CSV`
- CLI: `--input`

## Preprocessing Rules

`src/trump_graph/preprocess.py` handles deterministic preprocessing:

- parse `date` with coercion
- drop rows with invalid date or empty text
- include retweets by default (configurable)
- unescape HTML entities in text
- extract mentions with `@([A-Za-z0-9_]{1,15})`
- normalize mentions to lowercase
- dedupe mentions inside a tweet
- add ISO week metadata (`week_id`, `week_start`, `week_end`)

## Weekly Artifact Build

For each week:

- build undirected weighted co-mention graph
- node weight = tweets mentioning account
- edge weight = tweets where account pair co-appears
- compute metrics
- write:
  - `weeks/<week_id>/nodes.csv`
  - `weeks/<week_id>/edges.csv`
  - `weeks/<week_id>/metrics.json`

Global index outputs:

- `week_index.csv`
- `weekly_summary.csv`

## Global Animation Artifact

`global_animation/animation_state.json` includes:

- fixed global nodes and edges
- deterministic positions
- sparse weekly node and edge deltas
- normalization metadata

This supports stable placement across time while letting visual state evolve week by week.
