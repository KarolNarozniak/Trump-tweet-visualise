# Artifact Schema

Default output directory:

- `data/processed`

## Root Files

- `week_index.csv`
- `weekly_summary.csv`
- `global_animation/animation_state.json`

## Weekly Folder

For each `week_id`:

- `weeks/<week_id>/nodes.csv`
- `weeks/<week_id>/edges.csv`
- `weeks/<week_id>/metrics.json`

## `nodes.csv`

Columns:

- `node`
- `weight`

## `edges.csv`

Columns:

- `source`
- `target`
- `weight`

## `metrics.json`

Includes:

- week metadata
- tweets processed
- tweets with mentions
- unique mentions
- edge count
- graph density
- top mentions
- top weighted edges

## `animation_state.json`

Required keys:

- `weeks`
- `global_nodes`
- `global_edges`
- `node_week_deltas`
- `edge_week_deltas`
- `heat_decay`
- `heat_scale`
- `max_cumulative_edge`
- `hub_node_id`
- `top_label_nodes`

### `global_nodes`

- `id`
- `total_mentions`
- `size`
- `x`
- `y`
- `is_hub`

### `global_edges`

- `id`
- `source`
- `target`
- `total_co_mentions`

# Truth Social Semantic Artifacts

Default output directory:

- `data/processed_truth`

## Root Files

- `truth_week_index.csv`
- `truth_weekly_summary.csv`
- `truth_posts_enriched.parquet`

## Semantic Graph Folder

- `truth_semantic_graph/animation_state.json`
- `truth_semantic_graph/node_catalog.csv`

## Embeddings Folder

- `truth_embeddings/embeddings.npy`
- `truth_embeddings/index.csv`

## `truth_posts_enriched.parquet`

Important columns:

- `post_id`
- `created_at_utc`
- `week_id`
- `is_retruth`
- `topic_labels`
- `topic_scores`
- `entities`
- `sentiment_label`
- `sentiment_score`
- `semantic_nodes`
- `text`

List/dictionary-like fields are stored as compact JSON strings for stable export and app loading.

## Truth `animation_state.json`

Required keys:

- `weeks`
- `global_nodes`
- `global_edges`
- `delta_sets`
- `node_week_deltas`
- `edge_week_deltas`
- `available_node_types`
- `available_delta_sets`
- `heat_decay`
- `heat_scale`
- `max_cumulative_edge`

Delta sets include:

- `all`
- `original`
- `negative`
- `neutral`
- `positive`
- `original_negative`
- `original_neutral`
- `original_positive`
