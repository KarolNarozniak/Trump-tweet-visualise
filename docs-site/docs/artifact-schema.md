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
