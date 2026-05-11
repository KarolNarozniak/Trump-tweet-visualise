# Temporal Forecast

The Forecast page is the frontend entrypoint for temporal neural network outputs.

## Endpoint

```text
http://localhost:3001/?page=forecast
```

## Modes

- `model`: load trained output from forecast artifacts
- `baseline`: local preview built from semantic history while training is in progress

## Expected Model Artifact

Path:

- `data/processed_forecast/forecast_graph/animation_state.json`

Required keys:

- `weeks`
- `global_nodes`
- `global_edges`
- `node_week_deltas`
- `edge_week_deltas`
- `heat_decay`
- `heat_scale`
- `max_cumulative_edge`

Optional key:

- `delta_sets` (if omitted, app uses top-level deltas as `all`)

## UX Behavior

- layout remains fixed between history and forecast
- future weeks are appended after historical weeks
- node heat indicates predicted weekly activation
- edge width shows cumulative predicted co-activation over time

## Why Baseline Exists

Baseline mode is intentionally simple. It lets you verify:

- graph rendering
- week extension mechanics
- export and playback flow

before plugging in trained model outputs from remote training runs.
