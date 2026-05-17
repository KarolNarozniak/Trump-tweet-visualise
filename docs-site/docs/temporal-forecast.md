# Temporal Forecast

The Forecast page is the comparison workspace for temporal model outputs.

## Endpoint

```text
http://localhost:3001/?page=forecast
```

## 4-Panel Comparison Layout

The page renders a fixed 2x2 grid:

1. Original timeline (ground truth)
2. TGN forecast
3. EvolveGCN-H forecast
4. GConvGRU forecast

All four panels use the same node positions and independent timeline controls.

## Comparison Anchor

Each panel starts from the same anchor week so comparisons are fair:

- default anchor: 52 weeks before observed history ends
- configured by `forecast_app.comparison_lookback_weeks`
- env override: `TG_FORECAST_APP_COMPARISON_LOOKBACK_WEEKS`

This gives a side-by-side "last year of history + forecast horizon" view.

## Latest Training Snapshot

Current metrics from `data/processed_forecast/*/metrics.json`:

| Model | Best Val Loss | MAE | RMSE | Train Seconds | Device |
|---|---:|---:|---:|---:|---|
| TGN | 0.071403 | 0.154438 | 0.495866 | 46.441 | cuda |
| EvolveGCN-H | 0.134911 | 0.219941 | 0.754413 | 86.546 | cuda |
| GConvGRU | 0.064649 | 0.133381 | 0.506902 | 215.194 | cuda |

## Training CLI

```bash
python -m trump_graph train-forecast --model tgn --device cuda
python -m trump_graph train-forecast --model evolvegcn --device cuda
python -m trump_graph train-forecast --model gconvgru --device cuda
```

Tuned example:

```bash
python -m trump_graph train-forecast \
  --model evolvegcn \
  --horizon-weeks 16 \
  --validation-weeks 12 \
  --epochs 30 \
  --hidden-dim 32 \
  --learning-rate 0.002
```

For `evolvegcn` and `gconvgru`, install compatible PyG wheels before training.

## Model Cards

Generate or refresh model cards:

```bash
python -m trump_graph build-model-cards \
  --forecast-dir data/processed_forecast \
  --semantic-input-dir data/processed_unified
```

Outputs:

- `data/processed_forecast/tgn/MODEL_CARD.md`
- `data/processed_forecast/evolvegcn/MODEL_CARD.md`
- `data/processed_forecast/gconvgru/MODEL_CARD.md`
- `data/processed_forecast/MODEL_CARDS_INDEX.md`

## Required Artifact Path

- `data/processed_forecast/<model_key>/forecast_graph/animation_state.json`

Required keys:

- `weeks`
- `global_nodes`
- `global_edges`
- `node_week_deltas`
- `edge_week_deltas`
- `heat_decay`
- `heat_scale`
- `max_cumulative_edge`

Optional:

- `delta_sets`
- `data/processed_forecast/<model_key>/metrics.json`
