# Temporal Forecast

The Forecast page is the frontend entrypoint for temporal neural network outputs.

## Endpoint

```text
http://localhost:3001/?page=forecast
```

## Modes

- `tgn`: load trained TGN output
- `evolvegcn`: load trained EvolveGCN output
- `gconvgru`: load trained GConvGRU output
- `baseline`: local preview built from semantic history while training is in progress

## Training CLI

```bash
python -m trump_graph train-forecast --model tgn --device cuda
python -m trump_graph train-forecast --model evolvegcn --device cuda
python -m trump_graph train-forecast --model gconvgru --device cuda
```

You can tune horizon and optimization:

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

Generate model cards after training:

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

## Expected Model Artifact

Path:

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

Optional key:

- `delta_sets` (if omitted, app uses top-level deltas as `all`)

Optional sidecar metrics:

- `data/processed_forecast/<model_key>/metrics.json`

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
