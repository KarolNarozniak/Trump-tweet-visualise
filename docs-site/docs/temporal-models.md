# Temporal Models

The Forecast stack supports three trainable model keys on top of unified semantic artifacts.

## Implemented Model Keys

### `tgn`

- style: memory-based temporal node forecaster
- implementation: GRU memory over per-node weekly signal + normalized neighbor aggregation
- role: fast sequence baseline that still uses graph context

### `evolvegcn`

- style: EvolveGCN-H snapshot model
- implementation: `torch_geometric_temporal.nn.recurrent.EvolveGCNH`
- role: evolves graph convolution parameters across weekly snapshots

### `gconvgru`

- style: graph-convolutional recurrent model
- implementation: `torch_geometric_temporal.nn.recurrent.GConvGRU`
- role: recurrent graph forecasting baseline with explicit hidden state

## Training Command

```bash
python -m trump_graph train-forecast \
  --model evolvegcn \
  --input-dir data/processed_unified \
  --out data/processed_forecast \
  --device cuda \
  --horizon-weeks 16 \
  --validation-weeks 12 \
  --epochs 30
```

Supported `--model` values:

- `baseline`
- `tgn`
- `evolvegcn`
- `gconvgru`

## Produced Artifacts

Each model writes:

- `data/processed_forecast/<model_key>/forecast_graph/animation_state.json`
- `data/processed_forecast/<model_key>/metrics.json`

The Forecast page reads these directly.
