---
license: mit
library_name: pytorch
tags:
- temporal-graph
- graph-forecasting
- trump-graph
- model:evolvegcn
datasets:
- trump-graph-unified-semantic
---

# EvolveGCN-H Model Card

## Model Details

- model_key: `evolvegcn`
- model_name: Evolving Graph Convolutional Networks
- family: neural-temporal-graph
- category: Snapshot recurrent graph convolution
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/5984
- implementation_reference: https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html

## Intended Use

This model forecasts weekly semantic graph activity (node mentions and edge co-occurrence intensity) for exploratory temporal analysis in the Trump Graph educational project.

## Out-of-Scope Use

- not for factual or legal claims about individuals
- not for policy or political decision automation
- not calibrated as a causal inference model

## Training and Evaluation Data

- source: unified semantic timeline produced by `build-unified`
- time granularity: weekly snapshots
- total_weeks_in_payload: 780
- nodes_in_payload: 3009
- edges_in_payload: 99766
- forecast_weeks_in_payload: 0

## Training Procedure

- epochs: n/a
- validation_weeks: n/a
- horizon_weeks: n/a
- device_used: n/a

## Evaluation Results

| Metric | Value |
|---|---:|
| best_val_score | n/a |
| mae | n/a |
| rmse | n/a |
| train_seconds | n/a |

## Limitations, Bias, and Risks

- quality depends on upstream topic/entity labeling quality
- sparse or noisy weeks can destabilize long-horizon predictions
- model outputs represent learned temporal correlations, not ground truth intent

## Status

- status: **missing**
- generated_at_utc: `2026-05-17T17:37:38.614827+00:00`
- artifact_path: `data\processed_forecast\evolvegcn\forecast_graph\animation_state.json`
- metrics_path: `data\processed_forecast\evolvegcn\metrics.json`

### Missing Artifact Checklist

1. Train the model with `python -m trump_graph train-forecast --model evolvegcn --device cuda`.
2. Confirm `forecast_graph/animation_state.json` and `metrics.json` exist.
3. Restart Streamlit if table status is cached.
