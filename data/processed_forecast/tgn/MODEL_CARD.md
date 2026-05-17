---
license: mit
library_name: pytorch
tags:
- temporal-graph
- graph-forecasting
- trump-graph
- model:tgn
datasets:
- trump-graph-unified-semantic
model-index:
- name: Temporal Graph Network
  results:
  - task:
      type: graph-forecasting
      name: Weekly node activity forecast
    dataset:
      type: trump-graph-unified-semantic
      name: Trump Graph Unified Semantic Timeline
    metrics:
    - type: best_val_score
      value: 0.071403
    - type: mae
      value: 0.154438
    - type: rmse
      value: 0.495866
---

# TGN Model Card

## Model Details

- model_key: `tgn`
- model_name: Temporal Graph Network
- family: neural-temporal-graph
- category: Continuous-time event model
- paper: https://arxiv.org/abs/2006.10637
- implementation_reference: https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.models.TGNMemory.html

## Intended Use

This model forecasts weekly semantic graph activity (node mentions and edge co-occurrence intensity) for exploratory temporal analysis in the Trump Graph educational project.

## Out-of-Scope Use

- not for factual or legal claims about individuals
- not for policy or political decision automation
- not calibrated as a causal inference model

## Training and Evaluation Data

- source: unified semantic timeline produced by `build-unified`
- time granularity: weekly snapshots
- total_weeks_in_payload: 796
- nodes_in_payload: 3009
- edges_in_payload: 99766
- forecast_weeks_in_payload: 16

## Training Procedure

- epochs: 30
- validation_weeks: 12
- horizon_weeks: 16
- device_used: cuda

## Evaluation Results

| Metric | Value |
|---|---:|
| best_val_score | 0.071403 |
| mae | 0.154438 |
| rmse | 0.495866 |
| train_seconds | 46.441 |

## Limitations, Bias, and Risks

- quality depends on upstream topic/entity labeling quality
- sparse or noisy weeks can destabilize long-horizon predictions
- model outputs represent learned temporal correlations, not ground truth intent

## Status

- status: **ready**
- generated_at_utc: `2026-05-17T17:45:11.108914+00:00`
- artifact_path: `data/processed_forecast/tgn/forecast_graph/animation_state.json`
- metrics_path: `data/processed_forecast/tgn/metrics.json`
