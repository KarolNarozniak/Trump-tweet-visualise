---
license: mit
library_name: pytorch
tags:
- temporal-graph
- graph-forecasting
- trump-graph
- model:gconvgru
datasets:
- trump-graph-unified-semantic
model-index:
- name: Graph Convolutional Recurrent Network
  results:
  - task:
      type: graph-forecasting
      name: Weekly node activity forecast
    dataset:
      type: trump-graph-unified-semantic
      name: Trump Graph Unified Semantic Timeline
    metrics:
    - type: best_val_score
      value: 0.064649
    - type: mae
      value: 0.133381
    - type: rmse
      value: 0.506902
---

# GConvGRU Model Card

## Model Details

- model_key: `gconvgru`
- model_name: Graph Convolutional Recurrent Network
- family: neural-temporal-graph
- category: Snapshot graph recurrent sequence model
- paper: https://arxiv.org/abs/1612.07659
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
| best_val_score | 0.064649 |
| mae | 0.133381 |
| rmse | 0.506902 |
| train_seconds | 215.194 |

## Limitations, Bias, and Risks

- quality depends on upstream topic/entity labeling quality
- sparse or noisy weeks can destabilize long-horizon predictions
- model outputs represent learned temporal correlations, not ground truth intent

## Status

- status: **ready**
- generated_at_utc: `2026-05-17T17:37:38.868207+00:00`
- artifact_path: `data\processed_forecast\gconvgru\forecast_graph\animation_state.json`
- metrics_path: `data\processed_forecast\gconvgru\metrics.json`
