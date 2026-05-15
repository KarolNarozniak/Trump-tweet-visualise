# Temporal Models (Research Plan)

This project will train and compare three temporal graph models on the unified semantic graph timeline.

## Model 1: TGN

- name: Temporal Graph Network (TGN)
- type: continuous-time event model with memory
- paper: https://arxiv.org/abs/2006.10637
- implementation reference: https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.models.TGNMemory.html

Why here:

- fits event-level temporal interactions
- captures long-term dependencies with node memory
- strong baseline for dynamic link-style prediction

## Model 2: EvolveGCN-H

- name: Evolving Graph Convolutional Networks (EvolveGCN)
- type: recurrent evolution of GCN parameters over time snapshots
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/5984
- implementation reference: https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html

Why here:

- built for evolving graph snapshots
- parameter evolution can adapt across long weekly horizons
- no strict dependency on fixed node embeddings

## Model 3: GConvGRU

- name: Graph Convolutional Recurrent Network (GConvGRU cell)
- type: graph convolution + GRU sequence model over snapshots
- paper: https://arxiv.org/abs/1612.07659
- implementation reference: https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html

Why here:

- stable and efficient recurrent baseline
- good bias for weekly temporal smoothness
- easy to tune and interpret for first production training runs

## Comparison Protocol

Use one consistent split across all three:

1. train: earliest 70% weeks
2. validation: next 15% weeks
3. test: final 15% weeks

Primary metrics:

- MAE / RMSE on predicted weekly node and edge deltas
- MAP for ranking high-activity nodes/edges

All model outputs should be exported to:

- `data/processed_forecast/tgn/forecast_graph/animation_state.json`
- `data/processed_forecast/evolvegcn/forecast_graph/animation_state.json`
- `data/processed_forecast/gconvgru/forecast_graph/animation_state.json`

Optional per-model metrics file:

- `data/processed_forecast/<model_key>/metrics.json`
