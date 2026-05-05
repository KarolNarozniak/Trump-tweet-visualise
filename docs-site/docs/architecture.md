# Architecture

## Project Layout

```text
app/
  main.py                         # Streamlit entrypoint
config/
  defaults.toml                   # Default runtime + build settings
docs-site/
  ...                             # Docusaurus documentation site
scripts/
  setup_*.{ps1,sh}                # Setup scripts for Windows/Ubuntu
  run_*.{ps1,sh}                  # Run app + docs scripts
  deploy_*.{ps1,sh}               # Deployment scripts
  run_services.py                 # Cross-platform process launcher
  deploy.py                       # Cross-platform deployment workflow
src/trump_graph/
  __main__.py                     # CLI entrypoint
  settings.py                     # Typed config loader (TOML + .env + env vars)
  io.py                           # CSV read/write + artifact persistence
  preprocess.py                   # Cleaning, mention extraction, week bucketing
  graph_build.py                  # Weekly graph construction
  metrics.py                      # Weekly metrics
  global_animation.py             # Stable global graph + per-week deltas
  pipeline.py                     # End-to-end build pipeline
  truth_*.py                      # Truth Social semantic pipeline and graph builder
  app.py                          # Artifact loaders + vis-network HTML builder
tests/
  ...                             # Unit/integration/smoke tests
```

## Runtime Components

1. CLI build (`python -m trump_graph build`) creates all processed artifacts.
2. Streamlit reads processed artifacts and renders one stable graph with week-by-week transitions.
3. Docusaurus serves project documentation as a separate endpoint.

Layer 2 adds a separate Truth Social path:

1. CLI build (`python -m trump_graph build-truth`) enriches Truth posts with topics, entities, sentiment, and embeddings.
2. The app route `?page=truth` renders the semantic temporal graph from `data/processed_truth`.
3. Existing Twitter graph artifacts and UI remain unchanged.

## Design Principles

- deterministic outputs for the same input + seed
- config-driven defaults rather than hardcoded runtime values
- modular separation by concern (I/O, preprocessing, graphing, app rendering)
- compatibility with Windows and Ubuntu setup/run workflows
