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
  app.py                          # Artifact loaders + vis-network HTML builder
tests/
  ...                             # Unit/integration/smoke tests
```

## Runtime Components

1. CLI build (`python -m trump_graph build`) creates all processed artifacts.
2. Streamlit reads processed artifacts and renders one stable graph with week-by-week transitions.
3. Docusaurus serves project documentation as a separate endpoint.

## Design Principles

- deterministic outputs for the same input + seed
- config-driven defaults rather than hardcoded runtime values
- modular separation by concern (I/O, preprocessing, graphing, app rendering)
- compatibility with Windows and Ubuntu setup/run workflows
