# Trump Tweet Visualize

Time-dependent mention-network analytics for Trump tweet archive data, with a deterministic Python build pipeline and an interactive Streamlit visualization.

## What This Project Includes

- CSV preprocessing with deterministic week bucketing and mention extraction
- weekly co-mention graph artifacts and summary metrics
- stable global network layout for time-dependent animation
- Streamlit app for playback, filtering, tables, and exports
- Docusaurus documentation site
- cross-platform setup/run/deploy scripts for Windows and Ubuntu

## Default Endpoints

- Streamlit app: `http://localhost:3001`
- Docs site: `http://localhost:3002`

These defaults are configurable through `config/defaults.toml`, `.env`, or environment variables.

## Repository Structure

```text
app/
  main.py
config/
  defaults.toml
docs-site/
  ... Docusaurus project
scripts/
  setup_windows.ps1
  setup_ubuntu.sh
  run_windows.ps1
  run_ubuntu.sh
  deploy_windows.ps1
  deploy_ubuntu.sh
  run_services.py
  deploy.py
src/trump_graph/
  __main__.py
  settings.py
  io.py
  preprocess.py
  graph_build.py
  metrics.py
  global_animation.py
  pipeline.py
  app.py
tests/
```

## Prerequisites

- Python `>=3.11`
- Node.js `>=20`
- npm

## Setup

### Windows (PowerShell)

```powershell
.\scripts\setup_windows.ps1
```

### Ubuntu

```bash
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
```

Both scripts:

1. create `venv` (if missing)
2. install Python dependencies
3. install package in editable mode with dev dependencies
4. install Docusaurus dependencies in `docs-site`
5. create `.env` from `.env.example` if needed

## Build Artifacts

### Using defaults from config/env

```bash
python -m trump_graph build
```

### Explicit invocation

```bash
python -m trump_graph build \
  --input "tweets_01-08-2021.csv" \
  --out "data/processed" \
  --min-mention-count 1 \
  --global-min-mentions 8 \
  --heat-decay 0.85 \
  --layout-seed 42 \
  --include-retweets
```

## Run App and Docs Together

### Windows

```powershell
.\scripts\run_windows.ps1
```

### Ubuntu

```bash
./scripts/run_ubuntu.sh
```

Mode options:

- `all` (default): app + docs
- `app`: app only
- `docs`: docs only

Examples:

```powershell
.\scripts\run_windows.ps1 -Mode app
```

```bash
./scripts/run_ubuntu.sh docs
```

The Streamlit header includes an **Open Docs** button that links directly to the configured docs endpoint.

## Deployment Workflow

### Windows

```powershell
.\scripts\deploy_windows.ps1
```

### Ubuntu

```bash
./scripts/deploy_ubuntu.sh
```

What this runs:

1. `pytest` (unless skipped)
2. `python -m compileall src app`
3. artifact build (`python -m trump_graph build`)
4. docs static build (`npm run build` in `docs-site`)

## Configuration

Configuration resolution order:

1. environment variables
2. `.env`
3. `config/defaults.toml`
4. hard fallback values

### Key Runtime Vars

- `TG_RUNTIME_STREAMLIT_PORT` (default `3001`)
- `TG_RUNTIME_DOCS_PORT` (default `3002`)
- `TG_RUNTIME_DOCS_URL` (default `http://localhost:3002`)

### Key App Vars

- `TG_APP_PROCESSED_DIR`
- `TG_APP_PLAYBACK_SPEED`
- `TG_APP_LAYOUT_SPREAD`
- `TG_APP_NODE_SIZE_MULTIPLIER`
- `TG_APP_GRAPH_HEIGHT_PX`

### Key Build Vars

- `TG_BUILD_INPUT_CSV`
- `TG_BUILD_OUTPUT_DIR`
- `TG_BUILD_GLOBAL_MIN_MENTIONS`
- `TG_BUILD_HEAT_DECAY`
- `TG_BUILD_LAYOUT_SEED`

See full reference in docs site:

- `docs-site/docs/configuration.md`

## Produced Artifacts

Default output directory: `data/processed`

- `week_index.csv`
- `weekly_summary.csv`
- `weeks/<week_id>/nodes.csv`
- `weeks/<week_id>/edges.csv`
- `weeks/<week_id>/metrics.json`
- `global_animation/animation_state.json`

## Testing

```bash
python -m pytest -q
```

If you hit Windows temp permission issues:

```bash
python -m pytest -q --basetemp=.pytest_tmp
```

## Documentation Site

Docusaurus project lives under `docs-site`.

Useful commands:

```bash
cd docs-site
npm run start
npm run build
npm run serve
```

## Data Source

- Trump Twitter Archive FAQ: https://www.thetrumparchive.com/faq
