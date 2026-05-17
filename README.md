# Trump Graph

Time-dependent mention-network analytics for Trump tweet archive data, with a deterministic Python build pipeline and an interactive Streamlit visualization.

Application display name: **Trump Graph**.

## What This Project Includes

- CSV preprocessing with deterministic week bucketing and mention extraction
- weekly co-mention graph artifacts and summary metrics
- stable global network layout for time-dependent animation
- Truth Social semantic temporal graph artifacts and Streamlit page
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
  truth_preprocess.py
  truth_semantics.py
  truth_graph.py
  truth_pipeline.py
  unified_preprocess.py
  unified_graph.py
  unified_io.py
  unified_pipeline.py
  app.py
tests/
```

## Prerequisites

- Python `>=3.11`
- Node.js `>=20` (LTS recommended)
- npm

## Setup

### Windows (PowerShell)

```powershell
.\scripts\setup_windows.ps1
```

To install the optional local Hugging Face inference stack:

```powershell
.\scripts\setup_windows.ps1 -InstallMl
```

### Ubuntu

```bash
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
```

To install optional ML dependencies:

```bash
./scripts/setup_ubuntu.sh --install-ml
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

## Build Truth Social Semantic Artifacts

The Truth Social pipeline is separate from the Twitter mention graph:

```bash
python -m trump_graph build-truth \
  --input "truthsocial.posts[Trump-FROM-10-8-25].txt" \
  --out "data/processed_truth"
```

Default semantic backend is local Hugging Face inference. For a fast smoke test without model downloads:

```bash
python -m trump_graph build-truth \
  --input "tests/fixtures/sample_truth_posts.csv" \
  --out "data/processed_truth" \
  --semantic-backend deterministic \
  --min-node-count 1
```

## Build Unified Twitter + Truth Semantic Artifacts

This command builds one combined labeled temporal graph and training-ready delta tables:

```bash
python -m trump_graph build-unified \
  --twitter-input "tweets_01-08-2021.csv" \
  --truth-input "truthsocial.posts[Trump-FROM-10-8-25].txt" \
  --out "data/processed_unified" \
  --semantic-backend hf \
  --device cuda
```

Smoke test mode (no model downloads):

```bash
python -m trump_graph build-unified \
  --twitter-input "tests/fixtures/sample_tweets.csv" \
  --truth-input "tests/fixtures/sample_truth_posts.csv" \
  --out "data/processed_unified" \
  --semantic-backend deterministic \
  --min-node-count 1
```

## Train Forecast Models (Layer 3)

After `build-unified`, train one model at a time:

```bash
python -m trump_graph train-forecast --model tgn --device cuda
python -m trump_graph train-forecast --model evolvegcn --device cuda
python -m trump_graph train-forecast --model gconvgru --device cuda
```

For `evolvegcn` and `gconvgru`, install PyG wheels that match your Torch/CUDA build first:

```bash
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.11.0+cu126.html
python -m pip install torch-geometric torch-geometric-temporal --no-deps
python -m pip install cython decorator==4.4.2
```

Recommended production command shape:

```bash
python -m trump_graph train-forecast \
  --model evolvegcn \
  --input-dir "data/processed_unified" \
  --out "data/processed_forecast" \
  --device cuda \
  --horizon-weeks 16 \
  --lookback-weeks 12 \
  --validation-weeks 12 \
  --epochs 30 \
  --hidden-dim 32 \
  --learning-rate 0.002 \
  --weight-decay 0.00001 \
  --seed 42
```

Baseline payload generation (no training loop):

```bash
python -m trump_graph train-forecast --model baseline
```

Generate/refresh model cards for all forecast model directories:

```bash
python -m trump_graph build-model-cards \
  --forecast-dir data/processed_forecast \
  --semantic-input-dir data/processed_unified
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
The app includes routes for `?page=graph`, `?page=semantic`, `?page=forecast`, and `?page=about`.
Legacy `?page=truth` is redirected to `?page=semantic`.

Forecast page behavior:

- renders a 2x2 comparison grid (Original, TGN, EvolveGCN-H, GConvGRU)
- all panels use fixed node coordinates for visual consistency
- timelines start from a shared anchor (default 52 weeks before history end)

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

Optional semantic builds:

- `python scripts/deploy.py --build-truth`
- `python scripts/deploy.py --build-unified`

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

### Key Truth Build Vars

- `TG_TRUTH_BUILD_INPUT_PATH`
- `TG_TRUTH_BUILD_OUTPUT_DIR`
- `TG_TRUTH_BUILD_SEMANTIC_BACKEND`
- `TG_TRUTH_BUILD_DEVICE`
- `TG_TRUTH_BUILD_TOPIC_THRESHOLD`
- `TG_TRUTH_BUILD_MIN_NODE_COUNT`

### Key Unified Build Vars

- `TG_UNIFIED_BUILD_TWITTER_INPUT_CSV`
- `TG_UNIFIED_BUILD_TRUTH_INPUT_PATH`
- `TG_UNIFIED_BUILD_OUTPUT_DIR`
- `TG_UNIFIED_BUILD_SEMANTIC_BACKEND`
- `TG_UNIFIED_BUILD_DEVICE`
- `TG_UNIFIED_BUILD_TOPIC_THRESHOLD`
- `TG_UNIFIED_BUILD_MIN_NODE_COUNT`

### Key Forecast Train Vars

- `TG_FORECAST_TRAIN_INPUT_DIR`
- `TG_FORECAST_TRAIN_OUTPUT_DIR`
- `TG_FORECAST_TRAIN_DEVICE`
- `TG_FORECAST_TRAIN_HORIZON_WEEKS`
- `TG_FORECAST_TRAIN_LOOKBACK_WEEKS`
- `TG_FORECAST_TRAIN_VALIDATION_WEEKS`
- `TG_FORECAST_TRAIN_EPOCHS`
- `TG_FORECAST_TRAIN_HIDDEN_DIM`
- `TG_FORECAST_TRAIN_LEARNING_RATE`
- `TG_FORECAST_TRAIN_WEIGHT_DECAY`
- `TG_FORECAST_TRAIN_SEED`

### Key Truth App Vars

- `TG_TRUTH_APP_PROCESSED_DIR`
- `TG_TRUTH_APP_INCLUDE_RETRUTHS`
- `TG_TRUTH_APP_NODE_TYPES`
- `TG_TRUTH_APP_SENTIMENT_FILTER`

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

Default Truth output directory: `data/processed_truth`

- `truth_week_index.csv`
- `truth_weekly_summary.csv`
- `truth_posts_enriched.parquet`
- `truth_semantic_graph/animation_state.json`
- `truth_semantic_graph/node_catalog.csv`
- `truth_embeddings/embeddings.npy`
- `truth_embeddings/index.csv`

Default Unified output directory: `data/processed_unified`

- `unified_week_index.csv`
- `unified_weekly_summary.csv`
- `unified_posts_enriched.parquet`
- `unified_semantic_graph/animation_state.json`
- `unified_semantic_graph/node_catalog.csv`
- `unified_semantic_graph/edge_catalog.csv`
- `unified_embeddings/embeddings.npy`
- `unified_embeddings/index.csv`
- `unified_training/temporal_node_deltas.parquet`
- `unified_training/temporal_edge_deltas.parquet`

Optional forecast model output directory: `data/processed_forecast`

- `tgn/forecast_graph/animation_state.json`
- `evolvegcn/forecast_graph/animation_state.json`
- `gconvgru/forecast_graph/animation_state.json`
- `<model_key>/metrics.json`
- `<model_key>/MODEL_CARD.md`
- `MODEL_CARDS_INDEX.md`

If a selected model artifact is not present, the Forecast page falls back to a baseline future preview generated from semantic artifacts.

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
