# Configuration

Configuration is resolved in this precedence order:

1. process environment variables
2. `.env` file values
3. `config/defaults.toml`
4. built-in fallback values

Loader implementation:

- `src/trump_graph/settings.py`

## Build Settings

- `TG_BUILD_INPUT_CSV`
- `TG_BUILD_OUTPUT_DIR`
- `TG_BUILD_MIN_MENTION_COUNT`
- `TG_BUILD_GLOBAL_MIN_MENTIONS`
- `TG_BUILD_INCLUDE_RETWEETS`
- `TG_BUILD_HEAT_DECAY`
- `TG_BUILD_LAYOUT_SEED`

## App Settings

- `TG_APP_PROCESSED_DIR`
- `TG_APP_INCLUDE_HUB`
- `TG_APP_ALWAYS_LABEL_TOP_NODES`
- `TG_APP_PLAYBACK_SPEED`
- `TG_APP_NODE_SIZE_MULTIPLIER`
- `TG_APP_LAYOUT_SPREAD`
- `TG_APP_INITIAL_ZOOM_BOOST`
- `TG_APP_GRAPH_HEIGHT_PX`

## Runtime Settings

- `TG_RUNTIME_STREAMLIT_HOST`
- `TG_RUNTIME_STREAMLIT_PORT`
- `TG_RUNTIME_DOCS_HOST`
- `TG_RUNTIME_DOCS_PORT`
- `TG_RUNTIME_DOCS_URL`
- `TG_RUNTIME_DOCS_SITE_DIR`

## Config Path Overrides

- `TG_CONFIG_PATH` to choose a TOML file
- `TG_ENV_PATH` to choose a dotenv file

## Example

```bash
TG_RUNTIME_STREAMLIT_PORT=3101 TG_RUNTIME_DOCS_PORT=3102 python scripts/run_services.py --mode all
```
