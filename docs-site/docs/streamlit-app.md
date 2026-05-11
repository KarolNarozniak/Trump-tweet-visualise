# Streamlit App

Entrypoint:

- `app/main.py`

Start command:

```bash
python -m streamlit run app/main.py --server.port 3001 --server.address 0.0.0.0
```

## Main UI Sections

1. Stable global graph
2. Semantic temporal graph (unified labels)
3. Temporal forecast graph (model or baseline preview)
4. Timeline controls (play/pause/stop/speed/week scrubber)
5. Weekly tables and export actions

## Sidebar Controls

- processed artifact directory
- hub inclusion toggle
- labeling options
- playback speed
- layout spread
- node size multiplier
- graph zoom and height

All defaults are loaded from `config/defaults.toml` and can be overridden by `.env` or environment variables.

## Docs Endpoint Access

Main page includes a direct button linking to the docs endpoint configured via:

- `runtime.docs_url`
- `TG_RUNTIME_DOCS_URL`

## Semantic Page

Open:

```text
http://localhost:3001/?page=semantic
```

This page reads `data/processed_unified` by default and renders the semantic temporal graph from combined labeled artifacts. It supports repost, sentiment, node-type, and minimum-count filters while leaving the original Graph page unchanged.

Legacy `?page=truth` is still accepted and redirected internally to `?page=semantic`.

## Temporal Forecast Page

Open:

```text
http://localhost:3001/?page=forecast
```

The Forecast page is prepared for temporal neural-network outputs and supports two sources:

- `model`: reads `data/processed_forecast/forecast_graph/animation_state.json`
- `baseline`: generates a local future preview from semantic artifacts

Both renderings keep stable node positions and extend weeks into future slots.
