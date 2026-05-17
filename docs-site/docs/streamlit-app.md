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
3. Temporal forecast comparison grid (original + 3 models)
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

The Forecast page renders four synchronized comparison panels:

- Original timeline (ground truth)
- `tgn`: `data/processed_forecast/tgn/forecast_graph/animation_state.json`
- `evolvegcn`: `data/processed_forecast/evolvegcn/forecast_graph/animation_state.json`
- `gconvgru`: `data/processed_forecast/gconvgru/forecast_graph/animation_state.json`

If any trained model artifact is missing, that panel falls back to a baseline forecast preview.

All panels keep fixed coordinates and start from the same comparison anchor:

- default: 52 weeks before history end
- config key: `forecast_app.comparison_lookback_weeks`
- shared top controller drives all panels at once
- each panel includes a live "top active nodes by type" table for numeric comparison
