# Streamlit App

Entrypoint:

- `app/main.py`

Start command:

```bash
python -m streamlit run app/main.py --server.port 3001 --server.address 0.0.0.0
```

## Main UI Sections

1. Stable global graph
2. Timeline controls (play/pause/stop/speed/week scrubber)
3. Top mentioned accounts table
4. Weekly export actions

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
