# Troubleshooting

## App says processed artifacts are missing

Run:

```bash
python -m trump_graph build
```

Confirm output directory matches:

- `TG_APP_PROCESSED_DIR`
- `app.processed_dir` in `config/defaults.toml`

## Docs endpoint does not open

1. Ensure docs dependencies are installed in `docs-site`.
2. Start docs service:
   - `npm run start` from `docs-site`
3. Confirm `runtime.docs_url` points to the active host/port.

## Graph is blank or collapsed

1. Rebuild artifacts with:
   - lower `global_min_mentions`
   - verify CSV path and date parsing
2. Check app sidebar:
   - processed directory
   - include hub toggle
   - layout spread and zoom values

## Node modules not installed on Ubuntu

Install Node.js `>=20` and npm, then run:

```bash
cd docs-site
npm install
```

## Pytest permission errors on Windows

Use a local writable base temp:

```bash
python -m pytest -q --basetemp=.pytest_tmp
```
