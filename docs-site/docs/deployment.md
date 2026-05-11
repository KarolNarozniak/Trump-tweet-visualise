# Deployment

## Standard Ports

- app: `3001`
- docs: `3002`

These are configurable through TOML and environment variables.

## One-Command Deploy Workflows

### Windows

```powershell
.\scripts\deploy_windows.ps1
```

### Ubuntu

```bash
./scripts/deploy_ubuntu.sh
```

The deployment script performs:

1. optional tests
2. Python bytecode compile checks
3. build of processed artifacts
4. Docusaurus static build

Truth Social semantic artifacts are optional because they may run local ML inference:

```bash
python scripts/deploy.py --build-truth
```

For a fast deployment smoke test:

```bash
python scripts/deploy.py --build-truth --truth-semantic-backend deterministic
```

Unified semantic artifacts are also optional and can be built in the same flow:

```bash
python scripts/deploy.py --build-unified
```

Deterministic unified smoke build:

```bash
python scripts/deploy.py --build-unified --unified-semantic-backend deterministic
```

## Serving Outputs

- Streamlit serves the app directly.
- Docusaurus static output is generated at `docs-site/build` for static hosting.

## CI Recommendation

- cache Python dependencies and npm modules
- run `python -m pytest -q`
- run `python scripts/deploy.py`
- publish `docs-site/build` via static site host
