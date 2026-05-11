# Quickstart

## Prerequisites

- Python `>=3.11`
- Node.js `>=20` (LTS recommended)
- npm

## 1) Setup

### Windows (PowerShell)

```powershell
.\scripts\setup_windows.ps1
```

### Ubuntu

```bash
./scripts/setup_ubuntu.sh
```

## 2) Build Processed Artifacts

```bash
python -m trump_graph build
```

Build the Truth Social semantic artifacts:

```bash
python -m trump_graph build-truth
```

For a no-model smoke test:

```bash
python -m trump_graph build-truth --semantic-backend deterministic --min-node-count 1
```

Build unified semantic artifacts (recommended for the new Semantic page):

```bash
python -m trump_graph build-unified
```

The command uses defaults from `config/defaults.toml` and optional overrides from `.env`.

## 3) Run App + Docs

### Windows

```powershell
.\scripts\run_windows.ps1
```

### Ubuntu

```bash
./scripts/run_ubuntu.sh
```

The wrapper runs:

- Streamlit app on `3001`
- Docusaurus docs on `3002`

Open key views with:

```text
http://localhost:3001/?page=graph
http://localhost:3001/?page=semantic
http://localhost:3001/?page=forecast
```
