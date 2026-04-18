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
