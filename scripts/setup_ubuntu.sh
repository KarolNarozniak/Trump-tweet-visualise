#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp ".env.example" ".env"
fi

if [[ ! -d "venv" ]]; then
  python3 -m venv venv
fi

PYTHON_EXE="$REPO_ROOT/venv/bin/python"

"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install -r requirements.txt
"$PYTHON_EXE" -m pip install -e ".[dev]"

if command -v npm >/dev/null 2>&1; then
  cd "$REPO_ROOT/docs-site"
  npm install
  cd "$REPO_ROOT"
else
  echo "npm is not available. Install Node.js >= 20, then run npm install in docs-site."
fi

echo "Setup complete."
