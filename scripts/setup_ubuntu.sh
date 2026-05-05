#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INSTALL_ML=0
for arg in "$@"; do
  if [[ "$arg" == "--install-ml" ]]; then
    INSTALL_ML=1
  fi
done

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp ".env.example" ".env"
fi

if [[ ! -d "venv" ]]; then
  python3 -m venv venv
fi

PYTHON_EXE="$REPO_ROOT/venv/bin/python"

"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install -r requirements.txt
if [[ "$INSTALL_ML" == "1" ]]; then
  "$PYTHON_EXE" -m pip install -r requirements-ml.txt
else
  echo "Skipping optional ML requirements (pass --install-ml to enable)."
fi
"$PYTHON_EXE" -m pip install -e ".[dev]"

if command -v npm >/dev/null 2>&1; then
  cd "$REPO_ROOT/docs-site"
  npm install
  cd "$REPO_ROOT"
else
  echo "npm is not available. Install Node.js >= 20, then run npm install in docs-site."
fi

echo "Setup complete."
