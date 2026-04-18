#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_EXE="$REPO_ROOT/venv/bin/python"
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Missing venv Python at $PYTHON_EXE. Run ./scripts/setup_ubuntu.sh first."
  exit 1
fi

"$PYTHON_EXE" "scripts/run_services.py" --mode "$MODE"
