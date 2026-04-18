from __future__ import annotations

import shutil
import sys
from pathlib import Path
import uuid

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture()
def sample_tweets_csv_path() -> Path:
    return ROOT_DIR / "tests" / "fixtures" / "sample_tweets.csv"


@pytest.fixture()
def local_temp_dir() -> Path:
    temp_root = ROOT_DIR / "tests_runtime_temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_dir = temp_root / f"run_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield run_dir
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
