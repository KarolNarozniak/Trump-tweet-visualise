from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trump_graph.settings import load_settings


def _npm_executable() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def _docs_cli_path(docs_dir: Path) -> Path:
    cli_name = "docusaurus.cmd" if os.name == "nt" else "docusaurus"
    return docs_dir / "node_modules" / ".bin" / cli_name


def _terminate_process(process: subprocess.Popen[bytes], timeout_seconds: float = 6.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    process.kill()


def _spawn_docs_process() -> subprocess.Popen[bytes]:
    settings = load_settings()
    docs_dir = settings.runtime.docs_site_dir
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    if not _docs_cli_path(docs_dir).exists():
        raise RuntimeError(
            "Docs dependencies are missing. Run scripts/setup_windows.ps1 (or npm install inside docs-site) first."
        )

    docs_command = [
        _npm_executable(),
        "run",
        "start",
        "--",
        "--host",
        settings.runtime.docs_host,
        "--port",
        str(settings.runtime.docs_port),
        "--no-open",
    ]
    try:
        return subprocess.Popen(docs_command, cwd=docs_dir)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Unable to start docs service: {_npm_executable()} executable was not found in PATH."
        ) from error


def _spawn_app_process() -> subprocess.Popen[bytes]:
    settings = load_settings()
    app_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/main.py",
        "--server.address",
        settings.runtime.streamlit_host,
        "--server.port",
        str(settings.runtime.streamlit_port),
    ]
    return subprocess.Popen(app_command, cwd=settings.project_root)


def run_docs_only() -> int:
    process = _spawn_docs_process()
    try:
        return int(process.wait())
    except KeyboardInterrupt:
        _terminate_process(process)
        return 130


def run_app_only() -> int:
    process = _spawn_app_process()
    try:
        return int(process.wait())
    except KeyboardInterrupt:
        _terminate_process(process)
        return 130


def run_both() -> int:
    settings = load_settings()
    docs_process = _spawn_docs_process()
    app_process = _spawn_app_process()

    print(f"App URL:  http://localhost:{settings.runtime.streamlit_port}")
    print(f"Docs URL: {settings.runtime.docs_url}")

    try:
        while True:
            app_exit = app_process.poll()
            docs_exit = docs_process.poll()

            if app_exit is not None:
                _terminate_process(docs_process)
                return int(app_exit)
            if docs_exit is not None:
                _terminate_process(app_process)
                return int(docs_exit)

            time.sleep(0.25)
    except KeyboardInterrupt:
        _terminate_process(app_process)
        _terminate_process(docs_process)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Streamlit and Docusaurus services for Trump Tweet Visualize."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "app", "docs"),
        default="all",
        help="Choose which service(s) to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "docs":
        return run_docs_only()
    if args.mode == "app":
        return run_app_only()
    return run_both()


if __name__ == "__main__":
    raise SystemExit(main())
