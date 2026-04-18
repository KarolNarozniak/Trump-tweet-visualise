from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trump_graph.settings import load_settings


def _run_command(command: Sequence[str], *, cwd: str | None = None) -> None:
    printable = " ".join(command)
    print(f"Running: {printable}")
    subprocess.run(command, check=True, cwd=cwd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deployment workflow for Trump Tweet Visualize."
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest execution.",
    )
    parser.add_argument(
        "--skip-artifact-build",
        action="store_true",
        help="Skip CSV -> processed artifact build.",
    )
    parser.add_argument(
        "--skip-docs-build",
        action="store_true",
        help="Skip Docusaurus static build.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = load_settings()

    if not args.skip_tests:
        _run_command([sys.executable, "-m", "pytest", "-q"], cwd=str(settings.project_root))

    _run_command([sys.executable, "-m", "compileall", "src", "app"], cwd=str(settings.project_root))

    if not args.skip_artifact_build:
        _run_command(
            [
                sys.executable,
                "-m",
                "trump_graph",
                "build",
                "--input",
                str(settings.build.default_input_csv),
                "--out",
                str(settings.build.default_output_dir),
                "--min-mention-count",
                str(settings.build.min_mention_count),
                "--global-min-mentions",
                str(settings.build.global_min_mentions),
                "--heat-decay",
                str(settings.build.heat_decay),
                "--layout-seed",
                str(settings.build.layout_seed),
            ],
            cwd=str(settings.project_root),
        )

    if not args.skip_docs_build:
        _run_command(["npm", "run", "build"], cwd=str(settings.runtime.docs_site_dir))

    print("Deployment workflow completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
