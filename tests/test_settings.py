from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from trump_graph.settings import clear_settings_cache, load_settings


def _clean_overrides(monkeypatch) -> None:
    for key in [
        "TG_CONFIG_PATH",
        "TG_ENV_PATH",
        "TG_BUILD_INPUT_CSV",
        "TG_BUILD_OUTPUT_DIR",
        "TG_BUILD_MIN_MENTION_COUNT",
        "TG_BUILD_GLOBAL_MIN_MENTIONS",
        "TG_BUILD_INCLUDE_RETWEETS",
        "TG_BUILD_HEAT_DECAY",
        "TG_BUILD_LAYOUT_SEED",
        "TG_APP_PROCESSED_DIR",
        "TG_APP_INCLUDE_HUB",
        "TG_APP_ALWAYS_LABEL_TOP_NODES",
        "TG_APP_PLAYBACK_SPEED",
        "TG_APP_NODE_SIZE_MULTIPLIER",
        "TG_APP_LAYOUT_SPREAD",
        "TG_APP_INITIAL_ZOOM_BOOST",
        "TG_APP_GRAPH_HEIGHT_PX",
        "TG_RUNTIME_STREAMLIT_HOST",
        "TG_RUNTIME_STREAMLIT_PORT",
        "TG_RUNTIME_DOCS_HOST",
        "TG_RUNTIME_DOCS_PORT",
        "TG_RUNTIME_DOCS_URL",
        "TG_RUNTIME_DOCS_SITE_DIR",
    ]:
        monkeypatch.delenv(key, raising=False)


def _create_local_temp_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    temp_root = project_root / "tests_runtime_temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_dir = temp_root / f"settings_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_load_settings_reads_toml_and_dotenv(monkeypatch) -> None:
    _clean_overrides(monkeypatch)
    temp_dir_path = _create_local_temp_dir()
    try:
        config_path = temp_dir_path / "defaults.toml"
        env_path = temp_dir_path / ".env"

        config_path.write_text(
            """
[build]
input_csv = "dataset.csv"
output_dir = "processed-output"
min_mention_count = 2
global_min_mentions = 10
include_retweets = false
heat_decay = 0.75
layout_seed = 77

[app]
processed_dir = "processed-output"
include_hub = true
always_label_top_nodes = false
playback_speed = 4.0
node_size_multiplier = 1.5
layout_spread = 1.2
initial_zoom_boost = 0.9
graph_height_px = 900

[runtime]
streamlit_host = "127.0.0.1"
streamlit_port = 4101
docs_host = "127.0.0.1"
docs_port = 4102
docs_url = "http://localhost:4102"
docs_site_dir = "docs-site"
""".strip(),
            encoding="utf-8",
        )
        env_path.write_text(
            """
TG_APP_INCLUDE_HUB=false
TG_APP_PLAYBACK_SPEED=6.0
""".strip(),
            encoding="utf-8",
        )

        clear_settings_cache()
        settings = load_settings(config_path=config_path, env_path=env_path)

        assert settings.build.default_input_csv.name == "dataset.csv"
        assert settings.build.min_mention_count == 2
        assert settings.build.include_retweets is False
        assert settings.app.include_hub is False
        assert settings.app.playback_speed == 6.0
        assert settings.runtime.streamlit_port == 4101
        assert settings.runtime.docs_port == 4102
    finally:
        shutil.rmtree(temp_dir_path, ignore_errors=True)


def test_process_env_overrides_dotenv_and_toml(monkeypatch) -> None:
    _clean_overrides(monkeypatch)
    temp_dir_path = _create_local_temp_dir()
    try:
        config_path = temp_dir_path / "defaults.toml"
        env_path = temp_dir_path / ".env"

        config_path.write_text(
            """
[runtime]
streamlit_port = 3001
docs_port = 3002
""".strip(),
            encoding="utf-8",
        )
        env_path.write_text("TG_RUNTIME_STREAMLIT_PORT=3111", encoding="utf-8")

        monkeypatch.setenv("TG_RUNTIME_STREAMLIT_PORT", "3222")

        clear_settings_cache()
        settings = load_settings(config_path=config_path, env_path=env_path)
        assert settings.runtime.streamlit_port == 3222
    finally:
        shutil.rmtree(temp_dir_path, ignore_errors=True)
