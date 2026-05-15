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
        "TG_TRUTH_BUILD_INPUT_PATH",
        "TG_TRUTH_BUILD_OUTPUT_DIR",
        "TG_TRUTH_BUILD_SEMANTIC_BACKEND",
        "TG_TRUTH_BUILD_TOPIC_MODEL_ID",
        "TG_TRUTH_BUILD_NER_MODEL_ID",
        "TG_TRUTH_BUILD_SENTIMENT_MODEL_ID",
        "TG_TRUTH_BUILD_EMBEDDING_MODEL_ID",
        "TG_TRUTH_BUILD_DEVICE",
        "TG_TRUTH_BUILD_BATCH_SIZE",
        "TG_TRUTH_BUILD_TOPIC_THRESHOLD",
        "TG_TRUTH_BUILD_MAX_TOPIC_LABELS",
        "TG_TRUTH_BUILD_ENTITY_SCORE_THRESHOLD",
        "TG_TRUTH_BUILD_MAX_CHUNK_CHARS",
        "TG_TRUTH_BUILD_MIN_NODE_COUNT",
        "TG_TRUTH_BUILD_INCLUDED_NODE_TYPES",
        "TG_TRUTH_BUILD_HEAT_DECAY",
        "TG_TRUTH_BUILD_LAYOUT_SEED",
        "TG_UNIFIED_BUILD_TWITTER_INPUT_CSV",
        "TG_UNIFIED_BUILD_TRUTH_INPUT_PATH",
        "TG_UNIFIED_BUILD_OUTPUT_DIR",
        "TG_UNIFIED_BUILD_INCLUDE_RETWEETS",
        "TG_UNIFIED_BUILD_SEMANTIC_BACKEND",
        "TG_UNIFIED_BUILD_TOPIC_MODEL_ID",
        "TG_UNIFIED_BUILD_NER_MODEL_ID",
        "TG_UNIFIED_BUILD_SENTIMENT_MODEL_ID",
        "TG_UNIFIED_BUILD_EMBEDDING_MODEL_ID",
        "TG_UNIFIED_BUILD_DEVICE",
        "TG_UNIFIED_BUILD_BATCH_SIZE",
        "TG_UNIFIED_BUILD_TOPIC_THRESHOLD",
        "TG_UNIFIED_BUILD_MAX_TOPIC_LABELS",
        "TG_UNIFIED_BUILD_ENTITY_SCORE_THRESHOLD",
        "TG_UNIFIED_BUILD_MAX_CHUNK_CHARS",
        "TG_UNIFIED_BUILD_MIN_NODE_COUNT",
        "TG_UNIFIED_BUILD_INCLUDED_NODE_TYPES",
        "TG_UNIFIED_BUILD_HEAT_DECAY",
        "TG_UNIFIED_BUILD_LAYOUT_SEED",
        "TG_APP_PROCESSED_DIR",
        "TG_APP_INCLUDE_HUB",
        "TG_APP_ALWAYS_LABEL_TOP_NODES",
        "TG_APP_PLAYBACK_SPEED",
        "TG_APP_NODE_SIZE_MULTIPLIER",
        "TG_APP_LAYOUT_SPREAD",
        "TG_APP_INITIAL_ZOOM_BOOST",
        "TG_APP_GRAPH_HEIGHT_PX",
        "TG_TRUTH_APP_PROCESSED_DIR",
        "TG_TRUTH_APP_INCLUDE_RETRUTHS",
        "TG_TRUTH_APP_NODE_TYPES",
        "TG_TRUTH_APP_SENTIMENT_FILTER",
        "TG_TRUTH_APP_MIN_NODE_COUNT",
        "TG_TRUTH_APP_PLAYBACK_SPEED",
        "TG_TRUTH_APP_NODE_SIZE_MULTIPLIER",
        "TG_TRUTH_APP_LAYOUT_SPREAD",
        "TG_TRUTH_APP_INITIAL_ZOOM_BOOST",
        "TG_TRUTH_APP_GRAPH_HEIGHT_PX",
        "TG_SEMANTIC_APP_PROCESSED_DIR",
        "TG_SEMANTIC_APP_INCLUDE_REPOSTS",
        "TG_SEMANTIC_APP_NODE_TYPES",
        "TG_SEMANTIC_APP_SENTIMENT_FILTER",
        "TG_SEMANTIC_APP_MIN_NODE_COUNT",
        "TG_SEMANTIC_APP_PLAYBACK_SPEED",
        "TG_SEMANTIC_APP_NODE_SIZE_MULTIPLIER",
        "TG_SEMANTIC_APP_LAYOUT_SPREAD",
        "TG_SEMANTIC_APP_INITIAL_ZOOM_BOOST",
        "TG_SEMANTIC_APP_GRAPH_HEIGHT_PX",
        "TG_FORECAST_APP_SEMANTIC_PROCESSED_DIR",
        "TG_FORECAST_APP_PROCESSED_DIR",
        "TG_FORECAST_APP_DEFAULT_MODE",
        "TG_FORECAST_APP_HORIZON_WEEKS",
        "TG_FORECAST_APP_LOOKBACK_WEEKS",
        "TG_FORECAST_APP_NODE_TYPES",
        "TG_FORECAST_APP_MIN_NODE_COUNT",
        "TG_FORECAST_APP_PLAYBACK_SPEED",
        "TG_FORECAST_APP_NODE_SIZE_MULTIPLIER",
        "TG_FORECAST_APP_LAYOUT_SPREAD",
        "TG_FORECAST_APP_INITIAL_ZOOM_BOOST",
        "TG_FORECAST_APP_GRAPH_HEIGHT_PX",
        "TG_FORECAST_TRAIN_INPUT_DIR",
        "TG_FORECAST_TRAIN_OUTPUT_DIR",
        "TG_FORECAST_TRAIN_DEVICE",
        "TG_FORECAST_TRAIN_HORIZON_WEEKS",
        "TG_FORECAST_TRAIN_LOOKBACK_WEEKS",
        "TG_FORECAST_TRAIN_VALIDATION_WEEKS",
        "TG_FORECAST_TRAIN_EPOCHS",
        "TG_FORECAST_TRAIN_HIDDEN_DIM",
        "TG_FORECAST_TRAIN_LEARNING_RATE",
        "TG_FORECAST_TRAIN_WEIGHT_DECAY",
        "TG_FORECAST_TRAIN_SEED",
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
        assert settings.unified_build.output_dir.name == "processed_unified"
        assert settings.unified_build.semantic_backend == "hf"
        assert settings.semantic_app.processed_dir.name == "processed_unified"
        assert settings.forecast_app.default_mode == "baseline"
        assert settings.forecast_train.output_dir.name == "processed_forecast"
        assert settings.forecast_train.epochs == 30
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
