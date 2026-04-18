from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.toml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        parsed[key] = value
    return parsed


def _read_nested(config: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise ValueError(f"Unable to parse boolean value from {value!r}")


def _to_int(value: Any) -> int:
    return int(str(value).strip()) if isinstance(value, str) else int(value)


def _to_float(value: Any) -> float:
    return float(str(value).strip()) if isinstance(value, str) else float(value)


def _to_str(value: Any) -> str:
    return str(value).strip()


def _to_path(value: Any, *, project_root: Path) -> Path:
    raw_path = Path(str(value).strip())
    if raw_path.is_absolute():
        return raw_path
    return (project_root / raw_path).resolve()


def _resolve_value(
    *,
    env_name: str,
    config_keys: tuple[str, ...],
    config_data: Mapping[str, Any],
    dotenv_data: Mapping[str, str],
    default_value: Any,
    parser: Any,
) -> Any:
    if env_name in os.environ:
        return parser(os.environ[env_name])
    if env_name in dotenv_data:
        return parser(dotenv_data[env_name])
    config_value = _read_nested(config_data, config_keys)
    if config_value is not None:
        return parser(config_value)
    return parser(default_value)


@dataclass(frozen=True)
class BuildSettings:
    default_input_csv: Path
    default_output_dir: Path
    min_mention_count: int
    global_min_mentions: int
    include_retweets: bool
    heat_decay: float
    layout_seed: int


@dataclass(frozen=True)
class AppSettings:
    processed_dir: Path
    include_hub: bool
    always_label_top_nodes: bool
    playback_speed: float
    node_size_multiplier: float
    layout_spread: float
    initial_zoom_boost: float
    graph_height_px: int


@dataclass(frozen=True)
class RuntimeSettings:
    streamlit_host: str
    streamlit_port: int
    docs_host: str
    docs_port: int
    docs_url: str
    docs_site_dir: Path


@dataclass(frozen=True)
class MetaSettings:
    app_name: str
    logo_path: Path
    data_source_url: str
    repository_url: str
    disclaimer_text: str
    about_text: str


@dataclass(frozen=True)
class ProjectSettings:
    project_root: Path
    config_path: Path
    env_path: Path
    build: BuildSettings
    app: AppSettings
    runtime: RuntimeSettings
    meta: MetaSettings


@lru_cache(maxsize=4)
def load_settings(config_path: Path | None = None, env_path: Path | None = None) -> ProjectSettings:
    project_root = PROJECT_ROOT

    config_override = os.environ.get("TG_CONFIG_PATH", "").strip()
    env_override = os.environ.get("TG_ENV_PATH", "").strip()

    chosen_config_path = Path(config_override) if config_override else (config_path or DEFAULT_CONFIG_PATH)
    chosen_env_path = Path(env_override) if env_override else (env_path or DEFAULT_ENV_PATH)

    if not chosen_config_path.is_absolute():
        chosen_config_path = (project_root / chosen_config_path).resolve()
    if not chosen_env_path.is_absolute():
        chosen_env_path = (project_root / chosen_env_path).resolve()

    config_data = _read_toml(chosen_config_path)
    dotenv_data = _read_dotenv(chosen_env_path)

    def _path_parser(value: Any) -> Path:
        return _to_path(value, project_root=project_root)

    build = BuildSettings(
        default_input_csv=_resolve_value(
            env_name="TG_BUILD_INPUT_CSV",
            config_keys=("build", "input_csv"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value="tweets_01-08-2021.csv",
            parser=_path_parser,
        ),
        default_output_dir=_resolve_value(
            env_name="TG_BUILD_OUTPUT_DIR",
            config_keys=("build", "output_dir"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value="data/processed",
            parser=_path_parser,
        ),
        min_mention_count=_resolve_value(
            env_name="TG_BUILD_MIN_MENTION_COUNT",
            config_keys=("build", "min_mention_count"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=1,
            parser=_to_int,
        ),
        global_min_mentions=_resolve_value(
            env_name="TG_BUILD_GLOBAL_MIN_MENTIONS",
            config_keys=("build", "global_min_mentions"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=8,
            parser=_to_int,
        ),
        include_retweets=_resolve_value(
            env_name="TG_BUILD_INCLUDE_RETWEETS",
            config_keys=("build", "include_retweets"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=True,
            parser=_to_bool,
        ),
        heat_decay=_resolve_value(
            env_name="TG_BUILD_HEAT_DECAY",
            config_keys=("build", "heat_decay"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=0.85,
            parser=_to_float,
        ),
        layout_seed=_resolve_value(
            env_name="TG_BUILD_LAYOUT_SEED",
            config_keys=("build", "layout_seed"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=42,
            parser=_to_int,
        ),
    )

    app = AppSettings(
        processed_dir=_resolve_value(
            env_name="TG_APP_PROCESSED_DIR",
            config_keys=("app", "processed_dir"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value="data/processed",
            parser=_path_parser,
        ),
        include_hub=_resolve_value(
            env_name="TG_APP_INCLUDE_HUB",
            config_keys=("app", "include_hub"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=True,
            parser=_to_bool,
        ),
        always_label_top_nodes=_resolve_value(
            env_name="TG_APP_ALWAYS_LABEL_TOP_NODES",
            config_keys=("app", "always_label_top_nodes"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=False,
            parser=_to_bool,
        ),
        playback_speed=_resolve_value(
            env_name="TG_APP_PLAYBACK_SPEED",
            config_keys=("app", "playback_speed"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=8.0,
            parser=_to_float,
        ),
        node_size_multiplier=_resolve_value(
            env_name="TG_APP_NODE_SIZE_MULTIPLIER",
            config_keys=("app", "node_size_multiplier"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=2.0,
            parser=_to_float,
        ),
        layout_spread=_resolve_value(
            env_name="TG_APP_LAYOUT_SPREAD",
            config_keys=("app", "layout_spread"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=1.0,
            parser=_to_float,
        ),
        initial_zoom_boost=_resolve_value(
            env_name="TG_APP_INITIAL_ZOOM_BOOST",
            config_keys=("app", "initial_zoom_boost"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=0.8,
            parser=_to_float,
        ),
        graph_height_px=_resolve_value(
            env_name="TG_APP_GRAPH_HEIGHT_PX",
            config_keys=("app", "graph_height_px"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=1000,
            parser=_to_int,
        ),
    )

    streamlit_host = _resolve_value(
        env_name="TG_RUNTIME_STREAMLIT_HOST",
        config_keys=("runtime", "streamlit_host"),
        config_data=config_data,
        dotenv_data=dotenv_data,
        default_value="0.0.0.0",
        parser=_to_str,
    )
    streamlit_port = _resolve_value(
        env_name="TG_RUNTIME_STREAMLIT_PORT",
        config_keys=("runtime", "streamlit_port"),
        config_data=config_data,
        dotenv_data=dotenv_data,
        default_value=3001,
        parser=_to_int,
    )
    docs_host = _resolve_value(
        env_name="TG_RUNTIME_DOCS_HOST",
        config_keys=("runtime", "docs_host"),
        config_data=config_data,
        dotenv_data=dotenv_data,
        default_value="0.0.0.0",
        parser=_to_str,
    )
    docs_port = _resolve_value(
        env_name="TG_RUNTIME_DOCS_PORT",
        config_keys=("runtime", "docs_port"),
        config_data=config_data,
        dotenv_data=dotenv_data,
        default_value=3002,
        parser=_to_int,
    )
    docs_url = _resolve_value(
        env_name="TG_RUNTIME_DOCS_URL",
        config_keys=("runtime", "docs_url"),
        config_data=config_data,
        dotenv_data=dotenv_data,
        default_value=f"http://localhost:{docs_port}",
        parser=_to_str,
    )
    docs_site_dir = _resolve_value(
        env_name="TG_RUNTIME_DOCS_SITE_DIR",
        config_keys=("runtime", "docs_site_dir"),
        config_data=config_data,
        dotenv_data=dotenv_data,
        default_value="docs-site",
        parser=_path_parser,
    )

    runtime = RuntimeSettings(
        streamlit_host=streamlit_host,
        streamlit_port=streamlit_port,
        docs_host=docs_host,
        docs_port=docs_port,
        docs_url=docs_url,
        docs_site_dir=docs_site_dir,
    )

    meta = MetaSettings(
        app_name=_resolve_value(
            env_name="TG_META_APP_NAME",
            config_keys=("meta", "app_name"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value="Trump Graph",
            parser=_to_str,
        ),
        logo_path=_resolve_value(
            env_name="TG_META_LOGO_PATH",
            config_keys=("meta", "logo_path"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value="logo.png",
            parser=_path_parser,
        ),
        data_source_url=_resolve_value(
            env_name="TG_META_DATA_SOURCE_URL",
            config_keys=("meta", "data_source_url"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value="https://www.thetrumparchive.com/",
            parser=_to_str,
        ),
        repository_url=_resolve_value(
            env_name="TG_META_REPOSITORY_URL",
            config_keys=("meta", "repository_url"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value="https://github.com/KarolNarozniak/Trump-tweet-visualise",
            parser=_to_str,
        ),
        disclaimer_text=_resolve_value(
            env_name="TG_META_DISCLAIMER_TEXT",
            config_keys=("meta", "disclaimer_text"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=(
                "Disclaimer: This visualization is a student educational project for studying "
                "time-dependent graph behavior. It is not intended as political endorsement, "
                "defamation, or factual/legal judgment about any person."
            ),
            parser=_to_str,
        ),
        about_text=_resolve_value(
            env_name="TG_META_ABOUT_TEXT",
            config_keys=("meta", "about_text"),
            config_data=config_data,
            dotenv_data=dotenv_data,
            default_value=(
                "Trump Graph is a student project focused on temporal network analysis. "
                "Its purpose is to explore how mentions and co-mentions evolve over time "
                "in archived tweet data. The project is purely educational."
            ),
            parser=_to_str,
        ),
    )

    return ProjectSettings(
        project_root=project_root,
        config_path=chosen_config_path,
        env_path=chosen_env_path,
        build=build,
        app=app,
        runtime=runtime,
        meta=meta,
    )


def clear_settings_cache() -> None:
    load_settings.cache_clear()
