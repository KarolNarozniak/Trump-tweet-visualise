from __future__ import annotations

from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trump_graph.app import (
    build_global_animation_html,
    build_truth_semantic_animation_html,
    load_forecast_animation_artifacts,
    load_global_animation_artifacts,
    load_unified_animation_artifacts,
    load_unified_edge_catalog,
    load_unified_enriched_posts,
    load_unified_node_catalog,
    load_unified_week_index,
    load_unified_weekly_summary,
    load_truth_animation_artifacts,
    load_truth_enriched_posts,
    load_truth_node_catalog,
    load_truth_week_index,
    load_truth_weekly_summary,
    load_week_artifacts,
    load_week_index,
)
from trump_graph.forecast import build_baseline_forecast_payload
from trump_graph.forecast_registry import FORECAST_MODEL_REGISTRY
from trump_graph.settings import ProjectSettings, load_settings


@st.cache_data(show_spinner=False)
def _cached_week_index(processed_dir: str):
    return load_week_index(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_week_artifacts(processed_dir: str, week_id: str):
    return load_week_artifacts(Path(processed_dir), week_id)


@st.cache_data(show_spinner=False)
def _cached_global_animation_payload(processed_dir: str):
    return load_global_animation_artifacts(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_truth_week_index(processed_dir: str):
    return load_truth_week_index(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_truth_weekly_summary(processed_dir: str):
    return load_truth_weekly_summary(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_truth_animation_payload(processed_dir: str):
    return load_truth_animation_artifacts(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_truth_enriched_posts(processed_dir: str):
    return load_truth_enriched_posts(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_truth_node_catalog(processed_dir: str):
    return load_truth_node_catalog(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_unified_week_index(processed_dir: str):
    return load_unified_week_index(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_unified_weekly_summary(processed_dir: str):
    return load_unified_weekly_summary(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_unified_animation_payload(processed_dir: str):
    return load_unified_animation_artifacts(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_unified_enriched_posts(processed_dir: str):
    return load_unified_enriched_posts(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_unified_node_catalog(processed_dir: str):
    return load_unified_node_catalog(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_unified_edge_catalog(processed_dir: str):
    return load_unified_edge_catalog(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_forecast_animation_payload(processed_dir: str):
    return load_forecast_animation_artifacts(Path(processed_dir))


def _inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
            .stMetric {
                border: 1px solid #D6E4FF;
                border-radius: 8px;
                padding: 0.4rem 0.6rem;
                background: #F8FBFF;
            }
            .stCaption {
                color: #334155;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_logo_path(settings: ProjectSettings) -> Path | None:
    logo_path = settings.meta.logo_path
    return logo_path if logo_path.exists() else None


def _current_page_slug() -> str:
    raw_page = st.query_params.get("page", "graph")
    if isinstance(raw_page, list):
        raw_page = raw_page[0] if raw_page else "graph"
    slug = str(raw_page).strip().lower()
    if slug == "truth":
        return "semantic"
    return slug if slug in {"graph", "semantic", "forecast", "about"} else "graph"


def _set_page_slug(page_slug: str) -> None:
    st.query_params["page"] = page_slug


def _graph_iframe_path(graph_html: str) -> Path:
    embed_dir = ROOT_DIR / ".streamlit" / "generated"
    embed_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(graph_html.encode("utf-8")).hexdigest()[:18]
    iframe_path = embed_dir / f"graph_{digest}.html"
    if not iframe_path.exists():
        iframe_path.write_text(graph_html, encoding="utf-8")
    return iframe_path


def _render_footer_links(settings: ProjectSettings) -> None:
    st.markdown("---")
    st.subheader("Data Source, Repository, And Documentation")
    st.markdown(
        f"- Data source: [The Trump Archive]({settings.meta.data_source_url})\n"
        f"- Code repository: [KarolNarozniak/Trump-tweet-visualise]({settings.meta.repository_url})\n"
        f"- Documentation: [Project Docs]({settings.runtime.docs_url})"
    )


def _render_about_page(settings: ProjectSettings, logo_path: Path | None) -> None:
    st.header("About")
    if logo_path is not None:
        st.image(str(logo_path), width=140)

    st.write(settings.meta.about_text)
    st.markdown(
        "- This is a student project for time-dependent graph study.\n"
        "- Home Graph page covers the original mention/co-mention timeline.\n"
        "- Semantic page covers labeled topic/entity temporal structure.\n"
        "- Forecast page is prepared for temporal neural network predictions.\n"
        "- It is purely educational and intended for learning and research practice."
    )
    st.info("About endpoint: add `?page=about` to the app URL.")
    _render_footer_links(settings)


def _parse_json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, list) else []


def _semantic_delta_set(include_reposts: bool, sentiment_filter: str) -> str:
    normalized_sentiment = sentiment_filter.strip().lower()
    base = "all" if include_reposts else "original"
    if normalized_sentiment == "all":
        return base
    return normalized_sentiment if include_reposts else f"original_{normalized_sentiment}"


def _semantic_week_posts(
    posts_df: pd.DataFrame,
    *,
    week_id: str,
    include_reposts: bool,
    sentiment_filter: str,
    repost_column: str,
) -> pd.DataFrame:
    filtered = posts_df.loc[posts_df["week_id"].astype(str) == str(week_id)].copy()
    if repost_column not in filtered.columns:
        filtered[repost_column] = False
    if not include_reposts:
        filtered = filtered.loc[~filtered[repost_column].astype(bool)].copy()
    if sentiment_filter != "all":
        filtered = filtered.loc[filtered["sentiment_label"].astype(str).str.lower() == sentiment_filter].copy()
    return filtered


def _semantic_top_nodes(
    posts_df: pd.DataFrame,
    *,
    included_node_types: set[str],
    min_count: int,
    limit: int = 30,
) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for value in posts_df.get("semantic_nodes", pd.Series(dtype="string")).tolist():
        for node_id in _parse_json_list(value):
            node_id_str = str(node_id)
            node_type = node_id_str.split("::", 1)[0]
            if node_type in included_node_types:
                counts[node_id_str] += 1

    rows = []
    for node_id, count in counts.most_common(limit * 2):
        if count < min_count:
            continue
        node_type, _, label = node_id.partition("::")
        if node_type == "topic":
            display_label = label.replace("_", " ").title()
        elif node_type == "hashtag":
            display_label = f"#{label}"
        elif node_type == "mention":
            display_label = f"@{label}"
        else:
            display_label = label.title()
        rows.append({"node": node_id, "label": display_label, "node_type": node_type, "count": int(count)})
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows, columns=["node", "label", "node_type", "count"])


def _truth_delta_set(include_retruths: bool, sentiment_filter: str) -> str:
    return _semantic_delta_set(include_reposts=include_retruths, sentiment_filter=sentiment_filter)


def _truth_week_posts(
    posts_df: pd.DataFrame,
    *,
    week_id: str,
    include_retruths: bool,
    sentiment_filter: str,
) -> pd.DataFrame:
    return _semantic_week_posts(
        posts_df,
        week_id=week_id,
        include_reposts=include_retruths,
        sentiment_filter=sentiment_filter,
        repost_column="is_retruth",
    )


def _truth_top_nodes(
    posts_df: pd.DataFrame,
    *,
    included_node_types: set[str],
    min_count: int,
    limit: int = 30,
) -> pd.DataFrame:
    return _semantic_top_nodes(
        posts_df,
        included_node_types=included_node_types,
        min_count=min_count,
        limit=limit,
    )


def _ensure_delta_sets(payload: dict[str, object]) -> dict[str, object]:
    if "delta_sets" in payload:
        return payload
    node_week_deltas = payload.get("node_week_deltas", [])
    edge_week_deltas = payload.get("edge_week_deltas", [])
    out = dict(payload)
    out["delta_sets"] = {
        "all": {
            "node_week_deltas": node_week_deltas,
            "edge_week_deltas": edge_week_deltas,
        }
    }
    return out


def _forecast_model_output_dir(forecast_root: Path, model_key: str) -> Path:
    return forecast_root / model_key


def _load_optional_model_metrics(model_dir: Path) -> dict[str, object]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _forecast_model_status_rows(forecast_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def _metric_as_text(raw_value: object) -> str:
        if raw_value is None or raw_value == "":
            return "n/a"
        try:
            return f"{float(raw_value):.6f}"
        except (TypeError, ValueError):
            return "n/a"

    for model in FORECAST_MODEL_REGISTRY:
        model_key = model["key"]
        model_dir = _forecast_model_output_dir(forecast_root, model_key)
        artifact_path = model_dir / "forecast_graph" / "animation_state.json"
        metrics = _load_optional_model_metrics(model_dir)
        model_card_path = model_dir / "MODEL_CARD.md"
        rows.append(
            {
                "model": model["label"],
                "category": model["category"],
                "status": "ready" if artifact_path.exists() else "missing",
                "artifact_path": str(artifact_path),
                "model_card": str(model_card_path) if model_card_path.exists() else "missing",
                "best_val_score": _metric_as_text(metrics.get("best_val_score")),
                "mae": _metric_as_text(metrics.get("mae")),
                "rmse": _metric_as_text(metrics.get("rmse")),
                "map": _metric_as_text(metrics.get("map")),
            }
        )
    return pd.DataFrame(rows)


def _format_semantic_dataset_summary(
    week_index: pd.DataFrame,
    enriched_posts: pd.DataFrame,
    node_catalog: pd.DataFrame,
    repost_column: str,
) -> tuple[list[tuple[str, str]], pd.DataFrame, pd.DataFrame]:
    total_posts = int(len(enriched_posts))
    repost_count = int(enriched_posts[repost_column].sum()) if repost_column in enriched_posts.columns else 0
    original_count = max(0, total_posts - repost_count)
    start_date = str(week_index["week_start"].iloc[0]) if not week_index.empty else "-"
    end_date = str(week_index["week_end"].iloc[-1]) if not week_index.empty else "-"
    sentiment_counts = (
        enriched_posts["sentiment_label"].astype(str).str.lower().value_counts() if "sentiment_label" in enriched_posts.columns else pd.Series(dtype="int64")
    )

    summary_items = [
        ("Weekly rows", f"{len(week_index):,}"),
        ("Date range", f"{start_date} to {end_date}"),
        ("Total posts", f"{total_posts:,}"),
        ("Original posts", f"{original_count:,} ({(original_count / total_posts * 100) if total_posts else 0:.2f}%)"),
        ("Reposts", f"{repost_count:,} ({(repost_count / total_posts * 100) if total_posts else 0:.2f}%)"),
        ("Negative", f"{int(sentiment_counts.get('negative', 0)):,} ({(sentiment_counts.get('negative', 0) / total_posts * 100) if total_posts else 0:.2f}%)"),
        ("Neutral", f"{int(sentiment_counts.get('neutral', 0)):,} ({(sentiment_counts.get('neutral', 0) / total_posts * 100) if total_posts else 0:.2f}%)"),
        ("Positive", f"{int(sentiment_counts.get('positive', 0)):,} ({(sentiment_counts.get('positive', 0) / total_posts * 100) if total_posts else 0:.2f}%)"),
    ]

    top_labels = node_catalog.copy()
    if not top_labels.empty:
        top_labels = top_labels.assign(
            label_display=top_labels["label"].astype(str),
            type_display=top_labels["node_type"].astype(str),
        )
        top_labels = top_labels.loc[:, ["label_display", "type_display", "total_count"]].rename(
            columns={"label_display": "Label", "type_display": "Type", "total_count": "Count"}
        )
        top_labels = top_labels.head(12).reset_index(drop=True)
    else:
        top_labels = pd.DataFrame(columns=["Label", "Type", "Count"])

    type_breakdown = (
        node_catalog.groupby("node_type", as_index=False)["total_count"].sum().sort_values("total_count", ascending=False)
        if not node_catalog.empty
        else pd.DataFrame(columns=["node_type", "total_count"])
    )
    if not type_breakdown.empty:
        type_breakdown = type_breakdown.rename(columns={"node_type": "Type", "total_count": "Count"}).reset_index(drop=True)
    return summary_items, top_labels, type_breakdown


def _render_semantic_page(settings: ProjectSettings) -> None:
    st.sidebar.header("Semantic Controls")
    source_mode = st.sidebar.selectbox(
        "Semantic source",
        options=["Unified (Twitter + Truth)", "Truth-only"],
        index=0,
    )

    default_processed_dir = (
        settings.semantic_app.processed_dir
        if source_mode.startswith("Unified")
        else settings.truth_app.processed_dir
    )
    processed_dir_input = st.sidebar.text_input(
        "Semantic processed data directory",
        value=str(default_processed_dir),
    )
    include_reposts_default = settings.semantic_app.include_reposts if source_mode.startswith("Unified") else settings.truth_app.include_retruths
    include_reposts = st.sidebar.checkbox(
        "Include reposts",
        value=include_reposts_default,
    )
    sentiment_default = settings.semantic_app.sentiment_filter if source_mode.startswith("Unified") else settings.truth_app.sentiment_filter
    sentiment_filter = st.sidebar.selectbox(
        "Sentiment",
        options=["all", "negative", "neutral", "positive"],
        index=["all", "negative", "neutral", "positive"].index(sentiment_default)
        if sentiment_default in {"all", "negative", "neutral", "positive"}
        else 0,
    )
    processed_dir = Path(processed_dir_input)
    if not processed_dir.exists():
        st.warning("Semantic artifacts not found. Run the build command first.")
        if source_mode.startswith("Unified"):
            st.code(
                f'python -m trump_graph build-unified --twitter-input "{settings.unified_build.twitter_input_csv}" --truth-input "{settings.unified_build.truth_input_path}" --out "{settings.unified_build.output_dir}"'
            )
        else:
            st.code(
                f'python -m trump_graph build-truth --input "{settings.truth_build.input_path}" --out "{settings.truth_build.output_dir}"'
            )
        return

    try:
        if source_mode.startswith("Unified"):
            week_index = _cached_unified_week_index(str(processed_dir))
            weekly_summary = _cached_unified_weekly_summary(str(processed_dir))
            animation_payload = _cached_unified_animation_payload(str(processed_dir))
            enriched_posts = _cached_unified_enriched_posts(str(processed_dir))
            node_catalog = _cached_unified_node_catalog(str(processed_dir))
            edge_catalog = _cached_unified_edge_catalog(str(processed_dir))
            repost_column = "is_repost"
        else:
            week_index = _cached_truth_week_index(str(processed_dir))
            weekly_summary = _cached_truth_weekly_summary(str(processed_dir))
            animation_payload = _cached_truth_animation_payload(str(processed_dir))
            enriched_posts = _cached_truth_enriched_posts(str(processed_dir))
            node_catalog = _cached_truth_node_catalog(str(processed_dir))
            edge_catalog = pd.DataFrame(
                [
                    {
                        "edge_id": edge.get("id"),
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "total_count": edge.get("total_co_occurrences", edge.get("total_co_mentions", 0)),
                    }
                    for edge in animation_payload.get("global_edges", [])
                ]
            )
            repost_column = "is_retruth"
    except FileNotFoundError:
        st.warning("Semantic artifacts are incomplete for this directory.")
        return
    except ValueError as error:
        st.error(f"Failed to read semantic artifacts: {error}")
        return

    available_node_types = animation_payload.get("available_node_types", ["topic", "per", "org", "loc"])
    default_node_types = [node_type for node_type in settings.semantic_app.node_types if node_type in available_node_types]
    selected_node_types = st.sidebar.multiselect(
        "Semantic node types",
        options=available_node_types,
        default=default_node_types or list(available_node_types[:4]),
    )
    if not selected_node_types:
        st.warning("Choose at least one semantic node type.")
        return

    max_node_count = int(node_catalog["total_count"].max()) if not node_catalog.empty else 1
    min_node_count = st.sidebar.slider(
        "Minimum semantic label count",
        min_value=1,
        max_value=max(1, min(max_node_count, 300)),
        value=int(max(1, min(settings.semantic_app.min_node_count, max(1, min(max_node_count, 300))))),
        step=1,
    )
    playback_speed = st.sidebar.slider(
        "Initial playback speed (weeks/sec)",
        min_value=0.5,
        max_value=8.0,
        value=float(max(0.5, min(8.0, settings.semantic_app.playback_speed))),
        step=0.5,
    )
    node_size_multiplier = st.sidebar.slider(
        "Node size multiplier",
        min_value=0.5,
        max_value=2.5,
        value=float(max(0.5, min(2.5, settings.semantic_app.node_size_multiplier))),
        step=0.05,
    )
    layout_spread = st.sidebar.slider(
        "Layout spread",
        min_value=0.7,
        max_value=4.0,
        value=float(max(0.7, min(4.0, settings.semantic_app.layout_spread))),
        step=0.1,
    )
    initial_zoom_boost = st.sidebar.slider(
        "Initial graph zoom",
        min_value=0.55,
        max_value=2.5,
        value=float(max(0.55, min(2.5, settings.semantic_app.initial_zoom_boost))),
        step=0.05,
    )
    graph_height = st.sidebar.slider(
        "Graph height (px)",
        min_value=560,
        max_value=1200,
        value=int(max(560, min(1200, settings.semantic_app.graph_height_px))),
        step=20,
    )

    if week_index.empty:
        st.warning("No semantic weeks are available in this artifact directory.")
        return

    st.subheader("Semantic Temporal Graph")
    delta_set_name = _semantic_delta_set(include_reposts=include_reposts, sentiment_filter=sentiment_filter)
    payload_with_sets = _ensure_delta_sets(animation_payload)
    graph_html = build_truth_semantic_animation_html(
        payload=payload_with_sets,
        included_node_types=set(selected_node_types),
        min_total_count=min_node_count,
        delta_set_name=delta_set_name,
        initial_speed=playback_speed,
        node_size_multiplier=node_size_multiplier,
        initial_zoom_boost=initial_zoom_boost,
        layout_spread=layout_spread,
        height_px=graph_height,
    )
    graph_iframe = _graph_iframe_path(graph_html)
    st.iframe(graph_iframe, width="stretch", height=graph_height + 180)
    st.caption(
        "Legend: warm nodes are active this week, white nodes are previously seen; "
        "edges thicken cumulatively and glow when active in the current week."
    )

    week_options = week_index["week_id"].astype(str).tolist()
    detail_week = st.selectbox("Week For Tables And Export", options=week_options, index=len(week_options) - 1)
    week_posts = _semantic_week_posts(
        enriched_posts,
        week_id=detail_week,
        include_reposts=include_reposts,
        sentiment_filter=sentiment_filter,
        repost_column=repost_column,
    )
    top_nodes_df = _semantic_top_nodes(
        week_posts,
        included_node_types=set(selected_node_types),
        min_count=1,
    )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Posts", len(week_posts))
    metric_col_2.metric("Reposts", int(week_posts[repost_column].sum()) if repost_column in week_posts.columns else 0)
    metric_col_3.metric("Active Nodes", len(top_nodes_df))
    metric_col_4.metric("Sentiment", sentiment_filter.title())

    st.subheader("Top Active Topics And Entities")
    st.dataframe(top_nodes_df, width="stretch", hide_index=True)

    if not edge_catalog.empty:
        top_edges = edge_catalog.sort_values("total_count", ascending=False, kind="mergesort").head(20).copy()
        if {"source", "target", "total_count"}.issubset(top_edges.columns):
            top_edges = top_edges.loc[:, ["source", "target", "total_count"]].rename(
                columns={"source": "Source", "target": "Target", "total_count": "Count"}
            )
        st.markdown("**Strongest Semantic Connections**")
        st.dataframe(top_edges, width="stretch", hide_index=True)

    with st.expander("Enriched Posts This Week"):
        visible_columns = [
            column
            for column in [
                "post_id",
                "source_platform",
                "created_at_utc",
                repost_column,
                "top_topic",
                "top_topic_score",
                "sentiment_label",
                "sentiment_score",
                "text",
            ]
            if column in week_posts.columns
        ]
        st.dataframe(week_posts[visible_columns].head(120), width="stretch", hide_index=True)

    st.subheader("Dataset Snapshot")
    summary_items, top_labels_df, type_breakdown_df = _format_semantic_dataset_summary(
        week_index=week_index,
        enriched_posts=enriched_posts,
        node_catalog=node_catalog,
        repost_column=repost_column,
    )
    summary_cols = st.columns(2)
    summary_cols[0].markdown("\n".join(f"- **{label}:** {value}" for label, value in summary_items[:4]))
    summary_cols[1].markdown("\n".join(f"- **{label}:** {value}" for label, value in summary_items[4:]))

    labels_col, types_col = st.columns(2)
    labels_col.markdown("**Most Present Labels**")
    labels_col.dataframe(top_labels_df, width="stretch", hide_index=True)
    types_col.markdown("**Most Common Node Types**")
    types_col.dataframe(type_breakdown_df, width="stretch", hide_index=True)

    st.subheader("Export Semantic Week")
    export_col_1, export_col_2, export_col_3 = st.columns(3)
    export_col_1.download_button(
        label="Download enriched posts",
        data=week_posts.to_csv(index=False).encode("utf-8"),
        file_name=f"{detail_week}_semantic_posts.csv",
        mime="text/csv",
    )
    export_col_2.download_button(
        label="Download top nodes",
        data=top_nodes_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{detail_week}_semantic_top_nodes.csv",
        mime="text/csv",
    )
    summary_row = weekly_summary.loc[weekly_summary["week_id"].astype(str) == str(detail_week)]
    export_col_3.download_button(
        label="Download summary.json",
        data=json.dumps(summary_row.to_dict(orient="records"), indent=2).encode("utf-8"),
        file_name=f"{detail_week}_semantic_summary.json",
        mime="application/json",
    )

    _render_footer_links(settings)


def _render_forecast_page(settings: ProjectSettings) -> None:
    st.sidebar.header("Temporal NN Controls")
    semantic_dir_input = st.sidebar.text_input(
        "Semantic source directory",
        value=str(settings.forecast_app.semantic_processed_dir),
    )
    forecast_root_input = st.sidebar.text_input(
        "Forecast models root directory",
        value=str(settings.forecast_app.forecast_processed_dir),
    )
    default_mode = settings.forecast_app.default_mode.strip().lower()
    forecast_source_options = ["baseline"] + [model["key"] for model in FORECAST_MODEL_REGISTRY]
    if default_mode not in forecast_source_options:
        default_mode = "baseline"
    mode = st.sidebar.selectbox(
        "Forecast source",
        options=forecast_source_options,
        index=forecast_source_options.index(default_mode),
        format_func=lambda key: (
            "Baseline preview"
            if key == "baseline"
            else next((f"{model['label']} ({model['category']})" for model in FORECAST_MODEL_REGISTRY if model["key"] == key), key)
        ),
    )
    horizon_weeks = st.sidebar.slider(
        "Future horizon (weeks)",
        min_value=2,
        max_value=52,
        value=int(max(2, min(52, settings.forecast_app.horizon_weeks))),
        step=1,
    )
    lookback_weeks = st.sidebar.slider(
        "Baseline lookback (weeks)",
        min_value=2,
        max_value=52,
        value=int(max(2, min(52, settings.forecast_app.lookback_weeks))),
        step=1,
    )
    playback_speed = st.sidebar.slider(
        "Initial playback speed (weeks/sec)",
        min_value=0.5,
        max_value=8.0,
        value=float(max(0.5, min(8.0, settings.forecast_app.playback_speed))),
        step=0.5,
    )
    node_size_multiplier = st.sidebar.slider(
        "Node size multiplier",
        min_value=0.5,
        max_value=2.5,
        value=float(max(0.5, min(2.5, settings.forecast_app.node_size_multiplier))),
        step=0.05,
    )
    layout_spread = st.sidebar.slider(
        "Layout spread",
        min_value=0.7,
        max_value=4.0,
        value=float(max(0.7, min(4.0, settings.forecast_app.layout_spread))),
        step=0.1,
    )
    initial_zoom_boost = st.sidebar.slider(
        "Initial graph zoom",
        min_value=0.55,
        max_value=2.5,
        value=float(max(0.55, min(2.5, settings.forecast_app.initial_zoom_boost))),
        step=0.05,
    )
    graph_height = st.sidebar.slider(
        "Graph height (px)",
        min_value=560,
        max_value=1200,
        value=int(max(560, min(1200, settings.forecast_app.graph_height_px))),
        step=20,
    )

    semantic_dir = Path(semantic_dir_input)
    if not semantic_dir.exists():
        st.warning("Semantic source artifacts not found. Build unified semantic artifacts first.")
        st.code(
            f'python -m trump_graph build-unified --twitter-input "{settings.unified_build.twitter_input_csv}" --truth-input "{settings.unified_build.truth_input_path}" --out "{settings.unified_build.output_dir}"'
        )
        return

    try:
        semantic_payload = _cached_unified_animation_payload(str(semantic_dir))
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Failed to read semantic source payload: {error}")
        return

    payload_to_render: dict[str, object]
    backend_label = "Baseline preview"
    forecast_root = Path(forecast_root_input)
    if mode != "baseline":
        model_dir = _forecast_model_output_dir(forecast_root, mode)
        try:
            payload_to_render = _cached_forecast_animation_payload(str(model_dir))
            payload_to_render = _ensure_delta_sets(payload_to_render)
            model_label = next((model["label"] for model in FORECAST_MODEL_REGISTRY if model["key"] == mode), mode)
            backend_label = f"{model_label} trained model output"
        except FileNotFoundError:
            st.warning(
                f"Model artifact not found for `{mode}` at `{model_dir / 'forecast_graph' / 'animation_state.json'}`. Using baseline preview."
            )
            payload_to_render = build_baseline_forecast_payload(
                semantic_payload,
                horizon_weeks=horizon_weeks,
                lookback_weeks=lookback_weeks,
            )
            payload_to_render = _ensure_delta_sets(payload_to_render)
        except ValueError as error:
            st.error(f"Forecast artifact schema issue: {error}")
            return
    else:
        payload_to_render = build_baseline_forecast_payload(
            semantic_payload,
            horizon_weeks=horizon_weeks,
            lookback_weeks=lookback_weeks,
        )
        payload_to_render = _ensure_delta_sets(payload_to_render)

    available_node_types = payload_to_render.get("available_node_types", ["topic", "per", "org", "loc"])
    default_node_types = [node_type for node_type in settings.forecast_app.node_types if node_type in available_node_types]
    selected_node_types = st.sidebar.multiselect(
        "Forecast node types",
        options=available_node_types,
        default=default_node_types or list(available_node_types[:4]),
    )
    if not selected_node_types:
        st.warning("Choose at least one node type.")
        return
    min_node_count = st.sidebar.slider(
        "Minimum semantic label count",
        min_value=1,
        max_value=300,
        value=int(max(1, min(300, settings.forecast_app.min_node_count))),
        step=1,
    )

    st.subheader("Temporal Neural Network Forecast")
    st.caption(
        f"Source: {backend_label}. This page is prepared for your trained model artifacts and can already preview future dynamics."
    )
    with st.expander("Three Planned Temporal Models"):
        for model in FORECAST_MODEL_REGISTRY:
            st.markdown(
                f"- **{model['label']}** ({model['category']}): {model['desc']} "
                f"[Paper]({model['paper_url']}) · [Implementation Reference]({model['impl_url']})"
            )
    graph_html = build_truth_semantic_animation_html(
        payload=payload_to_render,
        included_node_types=set(selected_node_types),
        min_total_count=min_node_count,
        delta_set_name="all",
        initial_speed=playback_speed,
        node_size_multiplier=node_size_multiplier,
        initial_zoom_boost=initial_zoom_boost,
        layout_spread=layout_spread,
        height_px=graph_height,
    )
    graph_iframe = _graph_iframe_path(graph_html)
    st.iframe(graph_iframe, width="stretch", height=graph_height + 180)
    st.caption(
        "Future weeks are appended after historical weeks; keep node positions fixed and compare heat/edge growth through time."
    )

    weeks = payload_to_render.get("weeks", [])
    history_weeks = sum(1 for week in weeks if not bool(week.get("is_forecast", False)))
    forecast_weeks = sum(1 for week in weeks if bool(week.get("is_forecast", False)))
    metric_cols = st.columns(4)
    metric_cols[0].metric("History weeks", f"{history_weeks:,}")
    metric_cols[1].metric("Forecast weeks", f"{forecast_weeks:,}")
    metric_cols[2].metric("Total nodes", f"{len(payload_to_render.get('global_nodes', [])):,}")
    metric_cols[3].metric("Total edges", f"{len(payload_to_render.get('global_edges', [])):,}")

    st.subheader("Model Comparison Readiness")
    model_status_df = _forecast_model_status_rows(forecast_root)
    st.dataframe(model_status_df, width="stretch", hide_index=True)

    st.subheader("Model Artifact Contract")
    st.markdown(
        "- Place trained payloads in `data/processed_forecast/<model_key>/forecast_graph/animation_state.json`.\n"
        "- Supported model keys in UI: `tgn`, `evolvegcn`, `gconvgru`.\n"
        "- Required keys: `weeks`, `global_nodes`, `global_edges`, `node_week_deltas`, `edge_week_deltas`, `heat_decay`, `heat_scale`, `max_cumulative_edge`.\n"
        "- Optional: include `delta_sets` for additional filters; if omitted, the app uses `all` from top-level deltas.\n"
        "- Optional model metrics: `data/processed_forecast/<model_key>/metrics.json` with fields like `best_val_score`, `mae`, `rmse`, `map` for comparison table."
    )

    payload_export = json.dumps(payload_to_render, indent=2).encode("utf-8")
    st.download_button(
        label="Download rendered forecast payload",
        data=payload_export,
        file_name="forecast_animation_state.json",
        mime="application/json",
    )

    _render_footer_links(settings)


def _render_truth_page(settings: ProjectSettings) -> None:
    st.sidebar.header("Truth Controls")
    processed_dir_input = st.sidebar.text_input(
        "Truth processed data directory",
        value=str(settings.truth_app.processed_dir),
    )
    include_retruths = st.sidebar.checkbox(
        "Include ReTruths",
        value=settings.truth_app.include_retruths,
    )
    sentiment_filter = st.sidebar.selectbox(
        "Sentiment",
        options=["all", "negative", "neutral", "positive"],
        index=["all", "negative", "neutral", "positive"].index(settings.truth_app.sentiment_filter)
        if settings.truth_app.sentiment_filter in {"all", "negative", "neutral", "positive"}
        else 0,
    )
    processed_dir = Path(processed_dir_input)
    if not processed_dir.exists():
        st.warning("Truth artifacts not found. Run the Truth build command first.")
        st.code(
            f'python -m trump_graph build-truth --input "{settings.truth_build.input_path}" --out "{settings.truth_build.output_dir}"'
        )
        return

    try:
        week_index = _cached_truth_week_index(str(processed_dir))
        weekly_summary = _cached_truth_weekly_summary(str(processed_dir))
        animation_payload = _cached_truth_animation_payload(str(processed_dir))
        enriched_posts = _cached_truth_enriched_posts(str(processed_dir))
        node_catalog = _cached_truth_node_catalog(str(processed_dir))
    except FileNotFoundError:
        st.warning("Truth artifacts are incomplete. Run the Truth build command first.")
        st.code(
            f'python -m trump_graph build-truth --input "{settings.truth_build.input_path}" --out "{settings.truth_build.output_dir}"'
        )
        return
    except ValueError as error:
        st.error(f"Failed to read Truth artifacts: {error}")
        return

    available_node_types = animation_payload.get("available_node_types", ["topic", "per", "org", "loc"])
    default_node_types = [node_type for node_type in settings.truth_app.node_types if node_type in available_node_types]
    selected_node_types = st.sidebar.multiselect(
        "Semantic node types",
        options=available_node_types,
        default=default_node_types or ["topic", "per", "org", "loc"],
    )
    if not selected_node_types:
        st.warning("Choose at least one semantic node type.")
        return

    max_node_count = int(node_catalog["total_count"].max()) if not node_catalog.empty else 1
    min_node_count = st.sidebar.slider(
        "Minimum semantic label count",
        min_value=1,
        max_value=max(1, min(max_node_count, 250)),
        value=int(max(1, min(settings.truth_app.min_node_count, max(1, min(max_node_count, 250))))),
        step=1,
    )
    playback_speed = st.sidebar.slider(
        "Initial playback speed (weeks/sec)",
        min_value=0.5,
        max_value=8.0,
        value=float(max(0.5, min(8.0, settings.truth_app.playback_speed))),
        step=0.5,
    )
    node_size_multiplier = st.sidebar.slider(
        "Node size multiplier",
        min_value=0.5,
        max_value=2.5,
        value=float(max(0.5, min(2.5, settings.truth_app.node_size_multiplier))),
        step=0.05,
    )
    layout_spread = st.sidebar.slider(
        "Layout spread",
        min_value=0.7,
        max_value=4.0,
        value=float(max(0.7, min(4.0, settings.truth_app.layout_spread))),
        step=0.1,
    )
    initial_zoom_boost = st.sidebar.slider(
        "Initial graph zoom",
        min_value=0.55,
        max_value=2.5,
        value=float(max(0.55, min(2.5, settings.truth_app.initial_zoom_boost))),
        step=0.05,
    )
    graph_height = st.sidebar.slider(
        "Graph height (px)",
        min_value=560,
        max_value=1200,
        value=int(max(560, min(1200, settings.truth_app.graph_height_px))),
        step=20,
    )

    if week_index.empty:
        st.warning("No Truth weeks are available in this artifact directory.")
        return

    st.subheader("Truth Social Semantic Temporal Graph")
    delta_set_name = _truth_delta_set(include_retruths, sentiment_filter)
    graph_html = build_truth_semantic_animation_html(
        payload=animation_payload,
        included_node_types=set(selected_node_types),
        min_total_count=min_node_count,
        delta_set_name=delta_set_name,
        initial_speed=playback_speed,
        node_size_multiplier=node_size_multiplier,
        initial_zoom_boost=initial_zoom_boost,
        layout_spread=layout_spread,
        height_px=graph_height,
    )
    graph_iframe = _graph_iframe_path(graph_html)
    st.iframe(graph_iframe, width="stretch", height=graph_height + 180)
    st.caption(
        "Legend: white nodes have appeared before; warm nodes are active in the current week; "
        "edge thickness is cumulative co-occurrence and warm edges are active this week."
    )

    week_options = week_index["week_id"].astype(str).tolist()
    detail_week = st.selectbox("Week For Tables And Export", options=week_options, index=len(week_options) - 1)
    week_posts = _truth_week_posts(
        enriched_posts,
        week_id=detail_week,
        include_retruths=include_retruths,
        sentiment_filter=sentiment_filter,
    )
    top_nodes_df = _truth_top_nodes(
        week_posts,
        included_node_types=set(selected_node_types),
        min_count=1,
    )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Posts", len(week_posts))
    metric_col_2.metric("ReTruths", int(week_posts["is_retruth"].sum()) if not week_posts.empty else 0)
    metric_col_3.metric("Active Nodes", len(top_nodes_df))
    metric_col_4.metric("Sentiment", sentiment_filter.title())

    st.subheader("Top Active Topics And Entities")
    st.dataframe(top_nodes_df, width="stretch", hide_index=True)

    with st.expander("Enriched Posts This Week"):
        visible_columns = [
            column
            for column in [
                "post_id",
                "created_at_utc",
                "is_retruth",
                "top_topic",
                "top_topic_score",
                "sentiment_label",
                "sentiment_score",
                "text",
            ]
            if column in week_posts.columns
        ]
        st.dataframe(week_posts[visible_columns].head(100), width="stretch", hide_index=True)

    st.subheader("Export Truth Week")
    export_col_1, export_col_2, export_col_3 = st.columns(3)
    export_col_1.download_button(
        label="Download enriched posts",
        data=week_posts.to_csv(index=False).encode("utf-8"),
        file_name=f"{detail_week}_truth_posts.csv",
        mime="text/csv",
    )
    export_col_2.download_button(
        label="Download top nodes",
        data=top_nodes_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{detail_week}_truth_top_nodes.csv",
        mime="text/csv",
    )
    summary_row = weekly_summary.loc[weekly_summary["week_id"].astype(str) == str(detail_week)]
    export_col_3.download_button(
        label="Download summary.json",
        data=json.dumps(summary_row.to_dict(orient="records"), indent=2).encode("utf-8"),
        file_name=f"{detail_week}_truth_summary.json",
        mime="application/json",
    )

    _render_footer_links(settings)


def main() -> None:
    settings = load_settings()
    logo_path = _resolve_logo_path(settings)

    page_config: dict[str, object] = {"page_title": settings.meta.app_name, "layout": "wide"}
    if logo_path is not None:
        page_config["page_icon"] = str(logo_path)
    st.set_page_config(**page_config)

    _inject_ui_styles()

    current_page = _current_page_slug()
    st.sidebar.header("Navigation")
    page_options = ["Graph", "Semantic", "Forecast", "About"]
    selected_page_label = st.sidebar.radio(
        "Page",
        options=page_options,
        index=page_options.index(current_page.title()),
    )
    selected_page_slug = selected_page_label.lower()
    if selected_page_slug != current_page:
        _set_page_slug(selected_page_slug)
        st.rerun()

    logo_col, title_col, nav_col = st.columns([0.12, 0.56, 0.32])
    if logo_path is not None:
        logo_col.image(str(logo_path), width=86)
    title_col.title(settings.meta.app_name)
    title_col.caption("Stable time-dependent mention network exploration.")

    nav_col.link_button("Open Docs", settings.runtime.docs_url, width="stretch")
    nav_col.markdown("[Graph](?page=graph) | [Semantic](?page=semantic) | [Forecast](?page=forecast) | [About](?page=about)")

    st.warning(settings.meta.disclaimer_text)

    if current_page == "about":
        _render_about_page(settings, logo_path)
        return
    if current_page == "semantic":
        _render_semantic_page(settings)
        return
    if current_page == "forecast":
        _render_forecast_page(settings)
        return

    st.sidebar.header("Graph Controls")
    st.sidebar.caption(f"Docs endpoint: {settings.runtime.docs_url}")
    processed_dir_input = st.sidebar.text_input("Processed data directory", value=str(settings.app.processed_dir))
    include_hub = st.sidebar.checkbox("Include @realdonaldtrump", value=settings.app.include_hub)
    always_label_top = st.sidebar.checkbox("Always label top 40 nodes", value=settings.app.always_label_top_nodes)
    playback_speed = st.sidebar.slider(
        "Initial playback speed (weeks/sec)",
        min_value=0.5,
        max_value=8.0,
        value=float(max(0.5, min(8.0, settings.app.playback_speed))),
        step=0.5,
    )
    node_size_multiplier = st.sidebar.slider(
        "Node size multiplier",
        min_value=0.5,
        max_value=2.5,
        value=float(max(0.5, min(2.5, settings.app.node_size_multiplier))),
        step=0.05,
    )
    layout_spread = st.sidebar.slider(
        "Layout spread",
        min_value=0.8,
        max_value=4.0,
        value=float(max(0.8, min(4.0, settings.app.layout_spread))),
        step=0.1,
    )
    initial_zoom_boost = st.sidebar.slider(
        "Initial graph zoom",
        min_value=0.55,
        max_value=2.5,
        value=float(max(0.55, min(2.5, settings.app.initial_zoom_boost))),
        step=0.05,
    )
    graph_height = st.sidebar.slider(
        "Graph height (px)",
        min_value=560,
        max_value=1200,
        value=int(max(560, min(1200, settings.app.graph_height_px))),
        step=20,
    )

    processed_dir = Path(processed_dir_input)
    if not processed_dir.exists():
        st.error(f"Directory not found: {processed_dir}")
        return

    try:
        week_index = _cached_week_index(str(processed_dir))
        global_payload = _cached_global_animation_payload(str(processed_dir))
    except FileNotFoundError:
        st.warning("Processed artifacts not found. Run the build command first.")
        st.code(
            f'python -m trump_graph build --input "{settings.build.default_input_csv}" --out "{settings.build.default_output_dir}" '
            f"--min-mention-count {settings.build.min_mention_count} --global-min-mentions {settings.build.global_min_mentions} "
            f"--heat-decay {settings.build.heat_decay} --layout-seed {settings.build.layout_seed}"
        )
        return
    except ValueError as error:
        st.error(f"Failed to read animation artifacts: {error}")
        return

    if week_index.empty:
        st.warning("No weekly artifacts available in this directory.")
        return

    st.subheader("Global Time-Dependent Graph")
    graph_html = build_global_animation_html(
        payload=global_payload,
        include_hub=include_hub,
        always_label_top_nodes=always_label_top,
        initial_week_index=None,
        initial_speed=playback_speed,
        node_size_multiplier=node_size_multiplier,
        initial_zoom_boost=initial_zoom_boost,
        layout_spread=layout_spread,
        height_px=graph_height,
    )
    graph_iframe = _graph_iframe_path(graph_html)
    st.iframe(graph_iframe, width="stretch", height=graph_height + 180)

    st.caption(
        "Legend: node size = global mention frequency; non-white node color = active mentions in current week; "
        "visible edges are historical co-mentions, with thickness showing cumulative strength and warm color marking current-week activity."
    )
    st.info(
        "What to expect: node positions stay stable over time, active weeks highlight where attention shifts, "
        "and cumulative edges expose persistent relationship patterns.",
    )

    st.markdown("---")
    week_options = week_index["week_id"].tolist()
    detail_week = st.selectbox("Week For Tables And Export", options=week_options, index=len(week_options) - 1)

    nodes_df, edges_df, metrics = _cached_week_artifacts(str(processed_dir), detail_week)

    st.subheader("Top Mentioned Accounts")
    st.dataframe(nodes_df.head(30), width="stretch", hide_index=True)

    with st.expander("Top Co-mention Edges"):
        st.dataframe(edges_df.head(30), width="stretch", hide_index=True)

    st.subheader("Export This Week")
    export_col_1, export_col_2, export_col_3 = st.columns(3)
    export_col_1.download_button(
        label="Download nodes.csv",
        data=nodes_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{detail_week}_nodes.csv",
        mime="text/csv",
    )
    export_col_2.download_button(
        label="Download edges.csv",
        data=edges_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{detail_week}_edges.csv",
        mime="text/csv",
    )
    export_col_3.download_button(
        label="Download metrics.json",
        data=json.dumps(metrics, indent=2, sort_keys=True).encode("utf-8"),
        file_name=f"{detail_week}_metrics.json",
        mime="application/json",
    )

    _render_footer_links(settings)


if __name__ == "__main__":
    main()
