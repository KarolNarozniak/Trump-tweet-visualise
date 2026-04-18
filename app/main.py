from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trump_graph.app import (
    build_global_animation_html,
    load_global_animation_artifacts,
    load_week_artifacts,
    load_week_index,
)
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
    return slug if slug in {"graph", "about"} else "graph"


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
        "- The app analyzes mention/co-mention structure over weekly time steps.\n"
        "- It is purely educational and intended for learning and research practice."
    )
    st.info("About endpoint: add `?page=about` to the app URL.")
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
    selected_page_label = st.sidebar.radio(
        "Page",
        options=["Graph", "About"],
        index=0 if current_page == "graph" else 1,
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
    nav_col.markdown("[Graph](?page=graph) | [About](?page=about)")

    st.warning(settings.meta.disclaimer_text)

    if current_page == "about":
        _render_about_page(settings, logo_path)
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
