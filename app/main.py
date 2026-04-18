from __future__ import annotations

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

DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


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


def main() -> None:
    st.set_page_config(page_title="Trump Mention Network", layout="wide")
    _inject_ui_styles()

    st.title("Trump Mention Network")
    st.caption("One stable graph over time: nodes flare with weekly heat and edges thicken cumulatively.")

    st.sidebar.header("Graph Controls")
    processed_dir_input = st.sidebar.text_input("Processed data directory", value=str(DEFAULT_PROCESSED_DIR))
    include_hub = st.sidebar.checkbox("Include @realdonaldtrump", value=False)
    always_label_top = st.sidebar.checkbox("Always label top 40 nodes", value=False)
    playback_speed = st.sidebar.slider("Initial playback speed (weeks/sec)", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
    node_size_multiplier = st.sidebar.slider("Node size multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
    layout_spread = st.sidebar.slider("Layout spread", min_value=1.0, max_value=4.0, value=2.2, step=0.1)
    initial_zoom_boost = st.sidebar.slider("Initial graph zoom", min_value=0.6, max_value=1.8, value=0.8, step=0.05)
    graph_height = st.sidebar.slider("Graph height (px)", min_value=560, max_value=980, value=760, step=20)

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
            "venv\\Scripts\\python.exe -m trump_graph build --input tweets_01-08-2021.csv --out data/processed --min-mention-count 1 --global-min-mentions 8 --include-retweets"
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
    st.components.v1.html(graph_html, height=graph_height + 170, scrolling=False)

    st.markdown("---")
    week_options = week_index["week_id"].tolist()
    default_week_index = len(week_options) - 1
    detail_week = st.selectbox("Week For Tables And Export", options=week_options, index=default_week_index)

    nodes_df, edges_df, metrics = _cached_week_artifacts(str(processed_dir), detail_week)

    st.subheader("Top Mentioned Accounts")
    st.dataframe(nodes_df.head(30), use_container_width=True, hide_index=True)

    with st.expander("Top Co-mention Edges"):
        st.dataframe(edges_df.head(30), use_container_width=True, hide_index=True)

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


if __name__ == "__main__":
    main()
