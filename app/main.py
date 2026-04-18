from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trump_graph.app import build_pyvis_html, load_week_artifacts, load_week_index


DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@st.cache_data(show_spinner=False)
def _cached_week_index(processed_dir: str) -> pd.DataFrame:
    return load_week_index(Path(processed_dir))


@st.cache_data(show_spinner=False)
def _cached_week_artifacts(processed_dir: str, week_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return load_week_artifacts(Path(processed_dir), week_id)


def _init_timeline_state(week_options: list[str]) -> None:
    if "timeline_week_id" not in st.session_state or st.session_state.timeline_week_id not in week_options:
        st.session_state.timeline_week_id = week_options[-1]
    if "timeline_week_widget" not in st.session_state or st.session_state.timeline_week_widget not in week_options:
        st.session_state.timeline_week_widget = st.session_state.timeline_week_id
    if "timeline_playing" not in st.session_state:
        st.session_state.timeline_playing = False
    if "timeline_speed" not in st.session_state:
        st.session_state.timeline_speed = 2.0

    if st.session_state.timeline_playing:
        # Keep the displayed slider week aligned with autoplay.
        st.session_state.timeline_week_widget = st.session_state.timeline_week_id
    elif st.session_state.timeline_week_widget != st.session_state.timeline_week_id:
        # User moved the slider while paused.
        st.session_state.timeline_week_id = st.session_state.timeline_week_widget


def _build_timeline_chart(week_index: pd.DataFrame, selected_week_id: str) -> alt.Chart:
    chart_df = week_index.copy()
    chart_df["week_start"] = pd.to_datetime(chart_df["week_start"], errors="coerce")
    chart_df = chart_df.dropna(subset=["week_start"]).reset_index(drop=True)

    metrics_df = chart_df.melt(
        id_vars=["week_id", "week_start"],
        value_vars=["tweets_with_mentions", "unique_mentions"],
        var_name="metric",
        value_name="value",
    )
    metrics_df["metric"] = metrics_df["metric"].map(
        {
            "tweets_with_mentions": "Tweets With Mentions",
            "unique_mentions": "Unique Mentioned Accounts",
        }
    )

    base = (
        alt.Chart(metrics_df)
        .mark_line(point=False, strokeWidth=2.5)
        .encode(
            x=alt.X("week_start:T", title="Week"),
            y=alt.Y("value:Q", title="Count"),
            color=alt.Color(
                "metric:N",
                title="",
                scale=alt.Scale(
                    domain=["Tweets With Mentions", "Unique Mentioned Accounts"],
                    range=["#22D3EE", "#F59E0B"],
                ),
            ),
            tooltip=[
                alt.Tooltip("week_start:T", title="Week"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=","),
            ],
        )
    )

    selected_rows = chart_df.loc[chart_df["week_id"] == selected_week_id, ["week_start"]]
    if selected_rows.empty:
        return base.properties(height=170)

    selected_date = selected_rows.iloc[0]["week_start"]
    selected_rule = alt.Chart(pd.DataFrame({"week_start": [selected_date]})).mark_rule(
        color="#F43F5E",
        strokeWidth=2,
    ).encode(x="week_start:T")
    return (base + selected_rule).properties(height=170)


def _advance_timeline(week_options: list[str]) -> None:
    if not week_options:
        return

    current_week = st.session_state.timeline_week_id
    current_index = week_options.index(current_week)
    next_index = (current_index + 1) % len(week_options)
    st.session_state.timeline_week_id = week_options[next_index]


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
    st.caption("Colorful weekly co-mention map with timeline playback controls.")

    st.sidebar.header("Controls")
    processed_dir_input = st.sidebar.text_input("Processed data directory", value=str(DEFAULT_PROCESSED_DIR))
    processed_dir = Path(processed_dir_input)

    if not processed_dir.exists():
        st.error(f"Directory not found: {processed_dir}")
        return

    try:
        week_index = _cached_week_index(str(processed_dir))
    except FileNotFoundError:
        st.warning("No processed data found yet. Run the build command first.")
        st.code(
            "python -m trump_graph build --input tweets_01-08-2021.csv --out data/processed --min-mention-count 1 --include-retweets"
        )
        return

    if week_index.empty:
        st.warning("No weekly artifacts available in this directory.")
        return

    week_options = week_index["week_id"].tolist()
    _init_timeline_state(week_options)

    max_edges = st.sidebar.slider("Displayed edges", min_value=50, max_value=2000, value=400, step=50)
    show_edges = st.sidebar.checkbox("Show edges", value=True)
    show_labels = st.sidebar.checkbox("Show labels", value=True)
    st.sidebar.caption("Node colors represent detected mention communities.")

    selected_week = st.session_state.timeline_week_id
    nodes_df, edges_df, metrics = _cached_week_artifacts(str(processed_dir), selected_week)
    network_html, displayed_edges, total_edges = build_pyvis_html(
        nodes_df,
        edges_df,
        max_edges=max_edges,
        show_labels=show_labels,
        show_edges=show_edges,
    )

    selected_week_row = week_index.loc[week_index["week_id"] == selected_week]
    week_range_label = selected_week
    if not selected_week_row.empty:
        row = selected_week_row.iloc[0]
        week_range_label = f"{selected_week} ({row['week_start']} to {row['week_end']})"

    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    col_1.metric("Tweets", f"{metrics.get('tweets_processed', 0):,}")
    col_2.metric("Tweets with mentions", f"{metrics.get('tweets_with_mentions', 0):,}")
    col_3.metric("Accounts", f"{metrics.get('unique_mentions', 0):,}")
    col_4.metric("Edges", f"{metrics.get('edge_count', 0):,}")
    col_5.metric("Density", f"{metrics.get('density', 0):.4f}")

    st.subheader(f"Week {week_range_label}")
    st.components.v1.html(network_html, height=780, scrolling=False)
    if not show_edges:
        st.caption("Edges are hidden. Enable them in the sidebar to inspect relationships.")
    elif displayed_edges < total_edges:
        st.caption(f"Showing {displayed_edges:,} of {total_edges:,} edges for rendering speed.")

    st.markdown("---")
    st.subheader("Timeline")
    st.caption("Use play/pause and speed to animate week-by-week network evolution.")

    timeline_cols = st.columns([1.1, 1.1, 2.0, 6.8])
    play_label = "Pause" if st.session_state.timeline_playing else "Play"
    if timeline_cols[0].button(play_label, use_container_width=True):
        st.session_state.timeline_playing = not st.session_state.timeline_playing

    if timeline_cols[1].button("Stop", use_container_width=True):
        st.session_state.timeline_playing = False

    with timeline_cols[2]:
        st.slider(
            "Speed (weeks/sec)",
            min_value=0.5,
            max_value=8.0,
            step=0.5,
            key="timeline_speed",
        )

    with timeline_cols[3]:
        st.select_slider("Week", options=week_options, key="timeline_week_widget")

    timeline_chart = _build_timeline_chart(week_index, st.session_state.timeline_week_id)
    st.altair_chart(timeline_chart, use_container_width=True)

    st.subheader("Top Mentioned Accounts")
    st.dataframe(nodes_df.head(30), use_container_width=True, hide_index=True)

    with st.expander("Top Co-mention Edges"):
        st.dataframe(edges_df.head(30), use_container_width=True, hide_index=True)

    st.subheader("Export This Week")
    export_col_1, export_col_2, export_col_3 = st.columns(3)
    export_col_1.download_button(
        label="Download nodes.csv",
        data=nodes_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_week}_nodes.csv",
        mime="text/csv",
    )
    export_col_2.download_button(
        label="Download edges.csv",
        data=edges_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_week}_edges.csv",
        mime="text/csv",
    )
    export_col_3.download_button(
        label="Download metrics.json",
        data=json.dumps(metrics, indent=2, sort_keys=True).encode("utf-8"),
        file_name=f"{selected_week}_metrics.json",
        mime="application/json",
    )

    if st.session_state.timeline_playing and len(week_options) > 1:
        delay_seconds = 1.0 / float(st.session_state.timeline_speed)
        time.sleep(delay_seconds)
        _advance_timeline(week_options)
        st.rerun()


if __name__ == "__main__":
    main()
