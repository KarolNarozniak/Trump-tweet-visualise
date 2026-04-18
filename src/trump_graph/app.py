from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_week_index(processed_dir: Path) -> pd.DataFrame:
    week_index_path = processed_dir / "week_index.csv"
    if not week_index_path.exists():
        raise FileNotFoundError(f"Missing week index: {week_index_path}")

    week_index = pd.read_csv(week_index_path)
    if week_index.empty:
        return week_index
    return week_index.sort_values(["week_start", "week_id"], kind="mergesort").reset_index(drop=True)


def load_week_artifacts(processed_dir: Path, week_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    week_dir = processed_dir / "weeks" / week_id
    nodes_path = week_dir / "nodes.csv"
    edges_path = week_dir / "edges.csv"
    metrics_path = week_dir / "metrics.json"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return nodes_df, edges_df, metrics


def load_global_animation_artifacts(processed_dir: Path) -> dict[str, Any]:
    animation_path = processed_dir / "global_animation" / "animation_state.json"
    if not animation_path.exists():
        raise FileNotFoundError(f"Missing global animation file: {animation_path}")

    payload = json.loads(animation_path.read_text(encoding="utf-8"))
    required_keys = {
        "weeks",
        "global_nodes",
        "global_edges",
        "node_week_deltas",
        "edge_week_deltas",
        "heat_decay",
        "heat_scale",
        "max_cumulative_edge",
        "hub_node_id",
        "top_label_nodes",
    }
    missing_keys = sorted(required_keys - set(payload.keys()))
    if missing_keys:
        raise ValueError(f"Global animation payload is missing keys: {', '.join(missing_keys)}")
    return payload


def _filtered_animation_payload(payload: dict[str, Any], include_hub: bool) -> dict[str, Any]:
    hub_node_id = str(payload.get("hub_node_id", "realdonaldtrump"))

    all_nodes = [dict(node) for node in payload.get("global_nodes", [])]
    if include_hub:
        visible_nodes = all_nodes
    else:
        visible_nodes = [node for node in all_nodes if str(node.get("id")) != hub_node_id]

    visible_node_ids = {str(node.get("id")) for node in visible_nodes}

    all_edges = [dict(edge) for edge in payload.get("global_edges", [])]
    visible_edges = [
        edge
        for edge in all_edges
        if str(edge.get("source")) in visible_node_ids and str(edge.get("target")) in visible_node_ids
    ]
    visible_edge_ids = {str(edge.get("id")) for edge in visible_edges}

    node_week_deltas = [
        [[str(node_id), int(delta)] for node_id, delta in week_entries if str(node_id) in visible_node_ids]
        for week_entries in payload.get("node_week_deltas", [])
    ]
    edge_week_deltas = [
        [[str(edge_id), int(delta)] for edge_id, delta in week_entries if str(edge_id) in visible_edge_ids]
        for week_entries in payload.get("edge_week_deltas", [])
    ]

    return {
        "version": int(payload.get("version", 1)),
        "heat_decay": float(payload.get("heat_decay", 0.85)),
        "global_min_mentions": int(payload.get("global_min_mentions", 8)),
        "heat_scale": float(payload.get("heat_scale", 1.0)),
        "max_cumulative_edge": int(payload.get("max_cumulative_edge", 0)),
        "hub_node_id": hub_node_id,
        "top_label_nodes": [node_id for node_id in payload.get("top_label_nodes", []) if str(node_id) in visible_node_ids],
        "weeks": payload.get("weeks", []),
        "global_nodes": visible_nodes,
        "global_edges": visible_edges,
        "node_week_deltas": node_week_deltas,
        "edge_week_deltas": edge_week_deltas,
    }


def build_global_animation_html(
    payload: dict[str, Any],
    include_hub: bool = False,
    always_label_top_nodes: bool = False,
    initial_week_index: int | None = None,
    initial_speed: float = 2.0,
    height_px: int = 760,
    transition_steps: int = 7,
) -> str:
    filtered_payload = _filtered_animation_payload(payload, include_hub=include_hub)
    weeks = filtered_payload.get("weeks", [])
    if not weeks:
        raise ValueError("Global animation payload has no weeks.")

    max_week_index = len(weeks) - 1
    if initial_week_index is None:
        initial_week_index = next(
            (index for index, entries in enumerate(filtered_payload.get("node_week_deltas", [])) if entries),
            0,
        )
    initial_week_index = max(0, min(max_week_index, int(initial_week_index)))
    speed_value = max(0.5, min(8.0, float(initial_speed)))

    json_payload = json.dumps(filtered_payload, separators=(",", ":"), ensure_ascii=True)
    label_flag = "true" if always_label_top_nodes else "false"

    return f"""
<div class="tg-root">
  <div id="tg-graph" style="height:{int(height_px)}px;"></div>
  <div class="tg-timeline">
    <div class="tg-control-row">
      <button id="tg-play" type="button">Play</button>
      <button id="tg-stop" type="button">Stop</button>
      <label class="tg-speed-label" for="tg-speed">Speed</label>
      <input id="tg-speed" type="range" min="0.5" max="8" step="0.5" value="{speed_value:.1f}" />
      <span id="tg-speed-value">{speed_value:.1f} w/s</span>
      <span id="tg-week-text"></span>
    </div>
    <input id="tg-week-slider" type="range" min="0" max="{max_week_index}" step="1" value="{initial_week_index}" />
    <div id="tg-week-meta"></div>
  </div>
</div>

<style>
  .tg-root {{
    width: 100%;
    border: 1px solid #0f172a;
    border-radius: 8px;
    overflow: hidden;
    background: #050a14;
  }}
  #tg-graph {{
    width: 100%;
    background: #050a14;
  }}
  .tg-timeline {{
    border-top: 1px solid #0f172a;
    padding: 10px 12px 12px 12px;
    background: #02050b;
    color: #e2e8f0;
  }}
  .tg-control-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }}
  .tg-control-row button {{
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    background: #0f172a;
    color: #e2e8f0;
    cursor: pointer;
    font-size: 13px;
  }}
  .tg-speed-label {{
    margin-left: 6px;
    font-size: 12px;
    color: #cbd5e1;
  }}
  #tg-speed {{
    width: 140px;
  }}
  #tg-speed-value {{
    min-width: 56px;
    font-size: 12px;
    color: #cbd5e1;
  }}
  #tg-week-text {{
    margin-left: 8px;
    font-size: 12px;
    color: #f8fafc;
  }}
  #tg-week-slider {{
    width: 100%;
  }}
  #tg-week-meta {{
    margin-top: 7px;
    font-size: 12px;
    color: #94a3b8;
  }}
</style>

<script src="https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js"></script>
<script>
(() => {{
  const payload = {json_payload};
  const alwaysLabelTopNodes = {label_flag};
  const transitionSteps = {int(max(1, transition_steps))};

  const graphElement = document.getElementById("tg-graph");
  const playButton = document.getElementById("tg-play");
  const stopButton = document.getElementById("tg-stop");
  const speedInput = document.getElementById("tg-speed");
  const speedValueElement = document.getElementById("tg-speed-value");
  const weekSlider = document.getElementById("tg-week-slider");
  const weekTextElement = document.getElementById("tg-week-text");
  const weekMetaElement = document.getElementById("tg-week-meta");

  if (typeof vis === "undefined") {{
    graphElement.innerHTML = "<div style='padding:20px;color:#f8fafc'>Unable to load vis-network library.</div>";
    return;
  }}

  const nodeData = payload.global_nodes || [];
  const edgeData = payload.global_edges || [];
  const weekData = payload.weeks || [];
  const nodeWeekDeltas = payload.node_week_deltas || [];
  const edgeWeekDeltas = payload.edge_week_deltas || [];
  const maxCumulativeEdge = Math.max(1, Number(payload.max_cumulative_edge || 1));
  const topLabelSet = new Set(payload.top_label_nodes || []);
  const positiveNodeDeltas = [];
  for (const weekEntries of nodeWeekDeltas) {{
    for (const entry of weekEntries) {{
      const delta = Number(entry[1] || 0);
      if (delta > 0) {{
        positiveNodeDeltas.push(delta);
      }}
    }}
  }}
  positiveNodeDeltas.sort((left, right) => left - right);
  const percentileIndex = positiveNodeDeltas.length
    ? Math.floor((positiveNodeDeltas.length - 1) * 0.95)
    : 0;
  const weeklyNodeScale = Math.max(1, Number(positiveNodeDeltas[percentileIndex] || 1));

  const nodeIds = nodeData.map((node) => String(node.id));
  const edgeIds = edgeData.map((edge) => String(edge.id));
  const nodeInfoById = new Map(nodeData.map((node) => [String(node.id), node]));
  const edgeInfoById = new Map(edgeData.map((edge) => [String(edge.id), edge]));

  const coolNodeColor = "rgba(15, 23, 42, 0.92)";
  const baseNodeBorderColor = "#334155";
  let selectedNodeId = null;

  function clamp(value, min, max) {{
    return Math.max(min, Math.min(max, value));
  }}

  function fireColor(intensity, alpha = 1.0) {{
    const stops = [
      [0.00, [8, 8, 24]],
      [0.18, [33, 12, 74]],
      [0.42, [118, 23, 101]],
      [0.64, [204, 60, 71]],
      [0.83, [249, 146, 6]],
      [1.00, [253, 255, 182]]
    ];
    const t = clamp(intensity, 0, 1);
    for (let i = 1; i < stops.length; i += 1) {{
      const left = stops[i - 1];
      const right = stops[i];
      if (t <= right[0]) {{
        const localT = (t - left[0]) / (right[0] - left[0]);
        const r = Math.round(left[1][0] + (right[1][0] - left[1][0]) * localT);
        const g = Math.round(left[1][1] + (right[1][1] - left[1][1]) * localT);
        const b = Math.round(left[1][2] + (right[1][2] - left[1][2]) * localT);
        return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha.toFixed(3)}})`;
      }}
    }}
    return `rgba(253, 255, 182, ${{alpha.toFixed(3)}})`;
  }}

  function edgeWidth(cumulative) {{
    if (cumulative <= 0) {{
      return 0;
    }}
    const normalized = clamp(cumulative / maxCumulativeEdge, 0, 1);
    return 0.5 + 5.5 * Math.sqrt(normalized);
  }}

  function nodeLabelForId(nodeId) {{
    if (selectedNodeId === nodeId) {{
      return `@${{nodeId}}`;
    }}
    if (alwaysLabelTopNodes && topLabelSet.has(nodeId)) {{
      return `@${{nodeId}}`;
    }}
    return "";
  }}

  const nodes = new vis.DataSet(
    nodeData.map((node) => {{
      const nodeId = String(node.id);
      return {{
        id: nodeId,
        label: nodeLabelForId(nodeId),
        title: `@${{nodeId}}<br>Total mentions: ${{Number(node.total_mentions || 0).toLocaleString()}}`,
        size: Number(node.size || 8),
        x: Number(node.x || 0),
        y: Number(node.y || 0),
        fixed: {{ x: true, y: true }},
        physics: false,
        color: {{
          background: coolNodeColor,
          border: baseNodeBorderColor,
          highlight: {{ background: coolNodeColor, border: "#e2e8f0" }},
          hover: {{ background: coolNodeColor, border: "#e2e8f0" }}
        }},
        font: {{ color: "#f8fafc", size: 12 }}
      }};
    }})
  );

  const edges = new vis.DataSet(
    edgeData.map((edge) => {{
      const edgeId = String(edge.id);
      return {{
        id: edgeId,
        from: String(edge.source),
        to: String(edge.target),
        hidden: true,
        width: 0,
        color: "rgba(100, 116, 139, 0.30)",
        smooth: false,
        title: `@${{edge.source}} <> @${{edge.target}}<br>Cumulative co-mentions: 0`
      }};
    }})
  );

  const network = new vis.Network(
    graphElement,
    {{ nodes, edges }},
    {{
      autoResize: true,
      physics: false,
      interaction: {{
        hover: true,
        navigationButtons: true,
        keyboard: true,
        tooltipDelay: 70
      }},
      nodes: {{
        shape: "dot",
        borderWidth: 1.1
      }},
      edges: {{
        smooth: false
      }}
    }}
  );
  network.fit({{ nodes: nodeIds, animation: false }});

  let currentWeekIndex = Number(weekSlider.value || 0);
  let playing = false;
  let playTimer = null;
  let animating = false;
  let playbackSpeed = Number(speedInput.value || 2.0);

  let nodeWeekly = Object.fromEntries(nodeIds.map((id) => [id, 0]));
  let nodeSeen = Object.fromEntries(nodeIds.map((id) => [id, 0]));
  let edgeCum = Object.fromEntries(edgeIds.map((id) => [id, 0]));

  function cloneState(stateObj) {{
    return Object.assign({{}}, stateObj);
  }}

  function resetState() {{
    nodeWeekly = Object.fromEntries(nodeIds.map((id) => [id, 0]));
    nodeSeen = Object.fromEntries(nodeIds.map((id) => [id, 0]));
    edgeCum = Object.fromEntries(edgeIds.map((id) => [id, 0]));
  }}

  function applyDeltaForWeek(index) {{
    for (const nodeId of nodeIds) {{
      nodeWeekly[nodeId] = 0;
    }}

    const nodeDeltas = nodeWeekDeltas[index] || [];
    for (const [nodeIdRaw, deltaRaw] of nodeDeltas) {{
      const nodeId = String(nodeIdRaw);
      if (!(nodeId in nodeWeekly)) {{
        continue;
      }}
      const delta = Number(deltaRaw);
      nodeWeekly[nodeId] = delta;
      nodeSeen[nodeId] += delta;
    }}

    const edgeDeltas = edgeWeekDeltas[index] || [];
    for (const [edgeIdRaw, deltaRaw] of edgeDeltas) {{
      const edgeId = String(edgeIdRaw);
      if (!(edgeId in edgeCum)) {{
        continue;
      }}
      edgeCum[edgeId] += Number(deltaRaw);
    }}
  }}

  function recomputeTo(targetWeekIndex) {{
    resetState();
    for (let i = 0; i <= targetWeekIndex; i += 1) {{
      applyDeltaForWeek(i);
    }}
  }}

  function weekRecord(index) {{
    return weekData[index] || {{}};
  }}

  function updateWeekLabels() {{
    const record = weekRecord(currentWeekIndex);
    const weekId = String(record.week_id || "");
    const start = String(record.week_start || "");
    const end = String(record.week_end || "");
    weekTextElement.textContent = `${{weekId}}  (${{start}} to ${{end}})`;

    const tweetsProcessed = Number(record.tweets_processed || 0).toLocaleString();
    const tweetsWithMentions = Number(record.tweets_with_mentions || 0).toLocaleString();
    const uniqueMentions = Number(record.unique_mentions || 0).toLocaleString();
    const edgeCount = Number(record.edge_count || 0).toLocaleString();
    weekMetaElement.textContent = `Tweets: ${{tweetsProcessed}} | Tweets with mentions: ${{tweetsWithMentions}} | Active nodes: ${{uniqueMentions}} | Active edges: ${{edgeCount}}`;
  }}

  function nodeUpdateForState(nodeId, weeklyValue, cumulativeMentions) {{
    const info = nodeInfoById.get(nodeId);
    if (cumulativeMentions <= 0) {{
      return {{
        id: nodeId,
        hidden: true,
        label: "",
        title: `@${{nodeId}}<br>Total mentions: ${{Number(info.total_mentions || 0).toLocaleString()}}`
      }};
    }}

    const normalizedWeekly = clamp(Number(weeklyValue || 0) / weeklyNodeScale, 0, 1);
    const intensity = normalizedWeekly > 0 ? Math.pow(normalizedWeekly, 0.65) : 0;
    const fillColor = normalizedWeekly > 0 ? fireColor(intensity, 0.95) : coolNodeColor;
    const title = `@${{nodeId}}<br>Total mentions: ${{Number(info.total_mentions || 0).toLocaleString()}}<br>This week: ${{Math.round(Number(weeklyValue || 0)).toLocaleString()}}<br>Cumulative: ${{Math.round(Number(cumulativeMentions || 0)).toLocaleString()}}`;
    return {{
      id: nodeId,
      hidden: false,
      label: nodeLabelForId(nodeId),
      title,
      color: {{
        background: fillColor,
        border: "#111827",
        highlight: {{ background: fillColor, border: "#f8fafc" }},
        hover: {{ background: fillColor, border: "#f8fafc" }}
      }}
    }};
  }}

  function edgeUpdateForState(edgeId, cumulativeValue) {{
    const info = edgeInfoById.get(edgeId);
    if (!info) {{
      return {{
        id: edgeId,
        hidden: true,
        width: 0,
        color: "rgba(100, 116, 139, 0.30)"
      }};
    }}

    if (cumulativeValue <= 0) {{
      return {{
        id: edgeId,
        hidden: true,
        width: 0,
        color: "rgba(100, 116, 139, 0.25)",
        title: `@${{info.source}} <> @${{info.target}}<br>Cumulative co-mentions: 0`
      }};
    }}

    const normalized = clamp(cumulativeValue / maxCumulativeEdge, 0, 1);
    const color = fireColor(normalized, 0.30 + normalized * 0.48);
    return {{
      id: edgeId,
      hidden: false,
      width: edgeWidth(cumulativeValue),
      color,
      title: `@${{info.source}} <> @${{info.target}}<br>Cumulative co-mentions: ${{Math.round(cumulativeValue).toLocaleString()}}`
    }};
  }}

  function renderFromState(weeklyState, seenState, edgeState) {{
    const nodeUpdates = nodeIds.map((nodeId) =>
      nodeUpdateForState(
        nodeId,
        Number(weeklyState[nodeId] || 0),
        Number(seenState[nodeId] || 0)
      )
    );
    const edgeUpdates = edgeIds.map((edgeId) => edgeUpdateForState(edgeId, Number(edgeState[edgeId] || 0)));
    nodes.update(nodeUpdates);
    edges.update(edgeUpdates);
    updateWeekLabels();
  }}

  function transitionRender(previousWeekly, previousSeen, previousEdge, nextWeekly, nextSeen, nextEdge, done) {{
    const totalSteps = Math.max(1, transitionSteps);
    let step = 0;

    function tick() {{
      step += 1;
      const t = step / totalSteps;

      const blendedWeekly = {{}};
      const blendedSeen = {{}};
      for (const nodeId of nodeIds) {{
        const prevWeekly = Number(previousWeekly[nodeId] || 0);
        const nextWeeklyValue = Number(nextWeekly[nodeId] || 0);
        blendedWeekly[nodeId] = prevWeekly + (nextWeeklyValue - prevWeekly) * t;

        const prevSeen = Number(previousSeen[nodeId] || 0);
        const nextSeenValue = Number(nextSeen[nodeId] || 0);
        blendedSeen[nodeId] = prevSeen + (nextSeenValue - prevSeen) * t;
      }}

      const blendedEdge = {{}};
      for (const edgeId of edgeIds) {{
        const prev = Number(previousEdge[edgeId] || 0);
        const next = Number(nextEdge[edgeId] || 0);
        blendedEdge[edgeId] = prev + (next - prev) * t;
      }}

      renderFromState(blendedWeekly, blendedSeen, blendedEdge);
      if (step < totalSteps) {{
        window.requestAnimationFrame(tick);
      }} else {{
        done();
      }}
    }}

    window.requestAnimationFrame(tick);
  }}

  function setWeek(targetIndex, animate = true) {{
    const boundedTarget = clamp(Number(targetIndex), 0, weekData.length - 1);
    if (animating || boundedTarget === currentWeekIndex) {{
      return;
    }}

    const previousWeekly = cloneState(nodeWeekly);
    const previousSeen = cloneState(nodeSeen);
    const previousEdge = cloneState(edgeCum);

    if (boundedTarget === currentWeekIndex + 1) {{
      applyDeltaForWeek(boundedTarget);
    }} else {{
      recomputeTo(boundedTarget);
    }}

    const nextWeekly = cloneState(nodeWeekly);
    const nextSeen = cloneState(nodeSeen);
    const nextEdge = cloneState(edgeCum);
    currentWeekIndex = boundedTarget;
    weekSlider.value = String(currentWeekIndex);

    if (!animate) {{
      renderFromState(nextWeekly, nextSeen, nextEdge);
      return;
    }}

    animating = true;
    transitionRender(previousWeekly, previousSeen, previousEdge, nextWeekly, nextSeen, nextEdge, () => {{
      animating = false;
      renderFromState(nodeWeekly, nodeSeen, edgeCum);
    }});
  }}

  function stopPlayback() {{
    if (playTimer) {{
      window.clearInterval(playTimer);
      playTimer = null;
    }}
    playing = false;
    playButton.textContent = "Play";
  }}

  function startPlayback() {{
    stopPlayback();
    playing = true;
    playButton.textContent = "Pause";
    const intervalMs = Math.max(80, Math.round(1000 / playbackSpeed));
    playTimer = window.setInterval(() => {{
      if (animating) {{
        return;
      }}
      const nextIndex = (currentWeekIndex + 1) % weekData.length;
      setWeek(nextIndex, true);
    }}, intervalMs);
  }}

  function updateSpeedLabel() {{
    speedValueElement.textContent = `${{playbackSpeed.toFixed(1)}} w/s`;
  }}

  speedInput.addEventListener("input", () => {{
    playbackSpeed = clamp(Number(speedInput.value || 2.0), 0.5, 8.0);
    updateSpeedLabel();
    if (playing) {{
      startPlayback();
    }}
  }});

  playButton.addEventListener("click", () => {{
    if (playing) {{
      stopPlayback();
    }} else {{
      startPlayback();
    }}
  }});

  stopButton.addEventListener("click", () => {{
    stopPlayback();
  }});

  weekSlider.addEventListener("input", () => {{
    stopPlayback();
    setWeek(Number(weekSlider.value || currentWeekIndex), true);
  }});

  network.on("selectNode", (params) => {{
    selectedNodeId = params.nodes && params.nodes.length ? String(params.nodes[0]) : null;
    renderFromState(nodeWeekly, nodeSeen, edgeCum);
  }});

  network.on("deselectNode", () => {{
    selectedNodeId = null;
    renderFromState(nodeWeekly, nodeSeen, edgeCum);
  }});

  recomputeTo(currentWeekIndex);
  updateSpeedLabel();
  renderFromState(nodeWeekly, nodeSeen, edgeCum);
}})();
</script>
"""
