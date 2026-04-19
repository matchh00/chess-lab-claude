"""Chess Lab Research Dashboard — streamlit run dashboard/app.py (from repo root)."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Chess Lab", layout="wide", page_icon="♟️")

RUNS_DIR = Path("data/runs")

# ── data loaders (all cached, all safe) ───────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in dirs]


@st.cache_data(show_spinner=False)
def load_manifest(run_id: str) -> dict:
    path = RUNS_DIR / run_id / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_run_report(run_id: str) -> dict | None:
    path = RUNS_DIR / run_id / "run_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_game_summary(run_id: str) -> pd.DataFrame | None:
    path = RUNS_DIR / run_id / "game_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) > 0 else None


@st.cache_data(show_spinner=False)
def load_move_log(run_id: str) -> pd.DataFrame | None:
    path = RUNS_DIR / run_id / "move_log.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) > 0 else None


@st.cache_data(show_spinner=False)
def load_attribution(run_id: str) -> pd.DataFrame | None:
    path = RUNS_DIR / run_id / "primitive_attribution.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) > 0 else None


@st.cache_data(show_spinner=False)
def load_learning_log(run_id: str) -> dict | None:
    path = RUNS_DIR / run_id / "learning_log.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("entries") else None


@st.cache_data(show_spinner=False)
def load_trace(run_id: str, game_id: str) -> list[dict]:
    path = RUNS_DIR / run_id / "traces" / f"{game_id}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


# ── sidebar ───────────────────────────────────────────────────────────────────

def _meta_row(label: str, value) -> None:
    st.markdown(f"<small>**{label}:** {value}</small>", unsafe_allow_html=True)


with st.sidebar:
    st.title("♟️ Chess Lab")

    runs = list_runs()
    if not runs:
        st.error("No runs found in data/runs/")
        st.stop()

    run_id = st.selectbox("Run", runs)
    manifest = load_manifest(run_id)

    cfg = manifest.get("config", {})
    st.divider()
    _meta_row("Run ID", f"`{manifest.get('run_id', run_id)}`")
    _meta_row("Date", (manifest.get("start_time") or "")[:10] or "—")
    _meta_row("Policy", manifest.get("policy_name", "—"))
    _meta_row("Prompt", manifest.get("prompt_version", "—"))
    _meta_row("Candidates", manifest.get("candidate_mode", "—"))
    _meta_row("Narrative", manifest.get("narrative_mode", "—"))
    _meta_row("Learning", cfg.get("learning_mode", "—"))
    if cfg.get("learning_mode"):
        _meta_row("Gain", cfg.get("gain", "—"))
    st.divider()

    compare = st.toggle("Comparison mode")
    run2_id = None
    if compare:
        other_runs = [r for r in runs if r != run_id]
        if other_runs:
            run2_id = st.selectbox("Compare run", other_runs)
        else:
            st.info("Only one run available.")
            compare = False

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Learning Curve",
    "🔬 Primitive Monitor",
    "🎯 Candidate Rank",
    "📖 Narrative Inspector",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Learning Curve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    game_df = load_game_summary(run_id)
    report = load_run_report(run_id)

    if game_df is None:
        st.info("No game_summary.csv for this run. Learning runs do not produce per-game CSVs.")
        st.stop()

    # 4 metric cards
    avg_cpl = game_df["avg_centipawn_loss"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total games", manifest.get("total_games", len(game_df)))
    c2.metric("Avg CPL", f"{avg_cpl:.1f}")
    c3.metric("Best game CPL", f"{game_df['avg_centipawn_loss'].min():.1f}")
    c4.metric("Worst game CPL", f"{game_df['avg_centipawn_loss'].max():.1f}")

    # Line chart
    gdf = game_df.copy().reset_index(drop=True)
    gdf["game_num"] = gdf.index + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gdf["game_num"], y=gdf["avg_centipawn_loss"],
        mode="lines+markers", name=run_id[:24], line=dict(color="#4C78A8"),
    ))
    run_avg = report["average_centipawn_loss"] if report else avg_cpl
    fig.add_hline(y=run_avg, line_dash="dash", line_color="grey",
                  annotation_text=f"run avg {run_avg:.1f}", annotation_position="top right")

    if compare and run2_id:
        gdf2 = load_game_summary(run2_id)
        if gdf2 is not None:
            gdf2 = gdf2.reset_index(drop=True)
            gdf2["game_num"] = gdf2.index + 1
            fig.add_trace(go.Scatter(
                x=gdf2["game_num"], y=gdf2["avg_centipawn_loss"],
                mode="lines+markers", name=run2_id[:24], line=dict(color="#F58518"),
            ))
            r2 = load_run_report(run2_id)
            avg2 = r2["average_centipawn_loss"] if r2 else gdf2["avg_centipawn_loss"].mean()
            fig.add_hline(y=avg2, line_dash="dot", line_color="#F58518",
                          annotation_text=f"{run2_id[:16]} avg {avg2:.1f}",
                          annotation_position="bottom right")

    fig.update_layout(
        xaxis_title="Game #", yaxis_title="Avg centipawn loss",
        height=380, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    display_cols = ["game_id", "result", "total_plies", "avg_centipawn_loss",
                    "blunder_count", "total_tokens"]
    tdf = gdf[[c for c in display_cols if c in gdf.columns]].copy()
    tdf["game_id"] = tdf["game_id"].str[:8]
    if "avg_centipawn_loss" in tdf.columns:
        tdf["avg_centipawn_loss"] = tdf["avg_centipawn_loss"].round(1)
    st.dataframe(tdf, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Primitive Monitor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    attr_df = load_attribution(run_id)
    llog = load_learning_log(run_id)
    is_learning = llog is not None

    # ── non-learning: attribution bar chart only ──────────────────────────────
    if not is_learning:
        if attr_df is None:
            st.info("No primitive_attribution.csv for this run.")
        else:
            adf = attr_df.copy()
            adf["abs_corr"] = adf["correlation"].abs()
            adf = adf.sort_values("abs_corr", ascending=False).reset_index(drop=True)
            adf["color"] = adf["correlation"].apply(lambda v: "positive" if v >= 0 else "negative")

            fig = px.bar(
                adf, x="correlation", y="primitive_id",
                orientation="h",
                color="color",
                color_discrete_map={"positive": "#4C78A8", "negative": "#E45756"},
                hover_data={"n_observations": True, "abs_corr": False, "color": False},
                title="Primitive correlation with centipawn loss (negative = better position → lower CPL)",
            )
            fig.update_layout(
                yaxis={"categoryorder": "array", "categoryarray": adf["primitive_id"].tolist()},
                height=700, showlegend=False, margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── learning run ──────────────────────────────────────────────────────────
    else:
        entries = llog["entries"]

        # Reconstruct weight timeline
        # initial weights: from first entry's adjustments .old_weight
        init_weights: dict[str, float] = {}
        for adj in entries[0]["adjustments"]:
            init_weights[adj["bare_key"]] = adj["old_weight"]

        # End-of-game weights per game
        running: dict[str, float] = dict(init_weights)
        weight_rows: list[dict] = []
        net_inf_cumulative: dict[str, float] = {k: 0.0 for k in init_weights}
        timeline_rows: list[dict] = []

        for entry in entries:
            gnum = entry["game_num"]
            for adj in entry["adjustments"]:
                key = adj["bare_key"]
                running[key] = adj["new_weight"]
                net_inf_cumulative[key] = net_inf_cumulative.get(key, 0.0) + adj["net_influence"]
            row = {"game_num": gnum}
            row.update(dict(running))
            timeline_rows.append(row)

        timeline_df = pd.DataFrame(timeline_rows)  # rows=games, cols=primitives

        # Summary table
        summary_rows = []
        for key, init_w in init_weights.items():
            final_w = running.get(key, init_w)
            total_ni = net_inf_cumulative.get(key, 0.0)
            clamped = any(
                adj.get("clamped", False)
                for e in entries for adj in e["adjustments"] if adj["bare_key"] == key
            )
            corr = None
            if attr_df is not None:
                match = attr_df[attr_df["primitive_id"] == f"self_{key}"]
                if not match.empty:
                    corr = float(match.iloc[0]["correlation"])
            summary_rows.append({
                "primitive": key,
                "correlation": round(corr, 3) if corr is not None else None,
                "start_weight": round(init_w, 4),
                "current_weight": round(final_w, 4),
                "total_delta": round(final_w - init_w, 4),
                "net_influence": round(total_ni, 1),
                "clamped": clamped,
            })

        sum_df = pd.DataFrame(summary_rows).sort_values("net_influence", ascending=False)

        def _color_row(row):
            ni = row["net_influence"]
            color = "background-color: #d4edda" if ni > 0 else (
                "background-color: #f8d7da" if ni < 0 else "background-color: #f0f0f0"
            )
            return [color] * len(row)

        st.subheader("Weight summary")
        st.dataframe(
            sum_df.style.apply(_color_row, axis=1),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Weight trajectory")
        all_keys = [c for c in timeline_df.columns if c != "game_num"]
        default_keys = all_keys[:5] if len(all_keys) >= 5 else all_keys
        selected_keys = st.multiselect("Primitives to plot", all_keys, default=default_keys)

        if selected_keys:
            traj_fig = go.Figure()
            for key in selected_keys:
                if key in timeline_df.columns:
                    # prepend the initial weight as game 0
                    x_vals = [0] + list(timeline_df["game_num"])
                    y_vals = [init_weights.get(key, float("nan"))] + list(timeline_df[key])
                    traj_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers", name=key))
            traj_fig.update_layout(
                xaxis_title="After game #", yaxis_title="Weight",
                height=350, margin=dict(t=20, b=40),
            )
            st.plotly_chart(traj_fig, use_container_width=True)

            st.subheader("Cumulative net influence")
            cum_inf_data: dict[str, list] = {k: [] for k in selected_keys}
            running_cum: dict[str, float] = {k: 0.0 for k in selected_keys}
            game_nums = []
            for entry in entries:
                gnum = entry["game_num"]
                game_nums.append(gnum)
                for adj in entry["adjustments"]:
                    key = adj["bare_key"]
                    if key in running_cum:
                        running_cum[key] += adj["net_influence"]
                for key in selected_keys:
                    cum_inf_data[key].append(running_cum.get(key, 0.0))

            inf_fig = go.Figure()
            for key in selected_keys:
                inf_fig.add_trace(go.Scatter(
                    x=game_nums, y=cum_inf_data[key], mode="lines+markers", name=key,
                ))
            inf_fig.add_hline(y=0, line_dash="dash", line_color="grey")
            inf_fig.update_layout(
                xaxis_title="Game #", yaxis_title="Cumulative net influence",
                height=300, margin=dict(t=20, b=40),
            )
            st.plotly_chart(inf_fig, use_container_width=True)

        # If attribution also available, show it below
        if attr_df is not None:
            st.subheader("Attribution (correlation with CPL)")
            adf = attr_df.copy()
            adf["abs_corr"] = adf["correlation"].abs()
            adf = adf.sort_values("abs_corr", ascending=False)
            afig = px.bar(
                adf, x="correlation", y="primitive_id", orientation="h",
                color=adf["correlation"].apply(lambda v: "positive" if v >= 0 else "negative"),
                color_discrete_map={"positive": "#4C78A8", "negative": "#E45756"},
                hover_data={"n_observations": True},
            )
            afig.update_layout(
                yaxis={"categoryorder": "array", "categoryarray": adf["primitive_id"].tolist()},
                height=600, showlegend=False, margin=dict(t=20),
            )
            st.plotly_chart(afig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Candidate Rank Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    move_df = load_move_log(run_id)
    report3 = load_run_report(run_id)

    if move_df is None:
        st.info("No move_log.csv for this run. Learning runs do not produce move-level CSVs.")
        st.stop()

    mdf = move_df.dropna(subset=["selected_internal_rank"]).copy().reset_index(drop=True)
    mdf["move_num"] = mdf.index + 1

    def _add_run_traces(fig, df, name, line_color):
        df = df.dropna(subset=["selected_internal_rank"]).reset_index(drop=True)
        df["move_num"] = df.index + 1
        df["rank_rolling"] = df["selected_internal_rank"].rolling(5, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["move_num"], y=df["rank_rolling"],
            mode="lines", name=name, line=dict(color=line_color),
        ))

    # Rolling rank trend
    mdf["rank_rolling"] = mdf["selected_internal_rank"].rolling(5, min_periods=1).mean()
    rank_fig = go.Figure()
    rank_fig.add_trace(go.Scatter(
        x=mdf["move_num"], y=mdf["rank_rolling"],
        mode="lines", name=run_id[:24], line=dict(color="#4C78A8"),
    ))
    if compare and run2_id:
        mdf2 = load_move_log(run2_id)
        if mdf2 is not None:
            _add_run_traces(rank_fig, mdf2, run2_id[:24], "#F58518")
    rank_fig.update_layout(
        title="Selected internal rank (rolling mean, window=5)",
        xaxis_title="Lab move #", yaxis_title="Rank (1 = best)",
        height=320, margin=dict(t=40, b=40),
    )
    st.plotly_chart(rank_fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Presentation index distribution
    with col_a:
        if report3 and "presentation_index_distribution" in report3:
            pidist = report3["presentation_index_distribution"]
            pi_df = pd.DataFrame(
                [(int(k), v) for k, v in pidist.items()],
                columns=["presentation_index", "count"],
            ).sort_values("presentation_index")
            pi_fig = px.bar(pi_df, x="presentation_index", y="count",
                            title="Presentation index distribution (positional bias check)",
                            color_discrete_sequence=["#4C78A8"])
            pi_fig.update_layout(height=320, margin=dict(t=40, b=40))
            st.plotly_chart(pi_fig, use_container_width=True)
        else:
            st.info("No presentation_index_distribution in run_report.json.")

    # CPL vs rank scatter
    with col_b:
        sdf = mdf.dropna(subset=["centipawn_loss", "selected_internal_rank"]).copy()
        if len(sdf) > 0:
            import numpy as np
            sdf["rank_jitter"] = sdf["selected_internal_rank"] + np.random.uniform(-0.2, 0.2, len(sdf))
            sdf["blunder_label"] = sdf["blunder_label"].fillna("ok")
            color_map = {"blunder": "#E45756", "mistake": "#F58518",
                         "inaccuracy": "#EECA3B", "ok": "#4C78A8"}
            sc_fig = px.scatter(
                sdf, x="rank_jitter", y="centipawn_loss",
                color="blunder_label", color_discrete_map=color_map,
                title="CPL vs selected rank",
                labels={"rank_jitter": "Internal rank (jittered)", "centipawn_loss": "CPL"},
            )
            sc_fig.update_layout(height=320, margin=dict(t=40, b=40))
            st.plotly_chart(sc_fig, use_container_width=True)
        else:
            st.info("No CPL/rank data available.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Narrative Inspector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    traces_dir = RUNS_DIR / run_id / "traces"
    game_ids_with_traces = sorted(f.stem for f in traces_dir.glob("*.json")) if traces_dir.exists() else []

    if not game_ids_with_traces:
        st.info("No trace files for this run. Traces are produced by standard (non-learning) runs.")
        st.stop()

    sel_game = st.selectbox("Game", game_ids_with_traces,
                             format_func=lambda g: g[:16])
    trace_entries = load_trace(run_id, sel_game)

    if not trace_entries:
        st.warning("Trace file is empty or unreadable.")
        st.stop()

    ply_labels = [f"Ply {e.get('ply_index', i)}" for i, e in enumerate(trace_entries)]
    ply_i = st.selectbox("Ply", range(len(trace_entries)), format_func=lambda i: ply_labels[i])
    entry = trace_entries[ply_i]

    st.divider()
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # Panel 1 — Primitive State
    with row1_col1:
        st.markdown("**Primitive State**")
        ws = entry.get("weighted_state", [])
        if not ws:
            st.info("No weighted state for this ply.")
        else:
            ws_df = pd.DataFrame(ws).sort_values("weighted_score", ascending=False).reset_index(drop=True)
            is_top5 = ws_df.index < 5
            ws_df["_top"] = is_top5
            colors = ["#4C78A8" if t else "#bdd7ee" for t in ws_df["_top"]]
            ws_fig = go.Figure(go.Bar(
                x=ws_df["weighted_score"],
                y=ws_df["primitive_id"],
                orientation="h",
                marker_color=colors,
                hovertext=ws_df.get("importance_label", ws_df["primitive_id"]),
            ))
            ws_fig.update_layout(
                yaxis={"categoryorder": "array",
                       "categoryarray": ws_df["primitive_id"].tolist()[::-1]},
                xaxis_title="Weighted score",
                height=420, margin=dict(t=10, b=30, l=220, r=10),
            )
            st.plotly_chart(ws_fig, use_container_width=True)

    # Panel 2 — Narrative
    with row1_col2:
        st.markdown("**Position Narrative**")
        narrative = entry.get("position_narrative", "")
        if narrative and narrative.strip():
            st.markdown(
                f'<div style="background:#f8f9fa;border-left:4px solid #4C78A8;'
                f'padding:12px 16px;border-radius:4px;font-size:0.9em;'
                f'line-height:1.6;white-space:pre-wrap;">{narrative}</div>',
                unsafe_allow_html=True,
            )
            words = entry.get("position_narrative_word_count", len(narrative.split()))
            tokens = entry.get("position_narrative_token_count", 0)
            st.caption(f"{words} words · {tokens} tokens")
        else:
            st.info("Narrative not available for this move.")

    # Panel 3 — Decision
    with row2_col1:
        st.markdown("**Decision**")
        dec = entry.get("decision_record")
        cpl = entry.get("centipawn_loss")
        candidates = entry.get("candidates", [])

        rank1_cand = None
        selected_cand = None
        if candidates:
            try:
                rank1_cand = min(candidates, key=lambda c: c.get("internal_rank", 999))
            except Exception:
                pass
            if dec:
                sel_uci = dec.get("selected_uci", "")
                selected_cand = next((c for c in candidates if c.get("uci") == sel_uci), None)

        if cpl is not None:
            indicator = "🟢" if cpl < 30 else ("🔴" if cpl > 100 else "🟡")
            st.markdown(f"### {indicator} CPL: **{cpl:.1f}**")
        else:
            st.markdown("### ⚪ CPL: —")

        if selected_cand:
            st.markdown(f"**Selected move:** `{selected_cand.get('san', '—')}`")
        elif dec:
            st.markdown(f"**Selected UCI:** `{dec.get('selected_uci', '—')}`")

        if rank1_cand:
            st.markdown(f"**Engine rank-1:** `{rank1_cand.get('san', '—')}`")

        if dec:
            conf = dec.get("confidence", 0.0)
            fb = dec.get("fallback_used", False)
            st.markdown(f"**Confidence:** {conf:.2f}")
            if fb:
                st.warning("⚠️ Fallback used — LLM response was invalid")
            else:
                st.success("✓ Valid LLM decision")

            rank_chosen = dec.get("selected_internal_rank")
            if rank_chosen is not None:
                st.markdown(f"**Rank chosen:** {rank_chosen} / {len(candidates)}")
        else:
            st.info("No decision record for this ply.")

    # Panel 4 — Candidates
    with row2_col2:
        st.markdown("**Candidates**")
        if not candidates:
            st.info("No candidates for this ply.")
        else:
            cand_rows = []
            for c in sorted(candidates, key=lambda x: x.get("internal_rank", 999)):
                risk = ", ".join(c.get("risk_flags", [])) or "none"
                narr = c.get("candidate_narrative", "") or ""
                cand_rows.append({
                    "SAN": c.get("san", ""),
                    "Rank": c.get("internal_rank", ""),
                    "Shown@": c.get("presentation_index", ""),
                    "Risk": risk,
                    "Narrative": narr[:100] + ("…" if len(narr) > 100 else ""),
                })
            st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, hide_index=True)
