#!/usr/bin/env python3
"""EcoLLM sustainable query-analysis dashboard — v2.

Additive to ecoLLM_benchmark.py: adds a Pareto efficiency-frontier view,
a reasoned leaderboard, dataset-level small multiples, a validated
colorblind-safe categorical palette, and a Tokens/kWh efficiency metric.
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecoLLM.analysis import build_run_catalog, pareto_front, read_artifact, sorted_frontier

RUNS = ROOT / "runs"
CONFIGS_DIR = ROOT / "configs"
TASKS_DIR = ROOT / "tasks"
UI_RUNS_CONFIG_DIR = CONFIGS_DIR / "ui_runs"
UI_RUNS_OUTPUT_DIR = RUNS / "ui_runs"
MODEL_REGISTRY_PATH = CONFIGS_DIR / "model_registry.yaml"
BENCH_MODULE = "ecoLLM.bench"

FAMILY_LABELS = {
    "family1_qa": "F1 · Question Answering",
    "family2_anomaly": "F2 · Anomaly Detection",
    "family3_forecast": "F3 · Forecasting",
    "family4_codegen": "F4 · Code Generation",
}

# Human-readable, unit-bearing names for every axis/column a chart or table
# might show — passed as Plotly's `labels=` so ticks/titles never surface a
# raw dataframe column name like "energy_wh".
AXIS_LABELS = {
    "model_id": "Model",
    "task_id": "Query",
    "family": "Dataset",
    "accuracy": "Accuracy (pass rate, 0–1)",
    "latency_s": "Latency (s)",
    "energy_wh": "Energy (Wh)",
    "emissions_g": "CO₂ Emissions (g)",
    "cost_usd": "Cost (USD)",
    "tokens_per_kwh": "Efficiency (Tokens / kWh)",
}

# Traffic-light status tones (background tint, foreground text) for the
# Query × Model matrix — reserved for relative rank, never reused as a
# categorical series color.
STATUS_STYLES = {
    "good": ("rgba(74,222,128,0.16)", "#4ade80"),
    "warning": ("rgba(251,191,36,0.18)", "#fbbf24"),
    "critical": ("rgba(248,113,113,0.18)", "#f87171"),
}

LOGO_ICON_PATH = Path(__file__).resolve().parent / "assets" / "ecoLLM_icon.png"

st.set_page_config(
    page_title="EcoLLM Dashboard",
    page_icon=str(LOGO_ICON_PATH) if LOGO_ICON_PATH.exists() else "🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#07101f;--panel:#0d192b;--panel2:#111f33;--line:#21324a;--text:#f1f5fb;--muted:#8fa1b8;--blue:#3b82f6;--purple:#8b5cf6;--green:#72b947;--cyan:#35d0ba;}
html,body,[data-testid="stAppViewContainer"]{background:linear-gradient(145deg,#07101f 0%,#0a1424 55%,#07101d 100%);color:var(--text);font-family:Inter,sans-serif}
[data-testid="stHeader"]{background:transparent!important;box-shadow:none!important}
[data-testid="stMainMenu"],[data-testid="stToolbarActions"],#MainMenu,footer{display:none!important}
[data-testid="stExpandSidebarButton"]{color:#d8e1ee!important}
.block-container{max-width:1800px;padding:0 1rem 2rem 1rem}
[data-testid="stSidebar"]{top:90px;height:calc(100vh - 90px);border-right:1px solid var(--line)}
[data-testid="stSidebar"]>div:first-child{background:#091426;padding-top:1.2rem}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] label span,
[data-testid="stSidebar"] label div{color:#d8e1ee!important}.stRadio>div{gap:.35rem}
[data-testid="stSidebar"] .stRadio label{padding:.7rem .8rem;border-radius:8px;width:100%;transition:.15s}
[data-testid="stSidebar"] .stRadio label:hover{background:#17253a}
/* Selected nav item: force a visible accent instead of Streamlit's default
   primaryColor, which reads as low-contrast against this dark panel. */
[data-testid="stSidebar"] .stRadio label[aria-checked="true"],
[data-testid="stSidebar"] .stRadio label:has(input:checked){background:#17263b}
[data-testid="stSidebar"] .stRadio label[aria-checked="true"] p,
[data-testid="stSidebar"] .stRadio label[aria-checked="true"] span,
[data-testid="stSidebar"] .stRadio label:has(input:checked) p,
[data-testid="stSidebar"] .stRadio label:has(input:checked) span{color:#67a7ff!important;font-weight:600}
.topbar{height:90px;margin:0 -1rem 14px;padding:0 22px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);background:rgba(6,15,29,.96)}
.brand{display:flex;align-items:center;gap:14px}.brand-icon{font-size:40px}.brand-icon-img{height:58px;width:auto;display:block}.brand-title{font-size:34px;font-weight:800;letter-spacing:-.01em;line-height:1.1}.brand-sub{font-size:14px;color:var(--muted);margin-top:4px}
.panel{background:linear-gradient(145deg,rgba(16,31,51,.96),rgba(11,24,41,.96));border:1px solid var(--line);border-radius:10px;padding:14px 16px;box-shadow:0 12px 35px rgba(0,0,0,.16);margin-bottom:12px}
.section-title{font-size:17px;font-weight:650;margin-bottom:12px}.section-sub{font-size:12.5px;color:var(--muted)}
.metric-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:9px;margin-bottom:12px}.metric-card{background:linear-gradient(145deg,#14233a,#101d30);border:1px solid #233753;border-radius:8px;padding:10px 11px}.metric-label{font-size:12.5px;font-weight:600}.metric-hint{font-size:11px;color:var(--muted);margin:3px 0}.metric-value{font-size:21px;font-weight:600;color:#58a3ff}.metric-card:nth-child(2) .metric-value{color:#a98aff}.metric-card:nth-child(3) .metric-value{color:#f0a92d}.metric-card:nth-child(4) .metric-value{color:#83c94f}.metric-card:nth-child(5) .metric-value{color:#3fd4be}
.query-box{font-family:'JetBrains Mono',monospace;font-size:13.5px;line-height:1.6;background:#091426;border:1px solid #253850;border-radius:7px;padding:14px;max-height:310px;overflow:auto;white-space:pre-wrap}.meta-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:11px}.pill{background:#17263b;border:1px solid #263a55;border-radius:6px;padding:6px 9px;font-size:12px;color:#dbe5f2}.expected{background:#132136;border-radius:7px;padding:11px;font-size:12.5px;color:#c8d3e2;margin-top:10px}
.status-card{margin-top:28px;border:1px solid var(--line);border-radius:9px;padding:18px 12px;text-align:center;color:var(--muted);font-size:12.5px}.status-icon{font-size:28px;margin-bottom:8px}.sidebar-brand{font-weight:700;font-size:17px;margin:0 0 17px 6px;color:white}.sidebar-note{font-size:12px;color:var(--muted);line-height:1.5}
div[data-baseweb="select"]>div{background:#101d30!important;border-color:#263a55!important;color:white!important;min-height:39px}div[data-baseweb="tag"]{background:#203d78!important}label,p,span{color:inherit}
.stTabs [data-baseweb="tab-list"]{gap:22px;border-bottom:1px solid var(--line)}.stTabs [data-baseweb="tab"]{font-size:13px;padding:8px 3px}.stTabs [aria-selected="true"]{color:#67a7ff!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px;overflow:hidden}.stDownloadButton button,.stButton button,.stFormSubmitButton button,[data-testid="stFormSubmitButton"] button{background:linear-gradient(90deg,#7447d7,#8c4bd8);color:white!important;border:0;border-radius:7px}
.stFormSubmitButton button:disabled,.stButton button:disabled{background:#2a3546;color:#6b7a91!important;opacity:1}
.badge{display:inline-block;background:#17263b;border:1px solid #263a55;border-radius:12px;padding:3px 10px;font-size:11.5px;color:#dbe5f2;margin:1px 3px 1px 0}
.log-box{font-family:'JetBrains Mono',monospace;font-size:12.5px;line-height:1.55;color:#c3d2e5;background:#070d18;border:1px solid #253850;border-radius:7px;padding:12px 14px;max-height:360px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;margin:8px 0}
.leaderboard-table{width:100%;border-collapse:collapse;font-size:13.5px}
.leaderboard-table th{text-align:left;color:var(--muted);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
.leaderboard-table td{padding:8px 10px;border-bottom:1px solid #1a2739;color:var(--text)}
.leaderboard-table tr:hover td{background:#101d30}
.dag-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0 18px}
.dag-node{background:#0b1424;border:1.5px solid #3987e5;border-radius:8px;padding:8px 15px;font-size:13px;font-weight:700;white-space:nowrap}
.dag-arrow{color:var(--muted);font-size:19px;font-weight:700}
.qm-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
.qm-table{width:100%;border-collapse:collapse;font-size:13px;min-width:640px}
.qm-table th{padding:8px 11px;text-align:center;border-bottom:1px solid var(--line);font-weight:700;white-space:nowrap;color:var(--text)}
.qm-table td{padding:7px 11px;text-align:center;border-bottom:1px solid #1a2739;white-space:nowrap}
.qm-table thead tr:first-child th{font-size:12.5px;letter-spacing:.02em}
.qm-tags{text-align:left!important}
.qm-qid{text-align:left!important;color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:12px}
.qm-val{font-weight:700;font-size:13px}
.qm-std{font-size:10.5px;color:var(--muted);margin-top:1px}
.qm-na{color:var(--muted)}
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:38px 20px;color:var(--muted)}
.empty-icon{font-size:34px;margin-bottom:10px;opacity:.85}
.empty-title{font-size:14.5px;font-weight:650;color:#c9d5e6}
.empty-hint{font-size:12.5px;color:var(--muted);margin-top:6px;max-width:440px;line-height:1.5}
.selection-bar{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:-2px 0 16px;padding:9px 16px;background:#0d1a2c;border:1px solid var(--line);border-radius:9px}
.sel-chip{font-size:12.5px;color:#c9d5e6;display:inline-flex;align-items:center;gap:6px}
.sel-chip b{color:#f1f5fb;font-weight:700}
@media(max-width:1100px){.metric-grid{grid-template-columns:repeat(2,1fr)}}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Validated categorical palette (dark-mode steps, see dataviz skill palette.md).
# Adjacent-use (bar/line) passes all 8 slots; scatter/bubble (all-pairs) is only
# guaranteed for the first 3 — the efficiency-frontier chart below uses a
# 2-state emphasis encoding (frontier vs. dominated) instead of per-model hue
# for exactly that reason.
# ----------------------------------------------------------------------------
MODEL_PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OVERFLOW_COLOR = "#5b6470"
FRONTIER_COLOR = "#3987e5"
DOMINATED_COLOR = "#5b6470"


def model_color_map(models: Iterable[str]) -> dict[str, str]:
    """Assign each model a fixed palette slot by sorted name so color follows
    the entity, not its current rank/filter — survivors keep their hue."""
    ordered = sorted(set(str(m) for m in models))
    return {m: (MODEL_PALETTE[i] if i < len(MODEL_PALETTE) else OVERFLOW_COLOR) for i, m in enumerate(ordered)}


def panel_start(title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="panel"><div class="section-title">{title}</div>{sub}', unsafe_allow_html=True)


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def empty_state(icon: str, title: str, hint: str = "") -> None:
    """A designed placeholder for 'nothing to show because of your current
    filters' — used instead of a bare st.info/st.warning line."""
    hint_html = f'<div class="empty-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="empty-state"><div class="empty-icon">{icon}</div>'
        f'<div class="empty-title">{title}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def chart_theme(fig: go.Figure, height: int = 255, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#b9c7d8", size=12.5),
        title_font=dict(size=14.5, color="#eef4fb"),
        margin=dict(l=24, r=12, t=42, b=24),
        showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)", linecolor="#34465e", title_font=dict(size=13), tickfont=dict(size=11.5)),
        yaxis=dict(gridcolor="rgba(255,255,255,.09)", linecolor="#34465e", title_font=dict(size=13), tickfont=dict(size=11.5)),
    )
    fig.update_traces(marker_line_color="rgba(255,255,255,.28)", marker_line_width=.8)
    return fig


@st.cache_data(ttl=30, show_spinner=False)
def catalog() -> pd.DataFrame:
    return build_run_catalog(RUNS)


@st.cache_data(ttl=30, show_spinner=False)
def load_results(paths: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for value in paths:
        run_dir = Path(value)
        frame = read_artifact(run_dir, "results")
        if frame is None:
            continue
        frame = frame.copy()
        frame["run_name"] = run_dir.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "is_warmup" in result:
        result = result.loc[~result["is_warmup"].fillna(False).astype(bool)]
    if "meta.table_rows" in result and "table_rows" not in result:
        result["table_rows"] = pd.to_numeric(result["meta.table_rows"], errors="coerce")
    if "meta.complexity" in result and "complexity" not in result:
        result["complexity"] = result["meta.complexity"].astype(str)
    return result


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for target, source, scale in [
        ("accuracy", "metric.task_pass", 1),
        ("latency_s", "latency_ms_total", .001),
        ("energy_wh", "energy_kwh", 1000),
        ("emissions_g", "emissions_kg", 1000),
        ("cost_usd", "api_cost_usd", 1),
    ]:
        frame[target] = pd.to_numeric(frame[source], errors="coerce") * scale if source in frame else np.nan
    tokens_in = pd.to_numeric(frame.get("tokens_in"), errors="coerce").fillna(0)
    tokens_out = pd.to_numeric(frame.get("tokens_out"), errors="coerce").fillna(0)
    tokens_total = tokens_in + tokens_out
    frame["tokens_per_kwh"] = np.where(frame["energy_wh"] > 0, tokens_total / (frame["energy_wh"] / 1000), np.nan)
    return frame


def metric_cards(metrics: pd.DataFrame) -> None:
    specs = [
        ("Accuracy", "accuracy", ".2f", "Higher is better"),
        ("Latency (s)", "latency_s", ".2f", "Lower is better"),
        ("Energy (Wh)", "energy_wh", ".4f", "Lower is better"),
        ("CO₂ Emissions (g)", "emissions_g", ".4f", "Lower is better"),
        ("Cost (USD)", "cost_usd", ".5f", "Lower is better"),
    ]
    cards = []
    for label, column, fmt, hint in specs:
        values = metrics[column].dropna()
        value = "—" if values.empty else (format(values.iloc[0], fmt) if len(values) == 1 else f"{format(values.min(), fmt)} – {format(values.max(), fmt)}")
        cards.append(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-hint">{hint}</div><div class="metric-value">{value}</div></div>')
    st.markdown('<div class="metric-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def aggregate(frame: pd.DataFrame, group) -> pd.DataFrame:
    return frame.groupby(group, as_index=False).agg(
        accuracy=("accuracy", "mean"), latency_s=("latency_s", "mean"),
        energy_wh=("energy_wh", "mean"), emissions_g=("emissions_g", "mean"),
        cost_usd=("cost_usd", "mean"), tokens_per_kwh=("tokens_per_kwh", "mean"),
        observations=("task_id", "size"),
    )


DEFAULT_SCORE_WEIGHTS = {"accuracy": 40.0, "energy_wh": 30.0, "cost_usd": 20.0, "latency_s": 10.0}
SCORE_WEIGHT_LABELS = {"accuracy": "Accuracy", "energy_wh": "Energy", "cost_usd": "Cost", "latency_s": "Latency"}
SCORE_WEIGHT_DIRECTION = {"accuracy": True, "energy_wh": False, "cost_usd": False, "latency_s": False}


def composite_score(summary: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted 0-10 score. Weights are shares (need not sum to 100 — they're
    normalized here), defaulting to 40% accuracy / 30% energy / 20% cost / 10% latency."""
    weights = weights or DEFAULT_SCORE_WEIGHTS
    total = sum(weights.values()) or 1.0

    def norm(column: str, higher_is_better: bool) -> pd.Series:
        span = summary[column].max() - summary[column].min()
        if not np.isfinite(span) or span == 0:
            return pd.Series(5.0, index=summary.index)
        score = (summary[column] - summary[column].min()) / span
        return score if higher_is_better else 1 - score

    return 10 * sum(
        (weights[column] / total) * norm(column, higher_is_better)
        for column, higher_is_better in SCORE_WEIGHT_DIRECTION.items()
    )


def _rebalance_weights(changed_key: str) -> None:
    """Callback for the Leaderboard weight sliders: keep all four summing to
    100 by redistributing the delta across the other three, proportional to
    their current share (a 'budget split' interaction, not independent 0-100s)."""
    weights = st.session_state["score_weights"]
    new_val = float(st.session_state[f"w_{changed_key}"])
    others = [k for k in weights if k != changed_key]
    old_others_sum = sum(weights[k] for k in others)
    new_others_sum = 100.0 - new_val
    if old_others_sum <= 0:
        share = new_others_sum / len(others)
        for k in others:
            weights[k] = share
    else:
        scale = new_others_sum / old_others_sum
        for k in others:
            weights[k] = max(0.0, weights[k] * scale)
    weights[changed_key] = new_val
    for k in weights:
        st.session_state[f"w_{k}"] = round(weights[k], 1)
    st.session_state["score_weights"] = weights


def _reset_weights() -> None:
    st.session_state["score_weights"] = dict(DEFAULT_SCORE_WEIGHTS)
    for k, v in DEFAULT_SCORE_WEIGHTS.items():
        st.session_state[f"w_{k}"] = v


def render_weight_sliders() -> dict[str, float]:
    st.session_state.setdefault("score_weights", dict(DEFAULT_SCORE_WEIGHTS))
    for k, v in st.session_state["score_weights"].items():
        st.session_state.setdefault(f"w_{k}", v)

    cols = st.columns([1, 1, 1, 1, 0.8])
    for col, key in zip(cols, DEFAULT_SCORE_WEIGHTS):
        col.slider(SCORE_WEIGHT_LABELS[key], 0.0, 100.0, step=1.0, key=f"w_{key}", on_change=_rebalance_weights, args=(key,))
    cols[4].markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    cols[4].button("Reset", on_click=_reset_weights, key="reset_weights")
    return {k: st.session_state[f"w_{k}"] for k in DEFAULT_SCORE_WEIGHTS}


def leaderboard_tags(board: pd.DataFrame) -> pd.Series:
    """One or more short 'why' badges per model, only for standouts."""
    tags: dict[int, list[str]] = {i: [] for i in board.index}
    picks = [
        ("composite_score", True, "🏆 Best overall"),
        ("accuracy", True, "🎯 Highest accuracy"),
        ("tokens_per_kwh", True, "⚡ Most efficient"),
        ("cost_usd", False, "💵 Cheapest"),
        ("latency_s", False, "🚀 Fastest"),
    ]
    for column, want_max, label in picks:
        series = board[column].dropna()
        if series.empty:
            continue
        idx = series.idxmax() if want_max else series.idxmin()
        tags[idx].append(label)
    return pd.Series({i: " ".join(f'<span class="badge">{t}</span>' for t in v) for i, v in tags.items()})


# ----------------------------------------------------------------------------
# Run Lab: discover tasks/models, manage a user-editable model registry, and
# launch bench.py as a tracked background subprocess with a live log tail.
# ----------------------------------------------------------------------------
TASK_FILE_PATTERN = re.compile(r"seds_(f\d)_(c\d)_sweep(_online)?\.yaml$", re.IGNORECASE)


def discover_task_files() -> pd.DataFrame:
    rows = []
    for path in sorted(TASKS_DIR.glob("family*/*.yaml")):
        match = TASK_FILE_PATTERN.search(path.name)
        if not match:
            continue
        family_code, complexity, online = match.group(1).upper(), match.group(2).upper(), bool(match.group(3))
        rows.append({
            "path": path,
            "family_dir": path.parent.name,
            "family_code": family_code,
            "complexity": complexity,
            "mode": "online" if online else "offline",
        })
    return pd.DataFrame(rows)


def load_model_pool() -> list[dict]:
    """Every ollama/openai model already referenced in a top-level config,
    plus anything registered through the Run Lab 'Add a model' form."""
    seen: dict[str, dict] = {}
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for model in cfg.get("models") or []:
            adapter = model.get("adapter")
            model_id = model.get("model_id")
            if adapter in {"ollama", "openai"} and model_id and model_id not in seen:
                seen[model_id] = {"model_id": model_id, "adapter": adapter, "params": model.get("params") or {}, "source": path.name}
    return sorted(seen.values(), key=lambda m: m["model_id"])


def save_model_to_registry(entry: dict) -> None:
    registry = {"models": []}
    if MODEL_REGISTRY_PATH.exists():
        registry = yaml.safe_load(MODEL_REGISTRY_PATH.read_text(encoding="utf-8")) or {"models": []}
    registry.setdefault("models", [])
    registry["models"] = [m for m in registry["models"] if m.get("model_id") != entry["model_id"]]
    registry["models"].append(entry)
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_PATH.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")


def ollama_installed_tags() -> set[str] | None:
    """None means the ollama CLI itself isn't reachable; empty set means it is, with nothing pulled."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()[1:]
    return {line.split()[0] for line in lines if line.strip()}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "run"


def launch_background(cmd: list[str], log_path: Path, cwd: Path, extra_env: dict | None = None) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with open(log_path, "w", encoding="utf-8") as log_file:
        return subprocess.Popen(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)


def tail_lines(path: Path, max_lines: int = 250) -> str:
    if not path.exists():
        return "(waiting for output…)"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"(could not read log: {exc})"
    return "\n".join(lines[-max_lines:]) or "(no output yet)"


@st.cache_data(show_spinner=False)
def load_logo_data_uri() -> str:
    """The EcoLLM logo, cropped to just the icon mark (reference_images/ecoLLM.png
    has the wordmark baked in below it, which would duplicate the text heading)."""
    if not LOGO_ICON_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_ICON_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_logo_uri = load_logo_data_uri()
_brand_icon_html = f'<img class="brand-icon-img" src="{_logo_uri}" alt="EcoLLM logo"/>' if _logo_uri else '<div class="brand-icon">🌿</div>'

st.markdown(
    f"""<div class="topbar"><div class="brand">{_brand_icon_html}<div><div class="brand-title">EcoLLM Dashboard</div><div class="brand-sub">Energy-aware LLM Bench for AI Data Systems</div></div></div></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌱 EcoLLM</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "Run Lab", "Efficiency Frontier", "Query Analysis", "Model Comparison", "Dataset Comparison", "Leaderboard", "Run Logs"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown('<div class="status-card"><div class="status-icon">🌱</div><b>Analyze sustainably.</b><br><span class="sidebar-note">Compare quality, latency, energy, emissions and cost across models and queries.</span></div>', unsafe_allow_html=True)

cat = catalog()
available = cat[(cat.get("has_results", False) == True) & (cat.get("status", "") == "complete")].copy() if len(cat) else pd.DataFrame()
if available.empty:
    empty_state("📭", "No completed benchmark runs found", "Use Run Lab to launch one, or check that runs/ contains a completed run.")
    st.stop()
preferred = available[available["run_name"].str.contains("complete", case=False, na=False)]
defaults = preferred.head(1)["run_path"].tolist() or available.head(1)["run_path"].tolist()
def _format_run_label(row: pd.Series) -> str:
    parts = [str(row["run_name"])]
    if pd.notna(row.get("models")):
        count = int(row["models"])
        parts.append(f"{count} model{'s' if count != 1 else ''}")
    if row.get("run_kind"):
        parts.append(str(row["run_kind"]))
    if pd.notna(row.get("modified_utc")):
        parts.append(row["modified_utc"].strftime("%b %d"))
    return " · ".join(parts)


RUN_LABELS = {row["run_path"]: _format_run_label(row) for _, row in available.iterrows()}
selected_paths = st.sidebar.multiselect(
    "Result runs",
    available["run_path"].tolist(),
    default=defaults,
    format_func=lambda value: RUN_LABELS.get(value, Path(value).name),
)
data = prepare(load_results(tuple(selected_paths))) if selected_paths else pd.DataFrame()
if data.empty and page not in {"Run Lab", "Run Logs"}:
    empty_state("🗂️", "No run selected", "Pick one or more runs from **Result runs** in the sidebar, or use Run Lab to create one.")
    st.stop()

GLOBAL_MODEL_COLORS = model_color_map(data["model_id"].dropna().unique()) if not data.empty else {}

if not data.empty:
    _n_runs, _n_datasets, _n_models, _n_obs = len(selected_paths), data["family"].nunique(), data["model_id"].nunique(), len(data)
    st.markdown(
        '<div class="selection-bar">'
        f'<span class="sel-chip">📦 <b>{_n_runs}</b> run{"s" if _n_runs != 1 else ""}</span>'
        f'<span class="sel-chip">🗂️ <b>{_n_datasets}</b> dataset{"s" if _n_datasets != 1 else ""}</span>'
        f'<span class="sel-chip">🤖 <b>{_n_models}</b> model{"s" if _n_models != 1 else ""}</span>'
        f'<span class="sel-chip">🔢 <b>{_n_obs:,}</b> observations</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Query Analysis extras: the operator-pipeline DAG (sourced from the task
# YAML — `operator_pattern` doesn't survive into results.parquet) and the
# per-query × per-model heatmap matrix.
# ----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_task_definitions() -> dict[str, dict]:
    """task_id -> its `meta` block (operator_pattern, difficulty, etc.), read
    straight from tasks/**/*.yaml since results.parquet only keeps a few
    flattened meta.* columns."""
    definitions: dict[str, dict] = {}
    for path in TASKS_DIR.glob("family*/*.yaml"):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for task in doc.get("tasks") or []:
            task_id = task.get("task_id")
            if task_id:
                definitions[str(task_id)] = task.get("meta") or {}
    return definitions


def render_operator_dag(operators: list[str]) -> str:
    """A left-to-right node chain: nodes = operators, edges = execution order."""
    if not operators:
        return '<div class="section-sub">No operator pipeline recorded for this query.</div>'
    colors = model_color_map(operators)
    parts = []
    for i, op in enumerate(operators):
        color = colors[op]
        label = html.escape(str(op).replace("_", " ").title())
        parts.append(f'<div class="dag-node" style="border-color:{color};color:{color};">{label}</div>')
        if i < len(operators) - 1:
            parts.append('<div class="dag-arrow">→</div>')
    return '<div class="dag-row">' + "".join(parts) + "</div>"


def render_query_matrix(subset: pd.DataFrame, model_ids: list[str], task_defs: dict, shown_tasks: list[str]) -> None:
    """SemBench-style heatmap: rows = queries, column-groups = models,
    sub-columns = Cost/Quality/Latency, colored relative to the other
    selected models on that row (green = best, red = worst). `shown_tasks`
    is the already-paginated slice of task ids the caller wants rendered."""
    if not model_ids:
        empty_state("🧩", "Pick at least one model", "Select models above to build the comparison matrix.")
        return
    if not shown_tasks:
        empty_state("🔍", "No queries in this dataset")
        return

    scoped = subset[subset["task_id"].astype(str).isin(shown_tasks) & subset["model_id"].isin(model_ids)].copy()
    scoped["task_id"] = scoped["task_id"].astype(str)
    grouped = scoped.groupby(["task_id", "model_id"], as_index=False).agg(
        cost_usd_mean=("cost_usd", "mean"), cost_usd_std=("cost_usd", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        latency_s_mean=("latency_s", "mean"), latency_s_std=("latency_s", "std"),
        difficulty=("difficulty", "first"),
    )
    lookup = {(r.task_id, r.model_id): r for r in grouped.itertuples()}
    metric_specs = [("cost_usd", "${:.4f}", False), ("accuracy", "{:.2f}", True), ("latency_s", "{:.1f}s", False)]

    rows_html = []
    for tid in shown_tasks:
        meta = task_defs.get(tid, {})
        ops = meta.get("operator_pattern") or []
        op_badge = html.escape(str(ops[-1]).replace("_", " ").title()) if ops else "—"
        difficulty = next((str(lookup[(tid, mid)].difficulty) for mid in model_ids if (tid, mid) in lookup and pd.notna(lookup[(tid, mid)].difficulty)), "")
        cells = f'<td class="qm-tags"><span class="badge">{html.escape(difficulty.title()) or "—"}</span><span class="badge">{op_badge}</span></td><td class="qm-qid">{html.escape(tid)}</td>'
        for metric_key, fmt, higher_better in metric_specs:
            values = {
                mid: getattr(lookup[(tid, mid)], f"{metric_key}_mean")
                for mid in model_ids
                if (tid, mid) in lookup and pd.notna(getattr(lookup[(tid, mid)], f"{metric_key}_mean"))
            }
            # Rank by distinct value, not by row position — tied values (e.g. $0
            # cost across every local model) must land in the same tier instead
            # of an arbitrary good/warning/critical split among equals.
            distinct_sorted = sorted({v for v in values.values()}, reverse=higher_better)
            rank_of_value = {v: i for i, v in enumerate(distinct_sorted)}
            tiers = len(distinct_sorted)
            def _status(v: float) -> str:
                r = rank_of_value[v]
                if tiers <= 1 or r == 0:
                    return "good"
                return "critical" if r == tiers - 1 else "warning"
            status_for = {mid: _status(v) for mid, v in values.items()}
            for mid in model_ids:
                row = lookup.get((tid, mid))
                mean_v = getattr(row, f"{metric_key}_mean") if row is not None else None
                if row is None or pd.isna(mean_v):
                    cells += '<td class="qm-cell qm-na">—</td>'
                    continue
                std_v = getattr(row, f"{metric_key}_std")
                bg, fg = STATUS_STYLES[status_for.get(mid, "warning")]
                std_txt = f'<div class="qm-std">±{std_v:.2g}</div>' if pd.notna(std_v) and std_v > 0 else ""
                cells += f'<td class="qm-cell" style="background:{bg};"><div class="qm-val" style="color:{fg};">{fmt.format(mean_v)}</div>{std_txt}</td>'
        rows_html.append(f"<tr>{cells}</tr>")

    header_groups = "".join(
        f'<th colspan="3" class="qm-group" style="background:{GLOBAL_MODEL_COLORS.get(mid, OVERFLOW_COLOR)}26;color:{GLOBAL_MODEL_COLORS.get(mid, OVERFLOW_COLOR)};">{html.escape(mid)}</th>'
        for mid in model_ids
    )
    sub_headers = "".join("<th>Cost</th><th>Quality</th><th>Latency</th>" for _ in model_ids)
    table_html = (
        '<div class="qm-scroll"><table class="qm-table"><thead>'
        f"<tr><th></th><th></th>{header_groups}</tr>"
        f"<tr><th>Tags</th><th>Query</th>{sub_headers}</tr>"
        f"</thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def query_analysis() -> None:
    panel_start("Query Analysis", "Select a workload, query and model set. Compare models or compare multiple queries for one model.")
    c1, c2, c3, c4 = st.columns([1.05, 1.25, 1.65, 1.05])
    families = sorted(data["family"].dropna().astype(str).unique())
    family = c1.selectbox("Select Dataset", families)
    subset = data[data["family"].astype(str) == family]
    task_ids = sorted(subset["task_id"].dropna().astype(str).unique())
    models = sorted(subset["model_id"].dropna().astype(str).unique())
    mode = c4.radio("Compare By", ["Models", "Queries"], horizontal=True)
    if mode == "Models":
        task = c2.selectbox("Select Query", task_ids)
        chosen_models = c3.multiselect("Select Model(s)", models, default=models[: min(3, len(models))])
        filtered = subset[(subset["task_id"].astype(str) == task) & subset["model_id"].isin(chosen_models)]
        group, detail_task = "model_id", task
    else:
        model = c2.selectbox("Select Model", models)
        chosen_tasks = c3.multiselect("Select Queries", task_ids, default=task_ids[: min(4, len(task_ids))])
        filtered = subset[(subset["model_id"] == model) & subset["task_id"].astype(str).isin(chosen_tasks)]
        group, detail_task = "task_id", (chosen_tasks[0] if chosen_tasks else task_ids[0])
    panel_end()

    metrics = aggregate(filtered, group) if len(filtered) else pd.DataFrame()
    left, right = st.columns([1.62, 1], gap="medium")
    with left:
        panel_start("Performance Analysis")
        if metrics.empty:
            empty_state("🎛️", "Nothing selected yet", "Choose a query + model(s), or a model + queries, above.")
        else:
            with st.spinner("Rendering charts…"):
                metric_cards(metrics)
                color_map = GLOBAL_MODEL_COLORS if group == "model_id" else None
                charts = [("accuracy", "Accuracy Comparison"), ("energy_wh", "Energy Consumption (Wh) Comparison"), ("emissions_g", "CO₂ Emissions (g) Comparison"), ("latency_s", "Latency (s) Comparison")]
                for start in (0, 2):
                    columns = st.columns(2)
                    for container, (metric, title) in zip(columns, charts[start:start + 2]):
                        plot_df = metrics.sort_values(metric, ascending=False)
                        fig = px.bar(plot_df, x=group, y=metric, color=group, text_auto=".3g", color_discrete_map=color_map, labels=AXIS_LABELS, title=title)
                        container.plotly_chart(chart_theme(fig), width="stretch")
                table = metrics.rename(columns={group: "Model" if group == "model_id" else "Query", "accuracy": "Accuracy", "latency_s": "Latency (s)", "energy_wh": "Energy (Wh)", "emissions_g": "CO₂ (g)", "cost_usd": "Cost (USD)", "tokens_per_kwh": "Tokens/kWh"})
                st.markdown("**Detailed Metrics Table**")
                st.dataframe(table, width="stretch", hide_index=True)
                st.caption("Metrics are averaged over measured repetitions. Warm-up executions are excluded.")
        panel_end()

    with right:
        panel_start("Query Details")
        details = subset[subset["task_id"].astype(str) == str(detail_task)]
        if details.empty:
            empty_state("📄", "Select a query to see its details")
        else:
            row = details.iloc[0]
            query_tab, plan_tab, results_tab, metadata_tab = st.tabs(["Query", "Execution Plan", "Results", "Metadata"])
            prompt = str(row.get("prompt", "")).split("\n\nCONTEXT:")[0].split("OUTPUT FORMAT")[0].strip()
            with query_tab:
                st.markdown("**Query Text**")
                st.markdown(f'<div class="query-box">{prompt}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta-row"><span class="pill">Query ID: {detail_task}</span><span class="pill">Dataset: {row.get("family")}</span><span class="pill">Rows: {row.get("table_rows", row.get("meta.table_rows", "—"))}</span></div>', unsafe_allow_html=True)
                expected = {column.removeprefix("expected."): row.get(column) for column in details.columns if column.startswith("expected.") and pd.notna(row.get(column))}
                st.markdown('<div class="expected"><b>Ground Truth (Expected Output)</b><br>' + (json.dumps(expected, default=str) if expected else "Structured scorer contract attached to this query.") + "</div>", unsafe_allow_html=True)
            with plan_tab:
                st.markdown("**Operator Pipeline (query DAG)**")
                task_meta = load_task_definitions().get(str(detail_task), {})
                st.markdown(render_operator_dag(task_meta.get("operator_pattern") or []), unsafe_allow_html=True)
                st.markdown("**Scoring Spec**")
                st.code(str(row.get("scoring_spec", "Deterministic validation")), language="text")
            with results_tab:
                columns = [column for column in ["model_id", "repeat_idx", "metric.task_pass", "latency_s", "energy_wh", "emissions_g", "metric.fail_reason"] if column in details]
                st.dataframe(details[columns], width="stretch", hide_index=True)
                best = details.sort_values(["metric.task_pass", "latency_ms_total"], ascending=[False, True], na_position="last").iloc[0]
                st.code(str(best.get("output_text", ""))[:9000], language="json")
            with metadata_tab:
                st.json({"complexity": row.get("complexity"), "difficulty": row.get("difficulty"), "table_rows": row.get("table_rows"), "energy_source": row.get("energy_source", "unspecified"), "run": row.get("run_name")})
        panel_end()

    panel_start(
        "Query × Model Matrix",
        "Every query in this dataset, compared across models. Colored relative to the other selected models on that row — green = best, red = worst.",
    )
    matrix_models = st.multiselect("Models in matrix (4 or fewer stays readable)", models, default=models[: min(4, len(models))], key="qm_models")

    page_size = 20
    task_ids_all = sorted(subset["task_id"].dropna().astype(str).unique())
    total_pages = max(1, -(-len(task_ids_all) // page_size))
    st.session_state.setdefault("qm_page", 1)
    st.session_state["qm_page"] = min(st.session_state["qm_page"], total_pages)

    pg1, pg2, pg3 = st.columns([1, 3, 1])
    if pg1.button("← Prev", disabled=st.session_state["qm_page"] <= 1, key="qm_prev"):
        st.session_state["qm_page"] -= 1
        st.rerun()
    start = (st.session_state["qm_page"] - 1) * page_size
    end = min(start + page_size, len(task_ids_all))
    pg2.markdown(
        f'<div style="text-align:center;color:var(--muted);font-size:12.5px;padding-top:8px;">'
        f'Queries {start + 1}–{end} of {len(task_ids_all)} · page {st.session_state["qm_page"]} of {total_pages}</div>',
        unsafe_allow_html=True,
    )
    if pg3.button("Next →", disabled=st.session_state["qm_page"] >= total_pages, key="qm_next"):
        st.session_state["qm_page"] += 1
        st.rerun()

    with st.spinner("Building matrix…"):
        render_query_matrix(subset, matrix_models, load_task_definitions(), task_ids_all[start:end])
    panel_end()


def overview() -> None:
    panel_start("Overview", "A high-level view of the selected benchmark results.")
    models = data["model_id"].nunique(); queries = data["task_id"].nunique(); runs = data["run_name"].nunique(); success = data["accuracy"].mean()
    st.markdown(f'<div class="metric-grid"><div class="metric-card"><div class="metric-label">Models</div><div class="metric-value">{models}</div></div><div class="metric-card"><div class="metric-label">Queries</div><div class="metric-value">{queries}</div></div><div class="metric-card"><div class="metric-label">Runs</div><div class="metric-value">{runs}</div></div><div class="metric-card"><div class="metric-label">Pass Rate</div><div class="metric-value">{success:.1%}</div></div><div class="metric-card"><div class="metric-label">Observations</div><div class="metric-value">{len(data)}</div></div></div>', unsafe_allow_html=True)
    with st.spinner("Rendering chart…"):
        summary = aggregate(data, "model_id")
        fig = px.scatter(
            summary, x="latency_s", y="accuracy", size="energy_wh", color="model_id",
            color_discrete_map=GLOBAL_MODEL_COLORS,
            text="model_id" if summary["model_id"].nunique() <= 8 else None,
            hover_data=["energy_wh", "emissions_g", "cost_usd", "tokens_per_kwh"],
            labels=AXIS_LABELS,
            title="Quality, latency and energy landscape",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(chart_theme(fig, 430, legend=True), width="stretch")
        st.caption("Bubble size = average energy per query (Wh). Every model keeps the same color across every page.")
    panel_end()


def efficiency_frontier() -> None:
    panel_start(
        "Efficiency Frontier",
        "Non-dominated models: no other model beats them on both accuracy and the chosen resource. "
        "The rest are strictly worse on at least one axis, all else equal.",
    )
    resource_options = {"Energy (Wh)": "energy_wh", "Cost (USD)": "cost_usd", "Latency (s)": "latency_s", "CO₂ Emissions (g)": "emissions_g"}
    label = st.selectbox("Resource axis (accuracy is always the quality axis)", list(resource_options))
    resource_col = resource_options[label]

    summary = aggregate(data, "model_id").dropna(subset=["accuracy", resource_col])
    if summary.empty:
        empty_state("📉", "Not enough data for a frontier", "Try a different resource axis, or select more models.")
        panel_end()
        return

    scored = pareto_front(summary, {"accuracy": "max", resource_col: "min"})
    frontier = sorted_frontier(scored, resource_col)
    dominated = scored[~scored["is_pareto"]]

    fig = go.Figure()
    if len(dominated):
        fig.add_trace(go.Scatter(
            x=dominated[resource_col], y=dominated["accuracy"], mode="markers",
            marker=dict(size=10, color=DOMINATED_COLOR, line=dict(width=0.8, color="rgba(255,255,255,.28)")),
            name="Dominated", text=dominated["model_id"],
            hovertemplate="<b>%{text}</b><br>Accuracy %{y:.2f}<br>" + label + " %{x:.4g}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=frontier[resource_col], y=frontier["accuracy"], mode="lines+markers+text",
        marker=dict(size=13, color=FRONTIER_COLOR, line=dict(width=1.2, color="white")),
        line=dict(color=FRONTIER_COLOR, width=2, dash="dot"),
        text=frontier["model_id"], textposition="top center",
        name="Efficient frontier",
        hovertemplate="<b>%{text}</b><br>Accuracy %{y:.2f}<br>" + label + " %{x:.4g}<extra></extra>",
    ))
    fig.update_layout(xaxis_title=label, yaxis_title="Accuracy")
    st.plotly_chart(chart_theme(fig, 480, legend=True), width="stretch")
    st.caption(f"{len(frontier)} of {len(scored)} models are on the efficient frontier for accuracy vs. {label.lower()}.")

    st.markdown("**Frontier models**")
    show = frontier[["model_id", "accuracy", resource_col, "tokens_per_kwh"]].rename(
        columns={"model_id": "Model", "accuracy": "Accuracy", resource_col: label, "tokens_per_kwh": "Tokens/kWh"}
    )
    st.dataframe(show, width="stretch", hide_index=True)
    panel_end()


def comparison(group: str, title: str) -> None:
    panel_start(title)
    metric_labels = {k: AXIS_LABELS[k] for k in ["accuracy", "latency_s", "energy_wh", "emissions_g", "cost_usd", "tokens_per_kwh"]}
    c1, c2 = st.columns([2, 1])
    metric = c1.selectbox("Metric", list(metric_labels), format_func=lambda m: metric_labels[m])
    facet = False
    if group == "model_id" and data["family"].nunique() > 1:
        facet = c2.checkbox("Break down by dataset", value=False, help="Small multiples: one panel per dataset instead of one average across all of them.")

    color_map = GLOBAL_MODEL_COLORS if group == "model_id" else None
    with st.spinner("Rendering chart…"):
        if facet:
            summary = data.groupby([group, "family"], as_index=False).agg(**{metric: (metric, "mean")})
            order = summary.groupby(group)[metric].mean().sort_values(ascending=False).index.tolist()
            fig = px.bar(
                summary, x=group, y=metric, color=group, facet_col="family", facet_col_wrap=3,
                category_orders={group: order}, color_discrete_map=color_map, text_auto=".3g",
                labels=AXIS_LABELS, title=f"{metric_labels[metric]} by dataset",
            )
            fig.update_xaxes(matches=None)
            rows = -(-summary["family"].nunique() // 3)
            st.plotly_chart(chart_theme(fig, 260 * max(rows, 1), legend=True), width="stretch")
        else:
            summary = aggregate(data, group).sort_values(metric, ascending=False)
            fig = px.bar(summary, x=group, y=metric, color=group, color_discrete_map=color_map, text_auto=".3g", labels=AXIS_LABELS, title=f"{metric_labels[metric]} comparison")
            st.plotly_chart(chart_theme(fig, 430), width="stretch")

        st.dataframe(summary.rename(columns=metric_labels), width="stretch", hide_index=True)
    panel_end()


@st.fragment(run_every="2s")
def _job_status_panel() -> None:
    job = st.session_state.get("lab_job")
    if not job:
        empty_state("🛰️", "No job running", "Configure a run above and click **Run benchmark**, or pull a missing Ollama model.")
        return
    proc: subprocess.Popen = job["proc"]
    code = proc.poll()
    elapsed = time.time() - job["started"]
    if code is None:
        status = '<span class="badge" style="background:#132136;color:#67a7ff;border-color:#28405f;">● Running</span>'
    elif code == 0:
        status = '<span class="badge" style="background:#0f2a1a;color:#4ade80;border-color:#1c4a2c;">✔ Completed</span>'
        if job["kind"] == "benchmark":
            catalog.clear()
    else:
        status = f'<span class="badge" style="background:#2a1414;color:#f87171;border-color:#4a1c1c;">✖ Failed (exit {code})</span>'
    st.markdown(
        f'**{job["label"]}** &nbsp; {status} &nbsp; <span class="section-sub">{elapsed:.0f}s elapsed</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<pre class="log-box">{html.escape(tail_lines(job["log_path"]))}</pre>', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 4])
    if code is None:
        if c1.button("Stop job"):
            proc.terminate()
            st.rerun()
    else:
        if c1.button("Clear"):
            st.session_state["lab_job"] = None
            st.rerun()
        if job["kind"] == "benchmark" and code == 0:
            c2.success("Pick this run under 'Result runs' in the sidebar to explore it.")


def run_lab() -> None:
    st.session_state.setdefault("lab_job", None)
    job = st.session_state["lab_job"]
    job_active = bool(job) and job["proc"].poll() is None

    panel_start(
        "Configure a Benchmark Run",
        "Pick a task family, complexity and one or more models, then launch bench.py. "
        "This executes real model calls — energy, latency and (for online models) API cost are all real.",
    )

    task_df = discover_task_files()
    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    families = sorted(task_df["family_dir"].unique()) if not task_df.empty else []
    family_sel = c1.multiselect("Task family", families, default=families[:1], format_func=lambda f: FAMILY_LABELS.get(f, f))
    complexity_sel = c2.multiselect("Complexity", ["C1", "C2", "C3", "C4"], default=["C1"])
    mode_sel = c3.radio("Task variant", ["offline", "online"], horizontal=True)

    chosen_tasks = task_df[
        task_df["family_dir"].isin(family_sel) & task_df["complexity"].isin(complexity_sel) & (task_df["mode"] == mode_sel)
    ] if not task_df.empty else task_df
    if chosen_tasks.empty:
        empty_state("🧭", "No matching task files", "Try a different family, complexity, or task variant.")
    else:
        st.caption(f"{len(chosen_tasks)} task file(s) will be included: " + ", ".join(chosen_tasks["path"].apply(lambda p: p.name)))

    pool = load_model_pool()
    pool_by_id = {m["model_id"]: m for m in pool}
    model_ids = st.multiselect(
        "Model(s) — single or multiple",
        list(pool_by_id),
        default=list(pool_by_id)[:1],
        format_func=lambda mid: f"{mid}  ·  {pool_by_id[mid]['adapter']}",
    )
    chosen_models = [pool_by_id[mid] for mid in model_ids]

    extra_env: dict[str, str] = {}
    ollama_models = [m for m in chosen_models if m["adapter"] == "ollama"]
    if ollama_models:
        tags = ollama_installed_tags()
        if tags is None:
            st.warning("Could not reach the `ollama` CLI — is it installed and on PATH? Local models can't be checked or pulled.")
        else:
            missing = [m for m in ollama_models if m["params"].get("model_name") not in tags]
            for m in missing:
                tag = m["params"].get("model_name")
                mc1, mc2 = st.columns([4, 1])
                mc1.markdown(f"⬇️ `{tag}` is **not pulled** locally yet (for `{m['model_id']}`).")
                if mc2.button("Pull", key=f"pull_{m['model_id']}", disabled=job_active):
                    log_path = UI_RUNS_OUTPUT_DIR / "_pull_logs" / f"{slugify(tag)}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"
                    proc = launch_background(["ollama", "pull", tag], log_path, cwd=ROOT)
                    st.session_state["lab_job"] = {"kind": "pull", "label": f"ollama pull {tag}", "proc": proc, "log_path": log_path, "started": time.time()}
                    st.rerun()

    openai_models = [m for m in chosen_models if m["adapter"] == "openai"]
    for m in openai_models:
        key_env = m["params"].get("api_key_env", "OPENAI_API_KEY")
        if not os.environ.get(key_env):
            value = st.text_input(f"{key_env} for {m['model_id']} (kept in-memory for this session only, never saved to disk)", type="password", key=f"key_{m['model_id']}")
            if value:
                extra_env[key_env] = value

    with st.expander("Advanced settings"):
        ac1, ac2, ac3, ac4 = st.columns(4)
        repeats = ac1.number_input("Repeats", min_value=1, max_value=10, value=1)
        temperature = ac2.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        max_tokens = ac3.number_input("Max tokens", min_value=50, max_value=4000, value=500, step=50)
        timeout_s = ac4.number_input("Timeout (s)", min_value=10, max_value=600, value=60, step=10)

    default_run_id = f"ui_{'-'.join(family_sel) or 'run'}_{'-'.join(complexity_sel) or 'na'}_{mode_sel}_{datetime.now(UTC).strftime('%H%M%S')}"
    run_id = st.text_input("Run ID", value=slugify(default_run_id))

    config = {
        "run": {
            "run_id": run_id,
            "output_dir": f"runs/ui_runs/{run_id}",
            "mode": "batch",
            "decoding": {"temperature": temperature, "max_tokens": int(max_tokens)},
            "constraints": {"timeout_s": int(timeout_s)},
            "repeats": int(repeats),
            "random_seed": 7,
            "electricity_price_eur_per_kwh": 0.30,
        },
        "tasks": {"include": [str(p.relative_to(ROOT)) for p in chosen_tasks["path"]] if not chosen_tasks.empty else []},
        "models": [{"model_id": m["model_id"], "adapter": m["adapter"], "params": m["params"]} for m in chosen_models],
    }
    with st.expander("Preview generated config"):
        st.code(yaml.safe_dump(config, sort_keys=False), language="yaml")

    st.markdown("---")
    confirm = st.checkbox("I understand this executes real model calls (energy, latency, and — for online models — API cost) and may take a while.")
    if job_active:
        st.info("A job is already running below — wait for it to finish (or stop it) before starting another.")
    can_run = (not chosen_tasks.empty) and bool(chosen_models) and confirm and not job_active
    if st.button("🚀 Run benchmark", disabled=not can_run, type="primary"):
        for env_key, env_val in extra_env.items():
            os.environ[env_key] = env_val
        UI_RUNS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = UI_RUNS_CONFIG_DIR / f"{run_id}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        output_dir = ROOT / config["run"]["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", BENCH_MODULE, "all", "--config", str(config_path)]
        log_path = output_dir / "subprocess.log"
        proc = launch_background(cmd, log_path, cwd=ROOT, extra_env=extra_env)
        st.session_state["lab_job"] = {"kind": "benchmark", "label": f"bench.py all — {run_id}", "proc": proc, "log_path": log_path, "started": time.time()}
        st.rerun()
    panel_end()

    panel_start("Add a Model to the Registry", "Register an Ollama (offline) or OpenAI (online) model so it's selectable above. Saved to configs/model_registry.yaml.")
    with st.form("add_model_form", clear_on_submit=True):
        fc1, fc2 = st.columns(2)
        new_model_id = fc1.text_input("Model ID (unique)")
        new_adapter = fc2.selectbox("Adapter", ["ollama", "openai"])
        if new_adapter == "ollama":
            pc1, pc2 = st.columns(2)
            new_model_name = pc1.text_input("Ollama tag", placeholder="qwen2.5:3b-instruct")
            new_host = pc2.text_input("Ollama host", value="http://localhost:11434")
            new_params = {"model_name": new_model_name, "host": new_host}
        else:
            pc1, pc2 = st.columns(2)
            new_model_name = pc1.text_input("OpenAI model name", placeholder="gpt-4.1-mini")
            new_key_env = pc2.text_input("API key environment variable", value="OPENAI_API_KEY")
            new_params = {"model_name": new_model_name, "api_key_env": new_key_env}
        submitted = st.form_submit_button("Save to registry")
        if submitted:
            if not new_model_id or not new_params.get("model_name"):
                st.error("Model ID and model name are required.")
            else:
                save_model_to_registry({"model_id": new_model_id, "adapter": new_adapter, "params": new_params})
                st.success(f"Saved '{new_model_id}'. It's now selectable above.")
    panel_end()

    panel_start("Job Status", "Live output from the most recent Run Lab job (auto-refreshes every 2s).")
    _job_status_panel()
    panel_end()


if page == "Query Analysis":
    query_analysis()
elif page == "Overview":
    overview()
elif page == "Run Lab":
    run_lab()
elif page == "Efficiency Frontier":
    efficiency_frontier()
elif page == "Model Comparison":
    comparison("model_id", "Model Comparison")
elif page == "Dataset Comparison":
    comparison("family", "Dataset Comparison")
elif page == "Leaderboard":
    panel_start("Leaderboard", "Ranked by a weighted composite score. Drag a slider — the other three rebalance to keep the split at 100%.")
    weights = render_weight_sliders()
    _w_total = sum(weights.values()) or 1.0
    st.caption(
        " · ".join(f"{SCORE_WEIGHT_LABELS[k]} {weights[k] / _w_total:.0%}" for k in DEFAULT_SCORE_WEIGHTS)
        + " — badges below call out category leaders."
    )
    board = aggregate(data, "model_id")
    board["composite_score"] = composite_score(board, weights)
    board = board.sort_values("composite_score", ascending=False).reset_index(drop=True)
    board["composite_score"] = board["composite_score"].round(2)
    board.insert(0, "Rank", range(1, len(board) + 1))
    why = leaderboard_tags(board)
    display = board.copy()
    display["accuracy"] = display["accuracy"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    display["latency_s"] = display["latency_s"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    display["energy_wh"] = display["energy_wh"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    display["emissions_g"] = display["emissions_g"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    display["cost_usd"] = display["cost_usd"].map(lambda v: f"${v:.5f}" if pd.notna(v) else "—")
    display["tokens_per_kwh"] = display["tokens_per_kwh"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    display = display.rename(columns={
        "model_id": "Model", "accuracy": "Accuracy", "latency_s": "Latency (s)", "energy_wh": "Energy (Wh)",
        "emissions_g": "CO₂ (g)", "cost_usd": "Cost (USD)", "tokens_per_kwh": "Tokens/kWh",
        "composite_score": "Score", "observations": "Observations",
    })
    display.insert(len(display.columns), "Why", why.values)
    st.markdown(
        display.to_html(escape=False, index=False, classes="leaderboard-table"),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.download_button("⇩ Export Table", board.to_csv(index=False), "ecollm_leaderboard.csv")
    panel_end()
else:
    panel_start("Run Logs", "Versioned result catalog and provenance status.")
    st.dataframe(cat.drop(columns=["run_path"], errors="ignore"), width="stretch", hide_index=True)
    panel_end()
