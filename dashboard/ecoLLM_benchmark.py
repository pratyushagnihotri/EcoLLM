#!/usr/bin/env python3
# streamlit run dashboard/etaGPT_control_tower_v5.py
"""
EcoLLM Control Tower
- functionality: configs/models, run explorer, tradeoffs default, leaderboard, scaling, inspector, execute task.
- Requires: streamlit, pandas, pyyaml, plotly
"""
from __future__ import annotations

import glob
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import yaml

# -----------------------------
# Repo paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_ROOT = REPO_ROOT / "configs"
TASKS_ROOT = REPO_ROOT / "tasks"
RUNS_ROOT = REPO_ROOT / "runs"

BENCH_OFFLINE = REPO_ROOT / "bench.py"
BENCH_ONLINE = REPO_ROOT / "bench_ecologits_online_merged.py"
if not BENCH_ONLINE.exists():
    alt = REPO_ROOT / "bench_ecologits_online.py"
    if alt.exists():
        BENCH_ONLINE = alt

# -----------------------------
# Page + theme CSS (match screenshot)
# -----------------------------
st.set_page_config(page_title="etaGPT Control Tower", page_icon="⚡", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
/* -------- StockPeers-like palette -------- */
:root{
  --bg: #0E1117;
  --panel: #161B22;
  --panel2: #11161D;
  --border: #D5DAE5;
  --text: #FFFFFF;
  --text-dark: #262730;
  --muted: #A3A8B8;
  --muted2: #808495;

  --primary: #1C83E1;
  --primary-hover: #3D9DF3;
  --primary-soft: rgba(28,131,225,0.14);

  --surface-hover: #F0F2F6;
  --danger: #FF4B4B;
  --success: #21C354;

  --shadow: 0 10px 24px rgba(0,0,0,0.18);
  --radius: 16px;
  --radius2: 12px;

  --font: "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --mono: "IBM Plex Mono", "Source Code Pro", monospace;
}

html, body, [data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg, #0E1117 0%, #111827 100%);
  color: var(--text);
  font-family: var(--font);
}

[data-testid="stHeader"]{
  background: rgba(9,16,28,0.45);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}

.block-container{
  max-width: 1650px;
  padding-top: 1.0rem;
  padding-bottom: 2.2rem;
}

a{ color: var(--primary) !important; }
a:hover{ color: var(--primary-hover) !important; }

/* -------- Cards -------- */
.et-card{
  background: rgba(22,27,34,0.92);
  border: 1px solid rgba(213,218,229,0.14);
  border-radius: var(--radius);
  padding: 18px 20px;
  box-shadow: var(--shadow);
  margin-top: 24px;
  margin-bottom: 24px;
}

.et-title{
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: 0;
  color: #FFFFFF;
}

.et-sub{
  margin-top: 6px;
  font-size: 1rem;
  color: var(--muted);
}

.et-section-title{
  font-size: 1.05rem;
  font-weight: 700;
  margin: 0 0 8px 0;
  color: #FFFFFF;
}

.et-note{
  font-size: 0.92rem;
  color: var(--muted);
}

.et-kpis{ display:flex; gap:10px; flex-wrap:wrap; margin-top: 10px; }
.et-kpi{
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(9,16,28,0.38);
  border-radius: var(--radius2);
  padding: 10px 12px;
  min-width: 170px;
}
.et-kpi-label{
  font-size: .78rem;
  letter-spacing:.08em;
  text-transform:uppercase;
  color: rgba(232,238,252,0.70);
}
.et-kpi-val{ font-size: 1.25rem; font-weight: 800; margin-top: 4px; }

/* -------- Sidebar -------- */
section[data-testid="stSidebar"]>div{
  background: linear-gradient(180deg, rgba(10,18,32,0.92) 0%, rgba(10,18,32,0.92) 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
  color: var(--text);
}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p{
  color: rgba(232,238,252,0.82) !important;
}

/* -------- Inputs / buttons -------- */
.stButton>button{
  background: linear-gradient(135deg, var(--violet), var(--violet2));
  color: white;
  border: 0;
  border-radius: 999px;
  padding: 0.48rem 1.2rem;
  font-weight: 700;
  box-shadow: 0 10px 26px rgba(109,94,252,0.24);
}
.stButton>button:hover{ filter: brightness(1.05); }

/* Tabs/radio visuals */
div[role="radiogroup"]{
  display: flex !important;
  gap: 10px;
  flex-wrap: wrap;
}

div[role="radiogroup"] > label{
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(213,218,229,0.22) !important;
  border-radius: 10px !important;
  padding: 8px 14px !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  transition: all 0.15s ease;
}

div[role="radiogroup"] > label:hover{
  background: rgba(240,242,246,0.10) !important;
  border-color: #615FFF !important;
}

div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked){
  background: #615FFF !important;
  border-color: #615FFF !important;
  color: #FFFFFF !important;
  box-shadow: 0 0 0 1px rgba(28,131,225,0.15);
}

div[role="radiogroup"] p,
div[role="radiogroup"] span{
  color: #FFFFFF !important;
}

/* Dataframe */
[data-testid="stDataFrame"]{
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}

/* -------- Inputs / buttons (match screenshot: subtle dark pills) -------- */
.stButton>button{
  background: #615FFF !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.5rem 1rem !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}

.stButton>button:hover{
  background: #3D9DF3 !important;
}

/* Make widget labels white like screenshot */
label, p, span, .stMarkdown, .stText, .stCaption, .stMetric,
.stSelectbox, .stMultiSelect, .stRadio, .stCheckbox, .stSlider{
  color: #FFFFFF !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div{
  color: #FFFFFF !important;
}

/* Selectbox/multiselect input areas */
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(213,218,229,0.24) !important;
  color: #FFFFFF !important;
  border-radius: 10px !important;
  min-height: 42px !important;
}

div[data-baseweb="select"] > div:hover{
  border-color: #615FFF !important;
}

div[data-baseweb="select"] *{
  color: #FFFFFF !important;
}

/* Selected items inside multiselect / select chips */
[data-baseweb="tag"] {
  background-color: #615FFF !important;
  border-radius: 8px !important;
  border: none !important;
}

[data-baseweb="tag"] span {
  color: #FFFFFF !important;
}

[data-baseweb="tag"] svg {
  fill: #FFFFFF !important;
}

/* Sometimes Streamlit uses this inner container for selected pills */
div[data-baseweb="select"] span[data-baseweb="tag"] {
  background-color: #615FFF !important;
  color: #FFFFFF !important;
  border: none !important;
}

/* Remove red close icon styling if it appears */
div[data-baseweb="select"] [data-baseweb="tag"] * {
  color: #FFFFFF !important;
  fill: #FFFFFF !important;
}

/* Radio pills look */
div[role="radiogroup"] label{
  color: var(--text) !important;
}

/* Slider label/value text */
.stSlider label,
.stSlider div,
.stSlider span,
.stSlider p {
  color: #FFFFFF !important;
}

/* Filled part of slider track */
.stSlider [data-baseweb="slider"] > div > div:first-child {
  background: #615FFF !important;
}

/* Unfilled track */
.stSlider [data-baseweb="slider"] > div > div:nth-child(2) {
  background: rgba(213,218,229,0.28) !important;
}

/* Slider thumb / knob */
.stSlider [role="slider"] {
  background: #615FFF !important;
  border: 2px solid #615FFF !important;
  box-shadow: 0 0 0 2px rgba(28,131,225,0.18) !important;
}

/* Numeric value above the slider */
.stSlider [data-testid="stThumbValue"] {
  color: #FFFFFF !important;
}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="et-card">
  <div class="et-title">⚡ EcoLLM — Energy-aware LLM Bench for Sustainable AI Data Systems</div>
  <div class="et-sub">
    Offline vs Online LLM benchmarking. Default view: <b>Tradeoffs</b> (quality vs latency/energy/cost).
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Plotly template to match theme
# -----------------------------
THEME = "plotly_dark"
pio.templates.default = THEME

def apply_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111827",
        font=dict(family="Source Sans 3, sans-serif", size=15, color="#FFFFFF"),
        title=dict(font=dict(size=24, color="#FFFFFF")),
        xaxis=dict(
            title_font=dict(color="#E5E7EB"),
            tickfont=dict(color="#D1D5DB"),
            gridcolor="rgba(255,255,255,0.10)"
        ),
        yaxis=dict(
            title_font=dict(color="#E5E7EB"),
            tickfont=dict(color="#D1D5DB"),
            gridcolor="rgba(255,255,255,0.10)"
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12, color="#FFFFFF"),
            bgcolor="rgba(17,24,39,0.88)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(r=240)
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1.5, color="rgba(255,255,255,0.65)")),
        opacity=0.95,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig

# -----------------------------
# Helpers
# -----------------------------
def card(border: bool = True):
    try:
        return st.container(border=border)  # streamlit>=1.31-ish
    except TypeError:
        return st.container()

def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def discover_configs() -> Tuple[List[Path], List[Path]]:
    offline = sorted([Path(p) for p in glob.glob(str(CONFIGS_ROOT / "run_seds_*.yaml")) if "_online" not in p])
    online = sorted([Path(p) for p in glob.glob(str(CONFIGS_ROOT / "run_seds_*_online.yaml"))])
    return offline, online

def task_files_from_config(cfg: Dict[str, Any]) -> List[Path]:
    inc = ((cfg.get("tasks") or {}).get("include") or [])
    out: List[Path] = []
    for p in inc:
        pp = (REPO_ROOT / p).resolve()
        if pp.exists():
            out.append(pp)
    return out

def tasks_from_yaml(task_file: Path) -> List[Dict[str, Any]]:
    doc = read_yaml(task_file)
    return doc.get("tasks", [])

def normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    if "meta.table_rows" in df.columns and "table_rows" not in df.columns:
        df["table_rows"] = pd.to_numeric(df["meta.table_rows"], errors="coerce")
    if "meta.complexity" in df.columns and "complexity" not in df.columns:
        df["complexity"] = df["meta.complexity"].astype(str)
    if "meta.dataset" in df.columns and "dataset" not in df.columns:
        df["dataset"] = df["meta.dataset"].astype(str)
    for c in ["family", "model_id", "task_id", "complexity", "dataset", "run_name", "run_kind"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "table_rows" in df.columns:
        df["table_rows"] = pd.to_numeric(df["table_rows"], errors="coerce")
    return df

def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out

def read_csv_or_parquet(run_dir: Path, base: str) -> Optional[pd.DataFrame]:
    csv_path = run_dir / "csv" / f"{base}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    pq_path = run_dir / f"{base}.parquet"
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    return None

def infer_run_kind(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "online" in name:
        return "online"
    try:
        lb = read_csv_or_parquet(run_dir, "leaderboard_by_family")
        if lb is not None and "model_id" in lb.columns:
            mids = lb["model_id"].astype(str).str.lower()
            if mids.str.startswith("openai-").any():
                return "online"
    except Exception:
        pass
    return "offline"

def discover_run_dirs_with_csv() -> List[Path]:
    out: List[Path] = []
    if not RUNS_ROOT.exists():
        return []
    for p in RUNS_ROOT.rglob("csv"):
        if (p / "leaderboard_by_family.csv").exists() or (p / "leaderboard_by_family_rows.csv").exists() or (p / "results.csv").exists():
            out.append(p.parent)
    return sorted(set(out), key=lambda d: d.stat().st_mtime, reverse=True)

def load_run_artifacts(run_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    lb = read_csv_or_parquet(run_dir, "leaderboard_by_family")
    lbr = read_csv_or_parquet(run_dir, "leaderboard_by_family_rows")
    res = read_csv_or_parquet(run_dir, "results")
    if lb is not None: lb = normalize_meta(lb)
    if lbr is not None: lbr = normalize_meta(lbr)
    if res is not None: res = normalize_meta(res)
    return lb, lbr, res

def pretty_run_name(run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(REPO_ROOT))
    except Exception:
        return str(run_dir)

def pick_quality_metric(df: pd.DataFrame) -> str:
    for c in ["pass_rate", "anomaly_f1_macro", "numeric_rmse"]:
        if c in df.columns:
            return c
    return "pass_rate"

def build_mini_run_config(
    out_dir: Path,
    run_id: str,
    tasks: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    decoding: Dict[str, Any],
    timeout_s: int,
    repeats: int,
    electricity_price: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_tasks = out_dir / "tmp_selected_tasks.yaml"
    with open(tmp_tasks, "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": tasks}, f, sort_keys=False, allow_unicode=True)
    cfg = {
        "run": {
            "run_id": run_id,
            "output_dir": str(out_dir.relative_to(REPO_ROOT)),
            "mode": "batch",
            "decoding": decoding,
            "constraints": {"timeout_s": int(timeout_s)},
            "repeats": int(repeats),
            "random_seed": 7,
            "electricity_price_eur_per_kwh": float(electricity_price),
            "progress_every": 1,
            "log_level": "INFO",
        },
        "tasks": {"include": [str(tmp_tasks.relative_to(REPO_ROOT))]},
        "models": models,
    }
    cfg_path = out_dir / "tmp_run.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg_path

def load_selected_runs(run_dirs: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lbs, lbrs, ress = [], [], []
    for rd in run_dirs:
        lb, lbr, res = load_run_artifacts(rd)
        rn = pretty_run_name(rd)
        rk = infer_run_kind(rd)
        if lb is not None:
            lb = lb.copy()
            lb["run_name"] = rn
            lb["run_kind"] = rk
            lbs.append(lb)
        if lbr is not None:
            lbr = lbr.copy()
            lbr["run_name"] = rn
            lbr["run_kind"] = rk
            lbrs.append(lbr)
        if res is not None:
            res = res.copy()
            res["run_name"] = rn
            res["run_kind"] = rk
            ress.append(res)
    return (
        pd.concat(lbs, ignore_index=True) if lbs else pd.DataFrame(),
        pd.concat(lbrs, ignore_index=True) if lbrs else pd.DataFrame(),
        pd.concat(ress, ignore_index=True) if ress else pd.DataFrame(),
    )

# -----------------------------
# Layout: left controls, right canvas
# -----------------------------
left, right = st.columns([1.08, 2.92], gap="large")

with left:
    with card(True):
        st.markdown('<div class="et-section-title">Controls</div>', unsafe_allow_html=True)
        st.caption("Configs → model lists. Runs → csv artifacts.")

        offline_cfgs, online_cfgs = discover_configs()
        #model_mode = st.radio("Model source", ["Offline", "Online", "Both"], index=0, key="ctl_mode_v5")
        model_mode = st.radio(
                        "Model source",
                        ["Offline", "Online", "Both"],
                        index=0,
                        key="ctl_mode_v5",
                        horizontal=True,
                    )

        cfg_off = cfg_on = None
        if model_mode == "Offline":
            cfg_off_path = st.selectbox("Offline config", options=offline_cfgs, format_func=lambda p: str(p.relative_to(REPO_ROOT)), key="cfg_off_v5")
            cfg_off = read_yaml(cfg_off_path)
        elif model_mode == "Online":
            cfg_on_path = st.selectbox("Online config", options=online_cfgs, format_func=lambda p: str(p.relative_to(REPO_ROOT)), key="cfg_on_v5")
            cfg_on = read_yaml(cfg_on_path)
        else:
            cfg_off_path = st.selectbox("Offline config", options=offline_cfgs, format_func=lambda p: str(p.relative_to(REPO_ROOT)), key="cfg_off_v5_b")
            cfg_on_path = st.selectbox("Online config", options=online_cfgs, format_func=lambda p: str(p.relative_to(REPO_ROOT)), key="cfg_on_v5_b")
            cfg_off = read_yaml(cfg_off_path)
            cfg_on = read_yaml(cfg_on_path)

        models_off = (cfg_off.get("models") or []) if cfg_off else []
        models_on = (cfg_on.get("models") or []) if cfg_on else []

        st.divider()
        st.markdown('<div class="et-section-title">Models</div>', unsafe_allow_html=True)

        sel_models_off: List[Dict[str, Any]] = []
        sel_models_on: List[Dict[str, Any]] = []

        if model_mode == "Offline":
            ids = [m.get("model_id","unknown") for m in models_off]
            sel = st.multiselect("Select models", options=ids, default=ids[:1] if ids else [], key="models_off_v5")
            sel_models_off = [m for m in models_off if m.get("model_id") in set(sel)]
        elif model_mode == "Online":
            ids = [m.get("model_id","unknown") for m in models_on]
            sel = st.multiselect("Select models", options=ids, default=ids[:1] if ids else [], key="models_on_v5")
            sel_models_on = [m for m in models_on if m.get("model_id") in set(sel)]
        else:
            ids_off = [m.get("model_id","unknown") for m in models_off]
            ids_on = [m.get("model_id","unknown") for m in models_on]
            sel_off = st.multiselect("Offline models", options=ids_off, default=ids_off[:1] if ids_off else [], key="models_off_v5_2")
            sel_on = st.multiselect("Online models", options=ids_on, default=ids_on[:1] if ids_on else [], key="models_on_v5_2")
            sel_models_off = [m for m in models_off if m.get("model_id") in set(sel_off)]
            sel_models_on = [m for m in models_on if m.get("model_id") in set(sel_on)]

        st.divider()
        st.markdown('<div class="et-section-title">Decode / constraints</div>', unsafe_allow_html=True)
        temperature = st.slider("temperature", 0.0, 1.0, 0.0, 0.05, key="temp_v5")
        max_tokens = st.slider("max_tokens", 64, 2000, 500, 50, key="max_tokens_v5")
        timeout_s = st.slider("timeout_s", 10, 300, 120, 10, key="timeout_v5")
        repeats = st.slider("repeats", 1, 5, 1, 1, key="repeats_v5")
        electricity_price = st.number_input("electricity_price_eur_per_kwh", value=0.30, step=0.01, key="eprice_v5")

    with card(True):
        st.markdown('<div class="et-section-title">Run explorer</div>', unsafe_allow_html=True)
        run_dirs = discover_run_dirs_with_csv()
        if not run_dirs:
            st.info("No runs found under runs/**/csv.")
            selected_run_dirs: List[Path] = []
        else:
            run_kinds = {rd: infer_run_kind(rd) for rd in run_dirs}
            kind_filter = st.multiselect("Kinds", ["offline","online"], default=["offline","online"], key="kind_filter_v5")
            filtered = [rd for rd in run_dirs if run_kinds.get(rd) in set(kind_filter)]

            newest_offline = next((rd for rd in filtered if run_kinds.get(rd) == "offline"), None)
            newest_online = next((rd for rd in filtered if run_kinds.get(rd) == "online"), None)
            default_runs = [r for r in [newest_offline, newest_online] if r is not None]
            if not default_runs and filtered:
                default_runs = [filtered[0]]

            selected_run_dirs = st.multiselect(
                "Runs",
                options=filtered,
                default=default_runs,
                format_func=lambda p: f"{pretty_run_name(p)}  ({run_kinds.get(p,'?')})",
                key="runs_v5",
            )

lb_all, lbr_all, res_all = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
if 'selected_run_dirs' in locals() and selected_run_dirs:
    lb_all, lbr_all, res_all = load_selected_runs(selected_run_dirs)

with right:
    # View switch
    view = st.radio(
        "View",
        ["Tradeoffs","Leaderboard","Table-size impact","Task inspector","Execute task"],
        horizontal=True,
        index=0,
        key="view_v5",
        label_visibility="collapsed",
    )

    # KPI strip
    with card(True):
        n_runs = len(selected_run_dirs) if 'selected_run_dirs' in locals() else 0
        n_models = int(lb_all["model_id"].nunique()) if not lb_all.empty and "model_id" in lb_all.columns else 0
        n_fams = int(lb_all["family"].nunique()) if not lb_all.empty and "family" in lb_all.columns else 0
        n_rows = int(res_all.shape[0]) if not res_all.empty else 0
        st.markdown(
            f"""
<div class="et-kpis">
  <div class="et-kpi"><div class="et-kpi-label">Selected runs</div><div class="et-kpi-val">{n_runs}</div></div>
  <div class="et-kpi"><div class="et-kpi-label">Models</div><div class="et-kpi-val">{n_models}</div></div>
  <div class="et-kpi"><div class="et-kpi-label">Families</div><div class="et-kpi-val">{n_fams}</div></div>
  <div class="et-kpi"><div class="et-kpi-label">Result rows</div><div class="et-kpi-val">{n_rows}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

    if view == "Tradeoffs":
        with card(True):
            st.markdown('<div class="et-section-title">Tradeoffs</div>', unsafe_allow_html=True)
            st.markdown('<div class="et-note">Quality vs latency/energy/cost. Color = model, symbol = offline/online.</div>', unsafe_allow_html=True)

            if lb_all.empty:
                st.info("Select runs in the left panel.")
            else:
                df = lb_all.copy()
                fams = sorted(df["family"].dropna().unique().tolist()) if "family" in df.columns else []
                kinds = sorted(df["run_kind"].dropna().unique().tolist()) if "run_kind" in df.columns else []
                c1, c2, c3, c4 = st.columns([1,1,1,1])
                with c1:
                    fam_pick = st.multiselect("Family", fams, default=fams, key="trade_fam_v5")
                with c2:
                    kind_pick = st.multiselect("Kind", kinds, default=kinds, key="trade_kind_v5")
                with c3:
                    x_opts = [c for c in ["p95_latency_ms","avg_latency_ms","avg_energy_kwh","avg_api_cost_usd"] if c in df.columns]
                    x = st.selectbox("X", x_opts, index=0, key="trade_x_v5") if x_opts else None
                with c4:
                    quality = pick_quality_metric(df)
                    y_opts = [c for c in [quality,"pass_rate","anomaly_f1_macro","numeric_rmse"] if c in df.columns]
                    y = st.selectbox("Y", y_opts, index=0, key="trade_y_v5") if y_opts else None

                if fam_pick and "family" in df.columns:
                    df = df[df["family"].isin(fam_pick)]
                if kind_pick and "run_kind" in df.columns:
                    df = df[df["run_kind"].isin(kind_pick)]

                if df.empty or x is None or y is None:
                    st.warning("Not enough data/columns for tradeoff plot.")
                else:
                    df = df.copy()

                    def short_model_label(model_id):
                        model_id = model_id.replace("ollama-", "").replace("openai-", "")
                        model_id = model_id.replace("-instruct", "")
                        model_id = model_id.replace("-latest", "")
                        return model_id

                    df["legend_label"] = [short_model_label(m) for m in df["model_id"]]

                    fig = px.scatter(
                        df,
                        x=x,
                        y=y,
                        color="legend_label" if "legend_label" in df.columns else None,
                        symbol="run_kind" if "run_kind" in df.columns else None,
                        size="avg_energy_kwh" if "avg_energy_kwh" in df.columns else None,
                        hover_data=[c for c in ["model_id","family","run_name","run_kind","n"] if c in df.columns],
                        title=f"{y} vs {x}",
                    )
                    fig = apply_plotly_theme(fig)
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

                    show_cols = [c for c in ["model_id","family","run_kind","run_name","n",y,x,"avg_energy_kwh","avg_api_cost_usd"] if c in df.columns]
                    st.dataframe(df.sort_values(y, ascending=False).head(12)[show_cols], use_container_width=True)

    elif view == "Leaderboard":
        with card(True):
            st.markdown('<div class="et-section-title">Leaderboard</div>', unsafe_allow_html=True)
            if lb_all.empty:
                st.info("Select runs in the left panel.")
            else:
                df = lb_all.copy()
                fams = sorted(df["family"].dropna().unique().tolist()) if "family" in df.columns else []
                kinds = sorted(df["run_kind"].dropna().unique().tolist()) if "run_kind" in df.columns else []
                c1, c2, c3 = st.columns([1,1,2])
                with c1:
                    fam_pick = st.multiselect("Family", fams, default=fams, key="lb_fam_v5")
                with c2:
                    kind_pick = st.multiselect("Kind", kinds, default=kinds, key="lb_kind_v5")
                with c3:
                    mids = sorted(df["model_id"].dropna().unique().tolist()) if "model_id" in df.columns else []
                    model_pick = st.multiselect("Models", mids, default=mids, key="lb_models_v5")

                if fam_pick and "family" in df.columns:
                    df = df[df["family"].isin(fam_pick)]
                if kind_pick and "run_kind" in df.columns:
                    df = df[df["run_kind"].isin(kind_pick)]
                if model_pick and "model_id" in df.columns:
                    df = df[df["model_id"].isin(model_pick)]

                st.dataframe(df, use_container_width=True, height=620)
                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "leaderboard_filtered.csv")

    elif view == "Table-size impact":
        with card(True):
            st.markdown('<div class="et-section-title">Table-size impact</div>', unsafe_allow_html=True)
            st.markdown('<div class="et-note">Requires leaderboard_by_family_rows.* in the selected runs.</div>', unsafe_allow_html=True)

            if lbr_all.empty:
                st.info("No leaderboard_by_family_rows found for selected runs.")
            else:
                df = lbr_all.copy()
                metric_opts = [c for c in ["pass_rate","p95_latency_ms","avg_latency_ms","avg_energy_kwh","avg_api_cost_usd","numeric_rmse","anomaly_f1_macro"] if c in df.columns]
                fams = sorted(df["family"].dropna().unique().tolist()) if "family" in df.columns else []
                kinds = sorted(df["run_kind"].dropna().unique().tolist()) if "run_kind" in df.columns else []
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    metric = st.selectbox("Metric", metric_opts, key="ts_metric_v5")
                with c2:
                    fam_pick = st.multiselect("Family", fams, default=fams, key="ts_fam_v5")
                with c3:
                    kind_pick = st.multiselect("Kind", kinds, default=kinds, key="ts_kind_v5")

                if fam_pick and "family" in df.columns:
                    df = df[df["family"].isin(fam_pick)]
                if kind_pick and "run_kind" in df.columns:
                    df = df[df["run_kind"].isin(kind_pick)]
                df = df.dropna(subset=["table_rows"])

                if df.empty:
                    st.info("No rows after filtering.")
                else:
                    fig = px.line(
                        df.sort_values("table_rows"),
                        x="table_rows",
                        y=metric,
                        color="model_id" if "model_id" in df.columns else None,
                        line_dash="run_kind" if "run_kind" in df.columns else None,
                        markers=True,
                        title=f"{metric} vs table_rows",
                    )
                    fig = apply_plotly_theme(fig)
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)

    elif view == "Task inspector":
        with card(True):
            st.markdown('<div class="et-section-title">Task inspector</div>', unsafe_allow_html=True)
            if res_all.empty:
                st.info("No results found for selected runs.")
            else:
                df = res_all.copy()
                fams = sorted(df["family"].dropna().unique().tolist()) if "family" in df.columns else []
                kinds = sorted(df["run_kind"].dropna().unique().tolist()) if "run_kind" in df.columns else []
                c1, c2, c3 = st.columns([1,1,2])
                with c1:
                    fam_pick = st.multiselect("Family", fams, default=fams, key="ins_fam_v5")
                with c2:
                    kind_pick = st.multiselect("Kind", kinds, default=kinds, key="ins_kind_v5")
                with c3:
                    mids = sorted(df["model_id"].dropna().unique().tolist()) if "model_id" in df.columns else []
                    model_pick = st.multiselect("Models", mids, default=mids, key="ins_models_v5")

                if fam_pick and "family" in df.columns:
                    df = df[df["family"].isin(fam_pick)]
                if kind_pick and "run_kind" in df.columns:
                    df = df[df["run_kind"].isin(kind_pick)]
                if model_pick and "model_id" in df.columns:
                    df = df[df["model_id"].isin(model_pick)]

                if df.empty:
                    st.info("No rows after filtering.")
                else:
                    tids = df["task_id"].astype(str).unique().tolist()
                    tid = st.selectbox("task_id", tids, key="ins_tid_v5")
                    sub = df[df["task_id"].astype(str) == str(tid)].copy()
                    if "metric.task_pass" in sub.columns and "latency_ms_total" in sub.columns:
                        sub = sub.sort_values(["metric.task_pass","latency_ms_total"], ascending=[False, True], na_position="last")
                    cols = [c for c in ["run_name","run_kind","model_id","metric.task_pass","metric.json_valid","metric.fail_reason","latency_ms_total","energy_kwh","api_cost_usd"] if c in sub.columns]
                    st.dataframe(sub[cols], use_container_width=True)

                    for _, r in sub.iterrows():
                        mid = r.get("model_id","model")
                        passed = r.get("metric.task_pass", np.nan)
                        header = f"{mid} — {'✅ PASS' if passed == 1 else '❌ FAIL'}"
                        with st.expander(header, expanded=False):
                            st.markdown("**Prompt**")
                            st.code(str(r.get("prompt",""))[:12000])
                            st.markdown("**Output**")
                            st.code(str(r.get("output_text",""))[:12000])
                            if "metric.fail_reason" in r:
                                st.markdown("**Fail reason**")
                                st.write(r.get("metric.fail_reason"))

    else:  # Execute task
        with card(True):
            st.markdown('<div class="et-section-title">Execute a task</div>', unsafe_allow_html=True)
            st.markdown('<div class="et-note">Runs one selected task and saves to <code>runs/ui_*</code>.</div>', unsafe_allow_html=True)

            # Task YAML list: from selected configs if available, else manual list
            task_files: List[Path] = []
            if 'cfg_off' in locals() and cfg_off:
                task_files += task_files_from_config(cfg_off)
            if 'cfg_on' in locals() and cfg_on:
                task_files += task_files_from_config(cfg_on)
            task_files = sorted(set(task_files))

            use_any = st.checkbox("Pick task YAML manually (tasks/**)", value=False, key="ex_any_v5")
            if use_any:
                all_yamls = sorted([Path(p) for p in glob.glob(str(TASKS_ROOT / "**" / "*.yaml"), recursive=True)])
                task_yaml = st.selectbox("Task YAML", all_yamls, format_func=lambda p: str(p.relative_to(REPO_ROOT)), key="ex_yaml_any_v5")
            else:
                if not task_files:
                    st.warning("No task YAMLs referenced by the selected config(s). Enable manual selection.")
                    st.stop()
                task_yaml = st.selectbox("Task YAML", task_files, format_func=lambda p: str(p.relative_to(REPO_ROOT)), key="ex_yaml_cfg_v5")

            tasks = tasks_from_yaml(task_yaml)
            if not tasks:
                st.error("No tasks found in selected YAML.")
                st.stop()

            fams = sorted({str(t.get("family","unknown")) for t in tasks})
            fam = st.selectbox("Family", fams, key="ex_fam_v5")
            tids = [str(t.get("task_id")) for t in tasks if str(t.get("family","unknown")) == fam]
            tid = st.selectbox("task_id", tids, key="ex_tid_v5")
            task_obj = next(t for t in tasks if str(t.get("task_id")) == tid)

            st.markdown("**Task**")
            st.code(task_obj.get("input",""), language="text")

            decoding = {"temperature": float(temperature), "max_tokens": int(max_tokens)}

            colA, colB = st.columns([1,2])
            run_single = colA.button("▶️ Run", use_container_width=True, key="ex_run_v5")
            run_multi = colB.button("🚀 Run across selected models", use_container_width=True, key="ex_run_multi_v5")

            if run_single or run_multi:
                off_list = sel_models_off if run_multi else (sel_models_off[:1] if sel_models_off else [])
                on_list = sel_models_on if run_multi else (sel_models_on[:1] if sel_models_on else [])

                def show_run_results(out_dir: Path) -> None:
                    res = read_csv_or_parquet(out_dir, "results")
                    if res is None:
                        st.warning("No results.csv/parquet found.")
                        return
                    res = normalize_meta(res)
                    cols = [c for c in ["model_id","metric.task_pass","metric.json_valid","metric.fail_reason","latency_ms_total","energy_kwh","api_cost_usd","tokens_out"] if c in res.columns]
                    st.dataframe(res[cols].sort_values(["metric.task_pass","latency_ms_total"], ascending=[False, True], na_position="last"),
                                 use_container_width=True)
                    for _, r in res.iterrows():
                        with st.expander(r.get("model_id","model"), expanded=False):
                            st.code(str(r.get("output_text",""))[:12000])

                if model_mode in ("Offline","Both") and off_list:
                    run_name = f"ui_offline_{Path(task_yaml).stem}_{tid}_{int(time.time())}"
                    out_dir = RUNS_ROOT / run_name
                    cfg_path = build_mini_run_config(out_dir, run_name, [task_obj], off_list, decoding, int(timeout_s), int(repeats), float(electricity_price))
                    with st.spinner("Running offline…"):
                        rc, out = run_cmd(["python", str(BENCH_OFFLINE), "all", "--config", str(cfg_path)])
                    if rc != 0:
                        st.error("Offline run failed.")
                        st.code(out[:12000])
                    else:
                        st.success(f"Offline run complete: {pretty_run_name(out_dir)}")
                        show_run_results(out_dir)

                if model_mode in ("Online","Both") and on_list:
                    run_name = f"ui_online_{Path(task_yaml).stem}_{tid}_{int(time.time())}"
                    out_dir = RUNS_ROOT / run_name
                    cfg_path = build_mini_run_config(out_dir, run_name, [task_obj], on_list, decoding, int(timeout_s), int(repeats), float(electricity_price))
                    with st.spinner("Running online…"):
                        rc, out = run_cmd(["python", str(BENCH_ONLINE), "all", "--config", str(cfg_path)])
                    if rc != 0:
                        st.error("Online run failed.")
                        st.code(out[:12000])
                    else:
                        st.success(f"Online run complete: {pretty_run_name(out_dir)}")
                        show_run_results(out_dir)

st.caption("v5: Matches StockPeers-style colors (navy + violet). If you want exact font (e.g., Inter), we can add it via CSS @import.")
