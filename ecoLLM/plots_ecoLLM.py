#!/usr/bin/env python3
from __future__ import annotations

"""
plots_ecoLLM.py

Adds support for:
- offline-only runs
- online-only runs (folder name contains "_online")
- combined (offline + online)
- all (generate 3 outputs)

Outputs:
  <out-root>/offline/...
  <out-root>/online/...
  <out-root>/combined/...

Usage:
  python scripts/plots_ecoLLM.py --runs-root runs --out-root plots_insight --mode all
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_PRETTY = {
    "pass_rate": "Pass rate",
    "json_valid_rate": "JSON-valid rate",
    "avg_latency_ms": "Avg latency (ms)",
    "p95_latency_ms": "P95 latency (ms)",
    "error_rate": "Error rate",
    "avg_tokens_out": "Avg tokens out",
    "avg_energy_kwh": "Avg energy (kWh)",
    "avg_energy_cost_eur": "Avg energy cost (€)",
    "avg_api_cost_usd": "Avg API cost (USD)",
    "avg_mem_rss_delta_mb": "Avg RSS Δ (MB)",
    "avg_cpu_user_s_delta": "Avg CPU user Δ (s)",
    "avg_cpu_system_s_delta": "Avg CPU sys Δ (s)",
    "numeric_mae": "Numeric MAE",
    "numeric_rmse": "Numeric RMSE",
    "anomaly_f1_macro": "Anomaly F1 (macro)",
    "avg_co2_kg": "Avg CO2 (kg)",
}

MODEL_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#003f5c", "#7a5195", "#ef5675", "#ffa600", "#2f4b7c",
    "#00a6a6", "#6a994e", "#f4a261", "#264653", "#e76f51",
]
HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "**", "--", "||"]
FAMILY_ORDER = {"family1_qa": 1, "family2_anomaly": 2, "family3": 3, "family4_codegen": 4, "family4": 4}
COMPLEXITY_ORDER = {"C1": 1, "C2": 2, "C3": 3, "C4": 4}

FIG_WIDTH_SCALE = 0.5
FIG_HEIGHT_SCALE = 0.5


def scaled_figsize(width: float, height: float) -> tuple[float, float]:
    return (max(1.0, width * FIG_WIDTH_SCALE), max(1.0, height * FIG_HEIGHT_SCALE))

# Recommended model order for plotting
MODEL_ORDER = [
    # Tier A: General-purpose instruct
    "ollama-qwen2.5-3b-instruct",
    "ollama-qwen2.5-7b-instruct",
    "ollama-mistral-7b-instruct",

    # Tier B: Strong general baselines
    "ollama-llama3-latest",
    "ollama-mistral-latest",
    "ollama-gemma-latest",

    # Tier C: Code specialists
    "ollama-deepseek-coder-latest",
    "ollama-codellama-latest",

    # Optional legacy baselines
    "ollama-llama2-latest",
    "ollama-qwen-latest",

    # Online baselines
    "openai-gpt-4.1-mini",
    "openai-o3-mini",
    "openai-gpt-4.1",
]

MODEL_SHORT_NAMES = {
    "ollama-qwen2.5-3b-instruct": "Qwen2.5-3B",
    "ollama-qwen2.5-7b-instruct": "Qwen2.5-7B",
    "ollama-mistral-7b-instruct": "Mistral-7B",
    "ollama-llama3-latest": "LLaMA3",
    "ollama-mistral-latest": "Mistral",
    "ollama-gemma-latest": "Gemma",
    "ollama-deepseek-coder-latest": "DeepSeek-C",
    "ollama-codellama-latest": "CodeLLaMA",
    "ollama-llama2-latest": "LLaMA2",
    "ollama-qwen-latest": "Qwen",
    "openai-gpt-4.1-mini": "GPT-4.1-mini",
    "openai-o3-mini": "o3-mini",
    "openai-gpt-4o-mini": "GPT-4o-mini",
    "openai-gpt-4.1": "GPT-4.1",
}


def short_model_name(name: str) -> str:
    return MODEL_SHORT_NAMES.get(str(name), str(name))


def short_model_names(names: List[str]) -> List[str]:
    return [short_model_name(n) for n in names]


MODEL_ORDER_INDEX = {m: i for i, m in enumerate(MODEL_ORDER)}

def sort_models(vals: List[str]) -> List[str]:
    vals = [v for v in vals if isinstance(v, str) and v not in ("", "nan", "unknown")]
    return sorted(vals, key=lambda x: (MODEL_ORDER_INDEX.get(str(x), 10000), str(x)))



def nice(metric: str) -> str:
    return METRIC_PRETTY.get(metric, metric)


def apply_style(font: int, title_font: int, tick_font: int, legend_font: int) -> None:
    plt.rcParams.update({
        "text.usetex": False,
        "font.size": font,
        "axes.titlesize": title_font,
        "axes.labelsize": font,
        "xtick.labelsize": tick_font,
        "ytick.labelsize": tick_font,
        "legend.fontsize": legend_font,
        "figure.titlesize": title_font + 1,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.25,
    })
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
        plt.rcParams["text.usetex"] = False
    except Exception:
        pass


def save_fig(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def color(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]


def hatch(i: int) -> str:
    return HATCHES[i % len(HATCHES)]




def format_bar_value(v: float) -> str:
    if pd.isna(v):
        return ""
    av = abs(float(v))
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 100:
        return f"{v:.1f}"
    if av >= 10:
        return f"{v:.2f}"
    if av >= 1:
        return f"{v:.3f}"
    if av >= 0.01:
        return f"{v:.4f}"
    return f"{v:.2e}"


def add_value_labels(ax: plt.Axes, bars, values: List[float], fontsize: int = 7, use_log: bool = False) -> None:
    clean_vals = [float(v) for v in values if not pd.isna(v) and np.isfinite(v)]
    if not clean_vals:
        return

    ymin, ymax = ax.get_ylim()

    if use_log and ax.get_yscale() == "log":
        pos_vals = [float(v) for v in clean_vals if float(v) > 0]
        if not pos_vals:
            return
        for rect, v in zip(bars, values):
            if pd.isna(v) or not np.isfinite(v):
                continue
            y = float(rect.get_height())
            if y <= 0:
                continue
            x = rect.get_x() + rect.get_width() / 2.0
            y_text = y * 1.08
            if np.isfinite(ymax) and y_text >= ymax:
                y_text = y * 1.03
            ax.text(x, y_text, format_bar_value(v), ha="center", va="bottom", fontsize=fontsize, rotation=90, clip_on=False)
        return

    yrange = ymax - ymin
    if not np.isfinite(yrange) or np.isclose(yrange, 0):
        yrange = max(abs(max(clean_vals)), 1.0)
    pad = 0.01 * yrange

    for rect, v in zip(bars, values):
        if pd.isna(v) or not np.isfinite(v):
            continue
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_height()
        va = "bottom" if y >= 0 else "top"
        y_text = y + pad if y >= 0 else y - pad
        ax.text(x, y_text, format_bar_value(v), ha="center", va=va, fontsize=fontsize, rotation=90, clip_on=False)



def positive_finite_values(values: List[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv) and fv > 0:
            out.append(fv)
    return out


def can_use_log_scale(values: List[float]) -> bool:
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not np.isfinite(fv) or fv <= 0:
            return False
        cleaned.append(fv)
    return len(cleaned) > 0


def maybe_set_log_y(ax: plt.Axes, values: List[float], use_log: bool) -> bool:
    if not use_log:
        return False
    if not can_use_log_scale(values):
        return False
    ax.set_yscale("log")
    return True


def accuracy_metric_candidates(df: pd.DataFrame) -> List[str]:
    candidates = ["anomaly_f1_macro", "pass_rate", "json_valid_rate"]
    return [c for c in candidates if c in df.columns]


def resolve_accuracy_metric(df: pd.DataFrame, requested: str = "auto") -> Optional[str]:
    if requested != "auto":
        return requested if requested in df.columns else None
    candidates = accuracy_metric_candidates(df)
    return candidates[0] if candidates else None


def metric_direction(metric: str) -> str:
    """Return optimization direction for a metric: 'max' or 'min'."""
    if metric in {"anomaly_f1_macro", "pass_rate", "json_valid_rate"}:
        return "max"
    return "min"


def tradeoff_pairs_for_subset(sub: pd.DataFrame, requested_accuracy: str = "auto") -> List[tuple[str, str]]:
    """
    Resolve trade-off pairs per subset.

    Includes:
    1. resource/cost vs accuracy pairs (subset-specific accuracy metric)
    2. requested pairwise cost-latency/resource trade-offs:
       - Energy vs Latency
       - CO2 vs Latency
       - API Cost vs Latency
       - Energy vs API Cost
       - CO2 vs API Cost
    """
    resource_metrics = ["avg_energy_kwh", "avg_latency_ms", "avg_co2_kg", "avg_api_cost_usd"]

    if requested_accuracy != "auto":
        accuracy_metrics = [requested_accuracy] if requested_accuracy in sub.columns else []
    else:
        accuracy_metrics = accuracy_metric_candidates(sub)

    requested_pairs = [
        ("avg_energy_kwh", "avg_latency_ms"),
        ("avg_co2_kg", "avg_latency_ms"),
        ("avg_api_cost_usd", "avg_latency_ms"),
        ("avg_energy_kwh", "avg_api_cost_usd"),
        ("avg_co2_kg", "avg_api_cost_usd"),
    ]

    pairs: List[tuple[str, str]] = []

    # Resource/cost vs accuracy
    for y_metric in accuracy_metrics:
        for x_metric in resource_metrics:
            if x_metric in sub.columns and y_metric in sub.columns:
                pairs.append((x_metric, y_metric))

    # Requested pairwise trade-offs
    for x_metric, y_metric in requested_pairs:
        if x_metric in sub.columns and y_metric in sub.columns:
            pairs.append((x_metric, y_metric))

    # Deduplicate while preserving order
    seen = set()
    deduped: List[tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return deduped


def build_tradeoff_df(sub: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    if x_metric not in sub.columns or y_metric not in sub.columns:
        return pd.DataFrame()
    g = (sub.groupby("model_id", dropna=False)[[x_metric, y_metric]]
           .mean(numeric_only=True)
           .reset_index())
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_metric, y_metric, "model_id"])
    if g.empty:
        return g
    g = g[g["model_id"].isin(sort_models(g["model_id"].tolist()))]
    g["_order"] = g["model_id"].map(lambda x: MODEL_ORDER_INDEX.get(str(x), 10000))
    g = g.sort_values(["_order", "model_id"]).drop(columns=["_order"])
    return g


def pareto_frontier_mask(g: pd.DataFrame, x_metric: str, y_metric: str, minimize_x: bool = True, maximize_y: bool = True) -> np.ndarray:
    if g.empty:
        return np.array([], dtype=bool)
    xs = g[x_metric].to_numpy(dtype=float)
    ys = g[y_metric].to_numpy(dtype=float)
    n = len(g)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        xi, yi = xs[i], ys[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = xs[j], ys[j]
            x_better_or_equal = (xj <= xi) if minimize_x else (xj >= xi)
            y_better_or_equal = (yj >= yi) if maximize_y else (yj <= yi)
            x_strictly_better = (xj < xi) if minimize_x else (xj > xi)
            y_strictly_better = (yj > yi) if maximize_y else (yj < yi)
            if x_better_or_equal and y_better_or_equal and (x_strictly_better or y_strictly_better):
                mask[i] = False
                break
    return mask


def frontier_dataframe(g: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    if g.empty:
        return g.copy()
    maximize_y = metric_direction(y_metric) == "max"
    mask = pareto_frontier_mask(g, x_metric, y_metric, minimize_x=True, maximize_y=maximize_y)
    front = g.loc[mask].copy()
    if front.empty:
        return front
    front = front.sort_values(
        [x_metric, y_metric],
        ascending=[True, not maximize_y],
    ).reset_index(drop=True)
    return front


def sanitize_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s)).strip("_")


def export_pareto_summary_csv(sub: pd.DataFrame, out_base: Path, x_metric: str, y_metric: str, scope_label: str) -> None:
    g = build_tradeoff_df(sub, x_metric, y_metric)
    if g.empty:
        return
    front = frontier_dataframe(g, x_metric, y_metric)
    if front.empty:
        return
    front = front.copy()
    front.insert(0, "scope", scope_label)
    front.insert(1, "x_metric", x_metric)
    front.insert(2, "y_metric", y_metric)
    front.insert(3, "pareto_rank", np.arange(1, len(front) + 1))
    front.rename(columns={x_metric: "x_value", y_metric: "y_value"}, inplace=True)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    front.to_csv(out_base.with_suffix('.csv'), index=False)


def plot_tradeoff_scatter(sub: pd.DataFrame, out_base: Path, x_metric: str, y_metric: str, title: str, use_log_x: bool = False, use_log_y: bool = False, highlight_pareto: bool = False) -> None:
    g = build_tradeoff_df(sub, x_metric, y_metric)
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=scaled_figsize(8.5, 6.0))
    xvals = g[x_metric].tolist()
    yvals = g[y_metric].tolist()
    front = frontier_dataframe(g, x_metric, y_metric) if highlight_pareto else pd.DataFrame()
    pareto_ids = set(front["model_id"].tolist()) if not front.empty else set()

    for i, row in g.reset_index(drop=True).iterrows():
        is_pareto = row["model_id"] in pareto_ids
        marker = 'D' if is_pareto else 'o'
        size = 120 if is_pareto else 70
        linewidth = 1.0 if is_pareto else 0.5
        ax.scatter(row[x_metric], row[y_metric], s=size, marker=marker,
                   color=color(i), edgecolor="black", linewidth=linewidth, label=short_model_name(row["model_id"]))
        ax.annotate(short_model_name(row["model_id"]), (row[x_metric], row[y_metric]), textcoords="offset points", xytext=(5, 4), fontsize=8)

    if highlight_pareto and not front.empty:
        ax.plot(front[x_metric], front[y_metric], linestyle='--', linewidth=1.5, color='black', alpha=0.9)

    if use_log_x and can_use_log_scale(xvals):
        ax.set_xscale("log")
    if use_log_y and can_use_log_scale(yvals):
        ax.set_yscale("log")
    ax.set_xlabel(nice(x_metric))
    ax.set_ylabel(nice(y_metric))
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, out_base)


def plot_accuracy_tradeoffs(df: pd.DataFrame, out_dir: Path, accuracy_metric: str) -> int:
    families = sort_families(df["family"].unique().tolist())
    complexities = sort_complexities(df["complexity"].unique().tolist())
    rows_vals = sorted([int(x) for x in df["table_rows"].dropna().unique().tolist()])

    total_pairs_generated = 0

    def export_scope_csv(subset: pd.DataFrame, scope_dir: Path, scope_label: str, x_metric: str, y_metric: str) -> None:
        base_name = f"{x_metric}_vs_{y_metric}"
        export_pareto_summary_csv(subset, scope_dir / f"{base_name}_pareto_summary", x_metric, y_metric, scope_label)

    def plot_scope(subset: pd.DataFrame, scope_dir: Path, title_prefix: str, scope_label: str) -> int:
        pairs = tradeoff_pairs_for_subset(subset, accuracy_metric)
        if not pairs:
            return 0

        generated = 0
        for x_metric, y_metric in pairs:
            base_name = f"{x_metric}_vs_{y_metric}"
            plot_tradeoff_scatter(subset, scope_dir / f"{base_name}_linear", x_metric, y_metric, f"{title_prefix} {nice(x_metric)} vs {nice(y_metric)}")
            plot_tradeoff_scatter(subset, scope_dir / f"{base_name}_logx", x_metric, y_metric, f"{title_prefix} {nice(x_metric)} vs {nice(y_metric)} (log-x)", use_log_x=True)
            plot_tradeoff_scatter(subset, scope_dir / f"{base_name}_pareto", x_metric, y_metric, f"{title_prefix} {nice(x_metric)} vs {nice(y_metric)}", highlight_pareto=True)
            plot_tradeoff_scatter(subset, scope_dir / f"{base_name}_pareto_logx", x_metric, y_metric, f"{title_prefix} {nice(x_metric)} vs {nice(y_metric)} (log-x)", use_log_x=True, highlight_pareto=True)
            export_scope_csv(subset, scope_dir, scope_label, x_metric, y_metric)
            generated += 1
        return generated

    total_pairs_generated += plot_scope(df, out_dir / "global", "Global", "global")

    for fam in families:
        for comp in complexities:
            sub_fc = df[(df["family"] == fam) & (df["complexity"] == comp)].copy()
            if sub_fc.empty:
                continue

            total_pairs_generated += plot_scope(
                sub_fc,
                out_dir / fam / comp,
                f"{fam} / {comp} —",
                f"{fam}/{comp}",
            )

            for tr in rows_vals:
                sub = sub_fc[sub_fc["table_rows"] == tr].copy()
                if sub.empty:
                    continue

                total_pairs_generated += plot_scope(
                    sub,
                    out_dir / fam / comp / f"rows_{tr}",
                    f"{fam} / {comp} / rows={tr} —",
                    f"{fam}/{comp}/rows_{tr}",
                )

    return total_pairs_generated


def normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "dataset": ["dataset", "meta.dataset"],
        "family": ["family", "meta.family"],
        "complexity": ["complexity", "meta.complexity"],
        "table_rows": ["table_rows", "meta.table_rows"],
        "model_id": ["model_id"],
    }
    for out, cols in aliases.items():
        if out in df.columns:
            continue
        for c in cols:
            if c in df.columns:
                df[out] = df[c]
                break
        if out not in df.columns:
            df[out] = np.nan if out == "table_rows" else "unknown"

    df["family"] = df["family"].astype(str).str.strip()
    df["complexity"] = df["complexity"].astype(str).str.strip()
    df["model_id"] = df["model_id"].astype(str).str.strip()

    if (df["complexity"].isin(["", "unknown", "nan"])).all() and "run_name" in df.columns:
        df["complexity"] = df["run_name"].astype(str).str.extract(r"(C[1-4])", expand=False).fillna("unknown")

    df["table_rows"] = pd.to_numeric(df["table_rows"], errors="coerce")
    return df


def add_co2_if_missing(df: pd.DataFrame, co2_kg_per_kwh: float) -> pd.DataFrame:
    if "avg_co2_kg" not in df.columns and "avg_energy_kwh" in df.columns:
        df["avg_co2_kg"] = pd.to_numeric(df["avg_energy_kwh"], errors="coerce") * float(co2_kg_per_kwh)
    return df


def available_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    return [m for m in metrics if m in df.columns]


def ensure_outdirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "family_fixed_complexity": root / "01_family_fixed_complexity",
        "family_compare_complexity": root / "02_family_compare_complexity",
        "complexity_compare_family": root / "03_complexity_compare_family",
        "family_scaling_grid": root / "04_family_scaling_grid",
        "master_heatmap": root / "05_master_heatmap",
        "accuracy_tradeoffs": root / "06_accuracy_tradeoffs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def read_rows_summary(run_dir: Path) -> pd.DataFrame:
    csv_candidates = list((run_dir / "csv").glob("leaderboard_by_family_rows*.csv")) if (run_dir / "csv").exists() else []
    pq_candidates = list(run_dir.glob("leaderboard_by_family_rows*.parquet"))
    if csv_candidates:
        return pd.read_csv(csv_candidates[0])
    if pq_candidates:
        return pd.read_parquet(pq_candidates[0])
    raise FileNotFoundError(f"Missing rows summary in {run_dir}")


def discover_run_dirs(runs_root: Path) -> List[Path]:
    out: List[Path] = []
    if not runs_root.exists():
        return out
    for p in runs_root.rglob("*"):
        if p.is_dir():
            has_csv = (p / "csv").exists() and list((p / "csv").glob("leaderboard_by_family_rows*.csv"))
            has_pq = list(p.glob("leaderboard_by_family_rows*.parquet"))
            if has_csv or has_pq:
                out.append(p)
    return sorted(set(out))


def run_kind(run_dir: Path) -> str:
    return "online" if "_online" in run_dir.name.lower() else "offline"


def combine_runs(runs_root: Path, keep_kinds: Optional[List[str]] = None) -> pd.DataFrame:
    keep_kinds = keep_kinds or ["offline", "online"]
    frames: List[pd.DataFrame] = []
    for rd in discover_run_dirs(runs_root):
        kind = run_kind(rd)
        if kind not in keep_kinds:
            continue
        try:
            df = read_rows_summary(rd)
            df["run_name"] = rd.name
            df["run_kind"] = kind
            frames.append(df)
            print(f"[load:{kind}] {rd} -> {len(df)} rows")
        except Exception as e:
            print(f"[skip] {rd}: {e}")
    if not frames:
        raise FileNotFoundError(f"No run summaries found under {runs_root} for kinds={keep_kinds}")
    return normalize_meta(pd.concat(frames, ignore_index=True))


def sort_complexities(vals: List[str]) -> List[str]:
    vals = [v for v in vals if isinstance(v, str) and v not in ("", "nan", "unknown")]
    return sorted(vals, key=lambda x: COMPLEXITY_ORDER.get(str(x), 999))


def sort_families(vals: List[str]) -> List[str]:
    vals = [v for v in vals if isinstance(v, str) and v not in ("", "nan", "unknown")]
    return sorted(vals, key=lambda x: FAMILY_ORDER.get(str(x), 999))


def draw_heatmap(piv: pd.DataFrame, title: str, xlabel: str, ylabel: str, out_base: Path) -> None:
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=scaled_figsize(max(6, 0.9 * len(piv.columns) + 2), max(4, 0.5 * len(piv.index) + 2)))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns], rotation=20 if len(piv.columns) > 6 else 0,
                       ha="right" if len(piv.columns) > 6 else "center")
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, "nan" if pd.isna(val) else f"{val:.3g}", ha="center", va="center",
                    fontsize=7, color="white")
    fig.tight_layout()
    save_fig(fig, out_base)


def plot_family_fixed_complexity(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    families = sort_families(df["family"].unique().tolist())
    complexities = sort_complexities(df["complexity"].unique().tolist())
    print(f"[group1] families={families} complexities={complexities}")
    for fam in families:
        for comp in complexities:
            sub = df[(df["family"] == fam) & (df["complexity"] == comp)].copy()
            if sub.empty:
                continue
            rows_vals = sorted([int(x) for x in sub["table_rows"].dropna().unique().tolist()])
            models = sort_models(sub["model_id"].unique().tolist())
            for metric in available_metrics(sub, metrics):
                plotted_value_sets: List[float] = []
                for use_log, suffix in [(False, "linear"), (True, "log")]:
                    fig, ax = plt.subplots(figsize=scaled_figsize(max(9, 1.25 * len(models) + 3), 5.5))
                    x = np.arange(len(models))
                    width = 0.8 / max(1, len(rows_vals))
                    all_vals: List[float] = []
                    bar_groups = []
                    for i, tr in enumerate(rows_vals):
                        vals = [float(sub[(sub["model_id"] == m) & (sub["table_rows"] == tr)][metric].mean())
                                if len(sub[(sub["model_id"] == m) & (sub["table_rows"] == tr)]) else np.nan
                                for m in models]
                        all_vals.extend([v for v in vals if not pd.isna(v)])
                        bars = ax.bar(x + (i - (len(rows_vals) - 1) / 2) * width, vals, width=width, label=f"rows={tr}",
                                      color=color(i), hatch=hatch(i), edgecolor="black", linewidth=0.35)
                        bar_groups.append((bars, vals))
                    if use_log and not maybe_set_log_y(ax, all_vals, use_log=True):
                        plt.close(fig)
                        continue
                    for bars, vals in bar_groups:
                        add_value_labels(ax, bars, vals, use_log=use_log)
                    ax.set_xticks(x); ax.set_xticklabels(short_model_names(models), rotation=25, ha="right")
                    ax.set_xlabel("Models"); ax.set_ylabel(nice(metric))
                    ax.set_title(f"{fam} / {comp} — {nice(metric)} vs models grouped by table size ({suffix} y)")
                    ax.grid(True, axis="y"); ax.legend(frameon=True, title="Table size")
                    ax.margins(y=0.18)
                    fig.tight_layout(); save_fig(fig, out_dir / fam / comp / f"bar_{metric}_{suffix}")
                piv = sub.pivot_table(index="model_id", columns="table_rows", values=metric, aggfunc="mean")
                if not piv.empty:
                    draw_heatmap(piv.reindex(index=models, columns=rows_vals),
                                 f"{fam} / {comp} — heatmap of {nice(metric)}",
                                 "Table rows", "Models", out_dir / fam / comp / f"heatmap_{metric}")

def plot_family_compare_complexity(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    families = sort_families(df["family"].unique().tolist())
    rows_vals_all = sorted([int(x) for x in df["table_rows"].dropna().unique().tolist()])
    for fam in families:
        fam_df = df[df["family"] == fam].copy()
        if fam_df.empty:
            continue
        complexities = sort_complexities(fam_df["complexity"].unique().tolist())
        models = sort_models(fam_df["model_id"].unique().tolist())
        for tr in rows_vals_all:
            sub = fam_df[fam_df["table_rows"] == tr].copy()
            if sub.empty:
                continue
            for metric in available_metrics(sub, metrics):
                for use_log, suffix in [(False, "linear"), (True, "log")]:
                    fig, ax = plt.subplots(figsize=scaled_figsize(max(9, 1.25 * len(models) + 3), 5.5))
                    x = np.arange(len(models))
                    width = 0.8 / max(1, len(complexities))
                    all_vals: List[float] = []
                    bar_groups = []
                    for i, comp in enumerate(complexities):
                        vals = [float(sub[(sub["model_id"] == m) & (sub["complexity"] == comp)][metric].mean())
                                if len(sub[(sub["model_id"] == m) & (sub["complexity"] == comp)]) else np.nan
                                for m in models]
                        all_vals.extend([v for v in vals if not pd.isna(v)])
                        bars = ax.bar(x + (i - (len(complexities) - 1) / 2) * width, vals, width=width, label=comp,
                                      color=color(i), hatch=hatch(i), edgecolor="black", linewidth=0.35)
                        bar_groups.append((bars, vals))
                    if use_log and not maybe_set_log_y(ax, all_vals, use_log=True):
                        plt.close(fig)
                        continue
                    for bars, vals in bar_groups:
                        add_value_labels(ax, bars, vals, use_log=use_log)
                    ax.set_xticks(x); ax.set_xticklabels(short_model_names(models), rotation=25, ha="right")
                    ax.set_xlabel("Models"); ax.set_ylabel(nice(metric))
                    ax.set_title(f"{fam} / rows={tr} — {nice(metric)} vs models grouped by complexity ({suffix} y)")
                    ax.grid(True, axis="y"); ax.legend(title="Complexity", frameon=True)
                    ax.margins(y=0.18)
                    fig.tight_layout(); save_fig(fig, out_dir / fam / f"rows_{tr}" / f"bar_{metric}_{suffix}")
                piv = sub.pivot_table(index="model_id", columns="complexity", values=metric, aggfunc="mean")
                if not piv.empty:
                    draw_heatmap(piv.reindex(index=models, columns=complexities),
                                 f"{fam} / rows={tr} — heatmap of {nice(metric)}",
                                 "Complexity", "Models", out_dir / fam / f"rows_{tr}" / f"heatmap_{metric}")

def plot_complexity_compare_family(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    complexities = sort_complexities(df["complexity"].unique().tolist())
    rows_vals = sorted([int(x) for x in df["table_rows"].dropna().unique().tolist()])
    families = sort_families(df["family"].unique().tolist())
    models = sort_models(df["model_id"].unique().tolist())
    print(f"[group3] complexities={complexities} families={families}")
    for comp in complexities:
        comp_df = df[df["complexity"] == comp].copy()
        if comp_df.empty:
            continue
        for tr in rows_vals:
            sub = comp_df[comp_df["table_rows"] == tr].copy()
            if sub.empty:
                continue
            for metric in available_metrics(sub, metrics):
                for use_log, suffix in [(False, "linear"), (True, "log")]:
                    fig, ax = plt.subplots(figsize=scaled_figsize(max(9, 1.25 * len(models) + 3), 5.5))
                    x = np.arange(len(models))
                    width = 0.8 / max(1, len(families))
                    all_vals: List[float] = []
                    bar_groups = []
                    for i, fam in enumerate(families):
                        vals = [float(sub[(sub["model_id"] == m) & (sub["family"] == fam)][metric].mean())
                                if len(sub[(sub["model_id"] == m) & (sub["family"] == fam)]) else np.nan
                                for m in models]
                        all_vals.extend([v for v in vals if not pd.isna(v)])
                        bars = ax.bar(x + (i - (len(families) - 1) / 2) * width, vals, width=width, label=fam,
                                      color=color(i), hatch=hatch(i), edgecolor="black", linewidth=0.35)
                        bar_groups.append((bars, vals))
                    if use_log and not maybe_set_log_y(ax, all_vals, use_log=True):
                        plt.close(fig)
                        continue
                    for bars, vals in bar_groups:
                        add_value_labels(ax, bars, vals, use_log=use_log)
                    ax.set_xticks(x); ax.set_xticklabels(short_model_names(models), rotation=25, ha="right")
                    ax.set_xlabel("Models"); ax.set_ylabel(nice(metric))
                    ax.set_title(f"{comp} / rows={tr} — {nice(metric)} vs models grouped by family ({suffix} y)")
                    ax.grid(True, axis="y"); ax.legend(title="Family", frameon=True)
                    ax.margins(y=0.18)
                    fig.tight_layout(); save_fig(fig, out_dir / comp / f"rows_{tr}" / f"bar_{metric}_{suffix}")
                piv = sub.pivot_table(index="model_id", columns="family", values=metric, aggfunc="mean")
                if not piv.empty:
                    draw_heatmap(piv.reindex(index=models, columns=families),
                                 f"{comp} / rows={tr} — heatmap of {nice(metric)}",
                                 "Family", "Models", out_dir / comp / f"rows_{tr}" / f"heatmap_{metric}")

def plot_family_scaling_grid(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    families = sort_families(df["family"].unique().tolist())
    complexity_order = [c for c in ["C1", "C2", "C3", "C4"] if c in set(df["complexity"].tolist())]
    print(f"[group4] families={families} complexity_order={complexity_order}")
    for fam in families:
        fam_df = df[df["family"] == fam].copy()
        if fam_df.empty:
            continue
        models = sort_models(fam_df["model_id"].unique().tolist())
        for metric in available_metrics(fam_df, metrics):
            for use_log, suffix in [(False, "linear"), (True, "log")]:
                nplots = max(1, len(complexity_order))
                ncols = 2
                nrows = int(np.ceil(nplots / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=scaled_figsize(13, 4.5 * nrows), squeeze=False)
                axes = axes.reshape(-1)
                any_axis_drawn = False
                all_vals: List[float] = []
                for ax, comp in zip(axes, complexity_order):
                    g = fam_df[fam_df["complexity"] == comp].copy().sort_values("table_rows")
                    if g.empty:
                        ax.set_title(comp); ax.axis("off"); continue
                    axis_has_data = False
                    for i, mid in enumerate(models):
                        mm = g[g["model_id"] == mid].groupby("table_rows", dropna=False)[metric].mean().reset_index().sort_values("table_rows")
                        if mm.empty:
                            continue
                        vals = mm[metric].tolist()
                        all_vals.extend([v for v in vals if not pd.isna(v)])
                        ax.plot(mm["table_rows"], mm[metric], marker="o", linewidth=2.0, label=short_model_name(mid), color=color(i))
                        axis_has_data = True
                    if axis_has_data:
                        any_axis_drawn = True
                        if use_log:
                            maybe_set_log_y(ax, all_vals, use_log=True)
                    ax.set_title(comp); ax.set_xlabel("Table rows"); ax.set_ylabel(nice(metric)); ax.grid(True)
                for ax in axes[len(complexity_order):]:
                    ax.remove()
                if use_log and not can_use_log_scale(all_vals):
                    plt.close(fig)
                    continue
                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=True)
                fig.suptitle(f"{fam} — scaling across complexities for {nice(metric)} ({suffix} y)", y=1.02)
                fig.tight_layout(); save_fig(fig, out_dir / fam / f"scaling_grid_{metric}_{suffix}")

def plot_master_heatmap(df: pd.DataFrame, out_dir: Path, metrics: List[str]) -> None:
    rows_vals = sorted([int(x) for x in df["table_rows"].dropna().unique().tolist()])
    families = sort_families(df["family"].unique().tolist())
    complexities = [c for c in ["C1", "C2", "C3", "C4"] if c in set(df["complexity"].tolist())]
    fam_comp_order = [f"{fam}|{comp}" for fam in families for comp in complexities]
    print(f"[group5] rows={rows_vals} fam_comp={fam_comp_order}")
    for tr in rows_vals:
        sub = df[df["table_rows"] == tr].copy()
        if sub.empty:
            continue
        sub["fam_comp"] = sub["family"].astype(str) + "|" + sub["complexity"].astype(str)
        for metric in available_metrics(sub, metrics):
            piv = sub.pivot_table(index="model_id", columns="fam_comp", values=metric, aggfunc="mean")
            cols = [c for c in fam_comp_order if c in piv.columns]
            piv = piv.reindex(columns=cols)
            if piv.empty:
                continue
            draw_heatmap(piv, f"rows={tr} — master heatmap of {nice(metric)}",
                         "Family | Complexity", "Models", out_dir / f"rows_{tr}" / f"master_heatmap_{metric}")


def run_one(df: pd.DataFrame, out_root: Path, metrics: List[str], accuracy_metric: str = "auto") -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = ensure_outdirs(out_root)
    df.to_csv(out_root / "combined_leaderboard_by_family_rows.csv", index=False)
    plot_family_fixed_complexity(df, dirs["family_fixed_complexity"], metrics)
    plot_family_compare_complexity(df, dirs["family_compare_complexity"], metrics)
    plot_complexity_compare_family(df, dirs["complexity_compare_family"], metrics)
    plot_family_scaling_grid(df, dirs["family_scaling_grid"], metrics)
    plot_master_heatmap(df, dirs["master_heatmap"], metrics)
    num_tradeoff_pairs = plot_accuracy_tradeoffs(df, dirs["accuracy_tradeoffs"], accuracy_metric)
    if num_tradeoff_pairs > 0:
        if accuracy_metric == "auto":
            print("[tradeoffs] generated per-subset plots using available accuracy metrics: anomaly_f1_macro / pass_rate / json_valid_rate")
        else:
            print(f"[tradeoffs] generated plots using requested accuracy metric={accuracy_metric}")
        print("[tradeoffs] x-metrics include avg_energy_kwh, avg_latency_ms, avg_co2_kg, avg_api_cost_usd")
    else:
        print("[tradeoffs] skipped: no valid trade-off pairs found in the loaded data")
    print(f"[ok] saved plots under: {out_root}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out-root", default="plots_insight_v4")
    ap.add_argument("--mode", choices=["offline", "online", "combined", "all"], default="offline",
                    help="offline: only *_sweep, online: only *_sweep_online, combined: both together, all: generate all three.")
    ap.add_argument("--font", type=int, default=12)
    ap.add_argument("--title-font", type=int, default=14)
    ap.add_argument("--tick-font", type=int, default=11)
    ap.add_argument("--legend-font", type=int, default=10)
    ap.add_argument("--co2-kg-per-kwh", type=float, default=0.4)
    ap.add_argument("--fig-width-scale", type=float, default=0.5,
                    help="Scale factor for all figure widths. Default 0.5 halves the current widths.")
    ap.add_argument("--fig-height-scale", type=float, default=0.5,
                    help="Scale factor for all figure heights. Default 0.5 halves the current heights.")
    ap.add_argument("--accuracy-metric", default="auto",
                    help="Accuracy metric to use for trade-off plots. Use auto to prefer anomaly_f1_macro, then pass_rate, then json_valid_rate.")
    ap.add_argument("--metrics", nargs="*", default=[
        "pass_rate", "json_valid_rate", "avg_latency_ms", "p95_latency_ms",
        "error_rate", "avg_tokens_out", "avg_energy_kwh", "avg_energy_cost_eur",
        "avg_api_cost_usd", "avg_mem_rss_delta_mb", "avg_cpu_user_s_delta",
        "avg_cpu_system_s_delta", "numeric_mae", "numeric_rmse", "anomaly_f1_macro", "avg_co2_kg",
    ])
    args = ap.parse_args()

    apply_style(args.font, args.title_font, args.tick_font, args.legend_font)

    global FIG_WIDTH_SCALE, FIG_HEIGHT_SCALE
    FIG_WIDTH_SCALE = max(0.05, float(args.fig_width_scale))
    FIG_HEIGHT_SCALE = max(0.05, float(args.fig_height_scale))

    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)

    def prepare(kinds: List[str]) -> pd.DataFrame:
        d = combine_runs(runs_root, keep_kinds=kinds)
        d = add_co2_if_missing(d, args.co2_kg_per_kwh)
        print(f"[loaded:{'+'.join(kinds)}] rows={len(d)} families={sorted(d['family'].unique().tolist())} complexities={sorted(d['complexity'].unique().tolist())}")
        return d

    if args.mode == "offline":
        df = prepare(["offline"])
        run_one(df, out_root / "offline", args.metrics, args.accuracy_metric)
    elif args.mode == "online":
        df = prepare(["online"])
        run_one(df, out_root / "online", args.metrics, args.accuracy_metric)
    elif args.mode == "combined":
        df = prepare(["offline", "online"])
        run_one(df, out_root / "combined", args.metrics, args.accuracy_metric)
    elif args.mode == "all":
        df_off = prepare(["offline"])
        run_one(df_off, out_root / "offline", args.metrics, args.accuracy_metric)
        df_on = prepare(["online"])
        run_one(df_on, out_root / "online", args.metrics, args.accuracy_metric)
        df_both = prepare(["offline", "online"])
        run_one(df_both, out_root / "combined", args.metrics, args.accuracy_metric)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
