#!/usr/bin/env python3
from __future__ import annotations
"""
Generate SEDS-based Family 2 (anomaly triage) and Family 3 (forecast) sweeps with table-size scaling.

Reads:
  data/processed/seds_industrial_consumption.parquet

Writes:
  tasks/family2_anomaly/seds_f2_C1_sweep.yaml
  tasks/family2_anomaly/seds_f2_C2_sweep.yaml
  tasks/family3_forecast/seds_f3_C1_sweep.yaml
  tasks/family3_forecast/seds_f3_C2_sweep.yaml

Run:
  python scripts/gen_seds_family23_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
"""
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

ROW_SIZES_DEFAULT = [20, 100, 250, 500, 1000]
ANOM_TYPES = ["spike", "drop", "level_shift", "trend_change"]


def md_table(rows: List[Dict[str, object]], cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["-" * len(c) for c in cols]) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def choose_msn(df: pd.DataFrame) -> str:
    vc = df["MSN"].astype(str).value_counts()
    if vc.empty:
        raise ValueError("No MSN values found.")
    return str(vc.index[0])


def build_pool(df: pd.DataFrame, msn: str) -> pd.DataFrame:
    d = df[df["MSN"].astype(str) == msn].copy()
    d["State"] = d["State"].astype(str).str.strip()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d.dropna(subset=["State", "Year", "Value"])
    d = d[d["State"].str.len().between(2, 3)]
    return d


def states_with_min_years(pool: pd.DataFrame, min_years: int) -> List[str]:
    counts = pool.groupby("State")["Year"].nunique()
    return counts[counts >= min_years].index.astype(str).tolist()


def sample_distractors(pool: pd.DataFrame, k: int, avoid: set[Tuple[str, int]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    tries = 0
    while len(out) < k and tries < k * 30:
        tries += 1
        r = pool.sample(1).iloc[0]
        key = (str(r["State"]), int(r["Year"]))
        if key in avoid:
            continue
        out.append({"State": str(r["State"]), "Year": int(r["Year"]), "Value": float(r["Value"]), "Unit": str(r.get("Unit",""))})
        avoid.add(key)
    if len(out) < k:
        extra = pool.sample(k - len(out), replace=True)
        for _, r in extra.iterrows():
            out.append({"State": str(r["State"]), "Year": int(r["Year"]), "Value": float(r["Value"]), "Unit": str(r.get("Unit",""))})
    return out


def inject_anomaly(series: List[Tuple[int, float]], kind: str, rng: random.Random) -> Tuple[List[Tuple[int, float]], int]:
    s = [(y, v) for (y, v) in series]
    n = len(s)
    if n < 6:
        return s, -1
    idx = rng.randint(2, n - 3)
    y, v = s[idx]
    if kind == "spike":
        s[idx] = (y, v * 1.6 + 1.0)
    elif kind == "drop":
        s[idx] = (y, max(0.0, v * 0.4))
    elif kind == "level_shift":
        shift = v * 0.25
        for j in range(idx, n):
            yy, vv = s[j]
            s[j] = (yy, vv + shift)
    elif kind == "trend_change":
        for j in range(idx, n):
            yy, vv = s[j]
            s[j] = (yy, vv + (j - idx) * (abs(v) * 0.05))
    return s, idx


def build_series_rows(state: str, series: List[Tuple[int, float]]) -> List[Dict[str, object]]:
    return [{"State": state, "Year": int(y), "Value": float(v)} for (y, v) in series]


def task_f2_C1(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    st_candidates = states_with_min_years(pool, min_years=20) or states_with_min_years(pool, min_years=10)
    st = rng.choice(st_candidates)
    st_df = pool[pool["State"] == st].sort_values("Year").drop_duplicates(subset=["Year"]).copy()
    years = [int(y) for y in st_df["Year"].dropna().unique().tolist()]
    if len(years) > 20:
        start = rng.randint(0, len(years) - 20)
        sel_years = years[start:start + 20]
        st_df = st_df[st_df["Year"].isin(sel_years)]
    series = [(int(r.Year), float(r.Value)) for r in st_df.itertuples()]
    kind = rng.choice(ANOM_TYPES)
    series2, _ = inject_anomaly(series, kind, rng)
    rows = build_series_rows(st, series2)

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(pool, need, avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value"]
    return {
        "task_id": f"seds_f2_C1_r{table_rows}_{idx:03d}",
        "family": "family2_anomaly",
        "difficulty": "easy",
        "meta": {"dataset": "SEDS", "complexity": "C1", "table_rows": table_rows, "msn": msn, "focus_state": st, "anomaly_type": kind},
        "input": "An alert was raised for unusual behavior in industrial energy consumption. Identify the likely anomaly type (spike/drop/level_shift/trend_change) and list two checks to confirm.",
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"labels": {"anomaly_type": kind}},
        "scoring": {"deterministic": [{"type": "label_match", "label": "anomaly_type"}]},
    }


def task_f2_C2(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    st_candidates = states_with_min_years(pool, min_years=15) or states_with_min_years(pool, min_years=10)
    st1, st2 = rng.sample(st_candidates, 2)
    st1_df = pool[pool["State"] == st1].sort_values("Year").drop_duplicates(subset=["Year"]).copy()
    years1 = [int(y) for y in st1_df["Year"].dropna().unique().tolist()]
    if len(years1) > 15:
        start = rng.randint(0, len(years1) - 15)
        st1_df = st1_df[st1_df["Year"].isin(years1[start:start + 15])]
    s1 = [(int(r.Year), float(r.Value)) for r in st1_df.itertuples()]

    st2_df = pool[pool["State"] == st2].sort_values("Year").drop_duplicates(subset=["Year"]).copy()
    years2 = [int(y) for y in st2_df["Year"].dropna().unique().tolist()]
    if len(years2) > 15:
        start = rng.randint(0, len(years2) - 15)
        st2_df = st2_df[st2_df["Year"].isin(years2[start:start + 15])]
    s2 = [(int(r.Year), float(r.Value)) for r in st2_df.itertuples()]

    kind = rng.choice(ANOM_TYPES)
    inject_to_first = rng.random() < 0.5
    if inject_to_first:
        s1b, _ = inject_anomaly(s1, kind, rng)
        s2b = s2
        focus = st1
    else:
        s2b, _ = inject_anomaly(s2, kind, rng)
        s1b = s1
        focus = st2

    rows = build_series_rows(st1, s1b) + build_series_rows(st2, s2b)
    if len(rows) > table_rows:
        rows = rng.sample(rows, table_rows)
    else:
        avoid = {(r["State"], r["Year"]) for r in rows}
        rows += sample_distractors(pool, table_rows - len(rows), avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value"]
    return {
        "task_id": f"seds_f2_C2_r{table_rows}_{idx:03d}",
        "family": "family2_anomaly",
        "difficulty": "medium",
        "meta": {"dataset": "SEDS", "complexity": "C2", "table_rows": table_rows, "msn": msn, "anomaly_type": kind, "focus_state": focus},
        "input": "Two states show unusual behavior. Identify the anomaly type (spike/drop/level_shift/trend_change) AND which State most likely contains it. Then give two checks to confirm.",
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"labels": {"anomaly_type": kind, "state": focus}},
        "scoring": {"deterministic": [
            {"type": "label_match", "label": "anomaly_type"},
            {"type": "label_match", "label": "state"},
        ]},
    }


def task_f3_C1(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    st_candidates = states_with_min_years(pool, min_years=12) or states_with_min_years(pool, min_years=8)
    st = rng.choice(st_candidates)
    st_df = pool[pool["State"] == st].sort_values("Year").drop_duplicates(subset=["Year"]).copy()
    years = [int(y) for y in st_df["Year"].dropna().unique().tolist()]
    if len(years) < 8:
        st = rng.choice(st_candidates)
        st_df = pool[pool["State"] == st].sort_values("Year").drop_duplicates(subset=["Year"]).copy()
        years = [int(y) for y in st_df["Year"].dropna().unique().tolist()]
    tpos = rng.randint(5, len(years) - 2)
    target_year = years[tpos + 1]
    hist_years = years[tpos - 5: tpos + 1]
    hist = st_df[st_df["Year"].isin(hist_years)].copy()
    target = st_df[st_df["Year"] == target_year].iloc[0]
    forecast_target = float(target["Value"])

    rows = [{"State": st, "Year": int(r.Year), "Value": float(r.Value)} for r in hist.itertuples()]
    avoid = {(r["State"], r["Year"]) for r in rows}
    rows += sample_distractors(pool, max(0, table_rows - len(rows)), avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value"]
    tol = max(0.0, abs(forecast_target) * 0.05)
    return {
        "task_id": f"seds_f3_C1_r{table_rows}_{idx:03d}",
        "family": "family3_forecast",
        "difficulty": "easy",
        "meta": {"dataset": "SEDS", "complexity": "C1", "table_rows": table_rows, "msn": msn, "state": st, "target_year": target_year},
        "input": f"Using the historical industrial energy consumption for State {st} in the table, forecast the Value for year {target_year}. Output a single numeric forecast.",
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"numeric_targets": [{"name": "forecast_value", "value": forecast_target, "tolerance_abs": tol}]},
        "scoring": {"deterministic": [{"type": "numeric_extract", "target": "forecast_value"}]},
    }


def task_f3_C2(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    st_candidates = states_with_min_years(pool, min_years=15) or states_with_min_years(pool, min_years=10)
    st = rng.choice(st_candidates)
    st_df = pool[pool["State"] == st].sort_values("Year").drop_duplicates(subset=["Year"]).copy()
    years = [int(y) for y in st_df["Year"].dropna().unique().tolist()]
    if len(years) < 10:
        st = rng.choice(st_candidates)
        st_df = pool[pool["State"] == st].sort_values("Year").drop_duplicates(subset=["Year"]).copy()
        years = [int(y) for y in st_df["Year"].dropna().unique().tolist()]
    tpos = rng.randint(6, len(years) - 3)
    yA, yB = years[tpos + 1], years[tpos + 2]
    hist_years = years[tpos - 6: tpos + 1]
    hist = st_df[st_df["Year"].isin(hist_years)].copy()
    vA = float(st_df[st_df["Year"] == yA].iloc[0]["Value"])
    vB = float(st_df[st_df["Year"] == yB].iloc[0]["Value"])
    delta = vB - vA

    rows = [{"State": st, "Year": int(r.Year), "Value": float(r.Value)} for r in hist.itertuples()]
    avoid = {(r["State"], r["Year"]) for r in rows}
    rows += sample_distractors(pool, max(0, table_rows - len(rows)), avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value"]
    tol = max(0.0, abs(delta) * 0.10 + 1e-6)
    return {
        "task_id": f"seds_f3_C2_r{table_rows}_{idx:03d}",
        "family": "family3_forecast",
        "difficulty": "medium",
        "meta": {"dataset": "SEDS", "complexity": "C2", "table_rows": table_rows, "msn": msn, "state": st, "yearA": yA, "yearB": yB},
        "input": f"Using the historical table for State {st}, estimate the change in Value between {yA} and {yB} (Value_{yB} - Value_{yA}). Output a single numeric estimate.",
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"numeric_targets": [{"name": "delta_future", "value": float(delta), "tolerance_abs": tol}]},
        "scoring": {"deterministic": [{"type": "numeric_extract", "target": "delta_future"}]},
    }


def write_yaml(path: Path, tasks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": tasks}, f, sort_keys=False, allow_unicode=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--input", default="data/processed/seds_industrial_consumption.parquet")
    ap.add_argument("--bases", type=int, default=30)
    ap.add_argument("--row-sizes", nargs="+", type=int, default=ROW_SIZES_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.repo_root).resolve()
    in_path = root / args.input
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_parquet(in_path)
    for col in ["MSN", "State", "Year", "Value"]:
        if col not in df.columns:
            raise ValueError(f"Input parquet missing {col}. Has: {df.columns.tolist()}")

    msn = choose_msn(df)
    pool = build_pool(df, msn)
    print(f"[ok] using MSN={msn} pool_rows={len(pool):,}")

    f2_C1, f2_C2, f3_C1, f3_C2 = [], [], [], []

    for nrows in args.row_sizes:
        for i in range(1, args.bases + 1):
            f2_C1.append(task_f2_C1(pool, nrows, i, msn, rng))
            f2_C2.append(task_f2_C2(pool, nrows, i, msn, rng))
            f3_C1.append(task_f3_C1(pool, nrows, i, msn, rng))
            f3_C2.append(task_f3_C2(pool, nrows, i, msn, rng))

    out_f2 = root / "tasks" / "family2_anomaly"
    out_f3 = root / "tasks" / "family3_forecast"
    write_yaml(out_f2 / "seds_f2_C1_sweep.yaml", f2_C1)
    write_yaml(out_f2 / "seds_f2_C2_sweep.yaml", f2_C2)
    write_yaml(out_f3 / "seds_f3_C1_sweep.yaml", f3_C1)
    write_yaml(out_f3 / "seds_f3_C2_sweep.yaml", f3_C2)

    print("[ok] wrote:")
    for p in ["tasks/family2_anomaly/seds_f2_C1_sweep.yaml",
              "tasks/family2_anomaly/seds_f2_C2_sweep.yaml",
              "tasks/family3_forecast/seds_f3_C1_sweep.yaml",
              "tasks/family3_forecast/seds_f3_C2_sweep.yaml"]:
        print("  -", p)


if __name__ == "__main__":
    main()
