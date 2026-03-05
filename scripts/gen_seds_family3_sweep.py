#!/usr/bin/env python3
"""Generate SEDS Family-3 forecasting tasks (C1..C4) with table-size sweeps.

Family 3 = numeric forecasting / projection from historical tables.
We keep scoring deterministic by defining a rule and computing the expected value.

Complexity:
  C1 (easy)   : forecast next-year value using mean(last 3 years)
  C2 (medium) : forecast next-year and report delta vs last actual year
  C3 (hard)   : compute CAGR over last k years and forecast h years ahead
  C4 (hard+)  : compute max rolling-5-year average value in the shown history

The prompt asks the model to output the numeric answer in numbers.<name>.

Input data: data/processed/seds_industrial_consumption.parquet

Run:
  python scripts/gen_seds_family3_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from _seds_task_utils import build_pool, choose_msn, md_table, sample_distractors, pick_years_for_state


def write_yaml(path: Path, tasks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": tasks}, f, sort_keys=False, allow_unicode=True)


def make_base_rows(pool: pd.DataFrame, state: str, years: List[int], include_unit: bool = True) -> List[Dict[str, object]]:
    rows = []
    for y in years:
        r = pool[(pool["State"] == state) & (pool["Year"] == y)].iloc[0]
        rows.append(
            {
                "State": str(state),
                "Year": int(y),
                "Value": float(r["Value"]),
                "Unit": str(r.get("Unit", "")) if include_unit else "",
            }
        )
    return rows


def task_template(
    task_id: str,
    complexity: str,
    difficulty: str,
    table_rows: int,
    prompt: str,
    rows: List[Dict[str, object]],
    expected_name: str,
    expected_value: float,
    msn: str,
    state: str,
    meta_extra: Dict[str, object],
) -> Dict:
    cols = ["State", "Year", "Value", "Unit"]
    return {
        "task_id": task_id,
        "family": "family3_forecast",
        "difficulty": difficulty,
        "meta": {
            "dataset": "SEDS",
            "complexity": complexity,
            "table_rows": table_rows,
            "msn": msn,
            "state": state,
            **meta_extra,
        },
        "input": prompt,
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {
            "numeric_targets": [
                {"name": expected_name, "value": float(expected_value), "tolerance_abs": 1e-6}
            ]
        },
        "scoring": {"deterministic": [{"type": "numeric_extract", "target": expected_name}]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--input", default="data/processed/seds_industrial_consumption.parquet")
    ap.add_argument("--bases", type=int, default=30)
    ap.add_argument("--row-sizes", nargs="+", type=int, default=[20, 100, 250, 500, 1000])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.repo_root).resolve()
    df = pd.read_parquet(root / args.input)
    msn = choose_msn(df)
    pool = build_pool(df, msn)

    # Pick candidate states with long histories
    state_counts = pool.groupby("State")["Year"].nunique().sort_values(ascending=False)
    candidate_states = [s for s, n in state_counts.items() if int(n) >= 10]
    if not candidate_states:
        candidate_states = state_counts.index.tolist()[:20]

    out_dir = root / "tasks" / "family3_forecast"

    tasks_C1, tasks_C2, tasks_C3, tasks_C4 = [], [], [], []

    for nrows in args.row_sizes:
        for i in range(1, args.bases + 1):
            state = rng.choice(candidate_states)
            years = pick_years_for_state(pool, state, k=8, rng=rng)
            hist = years[:-1]  # use first 7 as history, last is "current"
            y_last = hist[-1]

            # Build core table rows for history
            core_rows = make_base_rows(pool, state, hist)
            avoid = {(r["State"], r["Year"]) for r in core_rows}
            need = max(0, nrows - len(core_rows))
            rows = core_rows + sample_distractors(pool, need, avoid, rng)
            rng.shuffle(rows)

            # Extract history values in chronological order
            hist_vals = [
                float(pool[(pool["State"] == state) & (pool["Year"] == y)].iloc[0]["Value"])
                for y in hist
            ]

            # C1: mean of last 3 years
            mean3 = sum(hist_vals[-3:]) / 3.0
            y_next = y_last + 1
            tasks_C1.append(
                task_template(
                    f"seds_f3_C1_r{nrows}_{i:03d}",
                    "C1",
                    "easy",
                    nrows,
                    f"Using the table, forecast State {state}'s industrial energy consumption for {y_next} as the mean of the last 3 years shown. Put the number in numbers.forecast_value.",
                    rows,
                    "forecast_value",
                    mean3,
                    msn,
                    state,
                    {"year_next": y_next, "rule": "mean_last3"},
                )
            )

            # C2: same forecast + delta vs last actual
            delta = mean3 - hist_vals[-1]
            tasks_C2.append(
                task_template(
                    f"seds_f3_C2_r{nrows}_{i:03d}",
                    "C2",
                    "medium",
                    nrows,
                    f"Using the table, forecast State {state}'s industrial energy consumption for {y_next} as mean(last 3 years). Then compute delta = forecast - Value_{y_last}. Put delta in numbers.delta_value.",
                    rows,
                    "delta_value",
                    delta,
                    msn,
                    state,
                    {"year_last": y_last, "year_next": y_next, "rule": "mean_last3_delta"},
                )
            )

            # C3: CAGR over last 5 years (if possible) and forecast 3 years ahead
            k = 5
            if len(hist_vals) >= k:
                v0 = hist_vals[-k]
                v1 = hist_vals[-1]
                years_span = k - 1
                if v0 <= 0:
                    cagr = 0.0
                else:
                    cagr = (v1 / v0) ** (1.0 / years_span) - 1.0
                horizon = 3
                forecast_h = v1 * ((1.0 + cagr) ** horizon)
            else:
                forecast_h = mean3
                horizon = 3
            tasks_C3.append(
                task_template(
                    f"seds_f3_C3_r{nrows}_{i:03d}",
                    "C3",
                    "hard",
                    nrows,
                    f"Compute CAGR for State {state} using the last 5 years shown (CAGR=(V_end/V_start)^(1/(years-1))-1). Then forecast the value {horizon} years after the last year shown: forecast = V_end*(1+CAGR)^{horizon}. Put it in numbers.cagr_forecast.",
                    rows,
                    "cagr_forecast",
                    forecast_h,
                    msn,
                    state,
                    {"horizon_years": horizon, "rule": "cagr_last5"},
                )
            )

            # C4: max rolling-5-year average within the shown history
            roll_k = 5
            max_roll = None
            if len(hist_vals) >= roll_k:
                max_roll = max(
                    sum(hist_vals[j : j + roll_k]) / roll_k for j in range(0, len(hist_vals) - roll_k + 1)
                )
            else:
                max_roll = sum(hist_vals) / len(hist_vals)
            tasks_C4.append(
                task_template(
                    f"seds_f3_C4_r{nrows}_{i:03d}",
                    "C4",
                    "hard+",
                    nrows,
                    f"For State {state}, compute the maximum rolling 5-year average of Value across the years shown (use consecutive years as they appear). Put it in numbers.max_roll5_avg.",
                    rows,
                    "max_roll5_avg",
                    max_roll,
                    msn,
                    state,
                    {"rule": "max_rolling5"},
                )
            )

    write_yaml(out_dir / "seds_f3_C1_sweep.yaml", tasks_C1)
    write_yaml(out_dir / "seds_f3_C2_sweep.yaml", tasks_C2)
    write_yaml(out_dir / "seds_f3_C3_sweep.yaml", tasks_C3)
    write_yaml(out_dir / "seds_f3_C4_sweep.yaml", tasks_C4)

    print("[ok] wrote:")
    for p in [
        "seds_f3_C1_sweep.yaml",
        "seds_f3_C2_sweep.yaml",
        "seds_f3_C3_sweep.yaml",
        "seds_f3_C4_sweep.yaml",
    ]:
        print("  -", (out_dir / p).as_posix())


if __name__ == "__main__":
    main()
