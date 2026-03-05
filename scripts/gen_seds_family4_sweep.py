#!/usr/bin/env python3
"""Generate SEDS Family-4 code generation tasks (C1..C4) with table-size sweeps.

Family 4 = code generation (Python/pandas) operating on a provided table.
We measure *generation performance* (latency/energy/cost) while checking
basic correctness via deterministic smoke tests:
  - code_exec_smoke: code compiles (Python)

Complexity:
  C1 (easy)   : single-step (filter / max / simple groupby)
  C2 (medium) : top-k + aggregation + rename schema
  C3 (hard)   : YoY delta + pivot/merge logic
  C4 (hard+)  : rolling window stats + ranking per state

Output contract (matches your bench.py JSON requirement)
- Return EXACTLY one fenced ```json``` block.
- Put code under code.language="python" and code.content="...".

Run:
  python scripts/gen_seds_family4_sweep.py --repo-root . --bases 10 --row-sizes 20 100 250 500 1000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from _seds_task_utils import build_pool, choose_msn, md_table, sample_distractors, pick_year_with_many_states


def write_yaml(path: Path, tasks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": tasks}, f, sort_keys=False, allow_unicode=True)


def base_rows_for_year(pool: pd.DataFrame, year: int, k: int, rng: random.Random) -> List[Dict[str, object]]:
    year_df = pool[pool["Year"] == year].drop_duplicates(subset=["State"]).copy()
    if year_df.empty:
        raise ValueError("No rows for year")
    base = year_df.sample(min(k, len(year_df)), replace=False, random_state=rng.randint(0, 10**9))
    rows = []
    for r in base.itertuples():
        rows.append({"State": str(r.State), "Year": int(r.Year), "Value": float(r.Value), "Unit": str(getattr(r, "Unit", ""))})
    return rows


def task_C1(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """Single-step: max value in year."""
    rows = base_rows_for_year(pool, year, k=min(12, table_rows), rng=rng)
    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(pool[pool["Year"] == year], need, avoid, rng)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Unit"]

    prompt = (
        f"You are given a table of industrial energy consumption values for {year}. "
        "Write Python (pandas) code that reads this table into a dataframe df with columns "
        "State, Year, Value, Unit and returns the State with the maximum Value (argmax). "
        "Return a dataframe with columns: State, max_value."
    )

    return {
        "task_id": f"seds_f4_C1_r{table_rows}_{idx:03d}",
        "family": "family4_codegen",
        "difficulty": "easy",
        "meta": {"dataset": "SEDS", "complexity": "C1", "table_rows": table_rows, "msn": msn, "year": year},
        "input": prompt,
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"code_should_run": True},
        "scoring": {"deterministic": [{"type": "code_exec_smoke", "language": "python"}]},
    }


def task_C2(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """Top-k + aggregation + rename schema."""
    rows = base_rows_for_year(pool, year, k=min(20, table_rows), rng=rng)
    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(pool[pool["Year"] == year], need, avoid, rng)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Unit"]

    prompt = (
        f"Using the table for {year}, write Python (pandas) code that returns the TOP 3 States "
        "by Value, sorted descending. Rename the Value column to total_billion_btu, "
        "and return a dataframe with columns: State, total_billion_btu."
    )

    return {
        "task_id": f"seds_f4_C2_r{table_rows}_{idx:03d}",
        "family": "family4_codegen",
        "difficulty": "medium",
        "meta": {"dataset": "SEDS", "complexity": "C2", "table_rows": table_rows, "msn": msn, "year": year},
        "input": prompt,
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"code_should_run": True},
        "scoring": {"deterministic": [{"type": "code_exec_smoke", "language": "python"}]},
    }


def task_C3(pool: pd.DataFrame, year1: int, year2: int, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """YoY delta + pivot/merge logic."""
    # Build a table containing two years for multiple states
    d = pool[pool["Year"].isin([year1, year2])].drop_duplicates(subset=["State", "Year"]).copy()
    # sample some states that appear in both years
    counts = d.groupby("State")["Year"].nunique()
    states = counts[counts == 2].index.tolist()
    if len(states) < 8:
        # fall back to any states and allow missing; task still meaningful
        states = d["State"].unique().tolist()

    base_states = rng.sample(states, k=min(10, len(states)))
    base = d[d["State"].isin(base_states)].copy()

    rows = []
    for r in base.itertuples():
        rows.append({"State": str(r.State), "Year": int(r.Year), "Value": float(r.Value), "Unit": str(getattr(r, "Unit", ""))})

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(d, need, avoid, rng)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Unit"]

    prompt = (
        f"Write Python (pandas) code that computes, for each State, the year-over-year change "
        f"delta = Value({year2}) - Value({year1}). Use pivot/merge/groupby as needed. "
        "Return a dataframe with columns: State, delta_value."
    )

    return {
        "task_id": f"seds_f4_C3_r{table_rows}_{idx:03d}",
        "family": "family4_codegen",
        "difficulty": "hard",
        "meta": {"dataset": "SEDS", "complexity": "C3", "table_rows": table_rows, "msn": msn, "year1": year1, "year2": year2},
        "input": prompt,
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"code_should_run": True},
        "scoring": {"deterministic": [{"type": "code_exec_smoke", "language": "python"}]},
    }


def task_C4(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """Rolling window stats + ranking."""
    # sample a state with many years
    counts = pool.groupby("State")["Year"].nunique().sort_values(ascending=False)
    candidates = [s for s, n in counts.items() if int(n) >= 10]
    if not candidates:
        candidates = counts.index.tolist()[:20]
    state = rng.choice(candidates)

    st_df = pool[pool["State"] == state].drop_duplicates(subset=["Year"]).sort_values("Year")
    years = st_df["Year"].astype(int).tolist()
    if len(years) < 8:
        raise ValueError("Not enough years for rolling window")

    # select a contiguous slice
    start = rng.randint(0, max(0, len(years) - 8))
    sel_years = years[start : start + 8]
    core = st_df[st_df["Year"].isin(sel_years)].copy()

    rows = []
    for r in core.itertuples():
        rows.append({"State": str(state), "Year": int(r.Year), "Value": float(r.Value), "Unit": str(getattr(r, "Unit", ""))})

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(pool, need, avoid, rng)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Unit"]

    prompt = (
        f"Write Python (pandas) code that, for State {state}, computes the rolling 3-year mean of Value "
        "ordered by Year, then returns the Year with the maximum rolling mean. "
        "Return a dataframe with columns: Year, rolling_mean_3y."
    )

    return {
        "task_id": f"seds_f4_C4_r{table_rows}_{idx:03d}",
        "family": "family4_codegen",
        "difficulty": "hard+",
        "meta": {"dataset": "SEDS", "complexity": "C4", "table_rows": table_rows, "msn": msn, "state": state, "years": sel_years},
        "input": prompt,
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"code_should_run": True},
        "scoring": {"deterministic": [{"type": "code_exec_smoke", "language": "python"}]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--input", default="data/processed/seds_industrial_consumption.parquet")
    ap.add_argument("--bases", type=int, default=10)
    ap.add_argument("--row-sizes", nargs="+", type=int, default=[20, 100, 250, 500, 1000])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.repo_root).resolve()
    df = pd.read_parquet(root / args.input)
    msn = choose_msn(df)
    pool = build_pool(df, msn)

    year = pick_year_with_many_states(pool, min_states=40)
    years_sorted = sorted(pool["Year"].astype(int).unique().tolist())
    year1 = years_sorted[0]
    year2 = years_sorted[1] if len(years_sorted) > 1 else year1 + 1

    tasks_C1, tasks_C2, tasks_C3, tasks_C4 = [], [], [], []

    for nrows in args.row_sizes:
        for i in range(1, args.bases + 1):
            tasks_C1.append(task_C1(pool, year, nrows, i, msn, rng))
            tasks_C2.append(task_C2(pool, year, nrows, i, msn, rng))
            tasks_C3.append(task_C3(pool, year1, year2, nrows, i, msn, rng))
            tasks_C4.append(task_C4(pool, nrows, i, msn, rng))

    out_dir = root / "tasks" / "family4_codegen"
    write_yaml(out_dir / "seds_f4_C1_sweep.yaml", tasks_C1)
    write_yaml(out_dir / "seds_f4_C2_sweep.yaml", tasks_C2)
    write_yaml(out_dir / "seds_f4_C3_sweep.yaml", tasks_C3)
    write_yaml(out_dir / "seds_f4_C4_sweep.yaml", tasks_C4)

    print("[ok] wrote:")
    for p in ["seds_f4_C1_sweep.yaml", "seds_f4_C2_sweep.yaml", "seds_f4_C3_sweep.yaml", "seds_f4_C4_sweep.yaml"]:
        print("  -", (out_dir / p).as_posix())


if __name__ == "__main__":
    main()
