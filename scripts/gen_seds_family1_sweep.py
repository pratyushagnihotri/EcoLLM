#!/usr/bin/env python3
"""
Generate Family-1 SEDS QA sweep tasks with:
- complexities: C1..C4
- table sizes: configurable (default: 20,100,250,500,1000)
- deterministic expected values for numeric scoring

Reads:
  data/processed/seds_industrial_consumption.parquet

Writes:
  tasks/family1_qa/seds_f1_C1_sweep.yaml
  tasks/family1_qa/seds_f1_C2_sweep.yaml
  tasks/family1_qa/seds_f1_C3_sweep.yaml
  tasks/family1_qa/seds_f1_C4_sweep.yaml

Run:
  python scripts/gen_seds_family1_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

STRICT_JSON_TEMPLATE = """\
STRICT OUTPUT REQUIREMENTS:
Return EXACTLY one fenced ```json``` block (no extra text outside it) with keys:
- answer: string
- numbers: object
- label: object
- code: object
- checks: array
- evidence: array

IMPORTANT:
- numbers.{num_key} MUST be present and MUST be a JSON number (not a string).
{label_line}
- Do NOT leave numbers empty. If you cannot compute, set numbers.{num_key} to null and explain in answer.
"""

def add_contract_to_input(base_input: str, num_key: str, label_key: str | None = None) -> str:
    label_line = f"- label.{label_key} MUST be present as a string.\n" if label_key else ""
    contract = STRICT_JSON_TEMPLATE.format(num_key=num_key, label_line=label_line)
    return base_input + "\n\n" + contract


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
        raise ValueError("No MSN values found in input parquet.")
    return str(vc.index[0])


def build_pool(df: pd.DataFrame, msn: str) -> pd.DataFrame:
    d = df[df["MSN"].astype(str) == msn].copy()
    d["State"] = d["State"].astype(str).str.strip()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d.dropna(subset=["State", "Year", "Value"])
    d = d[d["State"].str.len().between(2, 3)]
    d = d[d["State"] != "US"]   # drop national total; keep only states
    return d


def pick_year_with_many_states(pool: pd.DataFrame, min_states: int = 40) -> int:
    g = pool.groupby("Year")["State"].nunique().sort_values(ascending=False)
    for year, n in g.items():
        if int(n) >= min_states:
            return int(year)
    return int(g.index[0])


def sample_distractors(pool: pd.DataFrame, k: int, avoid_keys: set[Tuple[str, int]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    tries = 0
    while len(out) < k and tries < k * 20:
        tries += 1
        r = pool.sample(1).iloc[0]
        key = (str(r["State"]), int(r["Year"]))
        if key in avoid_keys:
            continue
        out.append({
            "State": str(r["State"]),
            "Year": int(r["Year"]),
            "Value": float(r["Value"]),
            "Unit": str(r.get("Unit", "")),
            "Description": str(r.get("Description", "")),
        })
        avoid_keys.add(key)
    if len(out) < k:
        extra = pool.sample(k - len(out), replace=True)
        for _, r in extra.iterrows():
            out.append({
                "State": str(r["State"]),
                "Year": int(r["Year"]),
                "Value": float(r["Value"]),
                "Unit": str(r.get("Unit", "")),
                "Description": str(r.get("Description", "")),
            })
    return out


def task_C1(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str) -> Dict:
    year_df = pool[pool["Year"] == year].copy()
    base = year_df.sample(min(10, len(year_df)), replace=False).drop_duplicates(subset=["State"]).head(10)
    rows = [{
        "State": str(r.State), "Year": int(r.Year), "Value": float(r.Value),
        "Unit": str(getattr(r, "Unit", "")), "Description": str(getattr(r, "Description", ""))
    } for r in base.itertuples()]

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    distract_pool = year_df if len(year_df) > need else pool
    rows += sample_distractors(distract_pool, need, avoid)
    random.shuffle(rows)

    vmax = max(r["Value"] for r in rows)
    st = next(r["State"] for r in rows if r["Value"] == vmax)

    cols = ["State", "Year", "Value", "Unit"]
    return {
        "task_id": f"seds_f1_C1_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "easy",
        "meta": {"dataset": "SEDS", "complexity": "C1", "table_rows": table_rows, "msn": msn, "year": year},
        "input": add_contract_to_input(f"From the table, which State has the highest industrial energy consumption in {year}? Also report the Value.", num_key="max_value", label_key="state",),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {
            "numeric_targets": [{"name": "max_value", "value": float(vmax), "tolerance_abs": 1e-9}],
            "labels": {"state": st},
        },
        "scoring": {"deterministic": [
            {"type": "numeric_extract", "target": "max_value"},
            {"type": "label_match", "label": "state"}
        ]},
    }


def task_C2(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str) -> Dict:
    year_df = pool[pool["Year"] == year].copy().drop_duplicates(subset=["State"])
    if len(year_df) < 3:
        year = pick_year_with_many_states(pool, min_states=10)
        year_df = pool[pool["Year"] == year].copy().drop_duplicates(subset=["State"])
    base = year_df.sample(min(15, len(year_df)), replace=False)
    rows = [{
        "State": str(r.State), "Year": int(r.Year), "Value": float(r.Value),
        "Unit": str(getattr(r, "Unit", "")), "Description": str(getattr(r, "Description", ""))
    } for r in base.itertuples()]

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(year_df if len(year_df) > need else pool, need, avoid)
    random.shuffle(rows)

    sorted_rows = sorted(rows, key=lambda r: (-r["Value"], r["State"]))
    third = sorted_rows[2]
    cols = ["State", "Year", "Value", "Unit"]
    return {
        "task_id": f"seds_f1_C2_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "medium",
        "meta": {"dataset": "SEDS", "complexity": "C2", "table_rows": table_rows, "msn": msn, "year": year},
        "input": add_contract_to_input(
            f"In {year}, what is the 3rd-highest industrial energy consumption Value in the table, and which State has it?",
            num_key="third_value",
            label_key="state",
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {
            "numeric_targets": [{"name": "third_value", "value": float(third['Value']), "tolerance_abs": 1e-9}],
            "labels": {"state": str(third["State"])},
        },
        "scoring": {"deterministic": [
            {"type": "numeric_extract", "target": "third_value"},
            {"type": "label_match", "label": "state"}
        ]},
    }


def task_C3(pool: pd.DataFrame, year1: int, year2: int, table_rows: int, idx: int, msn: str) -> Dict:
    pivot = pool[pool["Year"].isin([year1, year2])].copy()
    counts = pivot.groupby("State")["Year"].nunique()
    candidates = counts[counts == 2].index.tolist()
    if not candidates:
        years = sorted([int(y) for y in pool["Year"].dropna().unique().tolist()])
        for i in range(len(years) - 1):
            y1, y2 = years[i], years[i + 1]
            pivot = pool[pool["Year"].isin([y1, y2])].copy()
            counts = pivot.groupby("State")["Year"].nunique()
            cand = counts[counts == 2].index.tolist()
            if cand:
                year1, year2 = y1, y2
                candidates = cand
                break
    st = random.choice(candidates)
    a = pool[(pool["State"] == st) & (pool["Year"] == year1)].iloc[0]
    b = pool[(pool["State"] == st) & (pool["Year"] == year2)].iloc[0]
    delta = float(b["Value"]) - float(a["Value"])

    core_rows = [
        {"State": st, "Year": int(year1), "Value": float(a["Value"]), "Unit": str(a.get("Unit","")), "Description": str(a.get("Description",""))},
        {"State": st, "Year": int(year2), "Value": float(b["Value"]), "Unit": str(b.get("Unit","")), "Description": str(b.get("Description",""))},
    ]
    avoid = {(st, int(year1)), (st, int(year2))}
    need = max(0, table_rows - len(core_rows))
    rows = core_rows + sample_distractors(pool, need, avoid)
    random.shuffle(rows)
    cols = ["State", "Year", "Value", "Unit"]
    return {
        "task_id": f"seds_f1_C3_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "medium",
        "meta": {"dataset": "SEDS", "complexity": "C3", "table_rows": table_rows, "msn": msn, "year1": year1, "year2": year2, "state": st},
        "input": add_contract_to_input(
            f"Compute the change in industrial energy consumption for State {st} from {year1} to {year2} (Value_{year2} - Value_{year1}).",
            num_key="delta_value",
            label_key=None,
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"numeric_targets": [{"name": "delta_value", "value": float(delta), "tolerance_abs": 1e-9}]},
        "scoring": {"deterministic": [{"type": "numeric_extract", "target": "delta_value"}]},
    }


def task_C4(pool: pd.DataFrame, table_rows: int, idx: int, msn: str) -> Dict:
    counts = pool.groupby("State")["Year"].nunique().sort_values(ascending=False)
    st = str(counts.index[0])
    st_df = pool[pool["State"] == st].copy().sort_values("Year")
    years = [int(y) for y in st_df["Year"].dropna().unique().tolist()]
    if len(years) < 5 and len(counts.index) > 1:
        st = str(counts.index[1])
        st_df = pool[pool["State"] == st].copy().sort_values("Year")
        years = [int(y) for y in st_df["Year"].dropna().unique().tolist()]
    start_i = random.randint(0, max(0, len(years) - 5))
    sel_years = years[start_i:start_i + 5]
    core = st_df[st_df["Year"].isin(sel_years)].drop_duplicates(subset=["Year"]).copy()
    avg_val = float(core["Value"].mean())

    core_rows = [{
        "State": st, "Year": int(r.Year), "Value": float(r.Value),
        "Unit": str(getattr(r, "Unit", "")), "Description": str(getattr(r, "Description", ""))
    } for r in core.itertuples()]

    avoid = {(st, int(y)) for y in sel_years}
    need = max(0, table_rows - len(core_rows))
    rows = core_rows + sample_distractors(pool, need, avoid)
    random.shuffle(rows)
    cols = ["State", "Year", "Value", "Unit"]
    y_min, y_max = min(sel_years), max(sel_years)
    return {
        "task_id": f"seds_f1_C4_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "hard",
        "meta": {"dataset": "SEDS", "complexity": "C4", "table_rows": table_rows, "msn": msn, "state": st, "years": sel_years},
        "input": add_contract_to_input(
            f"For State {st}, compute the average industrial energy consumption Value across the years {y_min}..{y_max} shown in the table.",
            num_key="avg_value",
            label_key=None,
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"numeric_targets": [{"name": "avg_value", "value": float(avg_val), "tolerance_abs": 1e-9}]},
        "scoring": {"deterministic": [{"type": "numeric_extract", "target": "avg_value"}]},
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
    ap.add_argument("--row-sizes", nargs="+", type=int, default=[20, 100, 250, 500, 1000])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.repo_root).resolve()
    in_path = root / args.input
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_parquet(in_path)
    required = {"MSN", "State", "Year", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}. Has: {df.columns.tolist()}")

    msn = choose_msn(df)
    pool = build_pool(df, msn)
    print(f"[ok] using MSN={msn} pool_rows={len(pool):,}")

    year = pick_year_with_many_states(pool, min_states=40)
    years_sorted = sorted([int(y) for y in pool["Year"].dropna().unique().tolist()])
    y1, y2 = years_sorted[0], years_sorted[1] if len(years_sorted) > 1 else (year, year + 1)

    tasks_C1, tasks_C2, tasks_C3, tasks_C4 = [], [], [], []

    for nrows in args.row_sizes:
        for i in range(1, args.bases + 1):
            tasks_C1.append(task_C1(pool, year, nrows, i, msn))
            tasks_C2.append(task_C2(pool, year, nrows, i, msn))
            tasks_C3.append(task_C3(pool, y1, y2, nrows, i, msn))
            tasks_C4.append(task_C4(pool, nrows, i, msn))

    out_dir = root / "tasks" / "family1_qa"
    write_yaml(out_dir / "seds_f1_C1_sweep.yaml", tasks_C1)
    write_yaml(out_dir / "seds_f1_C2_sweep.yaml", tasks_C2)
    write_yaml(out_dir / "seds_f1_C3_sweep.yaml", tasks_C3)
    write_yaml(out_dir / "seds_f1_C4_sweep.yaml", tasks_C4)

    print("[ok] wrote:")
    for p in ["seds_f1_C1_sweep.yaml", "seds_f1_C2_sweep.yaml", "seds_f1_C3_sweep.yaml", "seds_f1_C4_sweep.yaml"]:
        print("  -", (out_dir / p).as_posix())


if __name__ == "__main__":
    main()
