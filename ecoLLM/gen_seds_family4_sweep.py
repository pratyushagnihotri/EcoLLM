#!/usr/bin/env python3
"""
Family 4 = query / pipeline program synthesis for semantic data tasks.

Writes:
  tasks/family4_codegen/seds_f4_C1_sweep.yaml
  tasks/family4_codegen/seds_f4_C2_sweep.yaml
  tasks/family4_codegen/seds_f4_C3_sweep.yaml
  tasks/family4_codegen/seds_f4_C4_sweep.yaml

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


# ---------- generic helpers ----------

def write_yaml(path: Path, tasks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": tasks}, f, sort_keys=False, allow_unicode=True)


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

    if "Report_Text" not in d.columns:
        raise ValueError(
            "Input parquet is missing Report_Text. Rebuild your parquet with --add-report-text "
            "or merge external report text first."
        )

    d["Report_Text"] = d["Report_Text"].astype(str).fillna("").str.strip()
    if "Unit" not in d.columns:
        d["Unit"] = ""
    if "Description" not in d.columns:
        d["Description"] = ""

    d = d.dropna(subset=["State", "Year", "Value"])
    d = d[d["State"].str.len().between(2, 3)]
    d = d[d["State"] != "US"]
    d = d[d["Report_Text"].str.len() > 0]
    return d


def row_from_record(r: pd.Series) -> Dict[str, object]:
    return {
        "State": str(r["State"]),
        "Year": int(r["Year"]),
        "Value": float(r["Value"]),
        "Unit": str(r.get("Unit", "")),
        "Description": str(r.get("Description", "")),
        "Report_Text": str(r.get("Report_Text", "")),
    }


def operator_meta(complexity: str, operator_pattern: List[str], semantic_condition: str | None = None, num_relations: int = 1) -> Dict[str, object]:
    meta = {
        "dataset": "SEDS",
        "complexity": complexity,
        "operator_pattern": operator_pattern,
        "requires_text_interpretation": True,
        "num_relations": int(num_relations),
    }
    if semantic_condition is not None:
        meta["semantic_condition"] = semantic_condition
    return meta


# ---------- semantic helpers ----------

HEATWAVE_TERMS = ["heatwave", "extreme heat", "heat", "cooling-related demand"]
GROWTH_TERMS = ["increase", "increased", "growth", "surged", "higher demand", "elevated"]
DECREASE_TERMS = ["decrease", "declined", "reduced", "lower usage"]
NORMAL_TERMS = ["normal", "stable", "steady", "baseline", "usual"]


def contains_any(text: str, terms: List[str]) -> bool:
    t = str(text).lower()
    return any(term.lower() in t for term in terms)


def classify_event_type(report_text: str) -> str:
    t = str(report_text).lower()
    if contains_any(t, HEATWAVE_TERMS):
        return "heatwave"
    if contains_any(t, GROWTH_TERMS):
        return "growth"
    if contains_any(t, DECREASE_TERMS):
        return "decrease"
    return "normal"


def rewrite_text(row: Dict[str, object], mode: str, rng: random.Random) -> Dict[str, object]:
    row = dict(row)

    normal_texts = [
        "stable demand with normal operating conditions",
        "report indicates normal demand patterns",
        "steady industrial usage with no major disruption",
    ]
    growth_texts = [
        "demand increased due to industrial growth",
        "report indicates increased demand across industrial users",
        "usage surged due to stronger demand",
    ]
    heatwave_texts = [
        "heatwave conditions increased cooling-related demand",
        "report mentions heatwave-driven demand pressure",
        "extreme heat contributed to higher consumption",
    ]
    decrease_texts = [
        "slight decrease due to efficiency improvements",
        "report indicates reduced demand this year",
        "lower usage due to weaker industrial activity",
    ]

    if mode == "normal":
        txt = rng.choice(normal_texts)
    elif mode == "growth":
        txt = rng.choice(growth_texts)
    elif mode == "heatwave":
        txt = rng.choice(heatwave_texts)
    elif mode == "decrease":
        txt = rng.choice(decrease_texts)
    else:
        raise ValueError(mode)

    row["Report_Text"] = f"{row['State']} {row['Year']} {txt}."
    return row


def enrich_rows_with_events(rows: List[Dict[str, object]], rng: random.Random, heatwave_ratio: float = 0.20, growth_ratio: float = 0.30) -> List[Dict[str, object]]:
    rows = [dict(r) for r in rows]
    n = len(rows)
    idxs = list(range(n))
    rng.shuffle(idxs)

    n_heat = max(1, int(round(n * heatwave_ratio)))
    n_growth = max(1, int(round(n * growth_ratio)))

    heat_idxs = set(idxs[:n_heat])
    growth_idxs = set(idxs[n_heat:n_heat + n_growth])

    out = []
    for i, r in enumerate(rows):
        if i in heat_idxs:
            out.append(rewrite_text(r, "heatwave", rng))
        elif i in growth_idxs:
            out.append(rewrite_text(r, "growth", rng))
        else:
            out.append(rewrite_text(r, "normal", rng))
    return out


def build_reports_relation(energy_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    reports = []
    for r in energy_rows:
        reports.append({
            "State": r["State"],
            "Year": r["Year"],
            "Event_Type": classify_event_type(str(r["Report_Text"])),
            "Report_Text": str(r["Report_Text"]),
        })
    return reports


def sample_energy_rows(pool: pd.DataFrame, nrows: int, rng: random.Random) -> List[Dict[str, object]]:
    sample_n = min(max(12, nrows), len(pool))
    d = pool.sample(sample_n, random_state=rng.randint(0, 10**9)).copy()
    rows = [row_from_record(r) for _, r in d.iterrows()]
    rows = enrich_rows_with_events(rows, rng=rng, heatwave_ratio=0.20, growth_ratio=0.30)
    rng.shuffle(rows)
    return rows


# ---------- task builders ----------

def make_task(task_id: str, difficulty: str, prompt: str, context: Dict, meta_extra: Dict[str, object]) -> Dict:
    return {
        "task_id": task_id,
        "family": "family4_codegen",
        "difficulty": difficulty,
        "meta": meta_extra,
        "input": prompt,
        "context": context,
        "expected": {"code_should_run": True},
        "scoring": {"deterministic": [{"type": "code_exec_smoke", "language": "python"}]},
    }


def task_C1(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    text filter + aggregate
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)

    prompt = (
        "Return EXACTLY one fenced ```json``` block. Put Python code under code.language='python' and code.content='...'. "
        "Write pandas code that reads the energy table into dataframe df, keeps only rows whose Report_Text indicates increased demand or growth, "
        "and returns the maximum Value among those rows. "
        "Return a dataframe with one column named max_value."
    )

    context = {
        "type": "inline_table",
        "table_markdown": md_table(energy_rows, ["State", "Year", "Value", "Unit", "Report_Text"]),
    }

    return make_task(
        task_id=f"seds_f4_C1_r{table_rows}_{idx:03d}",
        difficulty="easy",
        prompt=prompt,
        context=context,
        meta_extra={
            **operator_meta("C1", ["filter_text", "aggregate"], "increase", num_relations=1),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


def task_C2(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    text filter + groupby + aggregate + rank
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)

    prompt = (
        "Return EXACTLY one fenced ```json``` block. Put Python code under code.language='python' and code.content='...'. "
        "Write pandas code that reads the energy table into dataframe df, keeps only rows whose Report_Text indicates increased demand or growth, "
        "groups by State, computes AVG(Value), sorts descending, keeps the TOP 3 states, "
        "and returns a dataframe with columns State and avg_value."
    )

    context = {
        "type": "inline_table",
        "table_markdown": md_table(energy_rows, ["State", "Year", "Value", "Unit", "Report_Text"]),
    }

    return make_task(
        task_id=f"seds_f4_C2_r{table_rows}_{idx:03d}",
        difficulty="medium",
        prompt=prompt,
        context=context,
        meta_extra={
            **operator_meta("C2", ["filter_text", "groupby", "aggregate", "rank", "topk"], "increase", num_relations=1),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


def task_C3(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    join + filter + aggregate
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)
    report_rows = build_reports_relation(energy_rows)

    prompt = (
        "Return EXACTLY one fenced ```json``` block. Put Python code under code.language='python' and code.content='...'. "
        "Write pandas code that reads two tables: energy_df from the energy table and reports_df from the reports table. "
        "Join them on State and Year, keep only joined rows where Event_Type == 'heatwave', "
        "group by State, compute AVG(Value), and return a dataframe with columns State and avg_value."
    )

    context = {
        "type": "multi_table",
        "tables": {
            "energy": md_table(energy_rows, ["State", "Year", "Value", "Unit", "Report_Text"]),
            "reports": md_table(report_rows, ["State", "Year", "Event_Type", "Report_Text"]),
        },
    }

    return make_task(
        task_id=f"seds_f4_C3_r{table_rows}_{idx:03d}",
        difficulty="hard",
        prompt=prompt,
        context=context,
        meta_extra={
            **operator_meta("C3", ["join", "filter", "groupby", "aggregate"], "heatwave", num_relations=2),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


def task_C4(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    join + filter + groupby + aggregate + rank + top-k
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)
    report_rows = build_reports_relation(energy_rows)

    prompt = (
        "Return EXACTLY one fenced ```json``` block. Put Python code under code.language='python' and code.content='...'. "
        "Write pandas code that reads energy_df and reports_df, joins them on State and Year, "
        "keeps only joined rows where Event_Type is 'growth' or 'heatwave', "
        "groups by State, computes AVG(Value), sorts descending, keeps the TOP 5 states, "
        "and returns a dataframe with columns State and avg_value."
    )

    context = {
        "type": "multi_table",
        "tables": {
            "energy": md_table(energy_rows, ["State", "Year", "Value", "Unit", "Report_Text"]),
            "reports": md_table(report_rows, ["State", "Year", "Event_Type", "Report_Text"]),
        },
    }

    return make_task(
        task_id=f"seds_f4_C4_r{table_rows}_{idx:03d}",
        difficulty="hard+",
        prompt=prompt,
        context=context,
        meta_extra={
            **operator_meta("C4", ["join", "filter", "groupby", "aggregate", "rank", "topk"], "growth_or_heatwave", num_relations=2),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


# ---------- main ----------

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
    in_path = root / args.input
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_parquet(in_path)
    required = {"MSN", "State", "Year", "Value", "Report_Text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}. Has: {df.columns.tolist()}")

    msn = choose_msn(df)
    pool = build_pool(df, msn)
    print(f"[ok] using MSN={msn} pool_rows={len(pool):,}")

    tasks_C1: List[Dict] = []
    tasks_C2: List[Dict] = []
    tasks_C3: List[Dict] = []
    tasks_C4: List[Dict] = []

    for nrows in args.row_sizes:
        for i in range(1, args.bases + 1):
            tasks_C1.append(task_C1(pool, nrows, i, msn, rng))
            tasks_C2.append(task_C2(pool, nrows, i, msn, rng))
            tasks_C3.append(task_C3(pool, nrows, i, msn, rng))
            tasks_C4.append(task_C4(pool, nrows, i, msn, rng))

    out_dir = root / "tasks" / "family4_codegen"
    write_yaml(out_dir / "seds_f4_C1_sweep.yaml", tasks_C1)
    write_yaml(out_dir / "seds_f4_C2_sweep.yaml", tasks_C2)
    write_yaml(out_dir / "seds_f4_C3_sweep.yaml", tasks_C3)
    write_yaml(out_dir / "seds_f4_C4_sweep.yaml", tasks_C4)

    print("[ok] wrote:")
    for p in [
        "seds_f4_C1_sweep.yaml",
        "seds_f4_C2_sweep.yaml",
        "seds_f4_C3_sweep.yaml",
        "seds_f4_C4_sweep.yaml",
    ]:
        print("  -", (out_dir / p).as_posix())


if __name__ == "__main__":
    main()