#!/usr/bin/env python3
"""
Family 1 = text-aware tabular reasoning over a single relation.

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
from typing import Dict, List, Tuple

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


def pick_year_with_many_states(pool: pd.DataFrame, min_states: int = 30) -> int:
    g = pool.groupby("Year")["State"].nunique().sort_values(ascending=False)
    for year, n in g.items():
        if int(n) >= min_states:
            return int(year)
    return int(g.index[0])


def row_from_record(r: pd.Series) -> Dict[str, object]:
    return {
        "State": str(r["State"]),
        "Year": int(r["Year"]),
        "Value": float(r["Value"]),
        "Unit": str(r.get("Unit", "")),
        "Description": str(r.get("Description", "")),
        "Report_Text": str(r.get("Report_Text", "")),
    }


def sample_distractors(pool: pd.DataFrame, k: int, avoid_keys: set[Tuple[str, int]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    tries = 0

    while len(out) < k and tries < k * 25:
        tries += 1
        r = pool.sample(1).iloc[0]
        key = (str(r["State"]), int(r["Year"]))
        if key in avoid_keys:
            continue
        out.append(row_from_record(r))
        avoid_keys.add(key)

    if len(out) < k:
        extra = pool.sample(k - len(out), replace=True)
        for _, r in extra.iterrows():
            out.append(row_from_record(r))

    return out


INCREASE_TERMS = ["increase", "increased", "growth", "surged", "higher demand", "elevated"]
NORMAL_TERMS = ["normal", "stable", "steady", "usual", "baseline"]


def contains_any(text: str, terms: List[str]) -> bool:
    t = str(text).lower()
    return any(term.lower() in t for term in terms)


def is_increase_row(row: Dict[str, object]) -> bool:
    return contains_any(str(row.get("Report_Text", "")), INCREASE_TERMS)


def is_normal_row(row: Dict[str, object]) -> bool:
    return contains_any(str(row.get("Report_Text", "")), NORMAL_TERMS)


def filter_rows(rows: List[Dict[str, object]], fn) -> List[Dict[str, object]]:
    return [r for r in rows if fn(r)]


def group_avg_by_state(rows: List[Dict[str, object]]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for r in rows:
        acc.setdefault(str(r["State"]), []).append(float(r["Value"]))
    return {s: (sum(vals) / len(vals)) for s, vals in acc.items() if vals}


def semantic_meta(complexity: str, operator_pattern: List[str], semantic_condition: str) -> Dict[str, object]:
    return {
        "dataset": "SEDS",
        "complexity": complexity,
        "operator_pattern": operator_pattern,
        "requires_text_interpretation": True,
        "num_relations": 1,
        "semantic_condition": semantic_condition,
    }


def rewrite_text(row: Dict[str, object], mode: str, rng: random.Random) -> Dict[str, object]:
    row = dict(row)

    normal_texts = [
        "stable demand with normal operating conditions",
        "report indicates normal demand patterns",
        "steady industrial usage with no major disruption",
        "baseline demand remained stable in the reporting period",
    ]
    increase_texts = [
        "demand increased due to industrial growth",
        "report indicates increased demand across industrial users",
        "usage surged due to stronger demand",
        "higher demand was observed in the reporting period",
    ]

    if mode == "normal":
        txt = rng.choice(normal_texts)
    elif mode == "increase":
        txt = rng.choice(increase_texts)
    else:
        raise ValueError(mode)

    row["Report_Text"] = f"{row['State']} {row['Year']} {txt}."
    return row


def force_semantic_mix(
    rows: List[Dict[str, object]],
    rng: random.Random,
    min_increase: int = 3,
    min_normal: int = 2,
) -> List[Dict[str, object]]:
    """
    Force a subset of rows to contain 'increase' and 'normal' semantic text,
    so generation never depends on the original parquet wording.
    """
    rows = [dict(r) for r in rows]
    n = len(rows)
    if n == 0:
        return rows

    idxs = list(range(n))
    rng.shuffle(idxs)

    inc_n = min(min_increase, n)
    norm_n = min(min_normal, max(0, n - inc_n))

    inc_idxs = idxs[:inc_n]
    norm_idxs = idxs[inc_n:inc_n + norm_n]

    for i in inc_idxs:
        rows[i] = rewrite_text(rows[i], "increase", rng)

    for i in norm_idxs:
        rows[i] = rewrite_text(rows[i], "normal", rng)

    return rows


def task_C1(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    year_df = pool[pool["Year"] == year].copy()
    if len(year_df) < 8:
        year = pick_year_with_many_states(pool, min_states=10)
        year_df = pool[pool["Year"] == year].copy()

    base = year_df.sample(min(12, len(year_df)), replace=False, random_state=rng.randint(0, 10**9))
    rows = [row_from_record(r) for _, r in base.iterrows()]
    rows = force_semantic_mix(rows, rng, min_increase=3, min_normal=2)

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(year_df if len(year_df) > need else pool, need, avoid)
    rows = force_semantic_mix(rows, rng, min_increase=3, min_normal=2)
    rng.shuffle(rows)

    inc_rows = filter_rows(rows, is_increase_row)
    max_val = max(float(r["Value"]) for r in inc_rows)

    cols = ["State", "Year", "Value", "Unit", "Report_Text"]
    return {
        "task_id": f"seds_f1_C1_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "easy",
        "meta": {
            **semantic_meta("C1", ["filter_text", "aggregate"], "increase"),
            "table_rows": table_rows,
            "msn": msn,
            "year": year,
        },
        "input": add_contract_to_input(
            f"From the table for year {year}, consider only rows whose Report_Text indicates increased demand or growth. "
            f"What is the maximum Value among those rows?",
            num_key="max_value",
            label_key=None,
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {
            "numeric_targets": [{"name": "max_value", "value": float(max_val), "tolerance_abs": 1e-9}],
        },
        "scoring": {"deterministic": [{"type": "numeric_extract", "target": "max_value"}]},
    }


def task_C2(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    year_df = pool[pool["Year"] == year].copy().drop_duplicates(subset=["State", "Year"])
    if len(year_df) < 8:
        year = pick_year_with_many_states(pool, min_states=10)
        year_df = pool[pool["Year"] == year].copy().drop_duplicates(subset=["State", "Year"])

    base = year_df.sample(min(18, len(year_df)), replace=False, random_state=rng.randint(0, 10**9))
    rows = [row_from_record(r) for _, r in base.iterrows()]
    rows = force_semantic_mix(rows, rng, min_increase=5, min_normal=3)

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(year_df if len(year_df) > need else pool, need, avoid)
    rows = force_semantic_mix(rows, rng, min_increase=5, min_normal=3)

    inc_rows = filter_rows(rows, is_increase_row)
    grouped = group_avg_by_state(inc_rows)

    if len(grouped) < 3:
        raise ValueError("Could not construct at least 3 increase-states for C2.")

    ranked = sorted(grouped.items(), key=lambda kv: (-kv[1], kv[0]))
    third_state, third_value = ranked[2]

    rng.shuffle(rows)
    cols = ["State", "Year", "Value", "Unit", "Report_Text"]
    return {
        "task_id": f"seds_f1_C2_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "medium",
        "meta": {
            **semantic_meta("C2", ["filter_text", "groupby", "aggregate", "rank"], "increase"),
            "table_rows": table_rows,
            "msn": msn,
            "year": year,
        },
        "input": add_contract_to_input(
            f"For year {year}, keep only rows whose Report_Text indicates increased demand or growth. "
            f"Group them by State, compute AVG(Value), sort descending, and return the 3rd-ranked State and its average Value.",
            num_key="third_avg_value",
            label_key="state",
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {
            "numeric_targets": [{"name": "third_avg_value", "value": float(third_value), "tolerance_abs": 1e-9}],
            "labels": {"state": str(third_state)},
        },
        "scoring": {"deterministic": [
            {"type": "numeric_extract", "target": "third_avg_value"},
            {"type": "label_match", "label": "state"},
        ]},
    }


def task_C3(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    year_df = pool[pool["Year"] == year].copy()
    if len(year_df) < 8:
        year = pick_year_with_many_states(pool, min_states=10)
        year_df = pool[pool["Year"] == year].copy()

    base = year_df.sample(min(18, len(year_df)), replace=False, random_state=rng.randint(0, 10**9))
    rows = [row_from_record(r) for _, r in base.iterrows()]
    rows = force_semantic_mix(rows, rng, min_increase=4, min_normal=4)

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(year_df if len(year_df) > need else pool, need, avoid)
    rows = force_semantic_mix(rows, rng, min_increase=4, min_normal=4)

    normal_rows = filter_rows(rows, is_normal_row)
    ref_avg = sum(float(r["Value"]) for r in normal_rows) / len(normal_rows)
    count_above = sum(1 for r in rows if float(r["Value"]) > ref_avg)

    rng.shuffle(rows)
    cols = ["State", "Year", "Value", "Unit", "Report_Text"]
    return {
        "task_id": f"seds_f1_C3_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "hard",
        "meta": {
            **semantic_meta("C3", ["filter_text", "aggregate", "compare_subset", "filter"], "normal"),
            "table_rows": table_rows,
            "msn": msn,
            "year": year,
        },
        "input": add_contract_to_input(
            f"For year {year}, first compute the average Value over rows whose Report_Text indicates normal or stable demand. "
            f"Then count how many rows in the full table have Value greater than that reference average.",
            num_key="count_above_reference",
            label_key=None,
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {
            "numeric_targets": [{"name": "count_above_reference", "value": float(count_above), "tolerance_abs": 1e-9}],
        },
        "scoring": {"deterministic": [{"type": "numeric_extract", "target": "count_above_reference"}]},
    }


def task_C4(pool: pd.DataFrame, year: int, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    year_df = pool[pool["Year"] == year].copy()
    if len(year_df) < 8:
        year = pick_year_with_many_states(pool, min_states=10)
        year_df = pool[pool["Year"] == year].copy()

    base = year_df.sample(min(25, len(year_df)), replace=False, random_state=rng.randint(0, 10**9))
    rows = [row_from_record(r) for _, r in base.iterrows()]
    rows = force_semantic_mix(rows, rng, min_increase=6, min_normal=3)

    avoid = {(r["State"], r["Year"]) for r in rows}
    need = max(0, table_rows - len(rows))
    rows += sample_distractors(year_df if len(year_df) > need else pool, need, avoid)
    rows = force_semantic_mix(rows, rng, min_increase=6, min_normal=3)

    inc_rows = filter_rows(rows, is_increase_row)
    grouped = group_avg_by_state(inc_rows)

    if len(grouped) < 5:
        raise ValueError("Could not construct at least 5 states for C4.")

    top5 = sorted(grouped.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    best_state, best_avg = top5[0]

    rng.shuffle(rows)
    cols = ["State", "Year", "Value", "Unit", "Report_Text"]
    return {
        "task_id": f"seds_f1_C4_r{table_rows}_{idx:03d}",
        "family": "family1_qa",
        "difficulty": "hard+",
        "meta": {
            **semantic_meta("C4", ["filter_text", "groupby", "aggregate", "rank", "topk"], "increase"),
            "table_rows": table_rows,
            "msn": msn,
            "year": year,
        },
        "input": add_contract_to_input(
            f"For year {year}, keep only rows whose Report_Text indicates increased demand or growth. "
            f"Group by State, compute AVG(Value), sort descending, keep the TOP 5 states, and return the top-ranked State and its average Value.",
            num_key="top_avg_value",
            label_key="state",
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {
            "numeric_targets": [{"name": "top_avg_value", "value": float(best_avg), "tolerance_abs": 1e-9}],
            "labels": {"state": str(best_state)},
        },
        "scoring": {"deterministic": [
            {"type": "numeric_extract", "target": "top_avg_value"},
            {"type": "label_match", "label": "state"},
        ]},
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

    year = pick_year_with_many_states(pool, min_states=30)

    tasks_C1, tasks_C2, tasks_C3, tasks_C4 = [], [], [], []

    for nrows in args.row_sizes:
        for i in range(1, args.bases + 1):
            tasks_C1.append(task_C1(pool, year, nrows, i, msn, rng))
            tasks_C2.append(task_C2(pool, year, nrows, i, msn, rng))
            tasks_C3.append(task_C3(pool, year, nrows, i, msn, rng))
            tasks_C4.append(task_C4(pool, year, nrows, i, msn, rng))

    out_dir = root / "tasks" / "family1_qa"
    write_yaml(out_dir / "seds_f1_C1_sweep.yaml", tasks_C1)
    write_yaml(out_dir / "seds_f1_C2_sweep.yaml", tasks_C2)
    write_yaml(out_dir / "seds_f1_C3_sweep.yaml", tasks_C3)
    write_yaml(out_dir / "seds_f1_C4_sweep.yaml", tasks_C4)

    print("[ok] wrote:")
    for p in [
        "seds_f1_C1_sweep.yaml",
        "seds_f1_C2_sweep.yaml",
        "seds_f1_C3_sweep.yaml",
        "seds_f1_C4_sweep.yaml",
    ]:
        print("  -", (out_dir / p).as_posix())


if __name__ == "__main__":
    main()