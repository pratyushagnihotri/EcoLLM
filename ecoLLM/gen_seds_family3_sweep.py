#!/usr/bin/env python3
"""
Family 3 = multi-step analytics over multiple relations.

Writes:
  tasks/family3_forecast/seds_f3_C1_sweep.yaml
  tasks/family3_forecast/seds_f3_C2_sweep.yaml
  tasks/family3_forecast/seds_f3_C3_sweep.yaml
  tasks/family3_forecast/seds_f3_C4_sweep.yaml

Run:
  python scripts/gen_seds_family3_sweep.py --repo-root . --bases 30 --row-sizes 20 100 250 500 1000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

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


def operator_meta(complexity: str, operator_pattern: List[str], semantic_condition: str | None = None) -> Dict[str, object]:
    meta = {
        "dataset": "SEDS",
        "complexity": complexity,
        "operator_pattern": operator_pattern,
        "requires_text_interpretation": True,
        "num_relations": 2,
    }
    if semantic_condition is not None:
        meta["semantic_condition"] = semantic_condition
    return meta


# ---------- semantic helpers ----------

HEATWAVE_TERMS = ["heatwave", "extreme heat", "heat", "cooling-related demand"]
GROWTH_TERMS = ["increase", "increased", "growth", "surged", "higher demand", "elevated"]
DECREASE_TERMS = ["decrease", "declined", "reduced", "lower usage", "weaker demand"]
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


def enrich_rows_with_events(rows: List[Dict[str, object]], rng: random.Random, heatwave_ratio: float = 0.20, growth_ratio: float = 0.25) -> List[Dict[str, object]]:
    """
    Deterministically-ish enrich rows with event-oriented report text so that
    the downstream reports relation has useful subsets.
    """
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


def joined_rows(energy_rows: List[Dict[str, object]], report_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rep_idx = {(r["State"], r["Year"]): r for r in report_rows}
    out = []
    for e in energy_rows:
        key = (e["State"], e["Year"])
        if key in rep_idx:
            r = rep_idx[key]
            out.append({
                "State": e["State"],
                "Year": e["Year"],
                "Value": float(e["Value"]),
                "Unit": e.get("Unit", ""),
                "Energy_Report_Text": e.get("Report_Text", ""),
                "Event_Type": r["Event_Type"],
                "Report_Text": r["Report_Text"],
            })
    return out


def group_avg_by_state(rows: List[Dict[str, object]]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for r in rows:
        acc.setdefault(str(r["State"]), []).append(float(r["Value"]))
    return {s: (sum(vs) / len(vs)) for s, vs in acc.items() if vs}


# ---------- task builders ----------

def make_task(
    task_id: str,
    family: str,
    difficulty: str,
    energy_rows: List[Dict[str, object]],
    report_rows: List[Dict[str, object]],
    prompt: str,
    expected: Dict,
    meta_extra: Dict[str, object],
) -> Dict:
    return {
        "task_id": task_id,
        "family": family,
        "difficulty": difficulty,
        "meta": meta_extra,
        "input": prompt,
        "context": {
            "type": "multi_table",
            "tables": {
                "energy": md_table(energy_rows, ["State", "Year", "Value", "Unit", "Report_Text"]),
                "reports": md_table(report_rows, ["State", "Year", "Event_Type", "Report_Text"]),
            },
        },
        "expected": expected,
        "scoring": {"deterministic": expected.pop("_scoring")},
    }


def task_C1(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    Join + filter
    Count distinct states in the heatwave subset.
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)
    report_rows = build_reports_relation(energy_rows)
    joined = joined_rows(energy_rows, report_rows)

    heatwave_states = sorted({r["State"] for r in joined if r["Event_Type"] == "heatwave"})
    count_states = len(heatwave_states)

    expected = {
        "numeric_targets": [{"name": "heatwave_state_count", "value": float(count_states), "tolerance_abs": 1e-9}],
        "_scoring": [{"type": "numeric_extract", "target": "heatwave_state_count"}],
    }

    return make_task(
        task_id=f"seds_f3_C1_r{table_rows}_{idx:03d}",
        family="family3_forecast",
        difficulty="easy",
        energy_rows=energy_rows,
        report_rows=report_rows,
        prompt=(
            "Join the energy table and the reports table on (State, Year). "
            "Keep only joined rows where reports.Event_Type = heatwave. "
            "Count the number of DISTINCT states in the joined result."
        ),
        expected=expected,
        meta_extra={
            **operator_meta("C1", ["join", "filter"], "heatwave"),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


def task_C2(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    Join + filter + groupby + aggregate + rank
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)
    report_rows = build_reports_relation(energy_rows)
    joined = joined_rows(energy_rows, report_rows)

    subset = [r for r in joined if r["Event_Type"] in ("growth", "heatwave")]
    grouped = group_avg_by_state(subset)
    if not grouped:
        raise ValueError("Could not build growth/heatwave subset for C2.")

    ranked = sorted(grouped.items(), key=lambda kv: (-kv[1], kv[0]))
    top_state, top_avg = ranked[0]

    expected = {
        "numeric_targets": [{"name": "avg_value", "value": float(top_avg), "tolerance_abs": 1e-9}],
        "labels": {"state": str(top_state)},
        "_scoring": [
            {"type": "numeric_extract", "target": "avg_value"},
            {"type": "label_match", "label": "state"},
        ],
    }

    return make_task(
        task_id=f"seds_f3_C2_r{table_rows}_{idx:03d}",
        family="family3_forecast",
        difficulty="medium",
        energy_rows=energy_rows,
        report_rows=report_rows,
        prompt=(
            "Join the energy table and the reports table on (State, Year). "
            "Keep only joined rows where Event_Type is growth or heatwave. "
            "Group by State, compute AVG(Value), sort descending, and return the top-ranked State and its average Value."
        ),
        expected=expected,
        meta_extra={
            **operator_meta("C2", ["join", "filter", "groupby", "aggregate", "rank"], "growth_or_heatwave"),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


def task_C3(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    Join + subset comparison
    Return heatwave_avg - normal_avg.
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)
    report_rows = build_reports_relation(energy_rows)
    joined = joined_rows(energy_rows, report_rows)

    heat_vals = [float(r["Value"]) for r in joined if r["Event_Type"] == "heatwave"]
    normal_vals = [float(r["Value"]) for r in joined if r["Event_Type"] == "normal"]

    if not heat_vals or not normal_vals:
        raise ValueError("Could not construct both heatwave and normal subsets for C3.")

    heat_avg = sum(heat_vals) / len(heat_vals)
    normal_avg = sum(normal_vals) / len(normal_vals)
    diff = heat_avg - normal_avg

    expected = {
        "numeric_targets": [{"name": "difference_value", "value": float(diff), "tolerance_abs": 1e-9}],
        "_scoring": [{"type": "numeric_extract", "target": "difference_value"}],
    }

    return make_task(
        task_id=f"seds_f3_C3_r{table_rows}_{idx:03d}",
        family="family3_forecast",
        difficulty="hard",
        energy_rows=energy_rows,
        report_rows=report_rows,
        prompt=(
            "Join the energy table and the reports table on (State, Year). "
            "Compute AVG(Value) over the joined heatwave subset and AVG(Value) over the joined normal subset. "
            "Return heatwave_avg - normal_avg."
        ),
        expected=expected,
        meta_extra={
            **operator_meta("C3", ["join", "filter", "aggregate", "compare_subsets"], "heatwave_vs_normal"),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


def task_C4(pool: pd.DataFrame, table_rows: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    Join + filter + groupby + aggregate + rank + top-k
    Return the 3rd-ranked state in top-5.
    """
    energy_rows = sample_energy_rows(pool, table_rows, rng)
    report_rows = build_reports_relation(energy_rows)
    joined = joined_rows(energy_rows, report_rows)

    subset = [r for r in joined if r["Event_Type"] in ("growth", "heatwave")]
    grouped = group_avg_by_state(subset)
    if len(grouped) < 5:
        raise ValueError("Could not construct at least 5 states for C4.")

    top5 = sorted(grouped.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    third_state, third_avg = top5[2]

    expected = {
        "numeric_targets": [{"name": "third_avg_value", "value": float(third_avg), "tolerance_abs": 1e-9}],
        "labels": {"state": str(third_state)},
        "_scoring": [
            {"type": "numeric_extract", "target": "third_avg_value"},
            {"type": "label_match", "label": "state"},
        ],
    }

    return make_task(
        task_id=f"seds_f3_C4_r{table_rows}_{idx:03d}",
        family="family3_forecast",
        difficulty="hard+",
        energy_rows=energy_rows,
        report_rows=report_rows,
        prompt=(
            "Join the energy table and the reports table on (State, Year). "
            "Keep only joined rows where Event_Type is growth or heatwave. "
            "Group by State, compute AVG(Value), sort descending, keep the TOP 5 states, "
            "and return the 3rd-ranked State and its average Value."
        ),
        expected=expected,
        meta_extra={
            **operator_meta("C4", ["join", "filter", "groupby", "aggregate", "rank", "topk"], "growth_or_heatwave"),
            "table_rows": table_rows,
            "msn": msn,
        },
    )


# ---------- main ----------

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

    out_dir = root / "tasks" / "family3_forecast"
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