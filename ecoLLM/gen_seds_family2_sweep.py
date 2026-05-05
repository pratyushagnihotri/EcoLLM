#!/usr/bin/env python3
"""
Family 2 = anomaly / validation classification over structured state-year histories.

Writes:
  tasks/family2_anomaly/seds_f2_C1_sweep.yaml
  tasks/family2_anomaly/seds_f2_C2_sweep.yaml
  tasks/family2_anomaly/seds_f2_C3_sweep.yaml
  tasks/family2_anomaly/seds_f2_C4_sweep.yaml

Run:
  python scripts/gen_seds_family2_sweep.py --repo-root . --lengths 20 100 250 500 1000 --bases 30
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


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


def row_from_record(r: pd.Series) -> Dict[str, object]:
    return {
        "State": str(r["State"]),
        "Year": int(r["Year"]),
        "Value": float(r["Value"]),
        "Unit": str(r.get("Unit", "")),
        "Description": str(r.get("Description", "")),
        "Report_Text": str(r.get("Report_Text", "")),
    }


def states_with_min_years(pool: pd.DataFrame, min_years: int) -> List[str]:
    counts = pool.groupby("State")["Year"].nunique().sort_values(ascending=False)
    return [str(s) for s, n in counts.items() if int(n) >= min_years]


def history_for_state(pool: pd.DataFrame, state: str, years: int) -> pd.DataFrame:
    d = (
        pool[pool["State"] == state]
        .sort_values("Year")
        .drop_duplicates(subset=["Year"])
        .copy()
    )
    if len(d) > years:
        # select a contiguous slice
        start = random.randint(0, len(d) - years)
        d = d.iloc[start : start + years].copy()
    return d


def operator_meta(complexity: str, operator_pattern: List[str], semantic_condition: str | None = None) -> Dict[str, object]:
    meta = {
        "dataset": "SEDS",
        "complexity": complexity,
        "operator_pattern": operator_pattern,
        "requires_text_interpretation": True,
        "num_relations": 1,
    }
    if semantic_condition is not None:
        meta["semantic_condition"] = semantic_condition
    return meta


# ---------- controlled text templates ----------

NORMAL_TEXTS = [
    "stable demand with normal operating conditions",
    "report indicates normal demand patterns",
    "consumption remained steady without unusual behavior",
    "steady industrial usage with no major disruption",
]

INCREASE_TEXTS = [
    "demand increased due to industrial growth",
    "report indicates increased demand across industrial users",
    "usage surged due to stronger demand",
    "higher demand was observed in the reporting period",
]

DECREASE_TEXTS = [
    "slight decrease due to efficiency improvements",
    "report indicates reduced demand this year",
    "lower usage due to weaker industrial activity",
    "consumption declined under softer demand",
]


def rewrite_text(row: Dict[str, object], mode: str, rng: random.Random) -> Dict[str, object]:
    row = dict(row)
    if mode == "normal":
        row["Report_Text"] = f"{row['State']} {row['Year']} {rng.choice(NORMAL_TEXTS)}."
    elif mode == "increase":
        row["Report_Text"] = f"{row['State']} {row['Year']} {rng.choice(INCREASE_TEXTS)}."
    elif mode == "decrease":
        row["Report_Text"] = f"{row['State']} {row['Year']} {rng.choice(DECREASE_TEXTS)}."
    else:
        raise ValueError(mode)
    return row


# ---------- anomaly injection helpers ----------

def inject_spike(rows: List[Dict[str, object]], idx: int, factor: float = 1.8) -> None:
    rows[idx]["Value"] = float(rows[idx]["Value"]) * factor
    rows[idx]["Report_Text"] = f"{rows[idx]['State']} {rows[idx]['Year']} {random.choice(INCREASE_TEXTS)}."


def inject_drop(rows: List[Dict[str, object]], idx: int, factor: float = 0.45) -> None:
    rows[idx]["Value"] = float(rows[idx]["Value"]) * factor
    rows[idx]["Report_Text"] = f"{rows[idx]['State']} {rows[idx]['Year']} {random.choice(DECREASE_TEXTS)}."


def inject_drift(rows: List[Dict[str, object]], start_idx: int, slope_frac: float = 0.30) -> None:
    n = len(rows)
    base = [float(r["Value"]) for r in rows]
    for i in range(start_idx, n):
        frac = (i - start_idx + 1) / max(1, (n - start_idx))
        rows[i]["Value"] = base[i] * (1.0 + slope_frac * frac)
        rows[i]["Report_Text"] = f"{rows[i]['State']} {rows[i]['Year']} {random.choice(INCREASE_TEXTS)}."


def inject_inconsistent(rows: List[Dict[str, object]], idx: int, factor: float = 1.6) -> None:
    rows[idx]["Value"] = float(rows[idx]["Value"]) * factor
    # force text to disagree with numeric direction
    rows[idx]["Report_Text"] = f"{rows[idx]['State']} {rows[idx]['Year']} {random.choice(DECREASE_TEXTS)}."


def make_rows_from_history(hist: pd.DataFrame, rng: random.Random, default_mode: str = "normal") -> List[Dict[str, object]]:
    rows = [rewrite_text(row_from_record(r), default_mode, rng) for _, r in hist.iterrows()]
    rows = sorted(rows, key=lambda x: (x["State"], x["Year"]))
    return rows


def sample_distractors(pool: pd.DataFrame, k: int, avoid: set[tuple[str, int]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    tries = 0
    while len(out) < k and tries < k * 30:
        tries += 1
        r = pool.sample(1).iloc[0]
        key = (str(r["State"]), int(r["Year"]))
        if key in avoid:
            continue
        out.append(row_from_record(r))
        avoid.add(key)

    if len(out) < k:
        extra = pool.sample(k - len(out), replace=True)
        for _, r in extra.iterrows():
            out.append(row_from_record(r))
    return out


# ---------- task builders ----------

def task_C1(pool: pd.DataFrame, length: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    One state, one obvious anomaly.
    Output: anomaly_type
    """
    candidates = states_with_min_years(pool, min_years=max(6, min(12, length)))
    if not candidates:
        candidates = states_with_min_years(pool, min_years=6)
    state = rng.choice(candidates)

    hist = history_for_state(pool, state, years=min(length, 8))
    rows = make_rows_from_history(hist, rng, default_mode="normal")

    if len(rows) < 6:
        raise ValueError("Not enough rows to construct C1 anomaly task.")

    anomaly_idx = max(2, min(len(rows) - 2, len(rows) // 2))
    inject_spike(rows, anomaly_idx)

    # add distractors if user asked for larger table size
    avoid = {(r["State"], r["Year"]) for r in rows}
    rows += sample_distractors(pool, max(0, length - len(rows)), avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Report_Text"]
    return {
        "task_id": f"seds_f2_C1_L{length}_{idx:03d}",
        "family": "family2_anomaly",
        "difficulty": "easy",
        "meta": {
            **operator_meta("C1", ["feature_extract", "classify"], None),
            "table_rows": int(length),
            "msn": msn,
            "focus_state": state,
        },
        "input": (
            "A state-year history is shown in the table. "
            "Classify the anomaly_type as one of: spike, drop, drift, inconsistent. "
            "Also list two checks to confirm."
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"labels": {"anomaly_type": "spike"}},
        "scoring": {"deterministic": [{"type": "label_match", "label": "anomaly_type"}]},
    }


def task_C2(pool: pd.DataFrame, length: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    Two candidate states, one has anomaly.
    Output: anomaly_type + state
    """
    candidates = states_with_min_years(pool, min_years=6)
    if len(candidates) < 2:
        raise ValueError("Need at least two states for C2.")

    st1, st2 = rng.sample(candidates, 2)

    h1 = history_for_state(pool, st1, years=min(7, max(6, length // 2)))
    h2 = history_for_state(pool, st2, years=min(7, max(6, length // 2)))

    rows1 = make_rows_from_history(h1, rng, default_mode="normal")
    rows2 = make_rows_from_history(h2, rng, default_mode="normal")

    inject_first = rng.random() < 0.5
    if inject_first:
        inject_drop(rows1, max(2, min(len(rows1) - 2, len(rows1) // 2)))
        focus_state = st1
    else:
        inject_drop(rows2, max(2, min(len(rows2) - 2, len(rows2) // 2)))
        focus_state = st2

    rows = rows1 + rows2
    if len(rows) > length:
        rows = rng.sample(rows, length)
    else:
        avoid = {(r["State"], r["Year"]) for r in rows}
        rows += sample_distractors(pool, length - len(rows), avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Report_Text"]
    return {
        "task_id": f"seds_f2_C2_L{length}_{idx:03d}",
        "family": "family2_anomaly",
        "difficulty": "medium",
        "meta": {
            **operator_meta("C2", ["feature_extract", "compare_groups", "classify"], None),
            "table_rows": int(length),
            "msn": msn,
            "candidate_states": [st1, st2],
            "focus_state": focus_state,
        },
        "input": (
            "Two state histories are mixed in the table. "
            "Identify which State most likely contains the anomaly and classify anomaly_type as one of: spike, drop, drift, inconsistent. "
            "Also list two checks to confirm."
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"labels": {"anomaly_type": "drop", "state": focus_state}},
        "scoring": {"deterministic": [
            {"type": "label_match", "label": "anomaly_type"},
            {"type": "label_match", "label": "state"},
        ]},
    }


def task_C3(pool: pd.DataFrame, length: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    Trend anomaly with supporting text.
    Output: anomaly_type
    """
    candidates = states_with_min_years(pool, min_years=7)
    if not candidates:
        raise ValueError("Need states with sufficient history for C3.")
    state = rng.choice(candidates)

    hist = history_for_state(pool, state, years=min(length, 8))
    rows = make_rows_from_history(hist, rng, default_mode="normal")

    if len(rows) < 6:
        raise ValueError("Not enough rows for C3.")

    start_idx = max(2, min(len(rows) - 3, len(rows) // 2))
    inject_drift(rows, start_idx)

    avoid = {(r["State"], r["Year"]) for r in rows}
    rows += sample_distractors(pool, max(0, length - len(rows)), avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Report_Text"]
    return {
        "task_id": f"seds_f2_C3_L{length}_{idx:03d}",
        "family": "family2_anomaly",
        "difficulty": "hard",
        "meta": {
            **operator_meta("C3", ["feature_extract", "trend_reasoning", "classify"], "increase"),
            "table_rows": int(length),
            "msn": msn,
            "focus_state": state,
        },
        "input": (
            "A state-year history is shown. Use both the numeric progression and the Report_Text narratives "
            "to classify anomaly_type as one of: spike, drop, drift, inconsistent. "
            "Also list two checks to confirm."
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"labels": {"anomaly_type": "drift"}},
        "scoring": {"deterministic": [{"type": "label_match", "label": "anomaly_type"}]},
    }


def task_C4(pool: pd.DataFrame, length: int, idx: int, msn: str, rng: random.Random) -> Dict:
    """
    Text-numeric inconsistency.
    Output: anomaly_type
    """
    candidates = states_with_min_years(pool, min_years=6)
    if not candidates:
        raise ValueError("Need states with sufficient history for C4.")
    state = rng.choice(candidates)

    hist = history_for_state(pool, state, years=min(length, 7))
    rows = make_rows_from_history(hist, rng, default_mode="normal")

    if len(rows) < 5:
        raise ValueError("Not enough rows for C4.")

    idx_bad = max(2, min(len(rows) - 2, len(rows) // 2))
    inject_inconsistent(rows, idx_bad)

    avoid = {(r["State"], r["Year"]) for r in rows}
    rows += sample_distractors(pool, max(0, length - len(rows)), avoid)
    rng.shuffle(rows)

    cols = ["State", "Year", "Value", "Report_Text"]
    return {
        "task_id": f"seds_f2_C4_L{length}_{idx:03d}",
        "family": "family2_anomaly",
        "difficulty": "hard+",
        "meta": {
            **operator_meta("C4", ["feature_extract", "text_numeric_consistency", "classify"], "decrease_vs_increase"),
            "table_rows": int(length),
            "msn": msn,
            "focus_state": state,
        },
        "input": (
            "A state-year history is shown. Detect whether there is a data-quality inconsistency between the numeric values "
            "and the Report_Text narratives. Classify anomaly_type as one of: spike, drop, drift, inconsistent. "
            "Also list two checks to confirm."
        ),
        "context": {"type": "inline_table", "table_markdown": md_table(rows, cols)},
        "expected": {"labels": {"anomaly_type": "inconsistent"}},
        "scoring": {"deterministic": [{"type": "label_match", "label": "anomaly_type"}]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--input", default="data/processed/seds_industrial_consumption.parquet")
    ap.add_argument("--bases", type=int, default=30)
    ap.add_argument("--lengths", nargs="+", type=int, default=[20, 100, 250, 500, 1000])
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

    for length in args.lengths:
        for i in range(1, args.bases + 1):
            tasks_C1.append(task_C1(pool, length, i, msn, rng))
            tasks_C2.append(task_C2(pool, length, i, msn, rng))
            tasks_C3.append(task_C3(pool, length, i, msn, rng))
            tasks_C4.append(task_C4(pool, length, i, msn, rng))

    out_dir = root / "tasks" / "family2_anomaly"
    write_yaml(out_dir / "seds_f2_C1_sweep.yaml", tasks_C1)
    write_yaml(out_dir / "seds_f2_C2_sweep.yaml", tasks_C2)
    write_yaml(out_dir / "seds_f2_C3_sweep.yaml", tasks_C3)
    write_yaml(out_dir / "seds_f2_C4_sweep.yaml", tasks_C4)

    print("[ok] wrote:")
    for p in [
        "seds_f2_C1_sweep.yaml",
        "seds_f2_C2_sweep.yaml",
        "seds_f2_C3_sweep.yaml",
        "seds_f2_C4_sweep.yaml",
    ]:
        print("  -", (out_dir / p).as_posix())


if __name__ == "__main__":
    main()