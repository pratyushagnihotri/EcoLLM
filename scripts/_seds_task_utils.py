#!/usr/bin/env python3
"""
Shared helpers for SEDS task generators.

This file is intentionally **backwards compatible** across generator versions.
In particular, `sample_distractors(...)` accepts an optional `rng` argument,
so older generators calling it with 3 args and newer ones with 4 args both work.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional

import pandas as pd


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
    return d


def pick_year_with_many_states(pool: pd.DataFrame, min_states: int = 40) -> int:
    g = pool.groupby("Year")["State"].nunique().sort_values(ascending=False)
    for year, n in g.items():
        if int(n) >= min_states:
            return int(year)
    return int(g.index[0])


def pick_years_for_state(pool: pd.DataFrame, state: str, rng: Optional[random.Random] = None, k: int = 8) -> List[int]:
    """
    Pick up to k years for which we have rows for the given state.
    rng is optional (for reproducibility in generators).
    """
    rr = rng or random
    st_df = pool[pool["State"] == state].copy()
    years = sorted([int(y) for y in st_df["Year"].dropna().unique().tolist()])
    if not years:
        return []
    if len(years) <= k:
        return years
    # sample without replacement but keep sorted order
    picked = rr.sample(years, k)
    return sorted(picked)


def sample_distractors(
    pool: pd.DataFrame,
    k: int,
    avoid_keys: set[Tuple[str, int]],
    rng: Optional[random.Random] = None,
) -> List[Dict[str, object]]:
    """
    Sample k distractor rows from pool, avoiding any (State, Year) in avoid_keys.

    Backwards compatible:
      - older generators call: sample_distractors(pool, k, avoid_keys)
      - newer generators call: sample_distractors(pool, k, avoid_keys, rng)
    """
    rr = rng or random
    out: List[Dict[str, object]] = []
    tries = 0
    # NOTE: using pool.sample(random_state=...) is awkward w/ Random; we use iloc sampling.
    n = len(pool)
    if n == 0:
        return out

    while len(out) < k and tries < k * 40:
        tries += 1
        r = pool.iloc[rr.randrange(0, n)]
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

    # If not enough unique keys, fill with replacement (still ok for table-size scaling)
    if len(out) < k:
        need = k - len(out)
        for _ in range(need):
            r = pool.iloc[rr.randrange(0, n)]
            out.append({
                "State": str(r["State"]),
                "Year": int(r["Year"]),
                "Value": float(r["Value"]),
                "Unit": str(r.get("Unit", "")),
                "Description": str(r.get("Description", "")),
            })
    return out
