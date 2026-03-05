#!/usr/bin/env python3
"""Generate SEDS Family-2 anomaly detection tasks (C1..C4) with length sweeps.

Family 2 = anomaly classification on a time series.
Complexity:
  C1 (easy)   : single clear spike
  C2 (medium) : sudden drop
  C3 (hard)   : gradual drift (trend change)
  C4 (hard+)  : flatline / sensor stuck (near-constant)

Notes
- Uses SEDS data only to pick a realistic scale for base consumption values.
- Series values are synthetic but seeded from real magnitudes.
- Context uses inline_timeseries points: [timestamp, value].
- Scoring: deterministic label_match on expected.labels.anomaly_type.

Run:
  python scripts/gen_seds_family2_sweep.py --repo-root . --bases 30 --lengths 20 100 250 500 1000
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from _seds_task_utils import choose_msn, build_pool


def write_yaml(path: Path, tasks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": tasks}, f, sort_keys=False, allow_unicode=True)


def ts_points(n: int, start: datetime, step: timedelta, values: List[float]) -> List[List[object]]:
    out: List[List[object]] = []
    t = start
    for i in range(n):
        out.append([t.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"), float(values[i])])
        t = t + step
    return out


def gen_base(n: int, base: float, noise_frac: float, rng: random.Random) -> List[float]:
    vals = []
    for _ in range(n):
        noise = (rng.random() * 2 - 1) * noise_frac  # [-noise_frac, +noise_frac]
        vals.append(base * (1.0 + noise))
    return vals


def inject_spike(vals: List[float], idx: int, factor: float) -> None:
    vals[idx] = vals[idx] * factor


def inject_drop(vals: List[float], idx: int, factor: float) -> None:
    vals[idx] = vals[idx] * factor


def inject_drift(vals: List[float], start_idx: int, slope_frac: float) -> None:
    n = len(vals)
    for i in range(start_idx, n):
        frac = (i - start_idx) / max(1, (n - 1 - start_idx))
        vals[i] = vals[i] * (1.0 + slope_frac * frac)


def inject_flatline(vals: List[float], start_idx: int, level: float) -> None:
    for i in range(start_idx, len(vals)):
        vals[i] = level


def make_task(task_id: str, complexity: str, anomaly_type: str, length: int, series: List[List[object]], msn: str, base_state: str) -> Dict:
    diff = {"C1": "easy", "C2": "medium", "C3": "hard", "C4": "hard+"}[complexity]
    return {
        "task_id": task_id,
        "family": "family2_anomaly",
        "difficulty": diff,
        "meta": {
            "dataset": "SEDS",
            "complexity": complexity,
            "table_rows": int(length),
            "msn": msn,
            "state_seed": base_state,
        },
        "input": (
            "An alert was raised for unusual behavior in the time series. "
            "Classify the anomaly_type as one of: spike, drop, drift, flatline. "
            "Also list two checks to confirm."
        ),
        "context": {
            "type": "inline_timeseries",
            "series_name": "industrial_energy_consumption",
            "points": series,
        },
        "expected": {"labels": {"anomaly_type": anomaly_type.lower()}},
        "scoring": {"deterministic": [{"type": "label_match", "label": "anomaly_type"}]},
    }


def sample_base_level(pool: pd.DataFrame, rng: random.Random) -> Tuple[str, float]:
    r = pool.sample(1, random_state=rng.randint(0, 10**9)).iloc[0]
    return str(r["State"]), float(r["Value"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--input", default="data/processed/seds_industrial_consumption.parquet")
    ap.add_argument("--bases", type=int, default=30)
    ap.add_argument("--lengths", nargs="+", type=int, default=[20, 100, 250, 500, 1000])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    df = pd.read_parquet(root / args.input)
    msn = choose_msn(df)
    pool = build_pool(df, msn, states_only=True)

    rng = random.Random(args.seed)

    out_dir = root / "tasks" / "family2_anomaly"

    tasks_C1: List[Dict] = []
    tasks_C2: List[Dict] = []
    tasks_C3: List[Dict] = []
    tasks_C4: List[Dict] = []

    for length in args.lengths:
        for i in range(1, args.bases + 1):
            base_state, base_val = sample_base_level(pool, rng)
            base_level = max(50.0, base_val)  # keep positive, realistic
            vals = gen_base(length, base_level, noise_frac=0.03, rng=rng)

            start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            step = timedelta(minutes=15)

            # C1: spike
            v1 = vals.copy()
            idx = rng.randint(max(2, length // 5), min(length - 3, (4 * length) // 5))
            inject_spike(v1, idx, factor=2.5)
            tasks_C1.append(
                make_task(
                    f"seds_f2_C1_L{length}_{i:03d}",
                    "C1",
                    "spike",
                    length,
                    ts_points(length, start, step, v1),
                    msn,
                    base_state,
                )
            )

            # C2: drop
            v2 = vals.copy()
            idx = rng.randint(max(2, length // 5), min(length - 3, (4 * length) // 5))
            inject_drop(v2, idx, factor=0.25)
            tasks_C2.append(
                make_task(
                    f"seds_f2_C2_L{length}_{i:03d}",
                    "C2",
                    "drop",
                    length,
                    ts_points(length, start, step, v2),
                    msn,
                    base_state,
                )
            )

            # C3: drift
            v3 = vals.copy()
            idx = rng.randint(max(2, length // 4), min(length - 5, (2 * length) // 3))
            inject_drift(v3, idx, slope_frac=0.6)
            tasks_C3.append(
                make_task(
                    f"seds_f2_C3_L{length}_{i:03d}",
                    "C3",
                    "drift",
                    length,
                    ts_points(length, start, step, v3),
                    msn,
                    base_state,
                )
            )

            # C4: flatline
            v4 = vals.copy()
            idx = rng.randint(max(2, length // 4), min(length - 5, (2 * length) // 3))
            level = float(v4[idx])
            inject_flatline(v4, idx, level=level)
            tasks_C4.append(
                make_task(
                    f"seds_f2_C4_L{length}_{i:03d}",
                    "C4",
                    "flatline",
                    length,
                    ts_points(length, start, step, v4),
                    msn,
                    base_state,
                )
            )

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
