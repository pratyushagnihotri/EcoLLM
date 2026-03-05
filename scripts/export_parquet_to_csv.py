#!/usr/bin/env python3
"""export_parquet_to_csv.py

Exports benchmark parquet files to CSV.

Looks for (relative to --run-dir):
- runs.parquet
- results.parquet
- leaderboard_by_family.parquet
- leaderboard_by_family_rows.parquet (optional)

Writes CSVs under <run-dir>/csv (or --out-dir).

Usage
    python scripts/export_parquet_to_csv.py --run-dir runs/seds_f1_C1_sweep
    python scripts/export_parquet_to_csv.py --run-dir runs/seds_f1_C1_sweep --out-dir runs/seds_f1_C1_sweep/csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

DEFAULT_FILES = [
    "runs.parquet",
    "results.parquet",
    "leaderboard_by_family.parquet",
    "leaderboard_by_family_rows.parquet",
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory containing parquet files")
    ap.add_argument("--out-dir", default=None, help="Output directory for CSVs (default: <run-dir>/csv)")
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="Parquet filenames to export")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for rel in args.files:
        p = run_dir / rel
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[skip] {p.name} (read failed): {e}")
            continue

        out_path = out_dir / (p.stem + ".csv")
        df.to_csv(out_path, index=False)
        exported += 1
        print(f"[ok] {p.name} -> {out_path}")

    if exported == 0:
        print("[warn] No parquet files exported. Check your --run-dir and filenames.")
    else:
        print(f"[done] Exported {exported} file(s) to {out_dir}")

if __name__ == "__main__":
    main()
