#!/usr/bin/env python3
"""
SEDS Fetch, Enrichment, and LLM-Ready Preparation Script.

Output:
- data/processed/seds_enriched.parquet
- (optional) data/processed/seds_industrial_consumption.parquet

Usage:
Basic run (download + process):
    python seds_fetch_prepare.py

Generate LLM-ready text column:
    python seds_fetch_prepare.py --add-report-text

Use LLM-style narrative text:
    python seds_fetch_prepare.py --add-report-text --report-style llm

Provide custom narrative text (CSV or Parquet):
    python seds_fetch_prepare.py --add-report-text --report-text-file my_reports.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import requests

import pyarrow as pa
import pyarrow.parquet as pq

SEDS_CSV_URL = "https://www.eia.gov/state/seds/CDF/Complete_SEDS.csv"
CODES_XLSX_URL = "https://www.eia.gov/state/seds/CDF/Codes_and_Descriptions.xlsx"
MSN_SHEET = "MSN descriptions"


def download(url: str, out_path: Path, overwrite: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists")
        return
    print(f"[dl] {url} -> {out_path}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def detect_cols(df: pd.DataFrame) -> Dict[str, str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    msn = cols.get("msn") or cols.get("series_id") or cols.get("series") or cols.get("code")
    year = cols.get("year") or cols.get("period")
    state = cols.get("statecode") or cols.get("state") or cols.get("geography") or cols.get("location")
    value = cols.get("data") or cols.get("value")
    if not (msn and year and state and value):
        raise ValueError(f"Could not detect required columns in Complete_SEDS.csv. Found: {list(df.columns)}")
    return {"msn": msn, "year": year, "state": state, "value": value}


def _normalize_code_cols(cols) -> Dict[str, str]:
    """Map raw column names to standardized MSN/Description/Unit if possible."""
    ren = {}
    for c in cols:
        cl = str(c).strip().lower()
        if cl in ("msn", "series", "series_id", "seriesid", "code"):
            ren[c] = "MSN"
        elif "msn" == cl.replace(" ", ""):
            ren[c] = "MSN"
        elif "description" in cl or cl in ("desc", "name", "series name", "series_name"):
            ren[c] = "Description"
        elif "unit" in cl or cl in ("units",):
            ren[c] = "Unit"
    return ren


def load_codes_best_effort(xlsx_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    if MSN_SHEET not in xl.sheet_names:
        raise ValueError(f"Expected sheet '{MSN_SHEET}' not found. Sheets: {xl.sheet_names}")

    best: Optional[pd.DataFrame] = None
    best_score = -1

    for header in range(0, 15):
        try:
            df = pd.read_excel(xlsx_path, sheet_name=MSN_SHEET, header=header)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        ren = _normalize_code_cols(df.columns)
        df = df.rename(columns=ren)

        if "MSN" not in df.columns:
            continue

        score = int(df["MSN"].notna().sum())
        if score > best_score:
            best, best_score = df, score

    if best is None or best_score < 10:
        raw = pd.read_excel(xlsx_path, sheet_name=MSN_SHEET, header=None)
        msn_col = None
        for c in raw.columns:
            s = raw[c].dropna().astype(str).str.strip()
            if s.empty:
                continue
            short = s[s.str.match(r"^[A-Z0-9]{2,6}$")]
            if len(short) >= max(10, int(0.3 * len(s))):
                msn_col = c
                break
        if msn_col is None:
            raise ValueError(f"Could not locate MSN-like column in '{MSN_SHEET}'.")

        desc_col = msn_col + 1 if (msn_col + 1) in raw.columns else None
        unit_col = msn_col + 2 if (msn_col + 2) in raw.columns else None
        df = pd.DataFrame({
            "MSN": raw[msn_col],
            "Description": raw[desc_col] if desc_col is not None else None,
            "Unit": raw[unit_col] if unit_col is not None else None,
        })
        best = df

    keep = [c for c in ("MSN", "Description", "Unit") if c in best.columns]
    codes = best[keep].dropna(subset=["MSN"]).copy()
    codes["MSN"] = codes["MSN"].astype(str).str.strip()
    codes = codes[codes["MSN"].str.len() > 0]
    codes = codes[~codes["MSN"].str.contains("MSN", case=False, na=False)]
    codes = codes.drop_duplicates(subset=["MSN"])
    return codes


def value_trend_phrase(value: float) -> str:
    """Very simple value-based phrase bucket. Tune thresholds as needed."""
    if pd.isna(value):
        return "reported energy activity"
    if value >= 5000:
        return "very high energy demand"
    if value >= 2000:
        return "high energy demand"
    if value >= 1000:
        return "elevated energy usage"
    if value >= 500:
        return "moderate energy usage"
    return "relatively low energy usage"


def build_report_text(row: pd.Series, style: str = "plain") -> str:
    state = str(row.get("State", "")).strip()
    year = row.get("Year", "")
    msn = str(row.get("MSN", "")).strip()
    desc = str(row.get("Description", "")).strip()
    unit = str(row.get("Unit", "")).strip()
    value = row.get("Value", None)

    desc_clean = desc if desc and desc.lower() != "nan" else "energy usage"
    unit_clean = unit if unit and unit.lower() != "nan" else "units"
    trend = value_trend_phrase(value)

    if pd.isna(value):
        value_str = "missing"
    else:
        value_str = f"{float(value):.2f}"

    if style == "compact":
        return f"{state} {year} {desc_clean}: {value_str} {unit_clean}."
    elif style == "llm":
        return (
            f"For {state} in {year}, the metric '{desc_clean}' "
            f"(MSN: {msn}) recorded a value of {value_str} {unit_clean}. "
            f"This suggests {trend} for this category."
        )
    else:
        return (
            f"{state} {year} {trend} for {desc_clean}. "
            f"Reported value: {value_str} {unit_clean}."
        )


def add_report_text_column(df: pd.DataFrame, style: str = "plain") -> pd.DataFrame:
    df = df.copy()
    df["Report_Text"] = df.apply(lambda row: build_report_text(row, style=style), axis=1)
    return df


def load_external_report_text(path: Path) -> pd.DataFrame:
    """
    Load external narratives from CSV or Parquet.

    Expected columns:
      required: State, Year, Report_Text
      optional: MSN

    Example CSV:
      State,Year,Report_Text
      TX,2020,Demand surged due to...
      CA,2020,Industrial growth usage...
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        ext = pd.read_csv(path)
    elif suffix == ".parquet":
        ext = pd.read_parquet(path)
    else:
        raise ValueError("External report text file must be .csv or .parquet")

    required_base = {"State", "Year", "Report_Text"}
    missing = required_base - set(ext.columns)
    if missing:
        raise ValueError(f"External report text file missing required columns: {sorted(missing)}")

    ext = ext.copy()
    ext["State"] = ext["State"].astype(str).str.strip()
    ext["Year"] = pd.to_numeric(ext["Year"], errors="coerce").astype("Int64")
    if "MSN" in ext.columns:
        ext["MSN"] = ext["MSN"].astype(str).str.strip()

    return ext


def merge_report_text(
    df: pd.DataFrame,
    ext_text: Optional[pd.DataFrame],
    style: str = "plain",
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """
    If external text is provided:
      - join by [State, Year, MSN] when MSN exists in external file
      - otherwise join by [State, Year]
    Missing texts are auto-generated from structured data.
    """
    df = df.copy()

    if ext_text is None:
        return add_report_text_column(df, style=style)

    use_msn = "MSN" in ext_text.columns
    join_cols: List[str] = ["State", "Year"] + (["MSN"] if use_msn else [])

    merged = df.merge(ext_text[join_cols + ["Report_Text"]], on=join_cols, how="left", suffixes=("", "_ext"))

    generated = merged.apply(lambda row: build_report_text(row, style=style), axis=1)

    if overwrite_existing:
        merged["Report_Text"] = merged["Report_Text"].fillna(generated)
    else:
        merged["Report_Text"] = merged["Report_Text"].where(merged["Report_Text"].notna(), generated)

    return merged


def stream_csv_to_parquet(
    csv_path: Path,
    codes: pd.DataFrame,
    out_path: Path,
    chunksize: int,
    add_report_text: bool = False,
    report_style: str = "plain",
    ext_text: Optional[pd.DataFrame] = None,
    overwrite_existing_text: bool = False,
) -> int:
    if out_path.exists():
        out_path.unlink()

    reader = pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)
    first = next(reader)
    colmap = detect_cols(first)

    def process(chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.rename(columns={
            colmap["msn"]: "MSN",
            colmap["year"]: "Year",
            colmap["state"]: "State",
            colmap["value"]: "Value",
        })
        df["MSN"] = df["MSN"].astype(str).str.strip()
        df["State"] = df["State"].astype(str).str.strip()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["MSN", "State", "Year", "Value"])
        df = df.merge(codes, on="MSN", how="left")

        if add_report_text:
            df = merge_report_text(
                df,
                ext_text=ext_text,
                style=report_style,
                overwrite_existing=overwrite_existing_text,
            )

        return df

    total = 0
    df0 = process(first)
    table0 = pa.Table.from_pandas(df0, preserve_index=False)
    writer = pq.ParquetWriter(out_path, table0.schema, compression="snappy")
    writer.write_table(table0)
    total += len(df0)

    for chunk in reader:
        dfi = process(chunk)
        if dfi.empty:
            continue
        writer.write_table(pa.Table.from_pandas(dfi, preserve_index=False))
        total += len(dfi)
        if total % 1_000_000 < len(dfi):
            print(f"[progress] wrote {total:,} rows...")

    writer.close()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--industrial-only", action="store_true")

    # New arguments
    ap.add_argument("--add-report-text", action="store_true",
                    help="Generate an LLM-friendly Report_Text column.")
    ap.add_argument("--report-style", choices=["plain", "compact", "llm"], default="plain",
                    help="Style for generated Report_Text.")
    ap.add_argument("--report-text-file", default=None,
                    help="Optional CSV/Parquet file containing State, Year, Report_Text, and optional MSN.")
    ap.add_argument("--overwrite-report-text", action="store_true",
                    help="If external report text is provided, allow generated text to fill missing values only.")

    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    raw = root / "data" / "raw"
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    download(SEDS_CSV_URL, raw / "Complete_SEDS.csv", overwrite=args.overwrite)
    download(CODES_XLSX_URL, raw / "Codes_and_Descriptions.xlsx", overwrite=args.overwrite)

    codes = load_codes_best_effort(raw / "Codes_and_Descriptions.xlsx")
    print(f"[ok] loaded codes rows={len(codes):,} cols={codes.columns.tolist()}")

    external_text_df = None
    if args.report_text_file:
        external_text_df = load_external_report_text(Path(args.report_text_file))
        print(f"[ok] loaded external report text rows={len(external_text_df):,}")

    out_full = proc / "seds_enriched.parquet"
    total = stream_csv_to_parquet(
        raw / "Complete_SEDS.csv",
        codes,
        out_full,
        chunksize=args.chunksize,
        add_report_text=args.add_report_text,
        report_style=args.report_style,
        ext_text=external_text_df,
        overwrite_existing_text=args.overwrite_report_text,
    )
    print(f"[ok] wrote {out_full} rows={total:,}")

    if args.industrial_only:
        cols = ["MSN", "State", "Year", "Value", "Description", "Unit"]
        if args.add_report_text:
            cols.append("Report_Text")

        df = pd.read_parquet(out_full, columns=cols)
        mask = (
            df["Description"].astype(str).str.contains("Industrial", case=False, na=False)
            & df["Description"].astype(str).str.contains("Consumption", case=False, na=False)
        )
        df2 = df[mask].copy()
        out_sub = proc / "seds_industrial_consumption.parquet"
        df2.to_parquet(out_sub, index=False)
        print(f"[ok] wrote {out_sub} rows={len(df2):,}")


if __name__ == "__main__":
    main()