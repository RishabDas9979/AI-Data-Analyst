"""Automated data cleaning with a transparent change log."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningResult:
    df: pd.DataFrame
    log: list[str] = field(default_factory=list)
    quality_before: float = 0.0
    quality_after: float = 0.0


def _snake(name: str) -> str:
    name = re.sub(r"[^\w\s]", "", str(name).strip())
    name = re.sub(r"\s+", "_", name)
    return name.lower()


def _text_cols(df: pd.DataFrame) -> list[str]:
    """Text columns, robust to pandas 2 (object) and pandas 3 (str) dtypes."""
    return [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]


def quality_score(df: pd.DataFrame) -> float:
    """0-100 score: penalises missing cells and duplicate rows."""
    if df.empty:
        return 0.0
    missing_ratio = df.isna().sum().sum() / df.size
    dup_ratio = df.duplicated().sum() / len(df)
    return round(max(0.0, 100 * (1 - missing_ratio) * (1 - dup_ratio)), 1)


def clean(df: pd.DataFrame) -> CleaningResult:
    """Clean a DataFrame and explain every change made."""
    out = df.copy()
    log: list[str] = []
    before = quality_score(out)

    # 1. Column names
    new_cols = [_snake(c) for c in out.columns]
    if new_cols != list(out.columns):
        out.columns = new_cols
        log.append("Standardised column names to snake_case.")

    # 2. Trim whitespace in text columns and turn blanks into NaN
    obj_cols = _text_cols(out)
    for col in obj_cols:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA})
    if len(obj_cols):
        log.append(f"Trimmed whitespace and normalised blank values in {len(obj_cols)} text column(s).")

    # 3. Convert numeric-looking text (e.g. "1,200", "$45") to numbers
    for col in _text_cols(out):
        stripped = out[col].astype("string").str.replace(r"[,$€£₹%\s]", "", regex=True)
        converted = pd.to_numeric(stripped, errors="coerce")
        non_null = out[col].notna().sum()
        if non_null and converted.notna().sum() / non_null >= 0.9:
            out[col] = converted
            log.append(f"Converted '{col}' from text to numeric.")

    # 4. Parse dates
    for col in _text_cols(out):
        looks_like_date = "date" in col or "time" in col or col.endswith("_at")
        if not looks_like_date:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce", format="mixed")
        non_null = out[col].notna().sum()
        if non_null and parsed.notna().sum() / non_null >= 0.8:
            out[col] = parsed
            log.append(f"Parsed '{col}' as datetime.")

    # 5. Remove duplicate rows
    dups = int(out.duplicated().sum())
    if dups:
        out = out.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {dups} duplicate row(s).")

    # 6. Drop columns that are almost entirely empty
    empty_cols = [c for c in out.columns if out[c].isna().mean() > 0.9]
    if empty_cols:
        out = out.drop(columns=empty_cols)
        log.append(f"Dropped mostly-empty column(s): {', '.join(empty_cols)}.")

    # 7. Fill missing values
    for col in out.columns:
        n_missing = int(out[col].isna().sum())
        if not n_missing:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            median = out[col].median()
            out[col] = out[col].fillna(median)
            log.append(f"Filled {n_missing} missing value(s) in '{col}' with the median ({median:g}).")
        elif pd.api.types.is_datetime64_any_dtype(out[col]):
            log.append(f"Left {n_missing} missing date(s) in '{col}' unfilled (can't guess dates).")
        else:
            out[col] = out[col].fillna("Unknown")
            log.append(f"Filled {n_missing} missing value(s) in '{col}' with 'Unknown'.")

    # 8. Flag (not remove) numeric outliers using the IQR rule
    for col in out.select_dtypes(include=np.number).columns:
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask = (out[col] < q1 - 1.5 * iqr) | (out[col] > q3 + 1.5 * iqr)
        n_out = int(mask.sum())
        if n_out:
            log.append(f"Flagged {n_out} possible outlier(s) in '{col}' (kept in data, review manually).")

    if not log:
        log.append("Data was already clean. No changes needed.")

    return CleaningResult(df=out, log=log, quality_before=before, quality_after=quality_score(out))
