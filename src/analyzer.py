"""Exploratory analysis helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def text_columns(df: pd.DataFrame) -> list[str]:
    """Text/categorical columns, robust to pandas 2 (object) and pandas 3 (str) dtypes."""
    return [
        c for c in df.columns
        if isinstance(df[c].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
    ]


def profile(df: pd.DataFrame) -> dict:
    """High-level profile of a DataFrame."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": text_columns(df),
        "datetime_columns": df.select_dtypes(include="datetime").columns.tolist(),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=np.number)
    return num.describe().T if not num.empty else pd.DataFrame()


def correlations(df: pd.DataFrame, min_abs: float = 0.3) -> pd.DataFrame:
    """Strongest pairwise correlations between numeric columns."""
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["column_a", "column_b", "correlation"])
    corr = num.corr()
    rows = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and abs(val) >= min_abs:
                rows.append((cols[i], cols[j], round(float(val), 3)))
    result = pd.DataFrame(rows, columns=["column_a", "column_b", "correlation"])
    return result.reindex(result["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def trend(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "M") -> pd.DataFrame:
    """Aggregate a value column over time."""
    tmp = df[[date_col, value_col]].dropna()
    # Period grouping works across pandas versions (resample's "M" was renamed "ME" in newer releases)
    grouped = tmp.groupby(tmp[date_col].dt.to_period(freq))[value_col].sum().reset_index()
    grouped[date_col] = grouped[date_col].dt.to_timestamp()
    grouped["change_pct"] = grouped[value_col].pct_change().mul(100).round(1)
    return grouped


def top_bottom(df: pd.DataFrame, group_col: str, value_col: str, n: int = 5) -> dict[str, pd.DataFrame]:
    totals = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    return {"top": totals.head(n).reset_index(), "bottom": totals.tail(n).reset_index()}


def find_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]


def auto_insights(df: pd.DataFrame) -> list[str]:
    """Plain-language observations generated without an LLM."""
    insights: list[str] = []
    p = profile(df)
    insights.append(f"Dataset has {p['rows']:,} rows and {p['columns']} columns.")

    corr = correlations(df, min_abs=0.6)
    for _, r in corr.head(3).iterrows():
        direction = "positively" if r["correlation"] > 0 else "negatively"
        insights.append(f"'{r['column_a']}' and '{r['column_b']}' are strongly {direction} correlated ({r['correlation']}).")

    for col in p["categorical_columns"][:3]:
        top = df[col].value_counts(normalize=True).head(1)
        if not top.empty:
            insights.append(f"Most common '{col}': {top.index[0]} ({top.iloc[0]:.0%} of rows).")
    return insights
