"""File ingestion: CSV, Excel and JSON into a pandas DataFrame."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

SUPPORTED = {".csv", ".tsv", ".xlsx", ".xls", ".json"}


def load_file(source: str | Path | BinaryIO, name: str | None = None) -> pd.DataFrame:
    """Load a file path or an uploaded file-like object (e.g. Streamlit's UploadedFile)."""
    filename = name or getattr(source, "name", None) or str(source)
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(SUPPORTED))}")

    if ext == ".csv":
        df = _read_csv(source, sep=",")
    elif ext == ".tsv":
        df = _read_csv(source, sep="\t")
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(source)
    else:
        df = pd.read_json(source)

    if df.empty:
        raise ValueError("The file loaded but contains no rows.")
    return df


def _read_csv(source, sep: str) -> pd.DataFrame:
    """Try common encodings so messy exports still load."""
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, sep=sep, encoding=enc)
        except UnicodeDecodeError as err:
            last_err = err
    raise ValueError(f"Could not decode file: {last_err}")
