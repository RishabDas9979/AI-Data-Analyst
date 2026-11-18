"""Basic tests for the non-LLM parts of the pipeline. Run with: pytest -q"""
import pandas as pd
import pytest

from src import analyzer
from src.cleaner import clean, quality_score
from src.loader import load_file
from src.safe_exec import UnsafeCodeError, check_code, run_code

SAMPLE = "data/sample_sales.csv"


@pytest.fixture(scope="module")
def raw():
    return load_file(SAMPLE)


def test_load(raw):
    assert len(raw) == 410
    assert "Order ID" in raw.columns


def test_clean_improves_quality(raw):
    res = clean(raw)
    assert res.quality_after > res.quality_before
    assert res.df.duplicated().sum() == 0
    assert res.df.isna().sum().sum() == 0 or res.df.select_dtypes("datetime").shape[1] > 0
    assert "revenue" in res.df.columns
    assert pd.api.types.is_numeric_dtype(res.df["revenue"])
    assert pd.api.types.is_datetime64_any_dtype(res.df["order_date"])


def test_region_whitespace_normalised(raw):
    df = clean(raw).df
    assert df["region"].str.strip().eq(df["region"]).all()


def test_analyzer(raw):
    df = clean(raw).df
    p = analyzer.profile(df)
    assert p["rows"] == len(df)
    assert not analyzer.correlations(df, min_abs=0.1).empty
    t = analyzer.trend(df, "order_date", "revenue")
    assert len(t) > 1


def test_safe_exec_runs_valid_code(raw):
    df = clean(raw).df
    result, _ = run_code("result = df.groupby('region')['revenue'].sum()", df)
    assert len(result) >= 4


@pytest.mark.parametrize("bad", [
    "import os\nresult = os.listdir('.')",
    "result = open('/etc/passwd').read()",
    "result = df.to_csv('x.csv')",
    "result = ().__class__.__bases__",
    "result = eval('1+1')",
    "from subprocess import run\nresult = run(['ls'])",
])
def test_safe_exec_blocks_unsafe_code(bad):
    with pytest.raises(UnsafeCodeError):
        check_code(bad)
