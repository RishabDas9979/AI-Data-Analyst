"""Streamlit UI for the AI Data Analyst."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analyzer
from src.ai_agent import DataAnalystAgent
from src.cleaner import clean
from src.loader import load_file

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Upload raw data. Get it cleaned, explored, and questioned in plain English.")

# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("CSV, Excel or JSON", type=["csv", "tsv", "xlsx", "xls", "json"])
    use_sample = st.button("Use sample dataset")

if use_sample:
    st.session_state["raw"] = load_file("data/sample_sales.csv")
    st.session_state.pop("history", None)
elif uploaded is not None and st.session_state.get("filename") != uploaded.name:
    try:
        st.session_state["raw"] = load_file(uploaded)
        st.session_state["filename"] = uploaded.name
        st.session_state.pop("history", None)
    except Exception as err:
        st.error(f"Could not load file: {err}")

if "raw" not in st.session_state:
    st.info("Upload a file or click **Use sample dataset** in the sidebar to begin.")
    st.stop()

raw: pd.DataFrame = st.session_state["raw"]
result = clean(raw)
df = result.df

tab_preview, tab_clean, tab_explore, tab_ask = st.tabs(["Preview", "Cleaning report", "Explore", "Ask AI"])

# ---------- Preview ----------
with tab_preview:
    st.subheader("Raw data")
    st.dataframe(raw.head(100), width="stretch")
    p = analyzer.profile(raw)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{p['rows']:,}")
    c2.metric("Columns", p["columns"])
    c3.metric("Missing cells", f"{p['missing_cells']:,}")
    c4.metric("Duplicate rows", p["duplicate_rows"])

# ---------- Cleaning ----------
with tab_clean:
    st.subheader("What was cleaned")
    c1, c2 = st.columns(2)
    c1.metric("Quality score (before)", result.quality_before)
    c2.metric("Quality score (after)", result.quality_after, delta=round(result.quality_after - result.quality_before, 1))
    for line in result.log:
        st.write(f"- {line}")
    st.subheader("Cleaned data")
    st.dataframe(df.head(100), width="stretch")
    st.download_button("Download cleaned CSV", df.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")

# ---------- Explore ----------
with tab_explore:
    st.subheader("Auto insights")
    for line in analyzer.auto_insights(df):
        st.write(f"- {line}")

    prof = analyzer.profile(df)
    stats = analyzer.summary_stats(df)
    if not stats.empty:
        st.subheader("Summary statistics")
        st.dataframe(stats, width="stretch")

    corr = analyzer.correlations(df)
    if not corr.empty:
        st.subheader("Notable correlations")
        st.dataframe(corr, width="stretch")

    st.subheader("Charts")
    num_cols, cat_cols, date_cols = prof["numeric_columns"], prof["categorical_columns"], prof["datetime_columns"]

    if date_cols and num_cols:
        dcol = st.selectbox("Date column", date_cols)
        vcol = st.selectbox("Value to trend", num_cols, key="trend_val")
        t = analyzer.trend(df, dcol, vcol)
        st.plotly_chart(px.line(t, x=dcol, y=vcol, markers=True, title=f"{vcol} over time"), width="stretch")

    if cat_cols and num_cols:
        gcol = st.selectbox("Group by", cat_cols)
        gval = st.selectbox("Measure", num_cols, key="grp_val")
        g = df.groupby(gcol)[gval].sum().sort_values(ascending=False).head(15).reset_index()
        st.plotly_chart(px.bar(g, x=gcol, y=gval, title=f"{gval} by {gcol}"), width="stretch")

# ---------- Ask AI ----------
with tab_ask:
    st.subheader("Ask a question about your data")
    st.caption("The model sees only the column names, types and a few sample rows, not your full dataset.")
    question = st.text_input("e.g. Which region had the highest revenue last quarter?")

    if st.button("Analyse", type="primary") and question:
        try:
            agent = DataAnalystAgent()
            with st.spinner("Thinking..."):
                answer = agent.ask(df, question)
            st.session_state.setdefault("history", []).insert(0, answer)
        except Exception as err:
            st.error(str(err))

    for ans in st.session_state.get("history", []):
        with st.container(border=True):
            st.markdown(f"**Q: {ans.question}**")
            if ans.error:
                st.warning(ans.error)
            st.write(ans.explanation)
            if isinstance(ans.result, (pd.DataFrame, pd.Series)):
                st.dataframe(ans.result)
            elif ans.result is not None:
                st.write(ans.result)
            with st.expander("Show generated code"):
                st.code(ans.code, language="python")
