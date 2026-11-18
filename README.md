# AI Data Analyst

Upload raw CSV/Excel data and the app will clean it, profile it, and answer
business questions in plain English using an LLM that writes and runs pandas code.

## Features
- **Load**: CSV, XLSX, JSON
- **Clean**: duplicates, whitespace, types, dates, missing values, with a full change log
- **Analyse**: summary stats, correlations, trends, outlier flags, data-quality score
- **Ask**: natural-language questions -> generated pandas code -> safe execution -> plain-English answer
- **Transparent**: every answer shows the code that produced it

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your ANTHROPIC_API_KEY
streamlit run app.py
```

Try it with `data/sample_sales.csv` (deliberately messy).

## Project structure

```
ai-data-analyst/
├── app.py                # Streamlit UI
├── src/
│   ├── loader.py         # file ingestion
│   ├── cleaner.py        # automated cleaning + change log
│   ├── analyzer.py       # profiling and statistics
│   ├── safe_exec.py      # AST-checked execution of generated code
│   └── ai_agent.py       # LLM question answering
├── data/sample_sales.csv
├── tests/test_pipeline.py
└── requirements.txt
```

## Safety note
Generated code is checked with an AST allow-list (no imports beyond pandas/numpy/math,
no file/network/dunder access) and run with restricted builtins. This is a best-effort
guard, not a full sandbox. For production, run it in a container or separate process.

## Roadmap
- SQL database connector
- PDF report export
- Chart auto-selection per question
- Conversation memory across questions
