"""LLM layer: turn a business question into pandas code, run it safely, explain the result."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

from .safe_exec import UnsafeCodeError, run_code

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_RETRIES = 2

CODE_SYSTEM = """You are a careful data analyst. You write pandas code to answer questions about a DataFrame named `df`.

Rules:
- Use only pandas (as pd), numpy (as np) and math. They are already imported. Do not import anything else.
- Do not read or write files, access the network, or use eval/exec.
- Do not modify `df` in place; work on copies.
- Assign the final answer to a variable named `result` (a DataFrame, Series, or scalar).
- Return ONLY a single ```python code block, no commentary."""

EXPLAIN_SYSTEM = """You are a data analyst explaining findings to a business stakeholder.
Be concise (3-6 sentences), use the actual numbers from the result, state any caveats, and end with one
concrete recommendation or next step when it makes sense. Do not invent numbers that are not in the result."""


@dataclass
class Answer:
    question: str
    code: str
    result: object
    explanation: str
    error: str | None = None


def _schema_summary(df: pd.DataFrame, n_rows: int = 5) -> str:
    lines = [f"Rows: {len(df)}", "Columns (name: dtype):"]
    lines += [f"- {c}: {t}" for c, t in df.dtypes.astype(str).items()]
    lines.append("\nSample rows:")
    lines.append(df.head(n_rows).to_string(index=False))
    return "\n".join(lines)


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def _result_to_text(result: object, limit: int = 40) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.head(limit).to_string()
    return str(result)


class DataAnalystAgent:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        import anthropic  # imported lazily so the rest of the app works without a key

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)

    def _ask_llm(self, system: str, user: str, max_tokens: int = 1500) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def ask(self, df: pd.DataFrame, question: str) -> Answer:
        schema = _schema_summary(df)
        prompt = f"DataFrame info:\n{schema}\n\nQuestion: {question}"
        code, result, error = "", None, None

        for attempt in range(MAX_RETRIES + 1):
            code = _extract_code(self._ask_llm(CODE_SYSTEM, prompt))
            try:
                result, _ = run_code(code, df)
                error = None
                break
            except UnsafeCodeError as err:
                error = f"Blocked by safety check: {err}"
                break  # do not retry unsafe code
            except Exception as err:  # runtime error in generated code: let the model fix it
                error = f"{type(err).__name__}: {err}"
                prompt += f"\n\nYour previous code failed with: {error}\nFix it and return the corrected code."

        if error or result is None:
            return Answer(question, code, None, "I couldn't produce a reliable answer for that question.", error or "No `result` variable was set.")

        explanation = self._ask_llm(
            EXPLAIN_SYSTEM,
            f"Question: {question}\n\nResult:\n{_result_to_text(result)}",
            max_tokens=500,
        )
        return Answer(question, code, result, explanation.strip())
