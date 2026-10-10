"""Shared pure helpers used by analysis modules."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def p_apa(p: float | None, decimals: int = 3) -> str:
    """Format a p-value in APA style."""
    if p is None or not np.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.{decimals}f}".replace("0.", ".", 1)


def clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric and remove missing/infinite values."""
    result = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return result.dropna()


def empty_result(title: str, n: int = 0) -> dict[str, Any]:
    """Create a standard empty result container."""
    return {"title": title, "n": int(n), "tables": [], "figures": [], "warnings": [], "interpretation": ""}


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that selected columns exist and are distinct."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {', '.join(missing)}")
    if len(columns) != len(set(columns)):
        raise ValueError("Please select different variables for each role.")


def round_tables(result: dict[str, Any], decimals: int = 3) -> dict[str, Any]:
    """Round numeric table values without changing the result structure."""
    copied = dict(result)
    copied["tables"] = [table.round(decimals) if isinstance(table, pd.DataFrame) else table for table in result.get("tables", [])]
    return copied
