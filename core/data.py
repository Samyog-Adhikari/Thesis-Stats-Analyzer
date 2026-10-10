"""Data loading and variable metadata helpers."""
from __future__ import annotations

from io import BytesIO
from typing import Dict, Iterable

import numpy as np
import pandas as pd

VALID_TYPES = ("nominal", "ordinal", "scale")


def list_excel_sheets(uploaded_file: bytes | BytesIO) -> list[str]:
    """Return sheet names from an in-memory Excel workbook."""
    return pd.ExcelFile(uploaded_file).sheet_names


def load_excel(uploaded_file: bytes | BytesIO, sheet_name: str = 0) -> pd.DataFrame:
    """Load one Excel sheet from memory, normalising empty column names."""
    data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    data.columns = [str(c).strip() or f"Variable_{i + 1}" for i, c in enumerate(data.columns)]
    return data


def infer_variable_type(series: pd.Series) -> str:
    """Infer nominal, ordinal, or scale from a pandas Series."""
    clean = series.dropna()
    if pd.api.types.is_numeric_dtype(clean):
        return "scale"
    return "nominal"


def infer_variable_types(df: pd.DataFrame) -> Dict[str, str]:
    """Infer a type for every column in a DataFrame."""
    return {str(column): infer_variable_type(df[column]) for column in df.columns}


def variables_of_type(var_types: Dict[str, str], kind: str | Iterable[str]) -> list[str]:
    """Return variables whose configured type matches one or more types."""
    wanted = {kind} if isinstance(kind, str) else set(kind)
    return [name for name, value in var_types.items() if value in wanted]


def validate_var_types(var_types: Dict[str, str], columns: Iterable[str]) -> None:
    """Raise ValueError when variable metadata is incomplete or invalid."""
    missing = [c for c in columns if c not in var_types]
    invalid = {c: var_types[c] for c in var_types if var_types[c] not in VALID_TYPES}
    if missing:
        raise ValueError(f"Missing variable types for: {', '.join(map(str, missing))}")
    if invalid:
        raise ValueError(f"Invalid variable types: {invalid}")
