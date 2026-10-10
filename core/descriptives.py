"""Descriptive statistics for scale and categorical research variables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import clean_numeric, empty_result, p_apa, validate_columns


def scale_descriptives(df: pd.DataFrame, variable: str, confidence: float = .95) -> dict:
    """Return thesis-ready descriptive statistics for one scale variable."""
    validate_columns(df, [variable])
    x = clean_numeric(df[variable])
    if len(x) < 2:
        raise ValueError("At least two non-missing numeric observations are required.")
    alpha = 1 - confidence
    mean, sd = x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(len(x))
    tcrit = stats.t.ppf(1 - alpha / 2, len(x) - 1)
    table = pd.DataFrame([{"Variable": variable, "N": len(x), "Mean": mean, "SD": sd, "SE": se,
                           "Median": x.median(), "Min": x.min(), "Max": x.max(),
                           "Skewness": stats.skew(x, bias=False), "Kurtosis": stats.kurtosis(x, bias=False),
                           f"CI {confidence:.0%} lower": mean - tcrit * se, f"CI {confidence:.0%} upper": mean + tcrit * se}])
    result = empty_result(f"Descriptives: {variable}", len(x))
    result["tables"] = [table]
    result["interpretation"] = f"The mean of {variable} was {mean:.3f} (SD = {sd:.3f}, N = {len(x)}), with a {confidence:.0%} confidence interval of [{mean - tcrit * se:.3f}, {mean + tcrit * se:.3f}]."
    return result


def frequency_table(df: pd.DataFrame, variable: str) -> dict:
    """Return frequency and percentage counts for a categorical variable."""
    validate_columns(df, [variable])
    clean = df[variable].dropna()
    if clean.empty:
        raise ValueError("The selected variable has no available observations.")
    counts = clean.value_counts(dropna=False).rename("Frequency")
    table = counts.to_frame()
    table["Percent"] = table["Frequency"] / len(clean) * 100
    table.index.name = variable
    result = empty_result(f"Frequencies: {variable}", len(clean))
    result["tables"] = [table.reset_index()]
    result["interpretation"] = f"The frequency table for {variable} includes {len(clean)} available observations across {len(table)} categories."
    return result


def crosstabulation(df: pd.DataFrame, row: str, column: str) -> dict:
    """Return counts, row percentages, and column percentages for two categorical variables."""
    validate_columns(df, [row, column])
    clean = df[[row, column]].dropna()
    if clean.empty:
        raise ValueError("No complete cases are available for the selected variables.")
    counts = pd.crosstab(clean[row], clean[column])
    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    col_pct = counts.div(counts.sum(axis=0), axis=1) * 100
    result = empty_result(f"Crosstabulation: {row} by {column}", len(clean))
    result["tables"] = [counts, row_pct.add_suffix(" (row %)") , col_pct.add_suffix(" (column %)")]
    result["interpretation"] = f"The crosstabulation uses {len(clean)} complete cases for {row} and {column}."
    return result
