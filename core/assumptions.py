"""Statistical assumption checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .utils import clean_numeric, empty_result, p_apa, validate_columns


def shapiro_test(df: pd.DataFrame, variable: str, alpha: float = .05) -> dict:
    """Run Shapiro-Wilk normality test and return a Q-Q plot."""
    import matplotlib.pyplot as plt
    from scipy.stats import probplot
    validate_columns(df, [variable])
    x = clean_numeric(df[variable])
    if len(x) < 3:
        raise ValueError("Shapiro-Wilk requires at least 3 non-missing observations.")
    if len(x) > 5000:
        x = x.sample(5000, random_state=1)
    statistic, pvalue = stats.shapiro(x)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    probplot(x, dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot: {variable}"); ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Ordered values")
    fig.tight_layout()
    result = empty_result(f"Shapiro-Wilk normality test: {variable}", len(x))
    result["tables"] = [pd.DataFrame([{"Variable": variable, "N": len(x), "W": statistic, "p": pvalue}])]
    if pvalue < alpha:
        result["warnings"].append(f"Normality was rejected ({p_apa(pvalue)}); consider a non-parametric alternative or inspect the Q-Q plot.")
    result["figures"] = [fig]
    result["interpretation"] = f"The Shapiro-Wilk test was {('significant' if pvalue < alpha else 'not significant')} ({p_apa(pvalue)})."
    return result


def levene_test(df: pd.DataFrame, outcome: str, grouping: str, alpha: float = .05) -> dict:
    """Run Levene's test for equality of variances across groups."""
    validate_columns(df, [outcome, grouping])
    clean = df[[outcome, grouping]].copy(); clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce"); clean = clean.dropna()
    groups = [g[outcome].to_numpy() for _, g in clean.groupby(grouping, observed=True)]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        raise ValueError("Levene's test needs at least two groups with two observations each.")
    statistic, pvalue = stats.levene(*groups, center="median")
    result = empty_result(f"Levene's test: {outcome} by {grouping}", len(clean))
    result["tables"] = [pd.DataFrame([{"Outcome": outcome, "Grouping": grouping, "N": len(clean), "Levene W": statistic, "p": pvalue}])]
    if pvalue < alpha:
        result["warnings"].append(f"Variances differ significantly ({p_apa(pvalue)}); use the Welch t-test or robust ANOVA row.")
    result["interpretation"] = f"Levene's test was {('significant' if pvalue < alpha else 'not significant')} ({p_apa(pvalue)})."
    return result


def vif_table(df: pd.DataFrame, predictors: list[str]) -> dict:
    """Calculate variance inflation factors for multiple numeric predictors."""
    validate_columns(df, predictors)
    x = df[predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(predictors) < 2 or len(x) <= len(predictors):
        raise ValueError("VIF requires at least two predictors and more complete cases than predictors.")
    values = pd.DataFrame({"Variable": predictors, "VIF": [variance_inflation_factor(x.to_numpy(), i) for i in range(x.shape[1])]})
    result = empty_result("Variance inflation factors", len(x)); result["tables"] = [values]
    high = values.loc[values["VIF"] >= 5, "Variable"].tolist()
    if high: result["warnings"].append(f"Potential multicollinearity: VIF ≥ 5 for {', '.join(high)}.")
    result["interpretation"] = "VIF values below 5 are commonly treated as acceptable, subject to the thesis methodology."
    return result
