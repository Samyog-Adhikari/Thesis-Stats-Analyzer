"""Relationships and predictive analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .utils import empty_result, p_apa, validate_columns


def correlation_matrix(df: pd.DataFrame, variables: list[str], method: str = "pearson", alpha: float = .05) -> dict:
    """Return pairwise correlations, p-values, N values, and a heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    validate_columns(df, variables)
    if len(variables) < 2: raise ValueError("Select at least two numeric variables.")
    r = pd.DataFrame(index=variables, columns=variables, dtype=float); p = r.copy(); n = r.copy()
    for a in variables:
        for b in variables:
            pair = df[[a,b]].apply(pd.to_numeric, errors="coerce").dropna(); n.loc[a,b] = len(pair)
            if a == b: r.loc[a,b], p.loc[a,b] = 1.0, 0.0; continue
            if len(pair) >= 3:
                test = stats.pearsonr(pair[a].to_numpy(), pair[b].to_numpy()) if method == "pearson" else stats.spearmanr(pair[a].to_numpy(), pair[b].to_numpy()); r.loc[a,b] = float(np.asarray(test.statistic).squeeze()); p.loc[a,b] = float(np.asarray(test.pvalue).squeeze())
    fig, ax = plt.subplots(figsize=(max(5, len(variables)*1.1), max(4, len(variables)*.9)), facecolor="white"); sns.heatmap(r, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax); ax.set_title(f"{method.title()} correlation heatmap"); fig.tight_layout()
    result = empty_result(f"{method.title()} correlation matrix", int(n.max().max())); result["tables"] = [r.add_suffix(" r"), p.add_suffix(" p"), n.add_suffix(" N")]; result["figures"] = [fig]; result["interpretation"] = f"The {method} matrix uses pairwise deletion; N is reported for each pair."; return result


def chi_square(df: pd.DataFrame, row: str, column: str, alpha: float = .05) -> dict:
    """Run chi-square association test with Cramer's V and expected-cell warnings."""
    validate_columns(df, [row, column]); clean = df[[row,column]].dropna(); observed = pd.crosstab(clean[row], clean[column]); chi, p, dof, expected = stats.chi2_contingency(observed); n = observed.to_numpy().sum(); v = np.sqrt(chi/(n*min(observed.shape[0]-1, observed.shape[1]-1))) if n else np.nan
    result = empty_result(f"Chi-square test: {row} by {column}", len(clean)); result["tables"] = [observed, pd.DataFrame(expected, index=observed.index, columns=observed.columns), pd.DataFrame([{"N": n, "χ²": chi, "df": dof, "p": p, "Cramer's V": v}])]
    if (expected < 5).any(): result["warnings"].append("At least one expected cell count is below 5; interpret chi-square cautiously or collapse sparse categories.")
    result["interpretation"] = f"There was {'a significant' if p < alpha else 'no significant'} association between {row} and {column} ({p_apa(p)}), Cramer's V = {v:.3f}."; return result


def linear_regression(df: pd.DataFrame, outcome: str, predictors: list[str], alpha: float = .05, confidence: float = .95) -> dict:
    """Fit OLS with coefficients, standardized betas, confidence intervals, R-squared, F-test, and VIF."""
    validate_columns(df, [outcome, *predictors]); clean = df[[outcome, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(predictors) < 1: raise ValueError("Select at least one predictor.")
    if len(clean) <= len(predictors)+1: raise ValueError("There are not enough complete cases for this model.")
    x = sm.add_constant(clean[predictors]); model = sm.OLS(clean[outcome], x).fit(); ci = model.conf_int(alpha=1-confidence); standardized = {}
    y_sd = clean[outcome].std(ddof=1)
    for col in predictors: standardized[col] = model.params[col] * clean[col].std(ddof=1) / y_sd
    standardized["const"] = np.nan
    table = pd.DataFrame({"Variable": model.params.index, "B": model.params.values, "Standardized beta": [standardized.get(v, np.nan) for v in model.params.index], "SE": model.bse.values, "t": model.tvalues.values, "p": model.pvalues.values, "CI lower": ci[0].values, "CI upper": ci[1].values})
    vif = pd.DataFrame({"Variable": predictors, "VIF": [variance_inflation_factor(clean[predictors].to_numpy(), i) for i in range(len(predictors))]}) if len(predictors) > 1 else pd.DataFrame({"Variable": predictors, "VIF": [1.0]})
    result = empty_result(f"Linear regression: {outcome}", len(clean)); result["tables"] = [table, pd.DataFrame([{"N": len(clean), "R²": model.rsquared, "Adjusted R²": model.rsquared_adj, "F": model.fvalue, "F p": model.f_pvalue}]), vif];
    if (vif["VIF"] >= 5).any(): result["warnings"].append("At least one predictor has VIF ≥ 5; inspect multicollinearity.")
    result["interpretation"] = f"The model explained {model.rsquared:.3f} of the variance in {outcome} (adjusted R² = {model.rsquared_adj:.3f}; F = {model.fvalue:.3f}, {p_apa(model.f_pvalue)})."; return result


def logistic_regression(df: pd.DataFrame, outcome: str, predictors: list[str], alpha: float = .05, confidence: float = .95) -> dict:
    """Fit binary logistic regression with coefficients, odds ratios, CIs, and model fit."""
    validate_columns(df, [outcome, *predictors]); clean = df[[outcome, *predictors]].copy(); clean[predictors] = clean[predictors].apply(pd.to_numeric, errors="coerce"); clean = clean.dropna();
    if len(predictors) < 1: raise ValueError("Select at least one predictor.")
    y = clean[outcome]
    if not pd.api.types.is_numeric_dtype(y): y = pd.Categorical(y).codes
    y = pd.to_numeric(y, errors="coerce"); clean = clean.loc[y.notna()].copy(); y = y.loc[clean.index]
    if y.nunique() != 2: raise ValueError("The logistic outcome must contain exactly two groups.")
    x = sm.add_constant(clean[predictors]); model = sm.Logit(y, x).fit(disp=False); ci = model.conf_int(alpha=1-confidence); table = pd.DataFrame({"Variable": model.params.index, "B": model.params.values, "SE": model.bse.values, "z": model.tvalues.values, "p": model.pvalues.values, "Odds ratio": np.exp(model.params.values), "OR CI lower": np.exp(ci[0].values), "OR CI upper": np.exp(ci[1].values)})
    llf, llnull = model.llf, model.llnull; pseudo_r2 = 1 - llf/llnull if llnull else np.nan; fit = pd.DataFrame([{"N": len(clean), "-2 Log Likelihood": -2*llf, "McFadden pseudo R²": pseudo_r2, "Likelihood-ratio p": model.llr_pvalue}]); result = empty_result(f"Logistic regression: {outcome}", len(clean)); result["tables"] = [table, fit];
    if model.mle_retvals.get("converged") is False: result["warnings"].append("The model did not converge; simplify predictors or inspect sparse cells.")
    result["interpretation"] = f"The logistic model had McFadden pseudo R² = {pseudo_r2:.3f}, likelihood-ratio {p_apa(model.llr_pvalue)}; odds ratios are reported for each predictor."; return result
