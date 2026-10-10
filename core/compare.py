"""Group comparison tests for independent, paired, one-way, factorial, and repeated designs."""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM

from .utils import empty_result, p_apa, validate_columns


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate pooled-SD Cohen's d for independent groups."""
    pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
    return float((np.mean(a)-np.mean(b))/pooled) if pooled else np.nan


def _paired_d(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cohen's dz for paired observations."""
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else np.nan


def independent_t_test(df: pd.DataFrame, outcome: str, grouping: str, alpha: float = .05, confidence: float = .95) -> dict:
    """Run Student and Welch independent-samples t-tests."""
    validate_columns(df, [outcome, grouping])
    clean = df[[outcome, grouping]].copy(); clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce"); clean = clean.dropna()
    groups = list(clean.groupby(grouping, observed=True))
    if len(groups) != 2: raise ValueError("The grouping variable must contain exactly two non-missing groups.")
    (name_a, ga), (name_b, gb) = groups; a, b = ga[outcome].to_numpy(), gb[outcome].to_numpy()
    if min(len(a), len(b)) < 2: raise ValueError("Each group needs at least two observations.")
    _, lev_p = stats.levene(a, b, center="median"); rows = []
    for label, equal in [("Student", True), ("Welch", False)]:
        t, p = stats.ttest_ind(a, b, equal_var=equal)
        dfree = len(a)+len(b)-2 if equal else (np.var(a,ddof=1)/len(a)+np.var(b,ddof=1)/len(b))**2 / ((np.var(a,ddof=1)/len(a))**2/(len(a)-1)+(np.var(b,ddof=1)/len(b))**2/(len(b)-1))
        rows.append({"Test": label, "N": len(a)+len(b), "Group 1": name_a, "Group 2": name_b, "t": t, "df": dfree, "p": p, "Cohen's d": _cohens_d(a,b)})
    result = empty_result(f"Independent t-test: {outcome} by {grouping}", len(a)+len(b)); result["tables"] = [pd.DataFrame(rows)]
    if lev_p < alpha: result["warnings"].append(f"Levene's test was significant ({p_apa(lev_p)}); interpret the Welch row.")
    chosen = rows[1] if lev_p < alpha else rows[0]; d = chosen["Cohen's d"]
    result["interpretation"] = f"{outcome} {'differed' if chosen['p'] < alpha else 'did not differ'} significantly between {name_a} and {name_b}, t({chosen['df']:.2f}) = {chosen['t']:.3f}, {p_apa(chosen['p'])}, d = {d:.3f}."
    return result


def mann_whitney(df: pd.DataFrame, outcome: str, grouping: str, alpha: float = .05) -> dict:
    """Run Mann-Whitney U for exactly two independent groups."""
    validate_columns(df, [outcome, grouping]); clean = df[[outcome, grouping]].copy(); clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce"); clean = clean.dropna(); groups = list(clean.groupby(grouping, observed=True))
    if len(groups) != 2: raise ValueError("The grouping variable must contain exactly two groups.")
    a, b = groups[0][1][outcome], groups[1][1][outcome]; u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    result = empty_result(f"Mann-Whitney U: {outcome} by {grouping}", len(clean)); result["tables"] = [pd.DataFrame([{"N": len(clean), "Group 1": groups[0][0], "Group 2": groups[1][0], "U": u, "p": p}])]; result["interpretation"] = f"The distributions {'differed' if p < alpha else 'did not differ'} significantly ({p_apa(p)})."; return result


def paired_t_test(df: pd.DataFrame, first: str, second: str, alpha: float = .05, confidence: float = .95) -> dict:
    """Run a paired-samples t-test with CI for the mean difference and Cohen's dz."""
    validate_columns(df, [first, second]); clean = df[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 2: raise ValueError("At least two complete paired observations are required.")
    a, b = clean[first].to_numpy(), clean[second].to_numpy(); diff = a - b; t, p = stats.ttest_rel(a, b); se = diff.std(ddof=1) / np.sqrt(len(diff)); crit = stats.t.ppf((1+confidence)/2, len(diff)-1); lo, hi = diff.mean()-crit*se, diff.mean()+crit*se; d = _paired_d(a,b)
    result = empty_result(f"Paired t-test: {first} vs {second}", len(clean)); result["tables"] = [pd.DataFrame([{"N": len(clean), "Mean difference": diff.mean(), "t": t, "df": len(clean)-1, "p": p, "CI lower": lo, "CI upper": hi, "Cohen's dz": d}])]; result["interpretation"] = f"The mean difference between {first} and {second} was {'significant' if p < alpha else 'not significant'}, t({len(clean)-1}) = {t:.3f}, {p_apa(p)}, dz = {d:.3f}."; return result


def wilcoxon_signed_rank(df: pd.DataFrame, first: str, second: str, alpha: float = .05) -> dict:
    """Run a two-sided Wilcoxon signed-rank test for paired observations."""
    validate_columns(df, [first, second]); clean = df[[first, second]].apply(pd.to_numeric, errors="coerce").dropna(); diff = clean[first] - clean[second]; diff = diff[diff != 0]
    if len(diff) < 2: raise ValueError("At least two non-zero paired differences are required.")
    statistic, p = stats.wilcoxon(diff, alternative="two-sided"); result = empty_result(f"Wilcoxon signed-rank: {first} vs {second}", len(clean)); result["tables"] = [pd.DataFrame([{"N": len(clean), "Non-zero differences": len(diff), "W": statistic, "p": p}])]; result["interpretation"] = f"The paired distributions {'differed' if p < alpha else 'did not differ'} significantly ({p_apa(p)})."; return result


def one_way_anova(df: pd.DataFrame, outcome: str, grouping: str, alpha: float = .05) -> dict:
    """Run one-way ANOVA with eta squared and Tukey post hoc when significant."""
    validate_columns(df, [outcome, grouping]); clean = df[[outcome, grouping]].copy(); clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce"); clean = clean.dropna(); groups = [g[outcome].to_numpy() for _, g in clean.groupby(grouping, observed=True)]
    if len(groups) < 2 or any(len(g)<2 for g in groups): raise ValueError("ANOVA needs at least two groups with two observations each.")
    f, p = stats.f_oneway(*groups); grand = clean[outcome].mean(); ss_between = sum(len(g)*(np.mean(g)-grand)**2 for g in groups); ss_total = sum((clean[outcome]-grand)**2); eta = ss_between/ss_total if ss_total else np.nan
    result = empty_result(f"One-way ANOVA: {outcome} by {grouping}", len(clean)); result["tables"] = [pd.DataFrame([{"N": len(clean), "F": f, "df_between": len(groups)-1, "df_within": len(clean)-len(groups), "p": p, "Eta squared": eta}])]
    if p < alpha:
        tukey = pairwise_tukeyhsd(clean[outcome], clean[grouping]); result["tables"].append(pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]))
    result["interpretation"] = f"The group means {'differed' if p < alpha else 'did not differ'} significantly, F({len(groups)-1}, {len(clean)-len(groups)}) = {f:.3f}, {p_apa(p)}, η² = {eta:.3f}."; return result


def kruskal_wallis(df: pd.DataFrame, outcome: str, grouping: str, alpha: float = .05, posthoc: bool = True) -> dict:
    """Run Kruskal-Wallis with optional Bonferroni-adjusted Dunn-style pairwise tests."""
    validate_columns(df, [outcome, grouping]); clean = df[[outcome, grouping]].copy(); clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce"); clean = clean.dropna(); grouped = list(clean.groupby(grouping, observed=True)); groups = [g[outcome].to_numpy() for _, g in grouped]
    if len(groups) < 2 or any(len(g)<2 for g in groups): raise ValueError("Kruskal-Wallis needs at least two groups with two observations each.")
    h, p = stats.kruskal(*groups); result = empty_result(f"Kruskal-Wallis: {outcome} by {grouping}", len(clean)); result["tables"] = [pd.DataFrame([{"N": len(clean), "H": h, "df": len(groups)-1, "p": p}])]
    if p < alpha and posthoc:
        pairs = []; raw = []
        for (name_a, ga), (name_b, gb) in itertools.combinations(grouped, 2):
            u, pv = stats.mannwhitneyu(ga[outcome], gb[outcome], alternative="two-sided"); pairs.append((name_a, name_b, u)); raw.append(pv)
        adjusted = multipletests(raw, method="bonferroni")[1]; result["tables"].append(pd.DataFrame([{"Group 1": a, "Group 2": b, "U": u, "p adjusted": pv} for (a,b,u), pv in zip(pairs, adjusted)]))
    result["interpretation"] = f"The group distributions {'differed' if p < alpha else 'did not differ'} significantly, H({len(groups)-1}) = {h:.3f}, {p_apa(p)}."; return result


def two_way_anova(df: pd.DataFrame, outcome: str, factor_a: str, factor_b: str, alpha: float = .05) -> dict:
    """Run Type III two-way ANOVA using sum-to-zero contrasts."""
    validate_columns(df, [outcome, factor_a, factor_b]); clean = df[[outcome, factor_a, factor_b]].copy(); clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce"); clean = clean.dropna()
    if clean[factor_a].nunique() < 2 or clean[factor_b].nunique() < 2: raise ValueError("Each factor must contain at least two groups.")
    model = smf.ols(f"Q('{outcome}') ~ C(Q('{factor_a}'), Sum) * C(Q('{factor_b}'), Sum)", data=clean).fit(); table = anova_lm(model, typ=3).reset_index().rename(columns={"index":"Effect"}); table["N"] = len(clean)
    ss_error = float(table.loc[table["Effect"] == "Residual", "sum_sq"].iloc[0]); table["Partial eta squared"] = table["sum_sq"] / (table["sum_sq"] + ss_error)
    result = empty_result(f"Two-way ANOVA: {outcome} by {factor_a} × {factor_b}", len(clean)); result["tables"] = [table]; result["interpretation"] = "The Type III ANOVA table reports main effects and the interaction using sum-to-zero contrasts."; return result


def repeated_measures_anova(df: pd.DataFrame, subject: str, measures: list[str], alpha: float = .05) -> dict:
    """Run one-factor repeated-measures ANOVA on wide-format repeated measures."""
    validate_columns(df, [subject, *measures]);
    if len(measures) < 2: raise ValueError("Select at least two repeated-measure columns.")
    clean = df[[subject, *measures]].copy(); clean[measures] = clean[measures].apply(pd.to_numeric, errors="coerce"); clean = clean.dropna(); long = clean.melt(id_vars=subject, value_vars=measures, var_name="Condition", value_name="Value")
    model = AnovaRM(long, depvar="Value", subject=subject, within=["Condition"]).fit(); table = model.anova_table.reset_index().rename(columns={"index":"Effect"}); result = empty_result(f"Repeated-measures ANOVA: {' / '.join(measures)}", len(clean)); result["tables"] = [table]; result["interpretation"] = "The repeated-measures ANOVA tests whether the mean outcome differs across conditions."; return result


def ancova(df: pd.DataFrame, outcome: str, factor: str, covariate: str, alpha: float = .05) -> dict:
    """Run Type III ANCOVA with a categorical factor and continuous covariate."""
    validate_columns(df, [outcome, factor, covariate]); clean = df[[outcome, factor, covariate]].copy(); clean[outcome] = pd.to_numeric(clean[outcome], errors="coerce"); clean[covariate] = pd.to_numeric(clean[covariate], errors="coerce"); clean = clean.dropna()
    if clean[factor].nunique() < 2: raise ValueError("The ANCOVA factor must contain at least two groups.")
    model = smf.ols(f"Q('{outcome}') ~ C(Q('{factor}'), Sum) + Q('{covariate}')", data=clean).fit(); table = anova_lm(model, typ=3).reset_index().rename(columns={"index":"Effect"}); ss_error = float(table.loc[table["Effect"] == "Residual", "sum_sq"].iloc[0]); table["Partial eta squared"] = table["sum_sq"] / (table["sum_sq"] + ss_error); result = empty_result(f"ANCOVA: {outcome} by {factor}, covariate {covariate}", len(clean)); result["tables"] = [table]; result["interpretation"] = "The Type III ANCOVA adjusts the group comparison for the selected continuous covariate."; return result
