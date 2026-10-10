import numpy as np
import pandas as pd

from core.data import infer_variable_types
from core.descriptives import scale_descriptives, frequency_table
from core.assumptions import shapiro_test, levene_test, vif_table
from core.compare import (ancova, independent_t_test, kruskal_wallis, mann_whitney, one_way_anova,
                          paired_t_test, repeated_measures_anova, two_way_anova, wilcoxon_signed_rank)
from core.relationships import correlation_matrix, chi_square, linear_regression, logistic_regression
from core.export_excel import figure_to_png, results_to_excel, figures_to_zip
from core.report_word import results_to_word


def sample_df():
    rng = np.random.default_rng(11)
    n = 36
    return pd.DataFrame({
        "id": range(n), "x": np.linspace(1, 36, n), "y": np.linspace(2, 72, n),
        "before": rng.normal(10, 1, n), "after": rng.normal(11, 1, n),
        "score": rng.normal(12, 1, n), "group": np.tile(["A", "B", "C"], 12),
        "factor_a": np.tile(["Low", "High"], 18), "factor_b": np.repeat(["No", "Yes"], 18),
        "binary": np.tile([0, 1], 18), "covariate": rng.normal(5, 1, n),
        "cat_a": np.tile(["Y", "N"], 18), "cat_b": np.repeat(["Y", "N"], 18),
    })


def test_type_inference_and_descriptives():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "status": ["A", "A", "B", "B"]})
    assert infer_variable_types(df) == {"x": "scale", "status": "nominal"}
    result = scale_descriptives(df, "x")
    assert result["n"] == 4 and result["tables"][0].iloc[0]["Mean"] == 2.5


def test_frequency_and_t_test():
    df = pd.DataFrame({"score": [1, 2, 3, 4, 5, 6], "group": ["A", "A", "A", "B", "B", "B"]})
    assert frequency_table(df, "group")["tables"][0]["Frequency"].sum() == 6
    assert independent_t_test(df, "score", "group")["n"] == 6
    assert mann_whitney(df, "score", "group")["n"] == 6


def test_assumptions_and_relationships():
    df = sample_df()
    assert shapiro_test(df, "score")["n"] == 36
    assert levene_test(df, "score", "group")["n"] == 36
    assert vif_table(df, ["x", "covariate"])["n"] == 36
    corr = correlation_matrix(df, ["x", "y"])
    assert np.isclose(corr["tables"][0].loc["x", "y r"], 1.0) and corr["figures"]
    assert chi_square(df, "cat_a", "cat_b")["n"] == 36
    assert linear_regression(df, "score", ["x", "covariate"])["n"] == 36
    assert logistic_regression(df, "binary", ["x", "covariate"])["n"] == 36


def test_all_group_comparisons():
    df = sample_df()
    assert paired_t_test(df, "before", "after")["n"] == 36
    assert wilcoxon_signed_rank(df, "before", "after")["n"] == 36
    assert one_way_anova(df, "score", "group")["n"] == 36
    assert kruskal_wallis(df, "score", "group")["n"] == 36
    assert two_way_anova(df, "score", "factor_a", "factor_b")["n"] == 36
    assert repeated_measures_anova(df, "id", ["before", "after"])["n"] == 36
    assert ancova(df, "score", "group", "covariate")["n"] == 36


def test_exports():
    df = sample_df(); results = [scale_descriptives(df, "score"), correlation_matrix(df, ["x", "y"])]
    assert len(results_to_excel(results)) > 1000
    assert len(figures_to_zip(results)) > 20
    assert figure_to_png(results[1]["figures"][0]).startswith(b"\x89PNG")
    assert len(results_to_word(results)) > 1000
