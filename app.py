"""Streamlit entry point for the complete thesis statistics app."""
from __future__ import annotations
from io import BytesIO
import pandas as pd
import streamlit as st

from core.assumptions import levene_test, shapiro_test, vif_table
from core.charts import bar_chart, boxplot, histogram
from core.compare import (ancova, independent_t_test, kruskal_wallis, mann_whitney, one_way_anova,
                          paired_t_test, repeated_measures_anova, two_way_anova, wilcoxon_signed_rank)
from core.data import infer_variable_types, list_excel_sheets, load_excel, variables_of_type
from core.descriptives import crosstabulation, frequency_table, scale_descriptives
from core.export_excel import figure_to_png, figures_to_zip, results_to_excel
from core.relationships import chi_square, correlation_matrix, linear_regression, logistic_regression
from core.report_word import results_to_word
from core.utils import round_tables

st.set_page_config(page_title="Thesis Stats App", page_icon="", layout="wide")
for key, default in {"df": pd.DataFrame(), "var_types": {}, "settings": {"alpha": .05, "confidence": .95, "decimals": 3}, "results": [], "upload_signature": None}.items():
    st.session_state.setdefault(key, default)


@st.cache_data(show_spinner=False)
def cached_sheet_names(raw: bytes) -> list[str]:
    """Cache Excel sheet-name discovery without writing the upload to disk."""
    return list_excel_sheets(BytesIO(raw))


@st.cache_data(show_spinner=False)
def cached_load_excel(raw: bytes, sheet_name: str) -> pd.DataFrame:
    """Cache the in-memory DataFrame produced from an uploaded workbook."""
    return load_excel(BytesIO(raw), sheet_name)


def save_result(result: dict) -> None:
    """Replace a matching prior analysis result or append a new one."""
    titles = [r.get("title") for r in st.session_state.results]
    if result.get("title") in titles: st.session_state.results[titles.index(result["title"])] = result
    else: st.session_state.results.append(result)


def show_result(result: dict) -> None:
    """Render and save a standardized analysis result."""
    result = round_tables(result, int(st.session_state.settings["decimals"])); save_result(result); st.success("Analysis completed and saved to Results."); st.subheader(result.get("title", "Result")); st.caption(f"N = {result.get('n', 'NA')}")
    for warning in result.get("warnings", []): st.warning(warning)
    for table in result.get("tables", []): st.dataframe(table, use_container_width=True)
    for figure in result.get("figures", []): st.pyplot(figure, clear_figure=False)
    if result.get("interpretation"): st.info(result["interpretation"])


def options(types: dict, kinds: str | list[str]) -> list[str]:
    """Return configured variables for one or more measurement types."""
    return variables_of_type(types, kinds)


def setup_page() -> None:
    """Render Excel upload, preview, and variable-type confirmation."""
    st.header("1. Data setup"); st.caption("Uploaded files are processed in memory only and are not sent to external services.")
    uploaded = st.file_uploader("Upload one Excel workbook (.xlsx)", type=["xlsx"])
    if not uploaded: st.info("Please upload data first."); return
    raw = uploaded.getvalue(); signature = (uploaded.name, len(raw), hash(raw))
    try:
        sheets = cached_sheet_names(raw); sheet = st.selectbox("Select sheet", sheets); df = cached_load_excel(raw, sheet)
        if st.session_state.upload_signature != signature: st.session_state.df = df; st.session_state.var_types = infer_variable_types(df); st.session_state.results = []; st.session_state.upload_signature = signature
        st.write(f"**{len(df):,} rows × {len(df.columns):,} variables**"); st.dataframe(df.head(20), use_container_width=True); st.subheader("Variable types")
        type_df = pd.DataFrame({"Variable": list(df.columns), "Type": [st.session_state.var_types.get(c, "nominal") for c in df.columns]})
        edited = st.data_editor(type_df, hide_index=True, disabled=["Variable"], column_config={"Type": st.column_config.SelectboxColumn("Type", options=["nominal", "ordinal", "scale"], required=True)}, use_container_width=True)
        if st.button("Confirm variable types", type="primary"): st.session_state.var_types = dict(zip(edited["Variable"], edited["Type"])); st.success("Variable types confirmed.")
    except Exception as exc: st.error(f"Could not read this workbook: {exc}")


def descriptives_page() -> None:
    """Render descriptives and chart controls."""
    st.header("2. Descriptives and charts")
    if st.session_state.df.empty: st.info("Please upload data first."); return
    df, types = st.session_state.df, st.session_state.var_types; analysis = st.selectbox("Analysis", ["Scale descriptives", "Frequency table", "Crosstabulation", "Histogram", "Bar chart", "Boxplot"])
    try:
        if analysis == "Scale descriptives":
            vars_ = options(types, "scale");
            if not vars_: st.warning("No scale variables are configured."); return
            var = st.selectbox("Scale variable", vars_)
            if st.button("Run descriptives"): show_result(scale_descriptives(df, var, st.session_state.settings["confidence"]))
        elif analysis == "Frequency table":
            vars_ = options(types, ["nominal", "ordinal"]);
            if not vars_: st.warning("No categorical variables are configured."); return
            var = st.selectbox("Categorical variable", vars_)
            if st.button("Run frequencies"): show_result(frequency_table(df, var))
        elif analysis == "Crosstabulation":
            vars_ = options(types, ["nominal", "ordinal"]);
            if len(vars_) < 2: st.warning("At least two categorical variables are required."); return
            a = st.selectbox("Rows", vars_); b = st.selectbox("Columns", [v for v in vars_ if v != a])
            if st.button("Run crosstab"): show_result(crosstabulation(df, a, b))
        elif analysis == "Histogram":
            vars_ = options(types, "scale");
            if not vars_: st.warning("No scale variables are configured."); return
            var = st.selectbox("Scale variable", vars_); normal = st.checkbox("Show normal curve", True)
            if st.button("Create histogram"): show_result({"title": f"Histogram: {var}", "n": int(pd.to_numeric(df[var], errors="coerce").notna().sum()), "tables": [], "figures": [histogram(df, var, normal)], "warnings": [], "interpretation": "Inspect the distribution and normal curve visually."})
        elif analysis == "Bar chart":
            vars_ = options(types, ["nominal", "ordinal"]);
            if not vars_: st.warning("No categorical variables are configured."); return
            var = st.selectbox("Categorical variable", vars_)
            if st.button("Create bar chart"): show_result({"title": f"Bar chart: {var}", "n": int(df[var].notna().sum()), "tables": [], "figures": [bar_chart(df, var)], "warnings": [], "interpretation": "The chart shows observed category frequencies."})
        else:
            vars_ = options(types, "scale");
            if not vars_: st.warning("No scale variables are configured."); return
            var = st.selectbox("Scale variable", vars_); groups = ["None"] + options(types, ["nominal", "ordinal"]); group = st.selectbox("Optional grouping", groups); group = None if group == "None" else group
            if st.button("Create boxplot"): show_result({"title": f"Boxplot: {var}", "n": int(pd.to_numeric(df[var], errors="coerce").notna().sum()), "tables": [], "figures": [boxplot(df, var, group)], "warnings": [], "interpretation": "The boxplot displays median, quartiles, and potential outliers."})
    except Exception as exc: st.error(str(exc))


def assumptions_page() -> None:
    """Render normality, variance, and multicollinearity checks."""
    st.header("3. Assumptions")
    if st.session_state.df.empty: st.info("Please upload data first."); return
    df, types, alpha = st.session_state.df, st.session_state.var_types, st.session_state.settings["alpha"]; analysis = st.selectbox("Check", ["Shapiro-Wilk", "Levene's test", "VIF"])
    try:
        if analysis == "Shapiro-Wilk":
            vars_ = options(types, "scale");
            if not vars_: st.warning("No scale variables are configured."); return
            var = st.selectbox("Scale variable", vars_)
            if st.button("Run Shapiro-Wilk"): show_result(shapiro_test(df, var, alpha))
        elif analysis == "Levene's test":
            outcomes, groups = options(types, "scale"), options(types, ["nominal", "ordinal"])
            if not outcomes or not groups: st.warning("A scale outcome and categorical grouping variable are required."); return
            outcome, group = st.selectbox("Outcome", outcomes), st.selectbox("Grouping", groups)
            if st.button("Run Levene's test"): show_result(levene_test(df, outcome, group, alpha))
        else:
            vars_ = options(types, "scale"); predictors = st.multiselect("Numeric predictors", vars_)
            if st.button("Calculate VIF"): show_result(vif_table(df, predictors))
    except Exception as exc: st.error(str(exc))


def compare_page() -> None:
    """Render all group-comparison analyses."""
    st.header("4. Compare groups")
    if st.session_state.df.empty: st.info("Please upload data first."); return
    df, types, alpha = st.session_state.df, st.session_state.var_types, st.session_state.settings["alpha"]; analysis = st.selectbox("Test", ["Independent t-test", "Paired t-test", "Mann-Whitney U", "Wilcoxon signed-rank", "One-way ANOVA", "Kruskal-Wallis", "Two-way ANOVA", "Repeated-measures ANOVA", "ANCOVA"]); scales, cats = options(types, "scale"), options(types, ["nominal", "ordinal"])
    try:
        if analysis in ("Independent t-test", "Mann-Whitney U", "One-way ANOVA", "Kruskal-Wallis"):
            if not scales or not cats: st.warning("A scale outcome and categorical grouping variable are required."); return
            outcome, group = st.selectbox("Scale outcome", scales), st.selectbox("Grouping variable", cats)
            if st.button("Run test"): show_result({"Independent t-test": independent_t_test, "Mann-Whitney U": mann_whitney, "One-way ANOVA": one_way_anova, "Kruskal-Wallis": kruskal_wallis}[analysis](df, outcome, group, alpha))
        elif analysis in ("Paired t-test", "Wilcoxon signed-rank"):
            if len(scales) < 2: st.warning("At least two scale variables are required."); return
            first, second = st.selectbox("First measurement", scales), st.selectbox("Second measurement", [v for v in scales if v != first])
            if st.button("Run paired test"): show_result((paired_t_test if analysis == "Paired t-test" else wilcoxon_signed_rank)(df, first, second, alpha, st.session_state.settings["confidence"]) if analysis == "Paired t-test" else wilcoxon_signed_rank(df, first, second, alpha))
        elif analysis == "Two-way ANOVA":
            if not scales or len(cats) < 2: st.warning("A scale outcome and two categorical factors are required."); return
            outcome, fa = st.selectbox("Outcome", scales), st.selectbox("Factor A", cats); fb = st.selectbox("Factor B", [v for v in cats if v != fa])
            if st.button("Run two-way ANOVA"): show_result(two_way_anova(df, outcome, fa, fb, alpha))
        elif analysis == "Repeated-measures ANOVA":
            if len(scales) < 2: st.warning("At least two repeated scale measurements are required."); return
            subject = st.selectbox("Subject ID", list(df.columns)); measures = st.multiselect("Repeated measurements", [v for v in scales if v != subject])
            if st.button("Run repeated-measures ANOVA"): show_result(repeated_measures_anova(df, subject, measures, alpha))
        else:
            if not scales or not cats: st.warning("A scale outcome, categorical factor, and scale covariate are required."); return
            outcome, factor = st.selectbox("Outcome", scales), st.selectbox("Factor", cats); cov = st.selectbox("Covariate", [v for v in scales if v != outcome])
            if st.button("Run ANCOVA"): show_result(ancova(df, outcome, factor, cov, alpha))
    except Exception as exc: st.error(str(exc))


def relationships_page() -> None:
    """Render correlations, chi-square, linear regression, and logistic regression."""
    st.header("5. Relationships")
    if st.session_state.df.empty: st.info("Please upload data first."); return
    df, types, alpha = st.session_state.df, st.session_state.var_types, st.session_state.settings["alpha"]; analysis = st.selectbox("Analysis", ["Pearson correlation", "Spearman correlation", "Chi-square", "Linear regression", "Logistic regression"]); scales, cats = options(types, "scale"), options(types, ["nominal", "ordinal"])
    try:
        if analysis in ("Pearson correlation", "Spearman correlation"):
            variables = st.multiselect("Scale variables", scales)
            if st.button("Run correlation"): show_result(correlation_matrix(df, variables, "pearson" if analysis.startswith("Pearson") else "spearman", alpha))
        elif analysis == "Chi-square":
            if len(cats) < 2: st.warning("At least two categorical variables are required."); return
            row = st.selectbox("Rows", cats); col = st.selectbox("Columns", [v for v in cats if v != row])
            if st.button("Run chi-square"): show_result(chi_square(df, row, col, alpha))
        elif analysis == "Linear regression":
            if not scales: st.warning("Scale variables are required."); return
            outcome = st.selectbox("Scale outcome", scales); predictors = st.multiselect("Scale predictors", [v for v in scales if v != outcome])
            if st.button("Run regression"): show_result(linear_regression(df, outcome, predictors, alpha, st.session_state.settings["confidence"]))
        else:
            if not scales: st.warning("Scale predictors are required."); return
            outcome = st.selectbox("Binary outcome", list(df.columns)); predictors = st.multiselect("Numeric predictors", [v for v in scales if v != outcome])
            if st.button("Run logistic regression"): show_result(logistic_regression(df, outcome, predictors, alpha, st.session_state.settings["confidence"]))
    except Exception as exc: st.error(str(exc))


def export_page() -> None:
    """Render Excel, PNG ZIP, and Word downloads for saved analyses."""
    st.header("6. Export")
    if not st.session_state.results: st.info("Completed analyses will appear here automatically."); return
    st.write(f"**{len(st.session_state.results)} saved result(s)**")
    for i, result in enumerate(st.session_state.results): st.checkbox(result.get("title", f"Result {i+1}"), value=True, key=f"result_{i}")
    selected = [r for i, r in enumerate(st.session_state.results) if st.session_state.get(f"result_{i}", True)]
    st.download_button("Download Excel workbook", results_to_excel(selected), "thesis_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download chart ZIP", figures_to_zip(selected), "thesis_charts.zip", "application/zip")
    st.download_button("Download Word report", results_to_word(selected), "thesis_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    st.subheader("Individual chart downloads")
    chart_number = 0
    for result_number, result in enumerate(selected, 1):
        for figure_number, figure in enumerate(result.get("figures", []), 1):
            chart_number += 1
            st.download_button(f"Download chart {chart_number}: {result.get('title', 'Result')}", figure_to_png(figure), f"result_{result_number}_figure_{figure_number}.png", "image/png", key=f"individual_chart_{result_number}_{figure_number}")


with st.sidebar:
    st.title("Thesis Stats App"); page = st.radio("Navigate", ["Data setup", "Descriptives", "Assumptions", "Compare groups", "Relationships", "Export"]); st.divider(); st.subheader("Settings")
    st.session_state.settings["alpha"] = st.number_input("Alpha", min_value=.001, max_value=.20, value=float(st.session_state.settings["alpha"]), step=.001, format="%.3f")
    st.session_state.settings["confidence"] = st.number_input("Confidence level", min_value=.80, max_value=.999, value=float(st.session_state.settings["confidence"]), step=.01, format="%.2f"); st.session_state.settings["decimals"] = st.number_input("Table decimals", min_value=0, max_value=6, value=int(st.session_state.settings["decimals"]), step=1); st.caption("All participants are female; no gender analysis is included.")
{"Data setup": setup_page, "Descriptives": descriptives_page, "Assumptions": assumptions_page, "Compare groups": compare_page, "Relationships": relationships_page, "Export": export_page}[page]()
