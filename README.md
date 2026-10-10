# Thesis Stats App

A self-contained Streamlit application for thesis analysis of pregnancy research data. Uploaded Excel workbooks are processed in memory only and are not sent to external services.

## Deploy to Streamlit Community Cloud

1. Upload the contents of this folder to a GitHub repository.
2. Create a new Streamlit Cloud app.
3. Select `app.py` as the main file.
4. Deploy. `requirements.txt` and `runtime.txt` are included automatically.

No secrets or environment variables are required.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the complete test suite with `pytest`.

## Complete feature set

- Excel `.xlsx` upload, sheet selection, preview, and editable nominal/ordinal/scale metadata
- Scale descriptives: N, mean, SD, SE, median, min, max, skewness, kurtosis, and configurable confidence interval
- Frequencies, percentages, and crosstabs with counts, row percentages, and column percentages
- Histogram with optional normal curve, bar chart, boxplot, and grouped boxplot
- Shapiro-Wilk with Q-Q plot, Levene's test, and VIF
- Independent Student and Welch t-tests with Cohen's d and Levene warning
- Paired t-test with confidence interval and Cohen's dz
- Mann-Whitney U and Wilcoxon signed-rank tests
- One-way ANOVA with eta squared and Tukey post hoc
- Kruskal-Wallis with Bonferroni-adjusted pairwise post hoc comparisons
- Type III two-way ANOVA with sum-to-zero contrasts and partial eta squared
- Repeated-measures ANOVA
- Type III ANCOVA with partial eta squared
- Pearson and Spearman correlation matrices with pairwise N, p-values, and heatmap
- Chi-square association with Cramer's V and expected-count warnings
- Linear regression with coefficients, standardized betas, confidence intervals, R², adjusted R², F-test, and VIF
- Binary logistic regression with coefficients, odds ratios, confidence intervals, likelihood-ratio fit, and pseudo-R²
- Session result list with tables, figures, warnings, and interpretations
- Excel workbook, PNG chart ZIP, individual PNG chart downloads, and Word report downloads

All result functions live in `core/` and return the same standardized result dictionary for consistent display and export.

## Privacy and study context

All participants are female, so the app intentionally contains no gender analysis or gender grouping. It does not create a separate missing-value summary; each analysis uses its appropriate complete/pairwise cases and reports N. Uploaded files remain in memory only. The workbook in `sample_data/` is synthetic and is included only for smoke testing.

Workbook sheet discovery and loading use Streamlit's in-memory `st.cache_data`; no uploaded file is written to disk.
