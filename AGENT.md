# AGENT.md — Thesis Stats App (SPSS/jamovi-style, Streamlit)

Read this file fully before making any change. Follow it unless the user explicitly overrides it.

## 1. Project goal

Build a **Streamlit web app** that lets the user upload an Excel file of thesis research data and run SPSS/jamovi-like statistical analyses through dropdowns and buttons, with no coding needed. Results must be thesis-ready: clean tables, charts, Excel export, and a Word report.

## 2. Study context

- Thesis topic: **pregnancy analysis in females**.
- **All participants are female.** There is no gender variable to analyse. Do NOT build value-label features, a gender-grouping feature, or any gender-specific logic.
- Data comes from a single Excel sheet (.xlsx), one row per participant, one column per variable.
- Do NOT build a missing-value summary. Analyses simply use the available cases for each test (pairwise/listwise as appropriate), and the number of cases used (N) must be shown in every result.

## 3. Tech stack

| Purpose | Library |
|---|---|
| UI | streamlit |
| Data | pandas, numpy, openpyxl |
| Tests | scipy, statsmodels, pingouin |
| Charts | matplotlib, seaborn |
| Excel output | XlsxWriter |
| Word report | python-docx |
| Testing | pytest |

Python 3.10+. Pin versions in `requirements.txt`. Do not add new libraries without a clear reason.

## 4. Folder structure

```
thesis-stats-app/
├── AGENT.md
├── requirements.txt
├── app.py                  # Streamlit entry point, sidebar navigation
├── pages/                  # One Streamlit page per phase/feature group
│   ├── 1_Data_Setup.py
│   ├── 2_Descriptives.py
│   ├── 3_Assumptions.py
│   ├── 4_Compare_Groups.py
│   ├── 5_Relationships.py
│   └── 6_Export.py
├── core/                   # Pure logic, NO streamlit imports here
│   ├── data.py             # load Excel, variable types
│   ├── descriptives.py
│   ├── assumptions.py
│   ├── compare.py
│   ├── relationships.py
│   ├── charts.py
│   ├── export_excel.py
│   └── report_word.py
├── tests/                  # pytest, with small known-answer datasets
└── sample_data/            # dummy data only, never real participant data
```

Rule: **all statistics live in `core/`** as plain functions that take a DataFrame and return DataFrames/figures. Pages only handle UI. This keeps the logic testable.

## 5. App workflow (how the app runs)

### 5.1 User journey

```
Upload Excel → Set variable types → Pick an analysis → Select variables
→ Check assumptions → Run test → View table + chart + interpretation
→ Result auto-saved to "Results" list → Repeat for other analyses
→ Export (Excel workbook / chart images / Word report)
```

1. **Start:** the user opens the app (`streamlit run app.py`) and lands on the Data Setup page.
2. **Upload:** the user uploads the .xlsx file and picks the sheet. The data preview appears.
3. **Variable types:** an editable table lists every column with its auto-detected type (nominal / ordinal / scale). The user corrects it and clicks **Confirm**. Other pages stay locked until this is done.
4. **Choose an analysis:** the user opens a page from the sidebar (Descriptives, Assumptions, Compare Groups, Relationships).
5. **Select variables:** dropdowns show only variables of the valid type for that test (e.g. t-test: one scale outcome + one nominal grouping variable with 2 groups).
6. **Run:** the user clicks **Run**. The app validates the inputs, runs the test, and shows the results.
7. **Output on screen:** each result shows (a) the table, (b) the chart, (c) warnings if assumptions are violated, (d) an APA-style interpretation sentence.
8. **Auto-save:** every completed analysis is stored in the Results list, so the user never has to copy tables manually.
9. **Export:** on the Export page the user ticks which results to include and downloads the Excel workbook, the zip of chart images, or the Word report.

### 5.2 Internal data flow

```
 Excel file
    │  (core/data.py: load_excel)
    ▼
 DataFrame ──► st.session_state["df"]
 Variable types ──► st.session_state["var_types"]
    │
    ▼
 Page (UI only): user selects variables + options
    │
    ▼
 core/ function (pure logic): validate → compute → return
    │      result = { title, tables[], figures[], warnings[], interpretation }
    ▼
 Page displays result  +  appends it to st.session_state["results"]
    │
    ▼
 Export page reads st.session_state["results"]
    ├─► core/export_excel.py → .xlsx (one sheet per result)
    ├─► core/charts.py       → PNG files / zip
    └─► core/report_word.py  → .docx
```

### 5.3 Session state keys

| Key | Holds |
|---|---|
| `df` | the loaded DataFrame |
| `var_types` | dict: column name → "nominal" / "ordinal" / "scale" |
| `settings` | alpha, CI level, decimals (set in the sidebar) |
| `results` | list of result dictionaries, in the order they were run |

### 5.4 Standard result format

Every function in `core/` that runs an analysis must return a dictionary with these keys, so the display, Excel, and Word code can handle all analyses the same way:

```python
{
    "title": "Independent t-test: Hemoglobin by Trimester",
    "n": 120,
    "tables": [DataFrame, ...],
    "figures": [matplotlib Figure, ...],
    "warnings": ["Levene's test significant: use Welch row"],
    "interpretation": "Hemoglobin differed significantly between ... t(98) = 2.31, p = .023, d = 0.46.",
}
```

### 5.5 Rules for the flow

- Pages that need data must check `st.session_state["df"]` and show "Please upload data first" if it is empty.
- If the user uploads a new file, reset `var_types` and `results` (after a confirmation prompt).
- Re-running the same analysis with the same settings replaces the earlier result instead of duplicating it.
- Never write the uploaded file to disk; keep it in memory only.
- A failed run must show a friendly message and leave existing results untouched.

## 6. Features in scope

**Phase 1 — Data setup**
- Upload .xlsx, choose sheet, preview data.
- Auto-detect each column as nominal / ordinal / scale; user can change it in an editable table.
- Store the choices in `st.session_state` and reuse them everywhere (variable dropdowns must only offer valid types for each analysis).

**Phase 2 — Descriptives and charts**
- Scale variables: N, mean, SD, SE, median, min, max, skewness, kurtosis, 95% CI of the mean.
- Categorical variables: frequency and percentage tables.
- Cross-tabulations with counts, row %, column %.
- Histogram (optional normal curve), bar chart, boxplot.

**Phase 3 — Assumption checks**
- Shapiro-Wilk with Q-Q plot.
- Levene's test (needed before t-test/ANOVA).
- VIF for multiple regression.

**Phase 4 — Compare groups**
- Independent t-test (Student and Welch) and paired t-test, with Cohen's d and 95% CI of the difference.
- Mann-Whitney U and Wilcoxon signed-rank.
- One-way ANOVA with Tukey post hoc, and Kruskal-Wallis (with a post hoc option, e.g. Dunn).
- Two-way ANOVA, repeated-measures ANOVA, ANCOVA (build last; confirm with the user whether they are needed).

**Phase 5 — Relationships**
- Pearson and Spearman correlation matrix with p-values and a heatmap.
- Chi-square test of association with Cramér's V and expected-count warning (flag cells with expected count < 5).
- Linear regression (simple and multiple): coefficients, standardized betas, 95% CI, R², adjusted R², F-test, VIF.
- Logistic regression: coefficients, odds ratios with 95% CI, model fit.

**Phase 6 — Output**
- One Excel workbook with one sheet per analysis run.
- All charts downloadable as PNG (and a zip of all).
- Word report: tables, charts, and an auto-written APA-style interpretation sentence per test.

## 7. Statistical defaults (match SPSS/jamovi so results can be verified)

- Alpha = 0.05; confidence intervals = 95%. Make both configurable in the sidebar.
- t-test: show both Student and Welch rows; recommend Welch if Levene's test is significant.
- ANOVA: Type III sums of squares with sum-to-zero (effect) contrasts, to match SPSS.
- Show effect sizes everywhere they exist (Cohen's d, eta squared / partial eta squared, Cramér's V, R²).
- Correlations: pairwise deletion; show N for each pair.
- Report p-values APA style: `p = .023`, and `p < .001` for very small values. Never print `p = 0.000`.
- Round to 3 decimals in tables by default.
- Each result must include the test name, N, test statistic, df, p-value, and effect size.
- Interpretation text must state facts only ("significant difference at the .05 level"). Do not claim causation.
- If an assumption is violated or a sample is too small, show a warning message beside the result and suggest the non-parametric alternative. Do not silently proceed.

## 8. Coding conventions

- Clear function names, type hints, and a short docstring on every function in `core/`.
- No hard-coded column names anywhere. Everything comes from the user's selections.
- Cache the loaded DataFrame with `st.cache_data`; keep user choices in `st.session_state`.
- Validate inputs before running (e.g. a t-test needs exactly 2 groups). Show friendly `st.error` messages, never raw tracebacks.
- Charts: white background, labelled axes, readable fonts, 300 dpi when exporting.
- Keep pages short; if a page grows past ~200 lines, move logic into `core/`.

## 9. Testing and verification

- Every function in `core/` gets a pytest test using a small dataset with known answers (from SPSS, jamovi, or a textbook).
- Before declaring a phase finished, run the same analysis on a real or sample dataset in jamovi/SPSS and compare numbers. Differences larger than rounding must be explained or fixed.
- Run `pytest` and `streamlit run app.py` and confirm both work before reporting done.

## 10. Data privacy

- The data is research data about pregnant women. Treat it as sensitive.
- Never commit real data files to git (add `*.xlsx` and `*.csv` to `.gitignore`, except `sample_data/`).
- Never upload data to external services or send it to any API. All processing stays local or in the user's own deployment.
- If deployed publicly, add a note in the app that uploaded data is not stored, and do not write uploads to disk.

## 11. How to work with the user

- Build **one phase at a time**, in order. After each phase: summarize what was added, how to run it, and what to test.
- Ask before adding anything not listed in section 6.
- Before writing code for a new analysis, state the test, its assumptions, and the default settings, briefly.
- When unsure which statistical option is correct for the thesis, ask the user and mention they should confirm it with their supervisor.
- Keep explanations simple. The user is a researcher, not a programmer.

## 12. Run commands

```bash
pip install -r requirements.txt
streamlit run app.py
pytest
```

## 13. Definition of done (per phase)

- [ ] Feature works in the app with real-looking sample data
- [ ] Logic is in `core/` with passing tests
- [ ] Numbers verified against jamovi/SPSS
- [ ] Friendly errors and assumption warnings in place
- [ ] Results exportable (Excel/PNG; Word once Phase 6 exists)
- [ ] No real data committed