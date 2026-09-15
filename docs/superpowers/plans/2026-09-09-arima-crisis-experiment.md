# ARIMA Crisis Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and report the separate 2024 `ARIMA(2,1,2)` crisis-only experiment for Germany and Austria using only `is_covid_period` and `is_post_invasion`, without overwriting existing model results.

**Architecture:** Extend the existing regression-ARIMA fit and rolling-validation infrastructure with a distinct crisis-only specification and `arima_intervention` output family. Reuse the existing `build_crisis_features` definitions, checkpoint/upsert logic, daily refitting, bridge forecasting, target-day scoring, and fit diagnostics. Add the new family to the deterministic central summary builder as an unselected comparator, so the existing selected registry entries remain unchanged.

**Tech Stack:** Python 3, pandas, NumPy, statsmodels `ARIMA`, pytest, existing project interpreter `.venv\\Scripts\\python.exe`.

---

### Task 1: Add the crisis-only specification contract

**Files:**
- Modify: `code/04_models/regression_arima_errors/regression_arima_errors_models.py`
- Modify: `code/04_models/regression_arima_errors/test_regression_arima_errors_models.py`

- [ ] **Step 1: Write the failing specification test**

Add a test that imports `CRISIS_SPECIFICATION_ID`, calls `specification_columns(CRISIS_SPECIFICATION_ID)`, and asserts the exact columns are `("is_covid_period", "is_post_invasion")`. Build exogenous data across ordinary, COVID, and post-invasion timestamps and assert the output has exactly those two finite binary columns and full rank.

- [ ] **Step 2: Run the focused model test and verify the expected failure**

Run:
`\.venv\\Scripts\\python.exe -m pytest code/04_models/regression_arima_errors/test_regression_arima_errors_models.py -k crisis_only -q`

Expected result: collection or assertion failure because the new specification identifier is not yet defined.

- [ ] **Step 3: Implement the minimal crisis-only contract**

Add `CRISIS_SPECIFICATION_ID = "arima_p2_d1_q2_crisis"` and a dedicated `crisis` specification group. Make `specification_columns` return only the two crisis columns for this identifier. Make `build_exog` reuse `build_crisis_features` with the country-local timestamp conversion, without adding calendar, temperature, load-forecast, or other columns. Keep `ERROR_ORDER == (2, 1, 2)`, `TREND == "n"`, and the existing `ARIMA` constructor flags unchanged.

- [ ] **Step 4: Run the focused model tests**

Run:
`\.venv\\Scripts\\python.exe -m pytest code/04_models/regression_arima_errors/test_regression_arima_errors_models.py -q`

Expected result: all model tests pass, including the new exact-column and binary-boundary checks.

### Task 2: Add a separate `arima_intervention` rolling-validation runner

**Files:**
- Create: `code/04_models/regression_arima_errors/arima_intervention_validation_2024.py`
- Create: `code/04_models/regression_arima_errors/test_arima_intervention_validation_2024.py`

- [ ] **Step 1: Write the failing runner contract tests**

Test that the runner constants are:
`MODEL_FAMILY = "arima_intervention"`, `SPECIFICATION_IDS = ("arima_p2_d1_q2_crisis",)`, and the six mini jobs are Germany/Austria crossed with 2024-02-15, 2024-03-31, and 2024-10-27. Test that full target dates contain exactly 366 local-calendar dates per country, all in 2024. Test the specification manifest contains six unique mini keys and the requested `(2, 1, 2)`/`n` configuration.

- [ ] **Step 2: Run the runner tests and verify the expected failure**

Run:
`\.venv\\Scripts\\python.exe -m pytest code/04_models/regression_arima_errors/test_arima_intervention_validation_2024.py -q`

Expected result: import failure because the new runner does not yet exist.

- [ ] **Step 3: Implement the runner as a thin wrapper**

Reuse `regression_arima_errors_validation_2024.run_validation` with `model_family="arima_intervention"` and the crisis-only specification tuple. Provide `--mini` and `--full` CLI stages, `--workers`, `--output-dir`, and `--processed-directory`. For `--full`, derive local 2024 dates through the existing `_validation_frame`/`COUNTRY_CONFIG` path. Do not change the forecast-path, information-set, checkpoint, retry, convergence, or failure behavior.

- [ ] **Step 4: Run runner tests and existing regression-ARIMA tests**

Run:
`\.venv\\Scripts\\python.exe -m pytest code/04_models/regression_arima_errors/test_arima_intervention_validation_2024.py code/04_models/regression_arima_errors/test_regression_arima_errors_validation_2024.py -q`

Expected result: all tests pass.

### Task 3: Run focused tests and the six-job mini validation

**Files:**
- Create: `results/arima_intervention/validation_2024/mini_validation/arima_intervention_*_2024.csv` and runtime records through the runner; do not modify existing result directories.

- [ ] **Step 1: Run the focused test set**

Run:
`\.venv\\Scripts\\python.exe -m pytest code/04_models/regression_arima_errors/test_regression_arima_errors_models.py code/04_models/regression_arima_errors/test_regression_arima_errors_validation_2024.py code/04_models/regression_arima_errors/test_arima_intervention_validation_2024.py code/02_preprocessing/test_crisis_features.py -q`

- [ ] **Step 2: Run the six-job mini validation with fixed BLAS thread counts**

Run with environment variables `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`:
`\.venv\\Scripts\\python.exe code/04_models/regression_arima_errors/arima_intervention_validation_2024.py --mini --workers 8 --output-dir results/arima_intervention/validation_2024/mini_validation --processed-directory data/processed`

- [ ] **Step 3: Verify mini output before full execution**

Require six unique completed jobs, no failed jobs, six 24/23/25-hour target-day lengths as applicable, target-day coverage 1.0, only the two crisis exogenous columns in the specification file, no timestamps beyond local 2024, and no future load or temperature columns in the forecast artifacts.

### Task 4: Run the full 732-job validation

**Files:**
- Create: `results/arima_intervention/validation_2024/full_validation/arima_intervention_*_2024.csv` and runtime records through checkpointed persistence.

- [ ] **Step 1: Run full validation with eight workers and fixed thread counts**

Run with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`:
`\.venv\\Scripts\\python.exe code/04_models/regression_arima_errors/arima_intervention_validation_2024.py --full --workers 8 --output-dir results/arima_intervention/validation_2024/full_validation --processed-directory data/processed`

- [ ] **Step 2: Resume if execution is interrupted**

Rerun the exact same command. Existing completed `(country, target_date, specification_id)` keys and forecast rows must be reused by the existing checkpoint/upsert logic; no duplicate job or timestamp keys may remain.

- [ ] **Step 3: Verify full coverage and validation rules**

Require 732 unique completed jobs, zero failed jobs, 8,784 evaluated observations per country, coverage 1.0, correct Germany 18:00 and Austria 08:00 previous-day origins, target-day-only evaluation, correct DST target lengths, no 2025 local dates, and no future observed values or external forecasts in the model inputs.

### Task 5: Build comparison and diagnostics report

**Files:**
- Create: `code/04_models/regression_arima_errors/arima_intervention_report_2024.py`
- Create: `results/arima_intervention/tables/arima_vs_crisis_comparison_2024.csv`
- Create: `results/arima_intervention/tables/arima_intervention_coefficients_2024.csv`
- Create: `results/arima_intervention/tables/arima_intervention_residual_diagnostics_2024.csv`
- Create: `results/arima_intervention/tables/arima_intervention_run_summary_2024.csv`
- Create: `results/arima_intervention/tables/arima_intervention_report_2024.md`

- [ ] **Step 1: Implement deterministic report calculations**

Read the new full summary and jobs files plus the existing plain ARIMA `(2,1,2)` rows. Produce Germany and Austria rows with MAE, RMSE, MAPE, coverage, absolute and percentage MAE/RMSE changes, and MAPE percentage-point change. Parse job-level parameter estimates for `is_covid_period` and `is_post_invasion`, and summarize counts, means, medians, standard deviations, finite standard errors, and signed-standard-error counts. Summarize convergence, retry, warning, residual effective counts, flagged ACF jobs, and significant Ljung-Box jobs using the established fit diagnostics.

- [ ] **Step 2: Write the report with the exact specification and restrictions**

Record `model_family = arima_intervention`, `specification_id = arima_p2_d1_q2_crisis`, `(2,1,2)`, `trend="n"`, the two intervention definitions, daily refitting, target-day scoring, DST handling, and the exclusion of calendar, temperature, official forecasts, future load, and 2025 data.

### Task 6: Update central summaries without selecting the experiment

**Files:**
- Modify: `code/04_models/common/build_model_validation_summaries.py`
- Modify: `code/04_models/common/test_build_model_validation_summaries.py` if required by the new unselected-family behavior.

- [ ] **Step 1: Add `arima_intervention` as an optional unselected source**

Add the family ordering and source path for `results/arima_intervention/validation_2024/full_validation/arima_intervention_validation_2024_summary.csv`. Allow a non-baseline family with no selection path to contribute to `all_models` while remaining unselected and absent from the frozen registry. Keep all existing family selection behavior unchanged.

- [ ] **Step 2: Run summary-builder tests before writing central outputs**

Run:
`\.venv\\Scripts\\python.exe -m pytest code/04_models/common/test_build_model_validation_summaries.py -q`

- [ ] **Step 3: Run the deterministic central summary builder**

Run:
`\.venv\\Scripts\\python.exe code/04_models/common/build_model_validation_summaries.py --output-directory results`

Expected result: central `model_validation_2024_all_models.csv` gains two unselected `arima_intervention` rows; existing selected rows remain unchanged; `selected_models_registry.csv` has no `arima_intervention` entry.

### Task 7: Final verification

**Files:**
- No further code changes unless verification identifies a concrete failure.

- [ ] **Step 1: Verify all requested outputs and metrics**

Check both countries, 8,784 observations, 1.0 coverage, all required comparison columns, coefficient rows for both crisis variables, residual diagnostics, convergence/retry counts, and the central summary entries.

- [ ] **Step 2: Verify non-overwrite and scope constraints**

Compare pre-run hashes for existing ARIMA, regression-ARIMA, intervention, SARIMA, SARIMAX, official benchmark, and raw data files. Confirm only the new `arima_intervention` outputs, the requested code/tests/plan, and the central summary tables changed. Confirm no SARIMA, SARIMAX, or 2025 outputs were created or modified.

- [ ] **Step 3: Run the final focused test set**

Run the model, validation, crisis-feature, and summary-builder tests again and report their exit status and counts.
