# Regression ARIMA Errors 2025 Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run only the frozen `reg_arima_h_d_m_hol` Regression + ARIMA errors specification over all 2025 Germany and Austria target dates and append its results to the existing 2025 model tables.

**Architecture:** Add an isolated `regression_arima_errors_test_2025.py` runner that verifies the authoritative 2024 selection, registry, and specification row; delegates fitting, forecasting, metrics, and diagnostics to the existing regression-family modules; and owns only 2025 checkpointing, integrity validation, and table merging. The runner will use the existing origin-bounded information set and spawn-based worker pattern without changing the 2024 validator or other model families.

**Tech Stack:** Python, pandas, NumPy, statsmodels through existing regression-family modules, pytest, project interpreter `.venv/bin/python`.

**Spec:** User-provided Regression + ARIMA errors 2025 final-test requirements in the conversation.

## Global Constraints

- Use only `reg_arima_h_d_m_hol` for Germany and Austria.
- Use only ARIMA errors `(2, 1, 2)` with trend `n`.
- Use only the frozen hour, weekday, month, and public-holiday predictors defined by the authoritative specification row.
- Use chronological origin-bounded information sets and no future load, temperature, SMARD forecast, intervention, crisis, or 2025 reselection inputs.
- Treat `data/raw/` as immutable and do not modify completed outputs for other model families.
- Do not initialize Git, add remotes, upload files, or use machine-learning methods.
- Do not launch the full 730-job simulation until focused tests and a safe worker benchmark pass.

---

### Task 1: Add failing 2025 runner tests

**Files:**
- Create: `code/04_models/regression_arima_errors/test_regression_arima_errors_test_2025.py`

**Interfaces:**
- Consumes: the planned `regression_arima_errors_test_2025` module and existing processed data.
- Produces: executable assertions for frozen configuration, manifest, validation, checkpointing, and table merge behavior.

- [ ] **Step 1: Write frozen-selection tests**

Assert that `verify_authoritative_selection()` finds exactly one complete, zero-failure, full-coverage row per country, that both rows select `reg_arima_h_d_m_hol`, and that the registry contains the same frozen model-family rows.

```python
selected = validation.verify_authoritative_selection()
assert selected.set_index("country")["selected_specification_id"].to_dict() == {
    "Germany": "reg_arima_h_d_m_hol",
    "Austria": "reg_arima_h_d_m_hol",
}
assert validation.ERROR_ORDER == (2, 1, 2)
assert validation.TREND == "n"
```

- [ ] **Step 2: Write specification and manifest tests**

Load the authoritative 2024 specification CSV and assert the exact 43 frozen exogenous columns, `specification_order == 4`, order `(2, 1, 2)`, trend `n`, 365 dates per country, 730 unique `(country, target_date, specification_id)` keys, and Germany/Austria origin hours 18/08 on the previous local day.

- [ ] **Step 3: Write information-bound and retry-delegation tests**

Assert January and July training counts equal the origin-bounded `information_set` and increase chronologically. Assert the runner delegates fitting to `fit_regression_arima` with the frozen specification and does not define an alternate optimizer or retry policy.

- [ ] **Step 4: Write forecast-integrity and DST tests**

Use a small fake fitted result with the existing `build_regression_forecast_path`, then validate ordinary, spring-DST, and autumn-DST target paths. Require finite forecasts, unique UTC timestamps, exact family/specification labels, origin/cutoff equality, and rejection of corrupted training counts or future local-date rows.

- [ ] **Step 5: Write checkpoint and table-merge tests**

Use a temporary output directory to assert completed jobs require forecasts, completed keys are skipped on restart, incomplete or failed keys remain pending, and repeated upserts are deterministic. Use temporary central tables containing the existing baseline, Holt-Winters, ARIMA, and SARIMA rows and assert the two new regression rows do not drop or alter them.

- [ ] **Step 6: Run the new tests before production implementation**

Run:

```bash
.venv/bin/python -m pytest code/04_models/regression_arima_errors/test_regression_arima_errors_test_2025.py -q
```

Expected: collection fails because `regression_arima_errors_test_2025.py` does not exist yet.

### Task 2: Implement the isolated 2025 runner

**Files:**
- Create: `code/04_models/regression_arima_errors/regression_arima_errors_test_2025.py`

**Interfaces:**
- Consumes: `regression_arima_errors_validation_2024._execute_job`, `build_regression_forecast_path`, `fit_regression_arima`, `forecast_values`, and the common forecasting framework.
- Produces: `run_test(...)`, `verify_authoritative_selection(...)`, `load_frozen_specification(...)`, `build_test_manifest(...)`, `validate_job_result(...)`, `validate_aggregate(...)`, `update_test_tables(...)`, and the CLI.

- [ ] **Step 1: Define frozen identity and isolated paths**

Define the selected specification, order, trend, 2025 expected counts, output directory `results/regression_arima_errors/test_2025`, job/forecast/summary filenames, authoritative selection/specification/registry paths, model-table paths, and explicit job/forecast schemas. Keep all paths separate from `validation_2024/`.

- [ ] **Step 2: Verify authoritative inputs**

Read the selected 2024 table, selected-model registry, and full-validation specification table. Require both countries, complete 366-job 2024 coverage, zero failures, exact frozen status, exact specification ID, exact `(2, 1, 2)` order, trend `n`, specification order 4, and `specification_columns(SPECIFICATION_ID)` equality. Reject intervention/crisis rows.

- [ ] **Step 3: Build the exact 730-job 2025 manifest**

Derive all local dates from the processed frames, require exactly `2025-01-01` through `2025-12-31`, and produce one fixed-specification row for every country/date with expected DST target observations. Reject duplicate keys, missing countries, non-2025 dates, or any manifest count other than 730.

- [ ] **Step 4: Reuse the proven fit and forecast implementation**

For each job, build the country context, fit on `information_set(frame, context.cutoff_utc)` with `build_exog(..., SPECIFICATION_ID)`, and delegate to the existing `_execute_job(..., model_family=MODEL_FAMILY)`. Add only 2025 job metadata for origin, cutoff, training observations, and expected observations. Do not modify `fit_regression_arima`, its approved `maxiter=1000` retry, or forecast construction.

- [ ] **Step 5: Implement per-result checkpointing**

Implement CSV normalization, key-based upsert, atomic writes, and a store whose completed keys require both a completed job and forecast rows. Persist jobs, forecasts, and summaries after every result. Use spawn-based workers and make `run_test` accept a worker count so a bounded benchmark can select the safe full-run count.

- [ ] **Step 6: Validate each job and the aggregate**

Require the exact model family/specification, finite forecasts, unique UTC timestamps, origin/cutoff equality, origin-bounded training counts, target lengths 23/24/25, no future local dates, 365 jobs per country, 8,760 evaluated target observations per country, monotonic expanding training histories, and zero incomplete/failed jobs before central tables are updated.

- [ ] **Step 7: Build summary and merge both central tables**

Delegate summary metrics to the existing summary builder, convert completed country summaries to `model_family=regression_arima_errors` and `n_observations=evaluated_observations`, then atomically upsert both central 2025 tables by `(country, model_family, specification_id)`. Preserve all existing rows and never update the selected-model registry.

- [ ] **Step 8: Add CLI and runtime reporting**

Provide `--processed-directory`, `--output-directory`, `--workers`, and JSON output containing counts, elapsed seconds, country metrics, convergence, and retry aggregates. Reject non-positive worker counts and leave the actual full run to the tmux launch step.

- [ ] **Step 9: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest code/04_models/regression_arima_errors/test_regression_arima_errors_test_2025.py code/04_models/regression_arima_errors/test_regression_arima_errors_validation_2024.py code/04_models/regression_arima_errors/test_regression_arima_errors_models.py -q
```

### Task 3: Benchmark and prepare the controlled full run

**Files:**
- Create at runtime: `results/regression_arima_errors/test_2025/regression_arima_errors_test_2025_jobs.csv`
- Create at runtime: `results/regression_arima_errors/test_2025/regression_arima_errors_test_2025_forecasts.csv`
- Create at runtime: `results/regression_arima_errors/test_2025/regression_arima_errors_test_2025_summary.csv`
- Modify at runtime after successful full completion: `results/model_test_2025_all_models.csv`
- Modify at runtime after successful full completion: `results/model_test_2025_overview.csv`

- [ ] **Step 1: Run a bounded worker benchmark**

Run a small representative subset through the runner’s real worker path with BLAS/OpenMP thread counts set to 1, measure workers 1 and 2 first, and only test 4 if memory and runtime remain safe. Do not write to central tables during the benchmark; use a temporary output directory.

- [ ] **Step 2: Snapshot protected outputs**

Record hashes for all existing 2024 outputs, completed 2025 baseline/Holt-Winters/ARIMA/SARIMA outputs, and the selected-model registry before the full run.

- [ ] **Step 3: Report the exact tmux command before launch**

Report the chosen worker count, expected output paths, validation gates, runtime estimate from the benchmark, and this command without executing it until the launch gate is explicitly reached:

```bash
tmux new-session -d -s reg_arima_2025 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python code/04_models/regression_arima_errors/regression_arima_errors_test_2025.py --workers <chosen_workers> >> results/regression_arima_errors/test_2025/run.log 2>&1'
```

- [ ] **Step 4: After launch, verify the complete run**

Require 730/730 completed jobs, zero failures, 8,760 target observations per country, full coverage, valid DST counts, unique timestamps, finite forecasts, exact information bounds, unchanged protected hashes, and central tables containing all prior rows plus the two new regression rows.

- [ ] **Step 5: Report 2024 versus 2025 metrics**

Report Germany and Austria MAE, RMSE, MAPE, observations, coverage, failures, convergence, retry counts, and runtime using the authoritative 2024 selected rows and the new 2025 summary. Do not reselection or launch other model families.
