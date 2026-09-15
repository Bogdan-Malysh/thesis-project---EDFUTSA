# Holt-Winters 2025 Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run only the frozen `hw_mul_s168_damped` Holt-Winters specification over all 2025 Germany and Austria target dates and append its results to the existing 2025 model tables.

**Architecture:** Add an isolated `holt_winters_test_2025.py` runner. It will read and verify the authoritative 2024 selection, build 730 jobs from the matching frozen-shortlist rows, and delegate fitting, forecasting, and metrics to the existing 2024 Holt-Winters machinery. A 2025-specific subclass of the existing checkpoint store will use test-oriented filenames, while table merging will preserve prior 2025 family rows by deterministic key.

**Tech Stack:** Python, pandas, NumPy, statsmodels through existing Holt-Winters modules, pytest, project interpreter `.venv\Scripts\python.exe`.

---

### Task 1: Add failing tests for frozen selection, expanding history, and table merge

**Files:**
- Create: `code/04_models/exponential_smoothing/test_holt_winters_test_2025.py`

- [ ] **Step 1: Write the authoritative-selection contract tests**

Import the new module and assert:

```python
assert validation.SPECIFICATION_ID == "hw_mul_s168_damped"
selected = validation.verify_authoritative_selection()
assert selected.set_index("country")["specification_id"].to_dict() == {
    "Germany": "hw_mul_s168_damped",
    "Austria": "hw_mul_s168_damped",
}
```

Assert the frozen rows have `seasonal_form == "mul"`, `seasonal_periods == 168`,
`damped_trend is True`, and a non-empty optimizer-attempt configuration.

- [ ] **Step 2: Write the 730-job and origin tests**

Load processed Germany and Austria frames, obtain the exact 2025 date lists, and call `build_test_jobs`. Assert 365 dates per country, 730 jobs total, unique job keys, and origins of 18:00 for Germany and 08:00 for Austria through the job origin fields.

- [ ] **Step 3: Write the expanding-history tests**

Build two Germany jobs for January and July 2025 and assert their expected training counts equal the lengths of `information_set(frame, country_forecast_origin(...))`. Assert the later job count is greater than the earlier count. This proves the runner uses the expanding origin-bounded history rather than a fixed four-year window.

- [ ] **Step 4: Write forecast-integrity and DST tests**

Use a real `ValidationResult`-shaped forecast frame or a real selected-specification forecast for `2025-03-30` and `2025-10-26`. Assert target lengths 23 and 25, unique target UTC timestamps, finite forecasts, exact `Holt-Winters` model label, exact specification ID, and equal forecast-origin/information-cutoff columns. Assert a future training count or mismatched cutoff is rejected by the validation helper.

- [ ] **Step 5: Write the metric-delegation test**

Monkeypatch `holt_winters_test_2025.build_summary` or the existing imported summary function with a recorder, run the summary helper, and assert the existing `calculate_metrics`/`evaluate_target_day` path is used. Do not add a new MAE, RMSE, or MAPE formula.

- [ ] **Step 6: Write merge-preservation tests**

Create temporary 2025 all-models and overview files containing the existing Weekly seasonal naive rows. Merge new Germany and Austria rows with `model_family = "holt_winters"` and `specification_id = "hw_mul_s168_damped"`. Assert all four rows remain, same-key reruns replace only Holt-Winters rows, ordering is deterministic, and repeated updates produce byte-identical files.

- [ ] **Step 7: Run the tests before implementation**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest code/04_models/exponential_smoothing/test_holt_winters_test_2025.py -q
```

Expected: collection fails because `holt_winters_test_2025.py` does not exist yet.

### Task 2: Implement the isolated Holt-Winters 2025 runner

**Files:**
- Create: `code/04_models/exponential_smoothing/holt_winters_test_2025.py`

- [ ] **Step 1: Define constants and test-oriented output names**

Import the existing 2024 validation module, Holt-Winters phase-2 shortlist helpers, model classes, and forecasting-framework functions. Define:

```python
SPECIFICATION_ID = "hw_mul_s168_damped"
VALIDATION_YEAR = 2025
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "exponential_smoothing" / "holt_winters" / "test_2025"
JOB_FILENAME = "holt_winters_test_2025_jobs.csv"
FORECAST_FILENAME = "holt_winters_test_2025_forecasts.csv"
SUMMARY_FILENAME = "holt_winters_test_2025_summary.csv"
```

Do not import or write the 2024 output directory as a destination.

- [ ] **Step 2: Verify 2024 selection and load frozen settings**

Read `holt_winters_selected_specifications_2024.csv` from the authoritative 2024 output directory and require exactly one row per Germany and Austria, both with `hw_mul_s168_damped`, `completed_jobs == 366`, `failed_jobs == 0`, and `coverage == 1.0`. Call `load_frozen_shortlists()` and retain only the two matching country/specification rows. Convert each row with `_specification_from_row` and `_optimizer_configuration_from_row`, requiring multiplicative form, period 168, damped trend, and valid optimizer metadata.

- [ ] **Step 3: Build the exact 2025 job manifest**

Enumerate the exact dates `2025-01-01` through `2025-12-31`. For each country/date, construct the existing `ValidationJob` dataclass from the verified frozen row, preserving `specification_order`, `optimizer_used`, and `minimize_kwargs`. Reject any date outside 2025, duplicate key, or specification other than `hw_mul_s168_damped`.

- [ ] **Step 4: Reuse the existing worker and fit path**

Call the existing `iter_job_results(jobs, processed_directory=..., workers=2)` without a custom worker function so it uses the existing worker initializer and `_execute_validation_job`. This preserves `forecast_holt_winters`, `fit_specification`, initialization, optimizer fallback configuration, bridge forecasting, `evaluate_target_day`, and existing metric formulas.

- [ ] **Step 5: Add a test-specific checkpoint store**

Subclass the existing `ValidationStore` only to replace its three paths with `holt_winters_test_2025_jobs.csv`, `holt_winters_test_2025_forecasts.csv`, and `holt_winters_test_2025_summary.csv` under `OUTPUT_DIRECTORY`. Reuse inherited read, upsert, and atomic-write behavior. Do not alter the 2024 class or constants.

- [ ] **Step 6: Validate expanding information and forecast paths**

For every completed result, compare `training_observations` to `len(information_set(frame, origin_utc))`; require the count to equal the exact origin-bounded information set and be nondecreasing over chronological 2025 jobs within each country. Require finite forecasts, unique UTC timestamps within each job and target day, 23/24/25 target lengths, exact model/specification labels, and equal origin/cutoff timestamps. After aggregation, require unique scored target timestamps by country and 23 intervals on `2025-03-30` and 25 on `2025-10-26`.

- [ ] **Step 7: Build the authoritative 2025 summary**

Reuse the existing `build_summary` with the 2025 manifest, jobs, forecasts, and expected observations of 8,760 per country. Require exactly two rows, 365 expected jobs per country, 730 total jobs, zero failures, 8,760 evaluated observations per country, and coverage 1.0. Write the result to `holt_winters_test_2025_summary.csv` only in the test directory.

- [ ] **Step 8: Append merge-safe 2025 model-table rows**

Convert the authoritative summary rows to the existing 2025 table schema:

```text
model_family = "holt_winters"
specification_id = "hw_mul_s168_damped"
n_observations = evaluated_observations
```

Read both existing 2025 table destinations, concatenate new rows, drop duplicate `(country, model_family, specification_id)` keys keeping the new rows, sort by Germany/Austria then family/specification, and atomically write both tables. Preserve the Weekly seasonal naive rows exactly.

- [ ] **Step 9: Add CLI and runtime reporting**

Provide `run_test(processed_directory=..., workers=2)` and a CLI that sets no thread variables itself but is run with the four requested environment variables. Return total/completed/failed/pending jobs, country summaries, and elapsed seconds. Do not expose any alternative model or specification loop.

- [ ] **Step 10: Run focused tests**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest code/04_models/exponential_smoothing/test_holt_winters_test_2025.py code/04_models/exponential_smoothing/test_holt_winters_validation_2024.py code/04_models/exponential_smoothing/test_holt_winters_models.py -q
```

Expected: all focused tests pass.

### Task 3: Run and verify the full 2025 Holt-Winters test

**Files:**
- Create at runtime: `results/exponential_smoothing/holt_winters/test_2025/holt_winters_test_2025_jobs.csv`
- Create at runtime: `results/exponential_smoothing/holt_winters/test_2025/holt_winters_test_2025_forecasts.csv`
- Create at runtime: `results/exponential_smoothing/holt_winters/test_2025/holt_winters_test_2025_summary.csv`
- Modify at runtime: `results/model_test_2025_all_models.csv`
- Modify at runtime: `results/model_test_2025_overview.csv`

- [ ] **Step 1: Snapshot protected files**

Record SHA-256 hashes for the authoritative 2024 Holt-Winters result files, the existing 2024 baseline result files, `results/baselines/forecasts/baseline_forecasts_2025.csv`, `results/baselines/tables/baseline_validation_2025.csv`, and `results/selected_models_registry.csv` before running.

- [ ] **Step 2: Run only the frozen 2025 Holt-Winters test**

Run from the project root:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"
& ".venv\Scripts\python.exe" code/04_models/exponential_smoothing/holt_winters_test_2025.py --workers 2
```

- [ ] **Step 3: Verify counts, DST, leakage, and runtime**

Require 730/730 jobs, zero failures, 8,760 target observations per country, 100% coverage, 23/25 DST target lengths, unique scored UTC timestamps, finite forecasts, exact frozen settings, expanding training counts, origin/cutoff equality, and runtime in the returned summary.

- [ ] **Step 4: Verify tables and protected outputs**

Require both 2025 tables to contain the prior Weekly seasonal naive rows unchanged plus two Holt-Winters rows. Recompute hashes and assert all protected 2024 files, baseline 2025 outputs, and selected registry are unchanged.

- [ ] **Step 5: Report 2024 comparison metrics**

Read 2024 Germany/Austria `hw_mul_s168_damped` metrics from the authoritative selected-specification output and 2025 metrics from the new test summary. Report MAE, RMSE, MAPE, evaluated observations, coverage, failures, and runtime. Do not update the registry or run another model.
