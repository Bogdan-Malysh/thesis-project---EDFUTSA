# Holt-Winters 2024 Resumable Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build resumable and configurable parallel infrastructure for 2024 Holt-Winters validation, then run only a small measured mini-validation.

**Architecture:** Add a 2024-only orchestration module around the existing frozen Holt-Winters shortlist. A parent process will own atomic CSV checkpoint/forecast upserts while independent spawned workers execute existing fits. Aggregate metrics and selection will use only target-day observations and the approved MAE/RMSE/MAPE ordering.

**Tech Stack:** Python, pandas, NumPy, standard-library `concurrent.futures`, `multiprocessing`, `csv/pathlib/os`, pytest, existing statsmodels Holt-Winters implementation.

---

### Task 1: Define validation jobs and checkpoint contracts

**Files:**
- Create: `code/04_models/exponential_smoothing/holt_winters_validation_2024.py`
- Test: `code/04_models/exponential_smoothing/test_holt_winters_validation_2024.py`

- [ ] **Step 1: Write failing manifest and checkpoint tests**

Test that a supplied two-country shortlist and target-date list creates exactly one job for each `(country, target_date, specification_id)` key in stable order, rejects duplicate keys, and marks only rows with `status == "completed"` as resumable work. Test that failed rows remain pending.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest code\04_models\exponential_smoothing\test_holt_winters_validation_2024.py -q
```

Expected: collection or assertion failures because the new module and contracts do not exist.

- [ ] **Step 3: Implement immutable job and shortlist helpers**

Add dataclasses for a validation job and serializable job result. Build jobs from `load_frozen_shortlists`, using `_specification_from_row` and `_optimizer_configuration_from_row` from the existing Phase 2A module. Preserve `specification_order`, the frozen optimizer configuration, and the key fields `country`, ISO `target_date`, and `specification_id`.

- [ ] **Step 4: Implement atomic CSV checkpoint helpers**

Add a small store that reads existing jobs and forecasts, upserts rows by explicit keys, sorts before writing, writes a sibling temporary CSV, and replaces the destination with `os.replace`. The forecast key must be `(country, target_date, specification_id, timestamp_utc)`; the job key must be `(country, target_date, specification_id)`. Write forecast rows before changing a successful job to `completed`.

- [ ] **Step 5: Run the focused tests and verify they pass**

Run the same focused pytest command. Expected: all manifest, completed-job, retry, and duplicate-upsert tests pass.

### Task 2: Add deterministic worker execution

**Files:**
- Modify: `code/04_models/exponential_smoothing/holt_winters_validation_2024.py`
- Test: `code/04_models/exponential_smoothing/test_holt_winters_validation_2024.py`

- [ ] **Step 1: Write failing worker tests**

Use a deterministic test worker or monkeypatched forecast function to verify that sequential and parallel scheduling return the same job keys, forecast values, status, and metric fields after sorting. Verify that `workers=1` is accepted and that invalid worker counts are rejected.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest code\04_models\exponential_smoothing\test_holt_winters_validation_2024.py -q
```

Expected: failures because the execution API is not implemented.

- [ ] **Step 3: Implement the spawn-safe worker**

Use a module-level worker initializer that loads the two processed country frames once per spawned process. The worker must call the existing `forecast_holt_winters` with the job's frozen `HoltWintersSpecification` and `OptimizerConfiguration`, then call `evaluate_target_day` on the returned forecast. Catch `HoltWintersForecastError` and return its fit record as a failed serializable result instead of terminating the run.

- [ ] **Step 4: Implement configurable execution**

Use `ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))` for `workers > 1`; invoke the identical worker function directly for `workers == 1`. Consume futures as they complete, but hand results to the parent store in sorted job-key order before writing so output order is deterministic.

- [ ] **Step 5: Run the focused tests and verify they pass**

Run the same focused pytest command. Expected: sequential/parallel consistency and worker validation tests pass.

### Task 3: Add progressive validation, aggregation, and selection

**Files:**
- Modify: `code/04_models/exponential_smoothing/holt_winters_validation_2024.py`
- Test: `code/04_models/exponential_smoothing/test_holt_winters_validation_2024.py`

- [ ] **Step 1: Write failing recovery and metric tests**

Test a partial store containing one completed job, one failed job, and one missing job. Verify that only the completed job is skipped, failed/missing jobs are rerun, forecast rows are not duplicated, and aggregate metrics use only `is_target_day` rows. Include 23-, 24-, and 25-interval target dates and assert expected-observation coverage.

- [ ] **Step 2: Implement the resumable runner**

Add `run_validation(processed_directory, shortlist_path, output_directory, target_dates=None, workers=1, minimum_coverage=1.0)`. If dates are omitted, derive every 2024 local date from the processed frame. Build the full manifest, filter completed keys, execute pending jobs, persist each result progressively, and recompute summary CSVs after each persisted result. Failed jobs must remain diagnostic rows and remain eligible on the next invocation.

- [ ] **Step 3: Implement aggregate summaries**

For each country/specification, concatenate only completed target-day rows, calculate RMSE, MAE, MAPE, evaluated observations, expected observations, and coverage using the existing `calculate_metrics` logic, and include job counts and failure counts. Do not calculate performance metrics from bridge rows.

- [ ] **Step 4: Implement final selection without changing the shortlist**

Write the selected-specifications CSV only when all 366-date full-year manifest jobs are completed. Exclude specifications below `minimum_coverage`; sort eligible candidates by numeric MAE, numeric RMSE, numeric MAPE, and the existing `specification_order`, separately within each country. Never use AIC, AICc, or BIC in this selection.

- [ ] **Step 5: Run recovery and selection tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest code\04_models\exponential_smoothing\test_holt_winters_validation_2024.py -q
```

Expected: all recovery, metric, coverage, tie-break, and incomplete-run tests pass.

### Task 4: Add the mini-validation CLI and tests

**Files:**
- Modify: `code/04_models/exponential_smoothing/holt_winters_validation_2024.py`
- Test: `code/04_models/exponential_smoothing/test_holt_winters_validation_2024.py`

- [ ] **Step 1: Write failing CLI contract tests**

Verify that repeated `--target-date` arguments are accepted, `--workers` is passed through, the default output directory is not used when an explicit mini directory is supplied, and an incomplete mini-run does not write a selected-specifications file.

- [ ] **Step 2: Implement the CLI**

Add arguments for `--processed-directory`, `--shortlist`, `--output-directory`, repeated `--target-date`, `--workers`, and `--minimum-coverage`. Print JSON containing requested/completed/pending/failed job counts, summary records, measured elapsed time, and a full-year estimate derived from measured successful fit time. Do not invoke `run_validation` with no dates from the mini command.

- [ ] **Step 3: Run all Holt-Winters tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest code\04_models\exponential_smoothing\test_holt_winters_models.py code\04_models\exponential_smoothing\test_holt_winters_phase2a.py code\04_models\exponential_smoothing\test_holt_winters_validation_2024.py -q
```

Expected: all existing and new tests pass.

### Task 5: Run only the measured mini-validation

**Files:**
- Create at runtime under `results/exponential_smoothing/holt_winters/mini_validation_2024/`: checkpoint, forecast, and summary CSVs.

- [ ] **Step 1: Run the ordinary and DST mini-validation**

Run:

```powershell
.\.venv\Scripts\python.exe code\04_models\exponential_smoothing\holt_winters_validation_2024.py --target-date 2024-01-15 --target-date 2024-03-31 --target-date 2024-10-27 --workers 2 --output-directory results\exponential_smoothing\holt_winters\mini_validation_2024
```

Expected: 24 requested jobs (3 dates x 2 countries x 4 shortlisted specifications), no full-year manifest execution, and no selected-specifications CSV.

- [ ] **Step 2: Inspect the mini outputs**

Confirm unique job keys, unique forecast keys, target-day counts of 23/24/25 for DST/ordinary dates, finite successful metrics, and no duplicate rows.

- [ ] **Step 3: Run the complete repository test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Report and stop**

Report mini job counts, failures, per-country/specification metrics, measured wall-clock and fit time, projected full-year runtime for 2,928 fits, and the exact output paths. Do not start the full 2024 validation or any 2025 evaluation.
