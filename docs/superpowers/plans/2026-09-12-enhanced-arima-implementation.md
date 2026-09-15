# Enhanced ARIMA Robustness Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an isolated enhanced-ARIMA ordinary and weekly-differenced robustness pipeline with reproducible 2024 selection, safe worker benchmarking, and descriptive frozen-specification 2025 evaluation.

**Architecture:** Add focused modules beside the existing ARIMA modules and reuse the common forecasting framework plus existing ordinary `fit_arima` semantics. Ordinary enhanced candidates use `fit_arima` unchanged; weekly candidates use a gated missing-aware statsmodels adapter that preserves the complete UTC hourly index and stores mapping-invalid differences as indexed missing observations. All enhanced artifacts are CSV files below `results/enhanced_arima/`; existing ARIMA, SARIMA, comparator outputs, registries, and central tables are read-only.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy, statsmodels 0.14.6, standard-library `ProcessPoolExecutor` with `spawn`, pytest, CSV checkpoints, and Linux `/proc` resource telemetry where available.

**Spec:** `docs/superpowers/specs/2026-09-12-enhanced-arima-design.md`

## Global Constraints

- Do not initialize Git, add a remote, upload data, or add dependencies.
- Treat `data/raw/` as immutable.
- Use ordinary Python files and the project interpreter.
- Use classical ARIMA methods only; no machine learning, Prophet, `pmdarima`, temperature inputs, calendar regressors, or official forecasts as model inputs.
- Do not modify the behavior of the existing ARIMA pipeline.
- Do not write enhanced results to `results/arima/` or update existing central model tables with enhanced results.
- Do not use 2025 observations or metrics numerically for candidate search, screening, worker selection, adequacy decisions, or freezing.
- The original frozen ARIMA remains the primary thesis result; enhanced 2025 results are descriptive post-hoc robustness results.
- Germany and Austria are selected and reported separately.
- Use `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` for enhanced model execution.
- No model execution begins until the missing-aware weekly adapter regression test passes.
- No Git commit step is included because project instructions prohibit Git initialization and commits.

---

## File Map

Create:

- `code/04_models/arima/enhanced_arima_models.py`: branch specifications, strict weekly mapping, missing-aware fitting, forecast reconstruction, and fit metadata.
- `code/04_models/arima/enhanced_arima_screening_2024.py`: original artifact audit, reproducibility/candidate/calendar manifests, screening, read-only ordinary-artifact reuse, and branch-local shortlists.
- `code/04_models/arima/enhanced_arima_benchmark.py`: valid 32-job workload, worker-count runs, resource telemetry, equivalence checks, and worker selection.
- `code/04_models/arima/enhanced_arima_validation_2024.py`: resumable shortlisted full-year validation, native/common summaries, adequacy report, and country-specific freeze.
- `code/04_models/arima/enhanced_arima_test_2025.py`: freeze gate, descriptive 2025 evaluation, native/common summaries, and six-model comparison CSV.
- `code/04_models/arima/test_enhanced_arima_models.py`
- `code/04_models/arima/test_enhanced_arima_screening_2024.py`
- `code/04_models/arima/test_enhanced_arima_benchmark.py`
- `code/04_models/arima/test_enhanced_arima_validation_2024.py`
- `code/04_models/arima/test_enhanced_arima_test_2025.py`

Do not modify the existing ARIMA, SARIMA, comparator, registry, or central table files.

Create output directories and CSVs only below:

```text
results/enhanced_arima/
  audit/
  specifications/
  screening_2024/ordinary/
  screening_2024/weekly_differenced/
  benchmark/
  validation_2024/ordinary/
  validation_2024/weekly_differenced/
  freeze_2024/
  test_2025/ordinary/
  test_2025/weekly_differenced/
  comparison/
```

### Task 1: Branch Contracts and Missing-Aware Adapter Tests

**Files:**
- Create: `code/04_models/arima/test_enhanced_arima_models.py`
- Create: `code/04_models/arima/enhanced_arima_models.py`

**Interfaces:**
- `EnhancedSpecification(branch: str, specification_id: str, order: tuple[int, int, int], trend: str, transform: str)`.
- `WeeklyTransformResult(values: pd.Series, issues: pd.DataFrame, valid_observations: int, missing_observations: int)`.
- `enhanced_specifications() -> tuple[EnhancedSpecification, ...]`.
- `derive_weekly_invalid_target_dates(frame: pd.DataFrame, year: int) -> pd.DataFrame`.
- `transform_weekly_series(frame: pd.DataFrame, cutoff_utc: pd.Timestamp) -> WeeklyTransformResult`.
- `fit_missing_aware_arima(values: pd.Series, order: tuple[int, int, int]) -> ArimaFitRecord`.
- `fit_enhanced(specification: EnhancedSpecification, values: object) -> ArimaFitRecord`.
- `build_weekly_forecast_path(frame: pd.DataFrame, country: str, target_date: date | str, specification: EnhancedSpecification, fitted_result: object) -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests for exact branch specifications.** Assert the 11 ordinary orders are exactly `(1,1,1)`, `(1,1,2)`, `(1,1,3)`, `(2,1,1)`, `(2,1,2)`, `(2,1,3)`, `(3,1,1)`, `(3,1,2)`, `(3,1,3)`, `(4,1,1)`, `(4,1,2)`, and the five weekly orders are exactly `(1,0,1)`, `(2,0,0)`, `(2,0,1)`, `(2,0,2)`, `(3,0,1)`. Assert identifiers and branch values cannot collide.
- [ ] **Step 2: Write the missing-time regression test.** Construct a 96-row UTC hourly `Series`, set one internal transformed observation to `NaN`, fit `(1,0,0)` through the missing-aware path, and assert `endog.shape[0] == 96`, the original `DatetimeIndex` position and hourly frequency remain present, and the forecast differs from the `dropna()` 95-observation control with `rtol=1e-10`, `atol=1e-12`. Assert existing `fit_arima` rejects the non-finite array, proving the weekly adapter is a distinct path.
- [ ] **Step 3: Run the new model tests to verify they fail for missing implementations.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_models.py -q`. Expected: import or implementation failures.
- [ ] **Step 4: Implement specification records and strict source lookup.** Use the prepared frame's `(local_date, hour, local_occurrence)` key and map to `(local_date - 7 calendar days, hour, local_occurrence)`. Store zero-match and multi-match issues with target timestamp, source key, and reason.
- [ ] **Step 5: Implement `transform_weekly_series`.** Preserve every row of the origin-bounded prepared UTC hourly series. Set valid differences to `y_t - y_source`; set mapping-invalid positions to `NaN` at their original UTC timestamp. Never call `dropna()` or delete rows. Ensure the indexed series has an hourly frequency.
- [ ] **Step 6: Implement the missing-aware adapter.** Use the underlying statsmodels `ARIMA` with the indexed series, same order/trend/enforcement/optimizer retry/warning/convergence/root/finite-value conventions as `fit_arima`, and existing diagnostic primitives. Construct the existing `ArimaFitRecord` shape without changing `fit_arima`.
- [ ] **Step 7: Implement strict invalid-date derivation.** Scan every local date in the requested year and mark a date invalid when any target row has a missing or ambiguous prior-week source. Do not hard-code dates. Ensure current processed data derives `2024-04-07` and `2024-10-27` for both countries, while the spring-DST day and week following autumn DST are tested rather than assumed.
- [ ] **Step 8: Implement origin-available-only weekly reconstruction.** Resolve every bridge and target source to an actual row whose `interval_end_utc <= forecast_origin_utc`; assert this condition. Do not use recursive weekly source forecasts. Mark the whole target date `weekly_lag_invalid` when any target row is invalid, with no intra-day row dropping.
- [ ] **Step 9: Run the focused model tests.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_models.py -q`. Expected: all model-contract, index-preservation, strict-DST, and reconstruction tests pass.

### Task 2: Audit, Run Manifest, Candidate Definitions, and Screening Tests

**Files:**
- Create: `code/04_models/arima/test_enhanced_arima_screening_2024.py`
- Create: `code/04_models/arima/enhanced_arima_screening_2024.py`

**Interfaces:**
- `build_screening_calendar() -> pd.DataFrame`.
- `audit_existing_arima() -> pd.DataFrame`.
- `write_initial_artifacts(output_root: str | Path) -> dict[str, Path]`.
- `build_screening_manifest(calendar: pd.DataFrame, specifications: tuple[EnhancedSpecification, ...], frames: dict[str, pd.DataFrame], invalid_dates: pd.DataFrame) -> pd.DataFrame`.
- `run_screening(output_root: str | Path, workers: int = 1, processed_directory: str | Path = PROCESSED) -> dict[str, object]`.
- `build_shortlist(screening_jobs: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Write tests for the deterministic calendar and manifests.** Assert exactly the approved 16 dates, identical date values for Germany and Austria, 2024-only target dates, all 16 ordinary plus weekly specifications, unique `(country, target_date, branch, specification_id)` keys, and correct country origins.
- [ ] **Step 2: Write tests for the original-artifact audit.** Use temporary read-only copies of authoritative candidate-level tables and assert overlap, unavailable candidates, source paths, and reuse eligibility are recorded without changing source bytes.
- [ ] **Step 3: Write tests for the reproducibility manifest.** Assert required version, platform, CPU, memory, thread-setting, timestamp, selected-worker, and source-filename columns exist; selected worker may be blank before benchmarking and is atomically updated afterward.
- [ ] **Step 4: Run screening tests to verify they fail.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_screening_2024.py -q`. Expected: import or implementation failures.
- [ ] **Step 5: Implement constants and the exact calendar.** Use the approved 16 dates and labels; do not select dates from 2024 actual values or temperatures.
- [ ] **Step 6: Implement the audit.** Read `code/04_models/arima/arima_screening.py`, `results/arima/tables/arima_candidate_screening_training_2024.csv`, `results/arima/tables/arima_shortlists_2024.csv`, `results/arima/tables/arima_selected_2024.csv`, and `results/arima/validation_2024/full_validation/` read-only. Record the original grid, selected `(2,1,2)`, nearby tested orders, exact enhanced-order overlap, unexplored orders, and candidate-level evidence completeness.
- [ ] **Step 7: Implement initial CSV artifacts.** Write `audit/original_arima_audit.csv`, `audit/enhanced_arima_run_manifest.csv`, `specifications/enhanced_arima_candidate_definitions.csv`, and `specifications/screening_calendar_2024.csv` atomically before any screening fit. Gather versions from imported packages, platform/OS, logical CPU count, physical/available memory from standard-library/platform facilities, thread variables, UTC timestamp, and relevant source filenames.
- [ ] **Step 8: Implement the screening manifest.** Include all countries, 16 dates, 11 ordinary specifications, and five weekly specifications. Include branch, order, trend, transform, origin, expected target observations, and derived weekly invalid-date status.
- [ ] **Step 9: Implement read-only ordinary-artifact reuse.** Reuse a candidate-level job only when country, order, trend, fit settings, origin, cutoff, target timestamps, forecasts, diagnostics, and coverage all verify. Copy it under the enhanced identifier with `artifact_source`, `reuse_verified`, and source path fields. Refit when any evidence is missing; never fabricate or reuse weekly artifacts.
- [ ] **Step 10: Run screening tests.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_screening_2024.py -q`. Expected: all calendar, audit, manifest, and reuse tests pass.

### Task 3: Screening Execution, Shortlists, and CSV Checkpoints

**Files:**
- Modify: `code/04_models/arima/enhanced_arima_screening_2024.py`
- Modify: `code/04_models/arima/test_enhanced_arima_screening_2024.py`

**Interfaces:**
- `run_screening` writes branch-specific `jobs.csv`, `forecasts.csv`, `diagnostics.csv`, `summary.csv`, and weekly `mapping_issues.csv` under `results/enhanced_arima/screening_2024/`.
- `build_shortlist` writes `screening_2024/shortlist_2024.csv` with top 3 ordinary and top 2 weekly rows per country.

- [ ] **Step 1: Write checkpoint and shortlist tests.** Test atomic writes, resume skipping terminal completed/`weekly_lag_invalid` jobs, retrying unexpected failures, duplicate-free job/forecast keys, valid-only weekly aggregation, the 15/16 minimum, ordinary 16/16 requirement, and exact top-count output.
- [ ] **Step 2: Run the checkpoint tests to verify failure.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_screening_2024.py -q`. Expected: failures for missing checkpoint/shortlist behavior.
- [ ] **Step 3: Implement per-job execution.** For ordinary candidates, build the origin-bounded information set and call existing `fit_arima` unchanged. For weekly candidates, build the preserved-index transform and call the missing-aware adapter. Generate continuous bridge-plus-target paths and score target timestamps only.
- [ ] **Step 4: Implement CSV checkpoint stores.** Use stable keys `(country, target_date, branch, specification_id)` for jobs and add `timestamp_utc` for forecast keys. Persist after every job using temporary CSV replacement. Encode warnings/diagnostics in CSV fields.
- [ ] **Step 5: Implement invalid-date handling.** Write every invalid date, reason, affected timestamp, source key, and coverage. Exclude the whole date from weekly aggregate metrics only when mapping-invalid; never drop a row inside a valid target day.
- [ ] **Step 6: Implement branch-local summaries and shortlist.** Rank ordinary candidates over all 16 valid dates and weekly candidates over valid dates with at least 15/16. Rank by aggregate MAE, RMSE, MAPE, and deterministic specification order within each branch and country. Persist all rows before shortlist output.
- [ ] **Step 7: Run screening unit tests.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_screening_2024.py -q`. Expected: all checkpoint and shortlist tests pass.

### Task 4: Common Metrics, Full-Validation Stores, and Freeze Tests

**Files:**
- Create: `code/04_models/arima/test_enhanced_arima_validation_2024.py`
- Create: `code/04_models/arima/enhanced_arima_validation_2024.py`

**Interfaces:**
- `build_full_validation_manifest(shortlist: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame`.
- `run_full_validation(shortlist: pd.DataFrame, output_root: str | Path, workers: int, processed_directory: str | Path = PROCESSED, reuse_authoritative: bool = True) -> dict[str, object]`.
- `build_common_timestamp_set(forecasts: pd.DataFrame, candidates: pd.DataFrame, country: str) -> pd.DataFrame`.
- `build_native_and_common_summary(jobs: pd.DataFrame, forecasts: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame`.
- `build_adequacy_review(jobs: pd.DataFrame, forecasts: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame`.
- `freeze_specifications(summary: pd.DataFrame, adequacy: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Write tests for full manifests and common sets.** Assert 366 dates, country separation, expanding origin-bounded training observations, exact forecast path keys, common timestamp intersection, native/common metric columns, and whole-date exclusion for weekly invalidity.
- [ ] **Step 2: Write tests for adequacy and freeze gates.** Reject non-finite/convergence/data/implementation failures, require complete residual diagnostics for valid fits, preserve residual warnings, allow only derived weekly invalidity, and select one country-specific winner by common-set MAE, RMSE/MAPE, adequacy, and order.
- [ ] **Step 3: Run validation tests to verify failure.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_validation_2024.py -q`. Expected: missing implementation failures.
- [ ] **Step 4: Implement validation manifest and stores.** Build all 2024 jobs for the five shortlisted specifications per country. Use spawn workers, atomic upserts, completed/terminal-invalid resume, unexpected-failure retries, and branch-separated output directories.
- [ ] **Step 5: Implement native summaries.** Aggregate each candidate over its own valid target timestamps and retain MAE, RMSE, MAPE, observations, coverage, invalid target dates, and invalid reasons.
- [ ] **Step 6: Implement common-set summaries.** For each country, intersect valid target UTC timestamps across every shortlisted candidate. Calculate common MAE/RMSE/MAPE and common observations/coverage. Use common-set MAE as primary ordinary-versus-weekly comparison; retain native coverage explicitly.
- [ ] **Step 7: Implement adequacy review and freeze.** Require eligible convergence, finite parameters/criteria/forecasts, no implementation/data failure, and consumed residual diagnostics for every valid job. Record residual warning counts and reviewed status. Write exactly one frozen specification per country only after the required validation gates pass.
- [ ] **Step 8: Run validation tests.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_validation_2024.py -q`. Expected: all full-manifest, common-set, adequacy, and freeze tests pass.

### Task 5: Saturated Valid Worker Benchmark Tests and Runner

**Files:**
- Create: `code/04_models/arima/test_enhanced_arima_benchmark.py`
- Create: `code/04_models/arima/enhanced_arima_benchmark.py`

**Interfaces:**
- `build_benchmark_workload(shortlist: pd.DataFrame, invalid_dates: pd.DataFrame) -> pd.DataFrame`.
- `run_worker_benchmark(shortlist: pd.DataFrame, output_root: str | Path, processed_directory: str | Path = PROCESSED) -> pd.DataFrame`.
- `select_fastest_safe_worker(summary: pd.DataFrame) -> int`.

- [ ] **Step 1: Write tests for the fixed workload.** Assert exactly 32 jobs: both countries, two ordinary and two weekly shortlist rows per country, four derived-valid dates, unique keys, and no `weekly_lag_invalid` target date. Assert the same manifest is passed to every worker-count run.
- [ ] **Step 2: Write tests for worker result fields and safety.** Cover requested/active workers, elapsed time, jobs/minute, failures, peak RSS, swap, max forecast difference, equivalence, safe/unsafe, unsupported counts, and fastest-safe selection.
- [ ] **Step 3: Run benchmark tests to verify failure.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_benchmark.py -q`. Expected: implementation failures.
- [ ] **Step 4: Implement valid workload selection.** Start with preferred dates `2024-02-15`, `2024-03-31`, `2024-07-25`, and `2024-11-03`; derive invalid dates first and deterministically replace any invalid preference from the sorted set of all 2024 local dates not in the derived invalid set. Persist the final 32-job workload.
- [ ] **Step 5: Implement worker runs.** Run the identical workload for counts `1` through `8` when each is supported by logical CPU capacity and spawn startup. Do not use the full 366-day manifest or intentionally invalid DST targets. Persist each run under a separate CSV directory.
- [ ] **Step 6: Implement resource and equivalence telemetry.** Poll Linux `/proc` process trees for aggregate peak `VmRSS` and `VmSwap`, retaining explicit unavailable fields elsewhere. Compare each run’s forecasts with the one-worker run using `rtol=1e-10`, `atol=1e-12`; record maximum absolute/relative differences.
- [ ] **Step 7: Implement safety and selection.** Mark unsafe on unexpected failures, non-finite output, duplicate keys, bad target lengths, unexpected mapping invalidity, resource exhaustion, or failed equivalence. Select the fastest safe supported count and atomically update `audit/enhanced_arima_run_manifest.csv`.
- [ ] **Step 8: Run benchmark tests.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_benchmark.py -q`. Expected: all workload, telemetry, equivalence, and selection tests pass.

### Task 6: 2025 Freeze Gate, Descriptive Test, and Final Comparison

**Files:**
- Create: `code/04_models/arima/test_enhanced_arima_test_2025.py`
- Create: `code/04_models/arima/enhanced_arima_test_2025.py`

**Interfaces:**
- `verify_freeze(freeze_path: str | Path) -> pd.DataFrame`.
- `build_2025_manifest(freeze: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame`.
- `run_descriptive_2025_test(output_root: str | Path, processed_directory: str | Path = PROCESSED) -> dict[str, object]`.
- `build_final_comparison(enhanced_summary: pd.DataFrame, enhanced_forecasts: pd.DataFrame, comparator_forecasts: dict[str, pd.DataFrame], smard_benchmark: pd.DataFrame) -> pd.DataFrame`.

- [ ] **Step 1: Write tests for the 2025 gate.** Assert no 2025 manifest can be built before one frozen row per country exists, exactly one frozen enhanced specification is evaluated per country, and derived 2025 invalid dates are not hard-coded.
- [ ] **Step 2: Write tests for the comparison contract.** Assert each country has rows for original ARIMA `(2,1,2)`, enhanced ARIMA, original SARIMA, weekly seasonal naive, Holt-Winters, and SMARD; assert enhanced native coverage and common metrics are present.
- [ ] **Step 3: Run 2025 tests to verify failure.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_test_2025.py -q`. Expected: implementation failures.
- [ ] **Step 4: Implement freeze verification and manifest.** Require complete `freeze_2024/frozen_specifications_2024.csv` with one country-specific winner per country. Only then derive every 2025 weekly invalid date and build 365-date jobs.
- [ ] **Step 5: Implement resumable descriptive 2025 evaluation.** Persist branch-specific jobs, forecasts, native summaries, invalid mappings, and common timestamp metrics. Treat all enhanced 2025 metrics as descriptive post-hoc robustness output, not primary model ranking input.
- [ ] **Step 6: Implement read-only comparator import.** Load original ARIMA/SARIMA/Holt-Winters/baseline forecast CSVs and SMARD benchmark/aligned CSVs from their existing results paths. Do not update `results/model_test_2025_all_models.csv`, `results/model_test_2025_overview.csv`, `results/selected_models_registry.csv`, or any original result.
- [ ] **Step 7: Implement final native/common comparison.** Write `comparison/enhanced_arima_final_comparison_2025.csv` with all six required model rows per country, native metrics, coverage, invalid dates, and common-timestamp metrics whenever valid timestamp sets differ.
- [ ] **Step 8: Run 2025 unit tests.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_test_2025.py -q`. Expected: all freeze-gate and comparison-contract tests pass.

### Task 7: Integrated Tests and Static Preservation Checks

**Files:**
- Modify: `code/04_models/arima/test_enhanced_arima_models.py`
- Modify: `code/04_models/arima/test_enhanced_arima_screening_2024.py`
- Modify: `code/04_models/arima/test_enhanced_arima_benchmark.py`
- Modify: `code/04_models/arima/test_enhanced_arima_validation_2024.py`
- Modify: `code/04_models/arima/test_enhanced_arima_test_2025.py`
- Read-only verification: existing ARIMA/SARIMA code and result files.

- [ ] **Step 1: Run all enhanced tests.** Run `.venv/bin/python -m pytest code/04_models/arima/test_enhanced_arima_*.py -q`. Expected: all enhanced tests pass.
- [ ] **Step 2: Verify untouched authoritative artifacts.** Record hashes or bytes for representative original ARIMA and central table files before execution; after tests and runs, verify unchanged bytes. Include `results/arima/`, `results/arima_sarimax/`, `results/model_test_2025_all_models.csv`, `results/model_test_2025_overview.csv`, and `results/selected_models_registry.csv`.
- [ ] **Step 3: Run the existing ARIMA and SARIMA suites.** Run `.venv/bin/python -m pytest code/04_models/arima code/04_models/sarima -q`. Expected: all existing tests pass with no authoritative behavior changes.

### Task 8: Execute Artifacts, Screening, Benchmark, Validation, Freeze, and Test

**Files:**
- Runtime outputs only below `results/enhanced_arima/`.
- Runtime logs/checkpoints may be stored under `/tmp/opencode/`.

- [ ] **Step 1: Write initial manifests and run manifest.** Use the project interpreter and set all numerical-library thread variables to `1`. Confirm candidate definitions, exact calendar, audit, and reproducibility CSVs exist before fitting.
- [ ] **Step 2: Start 2024 screening immediately after manifest verification.** Run screening with incremental per-job CSV persistence and inspect ordinary/weekly counts, derived invalid dates, coverage, fit failures, and artifact reuse counts.
- [ ] **Step 3: Verify and persist shortlists.** Confirm top 3 ordinary and top 2 weekly candidates for Germany and Austria, with weekly minimum 15/16 valid screening dates.
- [ ] **Step 4: Run the 32-job valid benchmark.** Run supported worker counts 1 through 8 using the identical workload, persist every count’s metrics/resource/equivalence CSVs, and choose the fastest safe count.
- [ ] **Step 5: Update the run manifest with the selected worker count.** Verify the manifest records the selected count and all runtime metadata.
- [ ] **Step 6: Launch full 2024 validation in tmux.** Use the selected worker count exactly once as the production configuration, with checkpoint/resume and output paths under `results/enhanced_arima/validation_2024/`. Do not start 2025.
- [ ] **Step 7: Verify common/native 2024 summaries and freeze.** Confirm derived invalid dates only, native coverage, common-set metrics, adequacy review, and one frozen specification per country.
- [ ] **Step 8: Evaluate only frozen winners on 2025.** Launch the descriptive test in tmux after freeze verification, persist all jobs/forecasts/summaries/invalid dates/common metrics, and do not use the results for any selection.
- [ ] **Step 9: Write and verify the final comparison CSV.** Confirm all six required model rows per country, enhanced native/common metrics, post-hoc labels, and unchanged original tables.
- [ ] **Step 10: Run final verification.** Run `.venv/bin/python -m pytest code/04_models/arima code/04_models/sarima -q` and verify all required CSVs, coverage fields, derived DST records, worker-selection fields, and untouched authoritative files.
