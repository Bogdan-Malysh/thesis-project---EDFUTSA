# ARIMA Screening and Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the training-only ARIMA identification stage, frozen country-specific shortlists, ordinary/DST mini-validation, resumable 2024 validation infrastructure, and 1/2/3/4-worker runtime benchmark without running the full 2024 validation or using 2025 data.

**Architecture:** Keep model fitting and diagnostics in `arima_models.py`, training-only order identification and output writing in `arima_screening.py`, and resumable rolling forecast orchestration in `arima_validation_2024.py`. Reuse `common.forecasting_framework` for origins, information sets, timestamp paths and metrics. The CLI will expose mini-validation only for this stage; the validator API will support a complete 2024 manifest but no full-year command will be executed.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy/statsmodels 0.14.6, pytest, Windows `spawn` multiprocessing, the project interpreter `.venv\Scripts\python.exe`.

---

## File Map

Create the following files only:

- `code\04_models\arima\arima_models.py`: ARIMA order/trend records, statsmodels fitting, AICc, roots, convergence and residual diagnostics, and forecast generation.
- `code\04_models\arima\arima_screening.py`: 2020-2023 extraction, fixed ADF/KPSS diagnostics, differencing selection, fixed low-order candidates, training fits, IC narrowing and screening outputs.
- `code\04_models\arima\arima_validation_2024.py`: deterministic manifests, origin-by-origin fitting, bridge paths, atomic checkpoints, mini-validation CLI and summaries.
- `code\04_models\arima\test_arima_models.py`: model fitting, deterministic terms, roots, warnings, residual thresholds and metrics tests.
- `code\04_models\arima\test_arima_screening.py`: exact stationarity configuration, `d` selection, fixed candidate set and IC shortlist tests.
- `code\04_models\arima\test_arima_validation_2024.py`: manifest, leakage, DST path, checkpoint/upsert and deterministic worker tests.

Create these result directories only when screening or mini-validation runs:

- `results\arima\tables\`
- `results\arima\validation_2024\mini_validation\`
- `results\arima\validation_2024\benchmark_workers_1\`
- `results\arima\validation_2024\benchmark_workers_2\`
- `results\arima\validation_2024\benchmark_workers_3\`
- `results\arima\validation_2024\benchmark_workers_4\`

Do not modify Chapter 6, the outdated methodology artifacts, baseline outputs or Holt-Winters outputs.

### Task 1: Add failing model and diagnostic tests

**Files:**
- Create: `code\04_models\arima\test_arima_models.py`
- Create: `code\04_models\arima\arima_models.py`

- [ ] **Step 1: Write tests for the fixed deterministic-term convention and candidate identity**

Test that `trend_for_d(0) == "c"`, `trend_for_d(1) == "n"`, `trend_for_d(2) == "n"`, and that an order identity contains the exact `(p,d,q)` values without adding seasonal terms.

- [ ] **Step 2: Write tests for exact root thresholds**

Use a small helper accepting synthetic root arrays and assert that roots with modulus `1.010001` pass, roots with modulus `1.01` fail, and models with no corresponding roots pass vacuously.

- [ ] **Step 3: Write tests for the residual practical threshold and adequacy flag**

Assert that the threshold is `max(2.576 / sqrt(n_effective), 0.05)`, that a single significant Ljung-Box p-value does not set the joint flag, and that both p-values below `0.01` plus at least two ACF exceedances set the diagnostic adequacy flag without making the fit ineligible.

- [ ] **Step 4: Write tests for warning classification**

Assert that convergence/numerical-invalid warnings are hard failures, while messages containing Hessian inversion, covariance estimation, singular covariance or non-positive-definite Hessian remain diagnostic-only unless an invalid estimate or failed convergence is also supplied.

- [ ] **Step 5: Run the new model tests and confirm they fail**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code\04_models\arima\test_arima_models.py -q
```

Expected result: collection or import failures because the ARIMA module does not yet exist.

- [ ] **Step 6: Implement the minimal model records and pure diagnostic helpers**

Implement:

```python
def trend_for_d(d: int) -> str: ...
def order_id(order: tuple[int, int, int]) -> str: ...
def root_is_stable(roots: object, threshold: float = 1.01) -> bool: ...
def residual_acf_threshold(n_effective: int) -> float: ...
def residual_rejection(
    ljung_box_pvalues: dict[int, float],
    acf_values: dict[int, float],
    n_effective: int,
) -> tuple[bool, float, list[int]]: ...
def classify_warning(category_name: str, message: str) -> tuple[bool, str]: ...
```

Keep warning classification deterministic and preserve diagnostic-only Hessian/covariance warnings.

- [ ] **Step 7: Implement fitting and AICc metadata**

Implement `fit_arima(values, order)` using statsmodels `ARIMA` with `trend_for_d(order[1])`, `enforce_stationarity=True`, and `enforce_invertibility=True`. Capture warnings, `mle_retvals`, parameters, standard errors, log likelihood, AIC, BIC, AICc, roots, residual diagnostics and a hard rejection reason. AICc must be computed as:

```python
aicc = aic + (2 * k * (k + 1)) / (nobs - k - 1)
```

provided `nobs - k - 1 > 0`; otherwise the criterion is non-finite.

- [ ] **Step 8: Run the model tests and confirm they pass**

Run the same pytest command. Expected result: all model/diagnostic tests pass.

### Task 2: Add failing training-screening tests

**Files:**
- Create: `code\04_models\arima\test_arima_screening.py`
- Create: `code\04_models\arima\arima_screening.py`

- [ ] **Step 1: Test exact ADF/KPSS calls**

Monkeypatch `adfuller` and `kpss`, run the diagnostic function on a short series, and assert:

```python
adfuller(..., regression="ct", maxlag=24, autolag="AIC")
kpss(..., regression="ct", nlags="auto")
```

for `d=0`, and `regression="c"` for `d=1` and `d=2`.

- [ ] **Step 2: Test the smallest-adequate-`d` rule**

Supply mocked p-values where `d=0` fails, `d=1` satisfies `ADF p < .05` and `KPSS p >= .05`, and `d=2` also satisfies both. Assert that `d=1` is selected. Add a test asserting that no shortlist is returned when no tested `d` satisfies both conditions.

- [ ] **Step 3: Test the exact fixed candidate set**

Assert that `build_candidate_orders(1)` returns exactly:

```python
((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
 (2, 1, 0), (0, 1, 2), (2, 1, 1), (1, 1, 2),
 (2, 1, 2), (3, 1, 0), (0, 1, 3))
```

and that no order contains `p > 3` or `q > 3`.

- [ ] **Step 4: Test same-`d` IC narrowing**

Use synthetic fit records with known AIC, AICc and BIC ranks. Assert that the shortlist is the stable union of the top two eligible rows for each criterion, excludes only model-invalid candidates, and retains candidates carrying a residual adequacy flag.

- [ ] **Step 5: Run screening tests and confirm they fail**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code\04_models\arima\test_arima_screening.py -q
```

Expected result: import or missing-function failures before implementation.

- [ ] **Step 6: Implement training extraction and exact differencing diagnostics**

Implement `training_values(frame)` to retain only local dates from `2020-01-01` through `2023-12-31` in existing UTC order. Implement `stationarity_diagnostics(values)` for `d=0,1,2` with the exact ADF/KPSS configuration in the approved specification, returning test statistics, p-values, selected ADF lag, KPSS bandwidth and warnings. Implement `select_d()` with:

```python
adf_pvalue < 0.05 and kpss_pvalue >= 0.05
```

and the smallest qualifying `d`, or an explicit no-adequate-`d` result.

- [ ] **Step 7: Implement fixed candidate construction and training screening**

Implement `build_candidate_orders(d)`, ACF/PACF reporting through lag 24 without order expansion, country-specific candidate fitting, diagnostic flagging, model-validity rejection, and stable IC narrowing. Use only training values and write:

- `results\arima\tables\arima_differencing_diagnostics_2024.csv`
- `results\arima\tables\arima_candidate_screening_training_2024.csv`
- `results\arima\tables\arima_residual_diagnostics_training_2024.csv`
- `results\arima\tables\arima_shortlists_2024.csv`

The shortlist output must include country, specification ID, `p`, `d`, `q`, trend, IC values, diagnostic status, and stable specification order.

- [ ] **Step 8: Run screening tests and confirm they pass**

Run the screening pytest command. Expected result: all stationarity, candidate and IC tests pass.

### Task 3: Add failing forecast-path and resumability tests

**Files:**
- Create: `code\04_models\arima\test_arima_validation_2024.py`
- Create: `code\04_models\arima\arima_validation_2024.py`

- [ ] **Step 1: Test deterministic mini manifest construction**

Assert that manifest keys are `(country, target_date, specification_id)`, are unique, sorted by country configuration order/date/specification order, and contain exactly three requested dates for each country when supplied the mini date set.

- [ ] **Step 2: Test information leakage**

Build a synthetic prepared frame, alter actual values strictly after the country cutoff, run the same ARIMA forecast-path helper with the original fitted values, and assert that all generated forecasts before evaluation are unchanged.

- [ ] **Step 3: Test ordinary and DST target lengths**

Use the existing processed data or a small framework-compatible fixture and assert 24 intervals for `2024-02-15`, 23 for `2024-03-31`, and 25 for `2024-10-27` for both countries. Assert that metrics use target rows only.

- [ ] **Step 4: Test atomic checkpoint resume and upserts**

Write a partial jobs/forecasts state, reload it, assert that only incomplete jobs are returned, rerun the same result, and assert no duplicate job or forecast keys.

- [ ] **Step 5: Test deterministic worker result ordering**

Use a deterministic test worker function over a small manifest and assert that serial and spawned-worker outputs are sorted identically after parent persistence.

- [ ] **Step 6: Run validation tests and confirm they fail**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code\04_models\arima\test_arima_validation_2024.py -q
```

Expected result: missing-module or missing-function failures before implementation.

- [ ] **Step 7: Implement ARIMA forecast paths**

Implement a worker-safe forecast function that loads the country frame, builds the existing information context, fits the fixed order on values available at the origin, calls `get_forecast(steps=len(path))`, and attaches forecast values to the path starting at the first unavailable interval. Set `is_target_day` from the shared local date, retain bridge rows, and call `evaluate_target_day` only for target rows.

- [ ] **Step 8: Implement resumable parent-owned persistence**

Implement deterministic manifest construction, Windows `spawn` workers, parent-owned result persistence, atomic temporary-file replacement, completed-job skipping, incomplete-job retry, and forecast upsert by `(country, target_date, specification_id, timestamp_utc)`. Write job, forecast and aggregate-summary CSVs after each completed result batch.

- [ ] **Step 9: Implement the mini-only CLI guard**

Expose:

```text
python arima_validation_2024.py --mini --workers N --output-dir PATH
```

Require `--mini` for the development-stage CLI. Keep the full-year API and deterministic 366-date manifest available to later work, but do not expose a command that can accidentally launch it during this stage.

- [ ] **Step 10: Run validation tests and confirm they pass**

Run the validation pytest command. Expected result: all manifest, leakage, DST, checkpoint and deterministic-order tests pass.

### Task 4: Run training-only screening

**Files:**
- Modify only the newly created `code\04_models\arima\arima_screening.py` outputs.

- [ ] **Step 1: Run the training-only screen**

Run:

```powershell
& ".\.venv\Scripts\python.exe" code\04_models\arima\arima_screening.py
```

Expected behavior: only 2020-2023 values are loaded for fitting; no 2024 forecast path is created; no 2025 data is accessed.

- [ ] **Step 2: Verify screening completeness**

Assert that both countries have one selected `d`, exactly 11 attempted candidate orders, a recorded fit/diagnostic outcome for every attempted order, and a non-empty frozen shortlist. Verify that every shortlisted order belongs to the fixed 11-order set and has the country-selected `d`.

- [ ] **Step 3: Record screening results**

Report each country’s selected `d`, all candidate fit criteria, rejection reasons, eligible candidates, and frozen shortlist before running validation.

### Task 5: Run ordinary/DST mini-validation

**Files:**
- Create outputs under `results\arima\validation_2024\mini_validation\`.

- [ ] **Step 1: Run the actual mini workload with two workers**

Run:

```powershell
& ".\.venv\Scripts\python.exe" code\04_models\arima\arima_validation_2024.py --mini --workers 2 --output-dir results\arima\validation_2024\mini_validation
```

- [ ] **Step 2: Verify mini-validation coverage and path integrity**

Check all ordinary and DST dates for both countries, no duplicate keys, no missing target forecasts, correct 24/23/25 target lengths, country-specific origin timestamps, no future-information changes, and no final selection file.

- [ ] **Step 3: Report mini-validation metrics**

Report MAE, RMSE, MAPE, coverage, fit failures, convergence warnings and root diagnostics by country, date and shortlisted order.

### Task 6: Benchmark 1/2/3/4 workers and internal threading

**Files:**
- Create separate benchmark outputs under `results\arima\validation_2024\benchmark_workers_N\`.
- Modify only `results\computational_runtime_log.md` after verified measurements.

- [ ] **Step 1: Inspect numerical-library threading**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -c "import numpy as np; np.show_config()"
```

Record whether the installed BLAS/SciPy backend reports internal threading. Do not add a dependency.

- [ ] **Step 2: Benchmark one worker**

Run the actual mini workload with `--workers 1` into `benchmark_workers_1`, recording start time, finish time, elapsed seconds, exit code, job count and failures.

- [ ] **Step 3: Benchmark two workers**

Repeat with `--workers 2` and `benchmark_workers_2`.

- [ ] **Step 4: Benchmark three workers**

Repeat with `--workers 3` and `benchmark_workers_3`.

- [ ] **Step 5: Benchmark four workers**

Repeat with `--workers 4` and `benchmark_workers_4`.

- [ ] **Step 6: Compare controlled numerical threading if relevant**

If internal threading is active, repeat the recommended worker configuration with environment variables set before Python starts:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'
& ".\.venv\Scripts\python.exe" code\04_models\arima\arima_validation_2024.py --mini --workers 2 --output-dir results\arima\validation_2024\benchmark_threads_1
```

Compare with the default-thread setting and then clear the temporary environment variables in the shell. If the backend is effectively single-threaded, record that the controlled comparison is not material.

- [ ] **Step 7: Project the full 2024 runtime**

Use measured candidate-job wall times by country and scale the three representative dates to 366 dates per country. Report the observed worker timings, projected full-year wall time, throughput, failures/retries and the theoretical country-split benefit of a second machine. Do not execute on another machine or copy data.

- [ ] **Step 8: Update the runtime log**

Append only the verified ARIMA screening, mini-validation and benchmark stage to `results\computational_runtime_log.md`. Keep the completed Holt-Winters runtime section unchanged.

### Task 7: Final verification and stop boundary

- [ ] **Step 1: Run all targeted ARIMA tests**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code\04_models\arima -q
```

Expected result: all ARIMA targeted tests pass.

- [ ] **Step 2: Verify no full-year or 2025 outputs exist**

Confirm that no complete 2024 ARIMA job manifest, complete-year summary, selected full-year specification, or 2025 ARIMA output was created. The only validation outputs may be the ordinary/DST mini-validation and worker benchmark directories.

- [ ] **Step 3: Verify unchanged protected outputs**

Confirm that Chapter 6, baseline result files, Holt-Winters result files and the two outdated methodology artifacts were not modified.

- [ ] **Step 4: Stop and report**

Report selected `d`, every training candidate outcome, rejection reasons, eligible candidates, frozen country-specific shortlists, mini-validation results, worker/thread runtimes, projected full-2024 runtime, recommended worker configuration, theoretical second-laptop benefit and targeted test results. Do not start the full 2024 ARIMA validation or any 2025 command.
