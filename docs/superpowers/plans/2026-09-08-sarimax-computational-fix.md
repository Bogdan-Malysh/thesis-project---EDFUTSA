# SARIMAX Computational Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make forecast-only SARIMAX fits use bounded L-BFGS estimation without covariance inference or uncontrolled retry, then run exactly one full Germany benchmark.

**Architecture:** Keep the shared SARIMA fitter as the single place for optimizer invocation, fit validation, and metadata normalization. Add explicit keyword-only controls so ordinary SARIMA keeps its current behavior while `fit_sarimax` selects a forecast-only profile; final reporting can opt back into covariance. Extend fit metadata with optimizer iteration and function-call counts.

**Tech Stack:** Python 3.12, pandas, NumPy, statsmodels 0.14.6, pytest, existing project forecasting framework.

---

### Task 1: Add bounded fit controls and metadata

**Files:**
- Modify: `code/04_models/sarima/sarima_models.py:143-176,660-905`
- Test: `code/04_models/sarima/test_sarima_models.py:85-220`

- [ ] **Step 1: Add failing tests for bounded options and optional standard errors**

Extend the fake result with `mle_retvals = {"converged": True, "success": True, "status": 0, "warnflag": 0, "iterations": 3, "fcalls": 17}` and make the fake model capture arbitrary `fit(**kwargs)` calls. Add tests that call:

```python
record = fit_sarima(
    np.arange(200.0),
    SarimaOrder(1, 0, 0, 0, 0, 0),
    fit_kwargs={
        "method": "lbfgs",
        "maxiter": 50,
        "maxfun": 1000,
        "cov_type": "none",
        "low_memory": True,
    },
    require_standard_errors=False,
    retry_maxiter=None,
)
```

Assert exactly one fit call, the exact kwargs above, `record.optimizer_iterations == 3`, `record.optimizer_function_calls == 17`, `record.standard_errors_finite is False`, and `record.eligible` is true when all forecast checks pass. Add a second test with `converged=False` and `retry_maxiter=None`; assert one call, `record.optimizer_retry_count == 0`, `record.converged is False`, and `record.eligible is False`.

- [ ] **Step 2: Run the focused tests and verify they fail for the missing API**

Run:

```powershell
.venv\Scripts\python.exe -m pytest code/04_models/sarima/test_sarima_models.py -q
```

Expected: failure because `fit_sarima` does not yet accept `fit_kwargs`, `require_standard_errors`, or `retry_maxiter`, and the fit record has no optimizer count fields.

- [ ] **Step 3: Implement the explicit fitter controls**

Update `SarimaFitRecord` with defaulted fields:

```python
optimizer_iterations: int | None = None
optimizer_function_calls: int | None = None
```

Change `_fit_once` to accept `Mapping[str, object] | None` and pass those options directly to `model.fit`. Add keyword-only arguments to `fit_sarima`:

```python
fit_kwargs: Mapping[str, object] | None = None,
require_standard_errors: bool = True,
retry_maxiter: int | None = 1000,
```

Copy `fit_kwargs` once. Run the first fit with that mapping. If it does not explicitly converge and `retry_maxiter` is not `None`, copy the original mapping, replace only `maxiter`, and run the retry. If `retry_maxiter` is `None`, do not call `_fit_once` again.

Extract iteration counts from `mle_retvals` keys `iterations` or `nit`, and function counts from `fcalls`, `funcalls`, or `nfev`. Normalize missing values to `None`. Store them in the record and add them to the final metadata dictionary.

Only append the existing `"non-finite standard errors"` failure reason when `require_standard_errors` is true. Keep all convergence, optimizer status, finite-parameter, likelihood, forecast, and root checks unchanged, so a non-converged bounded result remains ineligible.

- [ ] **Step 4: Run the focused tests and confirm existing SARIMA retry behavior remains**

Run:

```powershell
.venv\Scripts\python.exe -m pytest code/04_models/sarima/test_sarima_models.py -q
```

Expected: all tests pass, including the existing ordinary-SARIMA `maxiter=1000` retry tests and the new forecast-only tests.

### Task 2: Configure exact forecast-only SARIMAX fitting

**Files:**
- Modify: `code/04_models/sarima/sarimax_models.py:23-126`
- Test: `code/04_models/sarima/test_sarimax_models.py:1-48`

- [ ] **Step 1: Add failing tests for the SARIMAX forecast profile**

Monkeypatch `sarimax_models.fit_sarima` with a recorder returning a valid fake `SarimaFitRecord`. Call:

```python
fit_sarimax(values, exog, COUNTRY_ORDERS["Germany"])
```

Assert that the call uses `method="lbfgs"`, `maxiter=50`, `maxfun=1000`, `cov_type="none"`, `low_memory=True`, `require_standard_errors=False`, `retry_maxiter=None`, and skips a second exogenous rank validation. Add a test calling `fit_sarimax(..., forecast_only=False)` and assert it uses the ordinary full-inference defaults.

- [ ] **Step 2: Run the focused SARIMAX tests and verify the new expectations fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest code/04_models/sarima/test_sarimax_models.py -q
```

Expected: failure because `fit_sarimax` currently has no forecast profile or `forecast_only` argument.

- [ ] **Step 3: Implement the forecast profile and shared exogenous validation**

Add a module constant:

```python
SARIMAX_FORECAST_FIT_KWARGS = {
    "method": "lbfgs",
    "maxiter": 50,
    "maxfun": 1000,
    "disp": 0,
    "cov_type": "none",
    "low_memory": True,
}
```

Add keyword-only `forecast_only: bool = True` to `fit_sarimax`. Use the constant, `require_standard_errors=False`, and `retry_maxiter=None` when true. Use `None`, `require_standard_errors=True`, and the existing retry default when false. Add `validate_exog: bool = True` to `fit_sarima`, and pass `validate_exog=False` from `fit_sarimax` after its existing exogenous matrix checks have completed.

- [ ] **Step 4: Run the SARIMAX tests and the shared focused suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest code/04_models/sarima/test_sarimax_models.py code/04_models/sarima/test_sarima_models.py -q
```

Expected: all tests pass, with the selected orders and exogenous columns unchanged.

### Task 3: Persist optimizer metadata in validation outputs

**Files:**
- Modify: `code/04_models/sarima/sarima_validation_2024.py:78-145,175-217,658-700`
- Modify: `code/04_models/sarima/sarimax_validation_2024.py:55-65,177-191`
- Modify: `code/04_models/sarima/sarimax_state_validation_2024.py:47-65`
- Test: `code/04_models/sarima/test_sarimax_validation_2024.py`

- [ ] **Step 1: Add failing metadata assertions**

Extend the fake fit metadata fixture and assert that `_fit_metadata` includes `optimizer_iterations` and `optimizer_function_calls`. Assert that the SARIMAX and fixed-state job/diagnostic column lists contain both fields and that a forecast-only fit may have `standard_errors_finite=False` without being converted into an error by metadata construction.

- [ ] **Step 2: Run the targeted validation tests and verify the assertions fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest code/04_models/sarima/test_sarimax_validation_2024.py code/04_models/sarima/test_sarimax_state_update_2024.py -q
```

Expected: failure because the two optimizer metadata columns do not yet exist.

- [ ] **Step 3: Add metadata columns and propagation**

Insert `optimizer_iterations` and `optimizer_function_calls` beside the existing optimizer status fields in the base job and diagnostic columns. Add both values in `sarima_validation_2024._fit_metadata`. The SARIMAX modules inherit these columns from the base lists; keep their exogenous and state-update columns unchanged.

- [ ] **Step 4: Run all SARIMA/SARIMAX focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest code/04_models/sarima/test_sarima_models.py code/04_models/sarima/test_sarima_validation_2024.py code/04_models/sarima/test_sarimax_models.py code/04_models/sarima/test_sarimax_validation_2024.py code/04_models/sarima/test_sarimax_state_update_2024.py -q
```

Expected: all tests pass.

### Task 4: Add the one-fit full-data benchmark

**Files:**
- Create: `code/04_models/sarima/benchmark_sarimax_fit_2024.py`
- Test: `code/04_models/sarima/test_sarimax_models.py`

- [ ] **Step 1: Add benchmark contract tests**

Test the benchmark result formatter with a synthetic fit record and assert that its output contains `observations`, `parameters`, `state_dimension`, `optimizer_method`, `function_evaluations`, `iterations`, `converged`, `log_likelihood`, `fit_seconds`, `rss_start_bytes`, `rss_finish_bytes`, and `covariance_skipped`.

- [ ] **Step 2: Implement the benchmark script**

Load Germany through `load_validation_country_data`, call `build_initial_training_set`, construct exactly the frozen `sarimax_calendar` exogenous matrix and Germany order, and call `fit_sarimax(..., fit_kwargs=SARIMAX_BENCHMARK_FIT_KWARGS)` once. This benchmark-only configuration copies the forecast profile and overrides `maxfun` to `3000`. Measure wall-clock time around that call. Use Windows `GetProcessMemoryInfo` through `ctypes` to record start and finish resident working set. Read state dimension from `fit.fitted_result.model.ssm.k_states`, optimizer method from the explicit profile, and counts/status/likelihood from the returned fit record. Write one JSON result under `results/arima_sarimax/validation_2024/` and print it.

The script must exit nonzero if the fit is ineligible, parameters are non-finite, roots are invalid, or the fitted result cannot produce a one-step forecast. It must not call any mini or full-year validation function.

- [ ] **Step 3: Run the benchmark contract test**

Run:

```powershell
.venv\Scripts\python.exe -m pytest code/04_models/sarima/test_sarimax_models.py -q
```

Expected: all tests pass.

### Task 5: Run the required benchmark and stop at the gate

**Files:**
- Output: `results/arima_sarimax/validation_2024/sarimax_full_fit_benchmark.json`
- Runtime log: `results/arima_sarimax/validation_2024/sarimax_fixed_state_runtime_segments.jsonl`

- [ ] **Step 1: Run the exact one-fit benchmark**

Run:

```powershell
.venv\Scripts\python.exe code/04_models/sarima/benchmark_sarimax_fit_2024.py
```

Use the project interpreter and allow only this one full Germany fit. Do not start the 12-job mini, full 2024 validation, or 2025 validation from the benchmark script.

- [ ] **Step 2: Verify the benchmark output**

Confirm the JSON reports all required fields, `covariance_skipped` is true, the optimizer counts are populated or explicitly unavailable, and the fit is either valid and converged or clearly recorded as a bounded non-convergence. Confirm no Python process remains after completion.

- [ ] **Step 3: Report the gate decision**

If the fit is valid and practically useful, report the measured values and wait for the next instruction before starting the 12-job mini. If it is not practically useful or not valid, report the measured bottleneck and stop without starting any mini or full validation.
