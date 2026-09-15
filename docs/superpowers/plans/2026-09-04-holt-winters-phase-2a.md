# Phase 2A Holt-Winters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement reproducible Holt-Winters screening, frozen shortlisting, and July/DST smoke testing while preserving Phase 1 behavior and avoiding complete 2024 validation.

**Architecture:** Keep Phase 1 files and outputs unchanged. Add one model module containing specification construction, fitting, eligibility, selection, and continuous-path forecasting, plus one runner that writes only screening and shortlist CSVs and performs in-memory smoke tests. Reuse Phase 1 public data, origin, information-set, target-day, and metric functions.

**Tech Stack:** Python 3.12, pandas, NumPy, statsmodels `ExponentialSmoothing`, pytest, project `.venv`.

---

### Task 1: Add the Holt-Winters dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements.lock`

- [ ] **Step 1: Add the direct dependency**

Add `statsmodels==0.14.6` to `requirements.txt` without changing existing pins.

- [ ] **Step 2: Record the resolved dependency closure**

Add the installed, exact versions of `statsmodels` and any packages required by it but absent from `requirements.lock`, including `patsy` and `scipy`, while preserving all existing lock entries.

- [ ] **Step 3: Verify the dependency metadata**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pip check
```

Expected: `No broken requirements found.`

### Task 2: Define the specification grid and fitting records

**Files:**
- Create: `code/07_forecasting_results/holt_winters_models.py`
- Create: `code/07_forecasting_results/test_holt_winters_models.py`

- [ ] **Step 1: Write the grid and metadata tests**

Test that `build_specification_grid()` returns exactly eight deterministic `HoltWintersSpecification` records covering:

```python
{("add", 24, False), ("add", 24, True),
 ("add", 168, False), ("add", 168, True),
 ("mul", 24, False), ("mul", 24, True),
 ("mul", 168, False), ("mul", 168, True)}
```

Also assert that specification IDs are unique and stable.

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\07_forecasting_results\test_holt_winters_models.py" -q
```

Expected: collection or assertion failure because the new module and grid do not yet exist.

- [ ] **Step 3: Implement the specification records**

Define:

```python
@dataclass(frozen=True)
class HoltWintersSpecification:
    specification_id: str
    seasonal_form: str
    seasonal_periods: int
    damped_trend: bool

def build_specification_grid() -> tuple[HoltWintersSpecification, ...]: ...
```

Use `seasonal_form` values accepted by statsmodels (`"add"`, `"mul"`) and `trend="add"` for every fit. Include deterministic grid order as an explicit specification-order field or stable tuple position.

- [ ] **Step 4: Add fitting-record types and the fit contract**

Define a fit-record structure containing the specification, fitted result when successful, AIC, AICc, BIC, convergence status, fit status, fitting time in seconds, `selection_criterion`, `selection_value`, training observation count, and error message. The fit function must call:

```python
ExponentialSmoothing(
    training_values,
    trend="add",
    damped_trend=spec.damped_trend,
    seasonal=spec.seasonal_form,
    seasonal_periods=spec.seasonal_periods,
    initialization_method="estimated",
    use_boxcox=False,
).fit(optimized=True, remove_bias=False)
```

Reject multiplicative fits before model construction when any training value is non-positive, using the clear error `multiplicative seasonal Holt-Winters requires strictly positive training values`.

- [ ] **Step 5: Run the grid and fit tests**

Run the focused model test file and confirm the grid and fit-record tests pass.

### Task 3: Implement eligibility and AICc shortlisting

**Files:**
- Modify: `code/07_forecasting_results/holt_winters_models.py`
- Modify: `code/07_forecasting_results/test_holt_winters_models.py`

- [ ] **Step 1: Write selection tests first**

Use synthetic screening rows to test that:

```python
select_shortlists(rows)
```

selects the lowest finite `selection_value` within each `(seasonal_form, seasonal_period)` group, selects the sole eligible candidate when its competitor is failed, and does not create a selection when both candidates are ineligible. Test deterministic tie-breaking by specification order.

Test that a finite AICc produces `selection_criterion == "AICc"`, a missing/non-finite AICc with finite AIC produces `"AIC"`, and neither finite criterion makes a candidate ineligible.

Test that explicit optimizer failure is ineligible, while a successful fit with missing convergence metadata records `convergence_status == "unknown"` and remains eligible when its criterion is finite.

- [ ] **Step 2: Run selection tests to verify they fail**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\07_forecasting_results\test_holt_winters_models.py" -q
```

Expected: failures for the unimplemented selection behavior.

- [ ] **Step 3: Implement criterion extraction and selection**

Use finite fitted AICc when available and finite fitted AIC only when AICc is unavailable. Treat `mle_retvals.success is False` or an equivalent explicit optimizer-failure flag as failed. Treat absent convergence metadata as `unknown`; do not reject it solely for being unknown.

Group eligible rows by seasonal form and period, sort by `selection_value` and deterministic specification order, and select the first row. Raise a clear `ShortlistingError` containing both failed candidate records when a group has no eligible candidates. Do not start smoke testing after that exception.

- [ ] **Step 4: Run selection tests to verify they pass**

Run the focused model test file and confirm all selection, fallback, explicit-failure, unknown-convergence, and no-force-selection tests pass.

### Task 4: Reuse Phase 1 information sets and implement continuous Holt-Winters paths

**Files:**
- Modify: `code/07_forecasting_results/holt_winters_models.py`
- Modify: `code/07_forecasting_results/test_holt_winters_models.py`

- [ ] **Step 1: Write path and leakage tests**

Build synthetic hourly frames and test that the forecast helper:

```python
forecast_holt_winters(frame, "Germany", date(2024, 1, 2), specification)
```

uses `information_set(frame, origin)` for training, starts at `timestamp_utc >= origin`, preserves target timestamps, and returns 30 rows for ordinary Germany and 40 rows for ordinary Austria. Test 23- and 25-hour target dates, no missing predictions, and that changing all observations after the origin does not change the fitted forecast values. Confirm temperature columns are not read.

- [ ] **Step 2: Run path tests to verify they fail**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\07_forecasting_results\test_holt_winters_models.py" -q
```

Expected: failures for the unimplemented forecast helper.

- [ ] **Step 3: Implement the approved data flow**

Use `country_forecast_origin`, `information_set`, and `extract_target_day` from `forecasting_framework`. Select the continuous path with `timestamp_utc >= origin` and `local_date <= target_date`. Fit on `actual_load_mwh` from the information set only, call `fit.forecast(len(path))`, and return path metadata plus `forecast_mwh`, `is_target_day`, and `evaluated` fields. Raise a recorded fit or prediction failure for optimizer failures, exceptions, length mismatches, or non-finite predictions.

- [ ] **Step 4: Run path tests to verify they pass**

Run the focused model test file and confirm path counts, target-day lengths, leakage protection, and finite forecasts pass.

### Task 5: Add screening and smoke orchestration

**Files:**
- Create: `code/07_forecasting_results/holt_winters_phase2a.py`
- Modify: `code/07_forecasting_results/test_holt_winters_models.py`

- [ ] **Step 1: Write runner contract tests**

Test that the first 2024 local target date is screened for both countries using only its information set, that screening creates 16 rows and four shortlist rows per country when all candidates are eligible, and that a failed group raises before the smoke function is called. Test smoke cases exactly at:

```python
("Germany", "2024-07-15")
("Austria", "2024-07-15")
("Germany", "2024-03-31")
("Germany", "2024-10-27")
```

Test that smoke forecasts remain in memory and report no missing predictions.

- [ ] **Step 2: Run runner tests to verify they fail**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\07_forecasting_results\test_holt_winters_models.py" -q
```

Expected: failures for the unimplemented screening and smoke runner.

- [ ] **Step 3: Implement screening and output writing**

Screen all eight specifications at the first 2024 origin for each country, record every fit result, select eligible candidates, and write only:

```text
results/tables/holt_winters_screening_2024.csv
results/tables/holt_winters_shortlists_2024.csv
```

If a group has no eligible candidate, write screening diagnostics, report both failures, and stop before smoke testing. Load the frozen shortlist CSV before smoke execution so the smoke uses the persisted identities.

- [ ] **Step 4: Implement the July/DST in-memory smoke runner**

Refit every selected specification at each smoke origin using that origin's expanding information set. Report target interval count, path count, bridge count, missing predictions, failure details, convergence status, and fitting time by specification. Do not write smoke forecasts.

- [ ] **Step 5: Implement the runtime estimate**

Use measured smoke fit times by country/specification to estimate 366 origins per country and four selected specifications per country, for 2,928 projected fits. Report the measured sample basis and projected duration.

- [ ] **Step 6: Run focused tests to verify the runner**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\07_forecasting_results\test_holt_winters_models.py" -q
```

Expected: all Phase 2A focused tests pass.

### Task 6: Run the requested checks and preserve Phase 1 artifacts

**Files:**
- Verify: `results/forecasts/baseline_forecasts_2024.csv`
- Verify: `results/tables/baseline_validation_2024.csv`
- Verify: `results/tables/validation_failures.csv`

- [ ] **Step 1: Run the focused tests**

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\07_forecasting_results\test_holt_winters_models.py" -q
```

- [ ] **Step 2: Run the Phase 2A runner without complete validation**

```powershell
& ".venv\Scripts\python.exe" "code\07_forecasting_results\holt_winters_phase2a.py"
```

Confirm the command performs screening and the four requested smoke cases only.

- [ ] **Step 3: Run the complete test suite**

```powershell
& ".venv\Scripts\python.exe" -m pytest -q
```

- [ ] **Step 4: Verify Phase 1 files were not changed**

Confirm the three Phase 1 baseline files retain their existing contents and that no 2024 baseline validation command was run during Phase 2A.

- [ ] **Step 5: Report results and stop**

Report changed files, screening rows and shortlist identities, smoke counts and failures, fitting time by specification, runtime estimate, focused tests, full tests, and the fact that complete 2024 validation and 2025 data were not used.

## Correction Plan Before Full Validation

### Task 7: Freeze and reuse optimizer configurations

**Files:**
- Modify: `code/07_forecasting_results/holt_winters_models.py`
- Modify: `code/07_forecasting_results/holt_winters_phase2a.py`
- Modify: `code/07_forecasting_results/test_holt_winters_models.py`
- Modify: `code/07_forecasting_results/test_holt_winters_phase2a.py`

- [ ] **Step 1: Add failing tests for frozen optimizer reuse**

Test that a forecast supplied with a frozen `least_squares` configuration calls that exact configuration first, and that a failed frozen attempt proceeds through fallback configurations without repeating an identical `(optimizer, minimize_kwargs)` pair. Test that the existing no-frozen-configuration screening path retains its current sequence.

- [ ] **Step 2: Implement configuration-aware fitting**

Add an optional frozen optimizer configuration to the fit/forecast path. Build the forecast sequence by prepending the frozen pair to the fixed fallback configurations and remove only exact duplicate pairs using a canonical JSON key. Preserve the existing conditional screening sequence when no frozen configuration is supplied.

- [ ] **Step 3: Persist and extract frozen settings**

Use `optimizer_used` and the final successful attempt’s `minimize_kwargs` from `optimizer_attempts_json`. Require the final successful attempt to match the persisted optimizer and pass the resulting pair into smoke and future rolling forecasts; do not branch on specification names.

### Task 8: Harden shortlist and forecast validation

**Files:**
- Modify: `code/07_forecasting_results/holt_winters_models.py`
- Modify: `code/07_forecasting_results/holt_winters_phase2a.py`
- Modify: `code/07_forecasting_results/test_holt_winters_models.py`
- Modify: `code/07_forecasting_results/test_holt_winters_phase2a.py`

- [ ] **Step 1: Add failing validation tests**

Test rejection of malformed, duplicated, incomplete, unknown-ID, inconsistent-identity, malformed-damping, ineligible, and non-finite-selection shortlist rows. Test that `select_shortlists()` rejects infinite selection values, that an ineligible fit cannot forecast, and that German spring/autumn smoke paths are 29 and 31 intervals.

- [ ] **Step 2: Implement strict shortlist loading and selection**

Require exactly the two configured countries and four distinct approved seasonal-form/period groups per country. Parse `damped_trend` and `eligible` from explicit boolean values, validate finite selection values, check specification IDs and all identity fields, and validate persisted optimizer metadata. Filter shortlisting candidates with `np.isfinite(selection_value)`.

- [ ] **Step 3: Enforce forecast eligibility and record optimizer metadata**

Reject forecasts when `record.eligible` is false. Include `optimizer_used` and `optimizer_attempt_count` in successful and failed smoke/forecast records.

### Task 9: Stop invalid runtime estimates and verify

**Files:**
- Modify: `code/07_forecasting_results/holt_winters_phase2a.py`
- Modify: `code/07_forecasting_results/test_holt_winters_phase2a.py`

- [ ] **Step 1: Add the runtime failure test**

Provide smoke results containing a failed fit and assert that runtime estimation raises before calculating a partial estimate.

- [ ] **Step 2: Implement the runtime guard**

Raise a clear runtime-estimation error whenever any smoke result is failed or lacks finite fitting time.

- [ ] **Step 3: Run the required checks**

Run the focused Phase 2A tests, the complete test suite, and `holt_winters_phase2a.py --smoke-only`. Confirm the shortlist identities remain unchanged, report the revised runtime estimate against 15.45 hours, and stop before `baseline_validation_2024.py` or any 2025 data is used.
