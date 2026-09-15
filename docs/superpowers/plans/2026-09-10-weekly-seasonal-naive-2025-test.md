# Weekly Seasonal Naive 2025 Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the unchanged 2024 Weekly seasonal naive implementation over all 2025 Germany and Austria target days and create merge-safe 2025 result tables.

**Architecture:** Add an isolated `baseline_test_2025.py` runner that imports the existing forecasting framework, calls only `forecast_path(..., "Weekly seasonal naive")`, and delegates all metric calculation to `evaluate_target_day`/`calculate_metrics`. The runner writes authoritative 2025 forecast paths and a country summary, then merges the summary into the two 2025 test tables by `(country, model_family, specification_id)` while retaining other completed families.

**Tech Stack:** Python, pandas, NumPy through existing framework functions, pytest, project interpreter `.venv\Scripts\python.exe`.

---

### Task 1: Add failing 2025 runner and table-contract tests

**Files:**
- Create: `code/04_models/baselines/test_baseline_test_2025.py`
- Test: existing `code/04_models/common/test_forecasting_framework.py` behavior remains unchanged

- [ ] **Step 1: Write tests for the fixed model identity and date manifest**

Add tests asserting:

```python
assert validation.MODEL_FAMILY == "baselines"
assert validation.SPECIFICATION_ID == "Weekly seasonal naive"
dates = validation.validation_dates(validation.load_country_data("Germany"))
assert len(dates) == 365
assert dates[0].isoformat() == "2025-01-01"
assert dates[-1].isoformat() == "2025-12-31"
```

- [ ] **Step 2: Write tests for path integrity and DST behavior**

For both countries and dates `2025-03-30` and `2025-10-26`, call the new runner's path-validation helper after building a path with the existing framework. Assert target lengths `23` and `25`, unique target UTC timestamps, finite forecasts, the expected model label, and `forecast_origin_utc == information_cutoff_utc`.

Also assert Germany's target-day origin is the 18:00 previous-day local origin and Austria's is the 08:00 previous-day local origin.

- [ ] **Step 3: Write tests for leakage validation**

Use a real weekly forecast path and assert the helper accepts observed source timestamps only when their interval ends by the cutoff, accepts forecast source timestamps only when the source row precedes the current row, and rejects a copied path whose source timestamp is changed to a future timestamp.

- [ ] **Step 4: Write tests for exact metric delegation**

Monkeypatch `validation.evaluate_target_day` with a recorder, call the country-summary helper, and assert the recorder receives the target-only forecast frame and expected observation count. Do not add an independent MAE, RMSE, or MAPE formula to the test or implementation.

- [ ] **Step 5: Write tests for merge-safe 2025 tables**

Create temporary existing `model_test_2025_all_models.csv` and `model_test_2025_overview.csv` containing a completed non-baseline row. Pass two new Weekly seasonal naive rows to the table-update helper. Assert:

```python
keys = ["country", "model_family", "specification_id"]
assert prior_family_key in set(updated[keys].itertuples(index=False, name=None))
assert len(updated) == 3
assert list(updated.columns) >= [
    "country", "model_family", "specification_id",
    "mae", "rmse", "mape", "n_observations", "coverage",
]
```

Run the helper twice and assert byte-identical CSV content after the second run. Assert replacing the same Weekly key changes only that key and retains the prior family.

- [ ] **Step 6: Run the new tests to verify the expected collection failure**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest code/04_models/baselines/test_baseline_test_2025.py -q
```

Expected: collection fails because `baseline_test_2025.py` does not exist yet.

### Task 2: Implement the isolated 2025 runner

**Files:**
- Create: `code/04_models/baselines/baseline_test_2025.py`

- [ ] **Step 1: Define constants and imports without touching 2024 code**

Import `BASELINE_MODELS`, `COUNTRY_CONFIG`, `PROCESSED`, `build_information_context`, `evaluate_target_day`, `extract_target_day`, `forecast_path`, and `load_country_data` from `common.forecasting_framework`. Define:

```python
MODEL_FAMILY = "baselines"
SPECIFICATION_ID = "Weekly seasonal naive"
VALIDATION_YEAR = 2025
```

Use separate 2025 output paths:

```python
FORECAST_PATH = PROJECT_ROOT / "results" / "baselines" / "forecasts" / "baseline_forecasts_2025.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "baselines" / "tables" / "baseline_validation_2025.csv"
ALL_MODELS_PATH = PROJECT_ROOT / "results" / "model_test_2025_all_models.csv"
OVERVIEW_PATH = PROJECT_ROOT / "results" / "model_test_2025_overview.csv"
```

Reject any attempt to run a model other than `SPECIFICATION_ID`.

- [ ] **Step 2: Implement exact 2025 date enumeration**

Implement `validation_dates(frame)` by selecting local dates beginning with `2025-`, converting them to `date`, sorting them, and requiring the exact set `date(2025, 1, 1)` through `date(2025, 12, 31)`. Raise `ValueError` for missing or extra dates.

- [ ] **Step 3: Implement path integrity and leakage checks**

Implement `validate_forecast_path(path, country, target_date, context)` to require:

- only the fixed baseline model and target date;
- target intervals in `{23, 24, 25}`;
- unique UTC timestamps per path and unique UTC timestamps among scored target rows for a country;
- finite `forecast_mwh` values;
- equal `forecast_origin_utc` and `information_cutoff_utc`;
- weekly source timestamps exactly `timestamp_utc - 168 hours`;
- observed source interval ends no later than the cutoff;
- forecast sources refer only to an earlier row in the same path;
- no fallback source for the weekly seasonal naive path.

- [ ] **Step 4: Implement country summaries using existing metrics**

For each country, build one path per date with:

```python
context = build_information_context(frame, country, target_date)
path = forecast_path(
    frame, country, target_date, SPECIFICATION_ID, context=context
)
```

Validate each path, append it, and call `evaluate_target_day` for metrics. After concatenating all paths, call `evaluate_target_day` again for the country summary with `expected_observations=8760`. Do not calculate metric formulas in the new file.

- [ ] **Step 5: Implement authoritative output writing**

After all 730 jobs pass validation, concatenate the full paths and write `FORECAST_PATH`. Write `SUMMARY_PATH` with exactly two country rows and fields including `country`, `model_family`, `specification_id`, `mae`, `rmse`, `mape`, `evaluated_observations`, `expected_observations`, `coverage`, and `validation_days`. Do not write until the complete run has passed.

- [ ] **Step 6: Implement merge-safe 2025 table updates**

Implement `update_test_tables(summary_path=SUMMARY_PATH, all_models_path=ALL_MODELS_PATH, overview_path=OVERVIEW_PATH)`:

1. Read the authoritative summary and rename `evaluated_observations` to `n_observations`.
2. Read each existing destination if present, requiring the key and metric columns.
3. Concatenate existing rows with the new rows.
4. Drop duplicate keys, keeping the new row for the same key only.
5. Sort deterministically by `country` order `Germany`, `Austria`, then `model_family`, then `specification_id`.
6. Atomically write both destinations with the required columns.

The first run produces only the two Weekly seasonal naive rows. Future runs retain other model-family rows.

- [ ] **Step 7: Add CLI entry point**

Provide `run_validation()` and a `main()` that runs the complete 2025 weekly test, updates both tables only after validation passes, and prints a JSON-serializable summary. No model selector or multi-model loop is permitted.

- [ ] **Step 8: Run focused tests**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest code/04_models/baselines/test_baseline_test_2025.py code/04_models/baselines/test_baseline_validation.py code/04_models/common/test_forecasting_framework.py -q
```

Expected: all focused tests pass and no 2024 result file is written by the tests.

### Task 3: Run and verify the full 2025 test

**Files:**
- Create at runtime: `results/baselines/forecasts/baseline_forecasts_2025.csv`
- Create at runtime: `results/baselines/tables/baseline_validation_2025.csv`
- Create at runtime: `results/model_test_2025_all_models.csv`
- Create at runtime: `results/model_test_2025_overview.csv`

- [ ] **Step 1: Snapshot protected 2024 outputs**

Record SHA-256 hashes for `results/baselines/tables/baseline_validation_2024.csv`, `results/baselines/forecasts/baseline_forecasts_2024.csv`, and `results/selected_models_registry.csv` before the run.

- [ ] **Step 2: Run only the 2025 Weekly seasonal naive test**

Run:

```powershell
& ".venv\Scripts\python.exe" code/04_models/baselines/baseline_test_2025.py
```

- [ ] **Step 3: Verify counts and forecast integrity**

Require 730 jobs, 365 dates per country, 8,760 evaluated observations per country, 100% country coverage, 23 target intervals on `2025-03-30`, 25 target intervals on `2025-10-26`, no duplicate scored UTC timestamps, finite forecasts, exact Weekly seasonal naive labels, and no source timestamp after the relevant information cutoff.

- [ ] **Step 4: Verify table contents and 2024 protection**

Read both 2025 tables and assert they contain exactly the two initial Weekly seasonal naive rows with the required schema. Compare the post-run protected hashes and assert they are unchanged. Assert no 2025 row is present in any 2024 result file.

- [ ] **Step 5: Report metrics and 2024 comparison**

Read 2025 metrics from the authoritative summary and 2024 metrics from the existing 2024 baseline summary. Report Germany and Austria MAE, RMSE, MAPE, observations, and coverage for both years. Do not update the selection registry or run another model.
