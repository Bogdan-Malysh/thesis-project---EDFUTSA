# Final Modelling Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine the completed Germany and Austria processed inputs into two validated final modelling tables without changing any existing input files.

**Architecture:** `modelling_datasets.py` reads the country-specific processed SMARD table as the canonical UTC timeline, selects only the actual target and official forecast, and merges temperature, calendar, and crisis frames on a normalized timezone-aware `timestamp_utc`. Calendar `timestamp_local` is the sole final local timestamp and is copied as the original offset-aware string. The module exposes an in-memory builder for tests and a CLI writer for the two final CSVs; no metadata or report files are written.

**Tech Stack:** Python 3, pandas, NumPy, pytest, timezone-aware `pandas.Timestamp` keys.

---

## File Map

- Create `code/02_preprocessing/modelling_datasets.py`: input configuration, key validation, exact one-to-one merging, output validation, and CLI output writing.
- Create `code/02_preprocessing/test_modelling_datasets.py`: full-country integration tests and duplicate/unmatched-key failure tests.
- Create `data/processed/modelling_germany_hourly.csv` and `data/processed/modelling_austria_hourly.csv` only after the focused tests pass.
- Do not modify SMARD, temperature, calendar, crisis, APG, raw, or validation files.

## Source Schemas

- `smard_{country}_hourly.csv`: `interval_start_utc`, `actual_grid_load_mwh`, `forecasted_grid_load_mwh`, `forecasted_grid_load_valid`; do not use `interval_start_local`.
- `temperature_{country}_hourly.csv`: `interval_start_utc`, `temperature_c`.
- `calendar_{country}_hourly.csv`: `timestamp_utc`, `timestamp_local`, `hour`, `day_of_week`, `month`, `is_weekend`, `is_public_holiday`.
- `crisis_{country}_hourly.csv`: `timestamp_utc`, `timestamp_local`, `is_covid_period`, `is_post_invasion`; select only the UTC key and flags.

## Output Schema

```python
OUTPUT_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "actual_grid_load_mwh",
    "forecasted_grid_load_mwh",
    "forecasted_grid_load_valid",
    "temperature_c",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday",
    "is_covid_period",
    "is_post_invasion",
]
```

### Task 1: Write the failing modelling-dataset tests

**Files:**
- Create `code/02_preprocessing/test_modelling_datasets.py`.

- [x] **Step 1: Add full-country output contract tests.**

Import `build_country_modelling_dataset`, `EXPECTED_ROWS`, and `OUTPUT_COLUMNS`. For Germany and Austria, assert 52,608 rows, exact output columns, unique sorted UTC keys, one-hour UTC differences, no missing or non-finite values outside `forecasted_grid_load_mwh`, binary flags including `forecasted_grid_load_valid`, one local column, no APG columns, and local repeated-hour offsets `{+01:00, +02:00}`. Assert Germany has exactly 24 missing forecast values from `2020-01-30 23:00:00+00:00` through `2020-01-31 22:00:00+00:00`, all with `forecasted_grid_load_valid=False`, and Austria has none.

- [x] **Step 2: Add exact source preservation and key-set tests.**

Read each SMARD source with pandas and assert output UTC keys equal `interval_start_utc`, actual load equals `actual_grid_load_mwh`, forecast equals `forecasted_grid_load_mwh`, and `forecasted_grid_load_valid` equals its source. Assert the output local column equals the calendar `timestamp_local` strings, not SMARD `interval_start_local`.

- [x] **Step 3: Add duplicate and unmatched input failure tests.**

Use the in-memory `build_modelling_dataset` API with full source frames. Duplicate one temperature UTC row and remove one calendar row; assert clear `ValueError` messages for duplicate and unmatched key sets.

- [x] **Step 4: Run the focused tests before production implementation.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/02_preprocessing/test_modelling_datasets.py -q
```

Expected result: collection fails because `modelling_datasets.py` does not yet exist.

### Task 2: Implement validation and in-memory merging

**Files:**
- Create `code/02_preprocessing/modelling_datasets.py`.

- [x] **Step 1: Add project paths, country configuration, and schemas.**

Define `PROJECT_ROOT`, `PROCESSED`, `EXPECTED_ROWS = 52_608`, `COUNTRY_CONFIG`, `INPUT_COLUMNS`, and the exact `OUTPUT_COLUMNS` above. Configure only Germany and Austria; do not include APG.

- [x] **Step 2: Normalize each input to a sorted UTC-key frame.**

Implement `_prepare_frame(frame, key_column, selected_columns, label)` to select columns, parse the key with `pd.to_datetime(..., utc=True, errors="coerce")`, reject missing or duplicate UTC keys, rename the key to `timestamp_utc`, and sort chronologically. Preserve calendar `timestamp_local` strings without parsing or rewriting.

- [x] **Step 3: Validate exact key sets and source values.**

Implement `_require_same_keys(expected_keys, actual_keys, label)` using sorted `DatetimeIndex.equals`. Raise a clear unmatched-key error. Validate the target has exactly 52,608 unique hourly UTC keys. Validate numeric output candidates are present and finite and calendar/crisis indicators are binary.

- [x] **Step 4: Merge with one-to-one validation.**

Implement `build_modelling_dataset(target, temperature, calendar, crisis, country)` using only `timestamp_utc` joins. Check each frame against the target key set, merge with `how="left"` and `validate="one_to_one"`, and raise if any required right-hand values are unmatched. Select `OUTPUT_COLUMNS` so only calendar `timestamp_local` survives.

- [x] **Step 5: Run focused tests and make them pass.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/02_preprocessing/test_modelling_datasets.py -q
```

Expected result: all focused tests pass before any modelling CSV is written.

### Task 3: Add file loading and final output writing

**Files:**
- Modify `code/02_preprocessing/modelling_datasets.py`.

- [x] **Step 1: Add country file loading.**

Implement `build_country_modelling_dataset(country, processed_directory=PROCESSED)` to read only the configured source files and pass their raw frames to the in-memory builder. Read values without transformations that alter timestamps or source numeric values.

- [x] **Step 2: Add the final CLI writer.**

Implement `run_pipeline(processed_directory=PROCESSED)` to build both country frames, verify them again, and write only `modelling_germany_hourly.csv` and `modelling_austria_hourly.csv`. Do not write metadata, hashes, summaries, or reports. Add `if __name__ == "__main__": run_pipeline()`.

- [x] **Step 3: Generate outputs only after tests pass.**

```powershell
& ".\.venv\Scripts\python.exe" code/02_preprocessing/modelling_datasets.py
```

Expected result: exactly two new 52,608-row CSV files.

### Task 4: Full verification

**Files:**
- Read both generated modelling CSVs and all source inputs.

- [x] **Step 1: Run all tests and dependency checks.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code -q
& ".\.venv\Scripts\python.exe" -m pip check
```

- [x] **Step 2: Independently verify both outputs.**

Assert exact columns, 52,608 rows, target UTC range, one-hour continuity, no missing/non-finite values outside the documented Germany benchmark exception, exact target/forecast/forecast-valid equality, exact calendar local strings, binary flags, repeated local offsets, no APG columns, and no unmatched joins. Report benchmark observations used as 52,584 for Germany and 52,608 for Austria.

- [x] **Step 3: Verify input immutability.**

Confirm the existing SMARD, temperature, calendar, crisis, APG, and raw files were not written or changed. No metadata or additional report files are generated.
