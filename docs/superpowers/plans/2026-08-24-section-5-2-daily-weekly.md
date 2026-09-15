# Section 5.2 Daily and Weekly Demand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Section 5.2 with local-calendar daily summaries, weekday/weekend comparisons, weekly hourly profiles, and a two-panel weekday-hour heatmap without changing existing Section 5.2 or Section 5.1 outputs.

**Architecture:** `section_5_2_daily_weekly.py` will read only `timestamp_local`, `actual_grid_load_mwh`, `hour`, and `day_of_week`, validate the established day-code mapping (`0=Monday` through `6=Sunday`) against local timestamp names, then build daily means and raw hourly weekday profiles. It will write only the requested 14-row summary, 336-row weekly profile, and Figure 5.3 PNG/PDF; all existing outputs are protected by byte-hash tests.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib, pytest.

---

## File Map

- Create `code/03_exploratory_analysis/section_5_2_daily_weekly.py`: loading, day-code validation, daily aggregation, weekday/hour profiles, heatmap rendering, and CLI.
- Create `code/03_exploratory_analysis/test_section_5_2_daily_weekly.py`: input, day mapping, DST, summary/profile, heatmap, and immutability tests.
- Create `outputs/chapter5/section_5_2/day_of_week_summary.csv`.
- Create `outputs/chapter5/section_5_2/weekly_hourly_profile.csv`.
- Create `outputs/chapter5/section_5_2/figure_5_3_weekly_demand_patterns.png`.
- Create `outputs/chapter5/section_5_2/figure_5_3_weekly_demand_patterns.pdf`.
- Do not modify existing Figure 5.1, Figure 5.2, hourly profile, modelling inputs, raw files, or validation files.

## Data Contract

- Inputs are `data/processed/modelling_germany_hourly.csv` and `data/processed/modelling_austria_hourly.csv`.
- Read only `timestamp_local`, `actual_grid_load_mwh`, `hour`, and `day_of_week`.
- Confirm the observed mapping against `pd.Timestamp(timestamp_local).day_name()`: `0 Monday`, `1 Tuesday`, `2 Wednesday`, `3 Thursday`, `4 Friday`, `5 Saturday`, `6 Sunday`.
- Derive `local_date` directly from `timestamp_local`; never derive local calendar values from UTC.
- Require 52,608 hourly rows, 2,192 local dates, local-date counts only in `{23, 24, 25}`, seven weekday codes, and all 24 hours for every country-weekday combination.
- Use sample standard deviation (`ddof=1`) and pandas’ default linear quartiles, consistent with earlier Section 5 analyses.
- Weekday and weekend averages are calculated from daily mean values. Weekday/Saturday/Sunday peak and minimum hours are calculated from original hourly observations filtered by the corresponding weekday groups.

## Output Contracts

`day_of_week_summary.csv` has exactly 14 rows ordered Germany Monday-Sunday, then Austria Monday-Sunday, with columns:

```text
country,day_of_week,day_name,number_of_days,mean_daily_mean_grid_load_mwh,median_daily_mean_grid_load_mwh,standard_deviation_mwh,first_quartile_mwh,third_quartile_mwh
```

`weekly_hourly_profile.csv` has exactly 336 rows ordered country, Monday-Sunday, hour 0-23, with columns:

```text
country,day_of_week,day_name,hour,number_of_observations,mean_hourly_grid_load_mwh,median_hourly_grid_load_mwh,first_quartile_mwh,third_quartile_mwh
```

Figure 5.3 uses two vertically stacked `cividis` heatmaps with independent country colour limits, Monday at the top, Sunday at the bottom, local hours 0-23 on x, comma-formatted colourbar values, 300 dpi PNG, vector PDF, and no main title.

### Task 1: Write failing tests

**Files:**
- Create `code/03_exploratory_analysis/test_section_5_2_daily_weekly.py`.

- [x] **Step 1: Test day-code mapping and full input validation.**

For both real inputs, capture input bytes, load only the four requested columns, assert 52,608 rows, finite actual load, exact mapping `0..6` to Monday-Sunday from `timestamp_local`, exactly 2,192 local dates, daily counts only 23/24/25, all seven days and all 24 hours. Assert input bytes are unchanged.

- [x] **Step 2: Test daily aggregation and day-of-week summary.**

Use a deterministic small fixture to verify local date means and counts, then assert real-country daily data contains 2,192 dates. Assert the summary contains exactly 14 rows, Monday-Sunday order, no missing/infinite values, and quartile ordering.

- [x] **Step 3: Test weekly hourly profile.**

Assert exactly 336 rows, all country-day-hour combinations, counts summing to 52,608 per country, no missing/infinite values, and quartile ordering. Independently compare profile means to direct groupby calculations.

- [x] **Step 4: Test weekday/weekend and extrema calculations.**

Assert weekday and weekend means use daily means, percentage reduction uses `(weekday - weekend) / weekday * 100`, and independently identify the highest/lowest mean-demand day and peak/minimum hours for weekday, Saturday, and Sunday groups.

- [x] **Step 5: Test heatmap output and protected artifacts.**

Run the new writer in a temporary directory, assert exactly the four new outputs exist, PNG resolution is at least 300 dpi, PDF is non-empty, and hashes of existing Section 5.1 and Section 5.2 outputs remain unchanged.

- [x] **Step 6: Run focused tests and confirm RED.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/03_exploratory_analysis/test_section_5_2_daily_weekly.py -q
```

Expected result: collection fails because `section_5_2_daily_weekly.py` does not yet exist.

### Task 2: Implement local daily and weekly calculations

**Files:**
- Create `code/03_exploratory_analysis/section_5_2_daily_weekly.py`.

- [x] **Step 1: Add constants and input loading.**

Define project paths, output paths, `EXPECTED_ROWS = 52_608`, `EXPECTED_DAILY_ROWS = 2_192`, `DAY_CODES = {0: "Monday", ..., 6: "Sunday"}`, `DAY_ORDER`, and the four-column input contract. Read only the requested columns.

- [x] **Step 2: Validate the local day coding.**

Parse local timestamps only to obtain `local_date` and local weekday names. Require every observed `day_of_week` code to match the corresponding local weekday name and reject missing, non-integer, or out-of-range codes.

- [x] **Step 3: Build daily observations and day-of-week summary.**

Group each country by derived local date to calculate `daily_mean_grid_load_mwh` and `number_of_hourly_observations`. Require 2,192 dates and counts in `{23,24,25}`. Group daily means by `day_of_week` and calculate day count, mean, median, sample standard deviation, first quartile, and third quartile in Monday-Sunday order.

- [x] **Step 4: Build the raw hourly weekly profile.**

Group original hourly actual load by validated `day_of_week` and `hour`, calculate count, mean, median, first quartile, and third quartile, and require every country-day-hour combination to be present with counts summing to 52,608 per country.

- [x] **Step 5: Implement descriptive comparison metrics.**

Calculate highest/lowest day from daily summary means, weekday and weekend averages from daily means, weekday-to-weekend percentage reduction, and peak/minimum hours from raw hourly groups for Monday-Friday, Saturday, and Sunday. Return these metrics from the run function for reporting and tests without writing an additional report file.

### Task 3: Implement Figure 5.3 and output writer

**Files:**
- Modify `code/03_exploratory_analysis/section_5_2_daily_weekly.py`.

- [x] **Step 1: Render the two heatmaps.**

Pivot each country’s weekly profile means to Monday-Sunday rows and hours 0-23 columns. Use `imshow` or `pcolormesh` with separate `vmin`/`vmax` per country, `cividis` colour maps, no cell annotations, readable day/hour labels, and one colorbar per panel labeled `Mean grid load [MWh]` with comma formatting.

- [x] **Step 2: Add the CLI writer.**

Implement `run_section_5_2_daily_weekly(output_directory=OUTPUT_DIRECTORY)` to load/validate both countries, compute all in-memory results, write only the two CSVs and two Figure 5.3 files, and return the summaries, profiles, daily data, and comparison metrics. Add the standard `__main__` entry point.

- [x] **Step 3: Run focused tests and generate outputs.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/03_exploratory_analysis/test_section_5_2_daily_weekly.py -q
& ".\.venv\Scripts\python.exe" code/03_exploratory_analysis/section_5_2_daily_weekly.py
```

Expected result: focused tests pass before repository outputs are generated, followed by exactly four new Section 5.2 daily/weekly files.

### Task 4: Full verification and descriptive reporting

**Files:**
- Read both inputs, new outputs, and protected Section 5.1/5.2 outputs.

- [x] **Step 1: Run focused tests, full suite, and dependency check.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/03_exploratory_analysis/test_section_5_2_daily_weekly.py -q
& ".\.venv\Scripts\python.exe" -m pytest code -q
& ".\.venv\Scripts\python.exe" -m pip check
```

- [x] **Step 2: Independently verify counts, mappings, quartiles, extrema, and averages.**

Recompute the day mapping, local date counts, 14-row summary, 336-row profile, weekday/weekend averages, percentage reductions, and highest/lowest mean days and hours directly from the inputs.

- [x] **Step 3: Verify heatmap properties and immutability.**

Check PNG resolution is at least 300 dpi, PDF is non-empty, exactly the four new outputs exist in the target output directory, and all existing Section 5.1 and Section 5.2 output hashes are unchanged.
