# Section 5.4 Temperature and Public Holidays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (inline execution is selected). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Section 5.4 daily temperature/load and nationwide public-holiday analyses with Figures 5.7 and 5.8, without changing any raw, processed, or Section 5.1-5.3 files.

**Architecture:** Add a standalone module that reads the processed modelling load and Chapter 4 population-weighted temperature CSVs, validates and joins them on the UTC hourly key, converts matched rows to the country timezone, and derives daily summaries. A fixed Easter-based calendar reproduces the repository’s approved nationwide holiday rules without web requests. All outputs are written only to `outputs/chapter5/section_5_4/`.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib Agg backend, `dateutil.easter`, `zoneinfo`, pytest, Pillow.

---

### Task 1: Specify Section 5.4 behavior with failing tests

**Files:**
- Create: `code/03_exploratory_analysis/test_section_5_4_temperature_public_holidays.py`

- [ ] **Step 1: Define imports, schemas, and protected files**

Import the future Section 5.4 public functions and constants. Define the exact expected columns:

```python
DAILY_COLUMNS = [
    "country", "local_date", "daily_mean_load_mwh",
    "daily_mean_temperature_c", "hourly_count", "weekday",
    "public_holiday_flag", "holiday_name", "day_type",
    "matching_baseline_mwh", "relative_load_deviation_pct",
]
TEMPERATURE_BIN_COLUMNS = [
    "country", "temperature_bin_lower_c", "temperature_bin_upper_c",
    "temperature_bin_midpoint_c", "number_of_days",
    "mean_daily_mean_load_mwh", "median_daily_mean_load_mwh",
    "standard_deviation_mwh", "first_quartile_mwh", "third_quartile_mwh",
]
HOLIDAY_SUMMARY_COLUMNS = [
    "country", "year", "month", "month_name", "weekday", "weekday_name",
    "ordinary_weekday_count", "weekday_public_holiday_count",
    "matching_baseline_mwh", "ordinary_weekday_mean_deviation_pct",
    "weekday_public_holiday_mean_deviation_pct",
    "weekday_public_holiday_median_deviation_pct",
    "weekday_public_holiday_first_quartile_pct",
    "weekday_public_holiday_third_quartile_pct",
]
```

Protect both modelling input files, both temperature input files, every Section 5.1-5.3 CSV/PNG/PDF, and record their SHA-256 hashes before/after isolated execution. The implementation must not read `data/raw/`.

- [ ] **Step 2: Test UTC alignment, local dates, coverage, and counts**

For Germany and Austria, assert the modelling and temperature inputs each have 52,608 unique continuous UTC keys, the one-to-one join returns 52,608 rows, and local conversion yields exactly 2,192 dates per country. Assert daily hourly counts are only 23, 24, or 25, finite load/temperature values exist, and the output has 4,384 rows.

- [ ] **Step 3: Test the fixed nationwide holiday calendar**

Assert the calendar returns the approved fixed/Easter-based nationwide dates:

```python
assert date(2023, 10, 3) in holiday_calendar("Germany", [2023])
assert date(2023, 11, 1) not in holiday_calendar("Germany", [2023])
assert date(2023, 10, 26) in holiday_calendar("Austria", [2023])
assert date(2023, 12, 8) in holiday_calendar("Austria", [2023])
```

Assert Germany has 9 dates/year and Austria 13 dates/year, regional holidays are absent, day types are mutually exclusive, and weekend public holidays are not treated as weekday holidays.

- [ ] **Step 4: Test daily baselines and deviations**

Assert ordinary weekday baselines group by country/year/month/weekday and exclude all public holidays. Assert weekday public-holiday rows receive the matching ordinary-weekday baseline, weekend rows have no holiday-analysis baseline, and ordinary weekday relative deviations average to zero within every non-empty baseline group.

- [ ] **Step 5: Test temperature bins and holiday summary statistics**

Build both summaries and assert one common 2°C bin range, all retained bins have `number_of_days >= 10`, finite values, and ordered quartiles. Assert the holiday summary contains only groups with at least one weekday public holiday, has finite baseline/deviation values, and its holiday deviations match direct daily calculations.

- [ ] **Step 6: Test figures and isolated output files**

Capture Figure 5.7 before closure and assert two vertical panels, separate y-axis scales, common x-limits, one shared legend, no suptitle, and the requested axis labels. Capture Figure 5.8 and assert two panels with identical y-limits, a zero reference line, and no suptitle. Run the orchestration into `tmp_path` and require exactly:

```python
{
    "daily_temperature_public_holidays.csv",
    "temperature_bin_summary.csv",
    "public_holiday_effect_summary.csv",
    "figure_5_7_temperature_load_relationship.png",
    "figure_5_7_temperature_load_relationship.pdf",
    "figure_5_8_public_holiday_effect.png",
    "figure_5_8_public_holiday_effect.pdf",
}
```

Check both PNGs have at least 299 dpi, both PDFs are one-page and non-empty, and protected bytes/hashes remain unchanged.

- [ ] **Step 7: Run the new test file to establish RED**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\03_exploratory_analysis\test_section_5_4_temperature_public_holidays.py" -q
```

Expected result: collection fails because the Section 5.4 module does not yet exist.

### Task 2: Implement validated UTC alignment and daily public-holiday data

**Files:**
- Create: `code/03_exploratory_analysis/section_5_4_temperature_public_holidays.py`

- [ ] **Step 1: Add constants, schemas, and loaders**

Define `PROJECT_ROOT`, `PROCESSED`, `OUTPUT_DIRECTORY`, `EXPECTED_ROWS = 52_608`, `EXPECTED_DAILY_ROWS = 2_192`, `YEARS = range(2020, 2026)`, `COUNTRIES = ["Germany", "Austria"]`, `COUNTRY_TIMEZONES`, UTC input mappings, fixed weekday/month names, dark/light Chapter 5 colours, and the three exact output schemas from Task 1.

Implement `load_load_data(country)` using `timestamp_utc`, `timestamp_local`, and `actual_grid_load_mwh`, and `load_temperature_data(country)` using `interval_start_utc`, `temperature_c`, and `temperature_valid`. Parse UTC timestamps with `pd.to_datetime(..., utc=True)` and reject duplicates, non-hourly continuity, invalid temperatures, and non-finite loads.

- [ ] **Step 2: Join matched hourly observations on the verified UTC key**

Implement `align_hourly_data(country)` that checks the load UTC index and temperature UTC index are exactly equal, performs a one-to-one merge on the UTC key, confirms 52,608 matched rows, and converts each matched UTC timestamp to `ZoneInfo(COUNTRY_TIMEZONES[country])`. Derive `local_date`, integer year/month, integer weekday number/name, load, and temperature from this matched frame. Do not use local timestamps as the join key.

- [ ] **Step 3: Implement the reproducible nationwide holiday calendar**

Use `dateutil.easter.easter(year)` and these rules, matching the existing approved calendar:

```python
GERMANY = {
    Jan 1: "New Year's Day", May 1: "Labour Day", Oct 3: "German Unity Day",
    Dec 25: "Christmas Day", Dec 26: "Boxing Day",
    Easter - 2: "Good Friday", Easter + 1: "Easter Monday",
    Easter + 39: "Ascension Day", Easter + 50: "Whit Monday",
}
AUSTRIA = {
    Jan 1: "New Year's Day", Jan 6: "Epiphany", May 1: "Labour Day",
    Aug 15: "Assumption Day", Oct 26: "Austrian National Day",
    Nov 1: "All Saints' Day", Dec 8: "Immaculate Conception",
    Dec 25: "Christmas Day", Dec 26: "St. Stephen's Day",
    Easter + 1: "Easter Monday", Easter + 39: "Ascension Day",
    Easter + 50: "Whit Monday", Easter + 60: "Corpus Christi",
}
```

Return `dict[date, str]`, with no observed/substitute dates and no regional holidays.

- [ ] **Step 4: Build daily temperature/load/holiday observations**

Aggregate each aligned country frame by `local_date`, calculating means over all matched hourly observations and retaining the hourly count. Add holiday flag/name and mutually exclusive day types in this order: weekday public holiday, weekend public holiday, weekend, ordinary weekday. Build ordinary-weekday baselines by `[country, year, month, weekday_number]`; merge them onto ordinary weekdays and weekday public holidays only; calculate `100 * (load / baseline - 1)`. Keep weekend rows in the daily table with missing baseline/deviation and exclude them from later analyses.

### Task 3: Implement temperature bins, holiday summary, and figures

**Files:**
- Modify: `code/03_exploratory_analysis/section_5_4_temperature_public_holidays.py`

- [ ] **Step 1: Build the common 2°C temperature-bin summary**

Derive `bin_lower = floor(joint_min / 2) * 2` and `bin_upper = ceil(joint_max / 2) * 2`, create common half-open 2°C intervals with the final interval right-closed, assign daily temperatures, and calculate per-country bin counts, mean, median, sample standard deviation, and quartiles. Retain only rows with `number_of_days >= 10`; validate finite values and quartile ordering.

- [ ] **Step 2: Render Figure 5.7**

Use two vertical panels in Germany/Austria order, no main title, common x-limits, separate y-limits, and 300 dpi/vector output. Plot daily observations in light blue, the binned mean line in dark blue with markers, and a light-blue IQR band. Add one shared figure-level legend with labels `Daily observations`, `Mean load by temperature bin`, and `Interquartile range`. Use exact axis labels:

```text
Daily mean population-weighted temperature [°C]
Daily mean grid load [MWh]
```

- [ ] **Step 3: Build the public-holiday effect summary**

For each country/year/month/weekday group that contains at least one weekday public holiday, calculate ordinary-weekday count, holiday count, ordinary baseline, ordinary deviation mean, holiday deviation mean/median/IQR, and emit `public_holiday_effect_summary.csv` with the defined schema. Never use weekend rows in the baseline or summary.

- [ ] **Step 4: Render Figure 5.8**

For each country, boxplot ordinary-weekday deviations and weekday-public-holiday deviations, use Chapter 5 light/dark blue styling, a horizontal zero reference line, identical y-limits across panels, Germany above Austria, no main title, and y-axis label `Relative load deviation [%]`. Save a one-page 300 dpi PNG and vector PDF.

- [ ] **Step 5: Add orchestration and short console summary**

Implement `run_section_5_4_temperature_public_holidays(output_directory=OUTPUT_DIRECTORY)` to generate only the seven Section 5.4 outputs and return validation, aligned hourly, daily, bin, and holiday summary frames. Print concise coverage, temperature range, retained-bin, weekday-holiday count, deviation, and protected-file status summaries from the script entry point; do not write thesis prose or causal interpretations.

### Task 4: Generate outputs and complete verification

**Files:**
- Create: `outputs/chapter5/section_5_4/daily_temperature_public_holidays.csv`
- Create: `outputs/chapter5/section_5_4/temperature_bin_summary.csv`
- Create: `outputs/chapter5/section_5_4/public_holiday_effect_summary.csv`
- Create: `outputs/chapter5/section_5_4/figure_5_7_temperature_load_relationship.png`
- Create: `outputs/chapter5/section_5_4/figure_5_7_temperature_load_relationship.pdf`
- Create: `outputs/chapter5/section_5_4/figure_5_8_public_holiday_effect.png`
- Create: `outputs/chapter5/section_5_4/figure_5_8_public_holiday_effect.pdf`

- [ ] **Step 1: Run focused tests until green**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\03_exploratory_analysis\test_section_5_4_temperature_public_holidays.py" -q
```

- [ ] **Step 2: Generate only Section 5.4 outputs**

Verify `outputs/chapter5` exists, then run:

```powershell
& ".venv\Scripts\python.exe" "code\03_exploratory_analysis\section_5_4_temperature_public_holidays.py"
```

The module must write only under `outputs/chapter5/section_5_4/`.

- [ ] **Step 3: Run focused tests, full suite, and dependency check**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\03_exploratory_analysis\test_section_5_4_temperature_public_holidays.py" -q
& ".venv\Scripts\python.exe" -m pytest -q
& ".venv\Scripts\python.exe" -m pip check
```

- [ ] **Step 4: Verify protected files and figures**

Compare SHA-256 hashes for all modelling/temperature inputs and every Section 5.1-5.3 output against pre-implementation baselines. Create a streaming hash manifest for all 89 raw files before and after execution outside the project, and require exact path/hash equality. Inspect both PNGs and PDFs, confirm one page per PDF, 300 dpi PNGs, expected dimensions, common temperature range, and identical Figure 5.8 y-limits.

## Self-Review

- UTC alignment is explicit and local dates are derived only after matching on UTC.
- DST day lengths are retained rather than normalized to 24 hours.
- Holiday dates are nationwide-only and reproduce the existing approved Germany/Austria rules without a live request.
- Weekend rows are retained in the daily CSV but excluded from baselines, holiday summaries, and Figure 5.8 analysis.
- Temperature bins share a common range and 2°C width, while load y-axes and Figure 5.6-style colour scales remain country-specific where requested.
- Existing Section 5.1-5.3 files and all raw files are read-only; no thesis prose or causal claims are generated.
- No Git commit is created because the project is not a Git repository and `AGENTS.md` prohibits initializing one.
