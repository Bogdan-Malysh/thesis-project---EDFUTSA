# Section 5.3 Monthly and Annual Demand Patterns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Section 5.3 monthly and year-month demand summaries plus Figures 5.5 and 5.6 from local-date daily means, without modifying any existing inputs or Section 5.1/5.2 outputs.

**Architecture:** Add a standalone Section 5.3 module rather than importing or changing Section 5.1/5.2 implementation details. The module will validate the existing modelling datasets, derive one daily mean per local calendar date using all available hourly observations, aggregate those daily means by month and year-month, write two formatted CSVs, and render the two requested two-panel figures.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib Agg backend, pytest, Pillow.

---

### Task 1: Define the Section 5.3 test contract

**Files:**
- Create: `code/03_exploratory_analysis/test_section_5_3_monthly_annual.py`

- [ ] **Step 1: Add imports, constants, and protected-file paths**

Import the Section 5.3 public builders, loader, validator, renderers, and orchestration function. Define `COUNTRIES = ["Germany", "Austria"]`, `EXPECTED_ROWS = 52_608`, `EXPECTED_DAILY_ROWS = 2_192`, and `PROJECT_ROOT`/`PROCESSED` paths. Add every existing Section 5.1/5.2 CSV, PNG, and PDF to a `PROTECTED_OUTPUTS` list, including Figures 5.3 and 5.4, so the isolated output test can compare their bytes before and after execution.

- [ ] **Step 2: Write real-input validation tests**

For each country, load the modelling data and assert:

```python
assert validation["row_count"] == 52_608
assert validation["local_date_count"] == 2_192
assert set(validation["daily_observation_counts"]) <= {23, 24, 25}
assert validation["local_years"] == [2020, 2021, 2022, 2023, 2024, 2025]
```

Also assert the daily frame has 2,192 rows, spans `2020-01-01` through `2025-12-31`, and contains the known DST counts of 23 and 25 observations on 2020-03-29 and 2020-10-25.

- [ ] **Step 3: Write aggregation and reconciliation tests**

Build daily observations for both countries, then build both summaries. Assert:

```python
assert len(monthly) == 24
assert len(year_month) == 144
assert monthly.groupby("country").size().to_dict() == {"Germany": 12, "Austria": 12}
assert year_month.groupby("country").size().to_dict() == {"Germany": 72, "Austria": 72}
assert monthly["month"].tolist() == list(range(1, 13)) * 2
assert sorted(year_month["year"].unique()) == [2020, 2021, 2022, 2023, 2024, 2025]
assert sorted(year_month["month"].unique()) == list(range(1, 13))
```

For every country/month and country/year/month, compare `number_of_days`, mean, median, sample standard deviation, and quartiles with direct pandas aggregations over that country’s daily frame. Assert all numeric values are finite and:

```python
(summary["first_quartile_mwh"] <= summary["median_daily_mean_grid_load_mwh"]).all()
assert (summary["median_daily_mean_grid_load_mwh"] <= summary["third_quartile_mwh"]).all()
```

For reconciliation, verify each monthly mean equals the weighted mean of the corresponding year-month means using `number_of_days` as weights.

- [ ] **Step 4: Write figure and isolated-output tests**

Capture the monthly renderer’s figure before closure and assert exactly one figure-level legend, legend labels in the requested order, upper-center location, and both y-axis labels equal `Daily mean grid load [MWh]`. Run the orchestration into `tmp_path` and assert exactly:

```python
{
    "monthly_demand_profile.csv",
    "year_month_summary.csv",
    "figure_5_5_monthly_demand_patterns.png",
    "figure_5_5_monthly_demand_patterns.pdf",
    "figure_5_6_year_month_heatmap.png",
    "figure_5_6_year_month_heatmap.pdf",
}
```

Check both PNGs have at least 299 dpi and non-trivial dimensions, both PDFs are non-empty, and all protected input/output bytes remain identical.

- [ ] **Step 5: Run the new test file to establish RED**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\03_exploratory_analysis\test_section_5_3_monthly_annual.py" -q
```

Expected result: collection fails because `section_5_3_monthly_annual.py` does not yet exist. Do not modify existing inputs or outputs to satisfy the RED state.

### Task 2: Implement validated local-date daily aggregation

**Files:**
- Create: `code/03_exploratory_analysis/section_5_3_monthly_annual.py`

- [ ] **Step 1: Add module constants and input loading**

Define `PROJECT_ROOT`, `PROCESSED`, `OUTPUT_DIRECTORY`, `EXPECTED_ROWS = 52_608`, `EXPECTED_DAILY_ROWS = 2_192`, `START_DATE`, `END_DATE`, `FIGURE_DPI = 300`, `COUNTRIES`, `MONTHS = tuple(range(1, 13))`, `MONTH_NAMES` from `calendar.month_name`, `YEARS = tuple(range(2020, 2026))`, `INPUT_COLUMNS = ["timestamp_local", "actual_grid_load_mwh"]`, and the existing country input mapping.

Implement:

```python
def load_country_data(country: str, processed_directory: str | Path = PROCESSED) -> pd.DataFrame:
    if country not in COUNTRY_INPUTS:
        raise ValueError(f"unsupported country: {country}")
    path = Path(processed_directory) / COUNTRY_INPUTS[country].name
    data = pd.read_csv(
        path,
        usecols=INPUT_COLUMNS,
        dtype={"timestamp_local": "string"},
    )
    return data[INPUT_COLUMNS]
```

- [ ] **Step 2: Validate timestamps, finite loads, period, and DST-aware daily counts**

Parse every `timestamp_local` value as an offset-aware `pd.Timestamp`, retain its local `date()`, and reject missing, naive, or invalid timestamps. Convert `actual_grid_load_mwh` with `pd.to_numeric(errors="coerce")` and reject missing/non-finite values. In `validate_country_data`, require 52,608 rows, the complete 2020-01-01 through 2025-12-31 local-date range, exactly 2,192 dates, and daily counts contained in `{23, 24, 25}`.

- [ ] **Step 3: Build one daily mean for every local calendar date**

Implement `build_daily_observations(data, country)` returning columns:

```python
[
    "local_date",
    "year",
    "month",
    "number_of_hourly_observations",
    "daily_mean_grid_load_mwh",
]
```

Group by the extracted local date and calculate the mean from all hourly observations in that date. Add integer year and month columns from the local date, then validate the 2,192-row range and 23/24/25 counts.

### Task 3: Implement monthly/year-month summaries and figures

**Files:**
- Modify: `code/03_exploratory_analysis/section_5_3_monthly_annual.py`

- [ ] **Step 1: Add shared summary-statistic validation and monthly aggregation**

Define the exact output column lists. Implement `build_monthly_demand_profile(daily_by_country)` by grouping each country’s daily frame by `month` and calculating `number_of_days`, mean, median, `std(ddof=1)`, 0.25 quantile, and 0.75 quantile. Emit Germany’s months 1-12 followed by Austria’s months 1-12, with `month_name` from `MONTH_NAMES`.

- [ ] **Step 2: Add year-month aggregation**

Implement `build_year_month_summary(daily_by_country)` with the same statistics grouped by `year` and `month`. Emit each country in `YEARS` order and month 1-12 order. Validate exactly 72 rows per country, all six years and twelve months, finite values, and ordered quartiles.

- [ ] **Step 3: Add formatted CSV writing**

Implement `_write_formatted(frame, path)` following the Section 5.2 convention: preserve text columns, write integer month/year/count columns as integers, format all other numeric values to two decimal places, create only the Section 5.3 output directory, and write `monthly_demand_profile.csv` and `year_month_summary.csv`.

- [ ] **Step 4: Render Figure 5.5 with the requested shared legend**

Implement `_render_monthly_figure(summary, png_path, pdf_path)` using two vertically stacked axes in Germany/Austria order, no shared y-axis, month 1-12 ordered January-December, dark blue line with markers, light blue IQR shading, and y-axis label `Daily mean grid load [MWh]`. Label the artists `Daily mean grid load` and `Interquartile range`, then use one `figure.legend(..., loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2)` and reserve top margin. Save 300 dpi PNG and PDF.

- [ ] **Step 5: Render Figure 5.6 as separate-scale cividis heatmaps**

Implement `_render_year_month_heatmap(summary, png_path, pdf_path)` with Germany above Austria. Pivot `mean_daily_mean_grid_load_mwh` to rows `YEARS` and columns `MONTHS`, use `imshow(..., cmap="cividis", interpolation="nearest")`, set separate `vmin`/`vmax` per country, label x-axis January-December and y-axis 2020-2025, and label each colorbar `Mean daily grid load [MWh]`. Save PNG/PDF at the same project figure conventions.

- [ ] **Step 6: Add orchestration**

Implement `run_section_5_3_monthly_annual(output_directory=OUTPUT_DIRECTORY)` to load and validate both countries, build daily observations, build both summaries, write the two CSVs, render Figures 5.5 and 5.6, and return validation, daily frames, and summaries. Do not call or modify Section 5.1/5.2 orchestration.

### Task 4: Execute outputs and complete verification

**Files:**
- Create: `outputs/chapter5/section_5_3/monthly_demand_profile.csv`
- Create: `outputs/chapter5/section_5_3/year_month_summary.csv`
- Create: `outputs/chapter5/section_5_3/figure_5_5_monthly_demand_patterns.png`
- Create: `outputs/chapter5/section_5_3/figure_5_5_monthly_demand_patterns.pdf`
- Create: `outputs/chapter5/section_5_3/figure_5_6_year_month_heatmap.png`
- Create: `outputs/chapter5/section_5_3/figure_5_6_year_month_heatmap.pdf`

- [ ] **Step 1: Run focused tests until green**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest "code\03_exploratory_analysis\test_section_5_3_monthly_annual.py" -q
```

Expected result: all Section 5.3 tests pass, including exact row contracts, reconciliation, figure artifacts, and protected-byte checks.

- [ ] **Step 2: Generate only Section 5.3 outputs**

Run the module with the project interpreter:

```powershell
& ".venv\Scripts\python.exe" "code\03_exploratory_analysis\section_5_3_monthly_annual.py"
```

The module must write only the six files under `outputs/chapter5/section_5_3/`.

- [ ] **Step 3: Re-run focused tests and inspect output contracts**

Verify the generated CSVs have 24 and 144 rows, six output files exist, figures are non-empty and 300 dpi, and no missing/infinite calculated values occur.

- [ ] **Step 4: Run the complete regression suite and dependency check**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest -q
& ".venv\Scripts\python.exe" -m pip check
```

Expected result: all tests pass and no broken requirements are reported. Recompute SHA-256 hashes for every Section 5.1/5.2 output and both modelling input files and confirm they match their pre-implementation hashes.

## Self-Review

- The design uses local dates and all available hourly observations, so DST dates are not normalized to 24 hours.
- Monthly and year-month statistics are calculated from daily means, not raw hourly values.
- Both output row contracts, month/year coverage, finite-value checks, quartile ordering, and weighted reconciliation are tested.
- Figure 5.5 uses the requested shared legend and independent y-axes; Figure 5.6 uses separate cividis scales and labelled colorbars.
- The new module and test are the only code files added; existing inputs and outputs are read-only during output generation and protected by byte comparisons.
- No Git commit is included because the project is not a Git repository and `AGENTS.md` prohibits initializing one.
