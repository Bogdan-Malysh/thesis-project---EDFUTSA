# SMARD Data Audit Design

## Purpose

This phase establishes a local, reproducible Python project and performs a read-only integrity audit of four SMARD CSV files containing hourly actual and forecast national grid load for Germany and Austria. It does not preprocess data, create figures, or begin modelling.

The primary future forecasting target is actual `Grid load [MWh]`. The corresponding official SMARD `Grid load [MWh]` forecast is an external benchmark, not a model input. Germany and Austria remain separate throughout the research.

## Constraints

- Keep the project and all data on this laptop.
- Do not create a GitHub repository, add a Git remote, push, upload, or share files.
- Do not initialize Git without explicit approval.
- Treat every file in `data/raw/` as immutable: never edit, rename, move, overwrite, clean, or delete it.
- Use ordinary `.py` files, not notebooks.
- Use classical time-series methods only; do not use machine learning, Prophet, or `pmdarima`.
- Do not begin modelling before preprocessing has been validated.
- Use chronological validation and prevent data leakage.
- Analyse Germany and Austria separately.
- Use official forecasts only as external benchmarks.
- Do not create audit figures. Visual diagnostics belong to `03_exploratory_analysis`.

## Inspected Project State

The project currently contains `opencode.json` and these raw files:

- `data/raw/smard/smard_germany_actual.csv`
- `data/raw/smard/smard_germany_forecasted.csv`
- `data/raw/smard/smard_austria_actual.csv`
- `data/raw/smard/smard_austria_forecasted.csv`

Preliminary read-only inspection found 52,609 lines in each file: one header and 52,608 observations. The audit must calculate and report counts and exact interval boundaries from the files rather than rely on these preliminary values.

The files use semicolon delimiters. Displayed numeric values use a decimal point and comma thousands separators. The timestamps are local civil-time labels without explicit UTC offsets. Spring transition days omit the local `02:00` start, and autumn transition days contain two `02:00` starts.

## Project Structure

```text
electricity-demand-thesis/
|-- .venv/
|-- AGENTS.md
|-- README.md
|-- requirements.txt
|-- code/
|   |-- 01_data_audit/
|   |   |-- load_smard.py
|   |   |-- audit_data.py
|   |   |-- test_load_smard.py
|   |   `-- test_audit_data.py
|   |-- 02_preprocessing/
|   |-- 03_exploratory_analysis/
|   |-- 04_models/
|   |   |-- baselines/
|   |   |-- exponential_smoothing/
|   |   |-- arima/
|   |   |-- sarima/
|   |   `-- sarimax/
|   |-- 05_validation/
|   `-- 06_model_comparison/
|-- data/
|   |-- raw/
|   |   `-- README.md
|   `-- processed/
|-- docs/
|   `-- superpowers/
|       `-- specs/
`-- results/
    |-- figures/
    |-- tables/
    `-- forecasts/
```

There is no root `tests/` directory. Tests for each research stage may live beside that stage's scripts. Scripts resolve project paths from their own locations and require no `src/` package, package installation, custom classes, or `sys.path` manipulation. Shared utility modules may be added later only when genuinely needed.

## Python Environment

The project targets 64-bit CPython 3.12. Python 3.14 is currently installed; it must remain installed and unchanged. After implementation-plan approval, install Python 3.12 alongside it and create the environment with:

```powershell
py -3.12 -m venv .venv
```

Every project command must use `.venv\Scripts\python.exe`. Verify the interpreter version and architecture before installing dependencies. Do not silently fall back to Python 3.14; stop and report any Python 3.12 installation or detection failure.

Initial libraries are:

- pandas
- numpy
- scipy
- statsmodels
- matplotlib
- seaborn
- pytest
- tzdata

Do not install Jupyter, Prophet, `pmdarima`, or machine-learning libraries. Pin exact tested package versions, including required transitive dependencies, in `requirements.txt`. Record the exact Python and installed package versions in audit provenance. Add a public-holiday library later only if it becomes necessary.

## Documentation

The root `README.md` documents the project purpose, CPython 3.12 environment setup, dependency installation, test command, audit command, exit codes, and generated outputs.

`AGENTS.md` records all project constraints listed above.

`data/raw/README.md` documents:

- SMARD as the source;
- hourly resolution;
- the country and actual/forecast role of each file;
- the available series in each file;
- the original download date if authoritative local information exists, otherwise that it was not recorded;
- exact SMARD column definitions and the target/benchmark interpretation; and
- the rule that raw files must never be modified.

Coverage documentation is populated from validated audit results, not hard-coded assumptions. It states the first interval start, final interval end, exact observation count, and the covered calendar years. Exact definitions are checked against authoritative SMARD metadata without uploading project content. No source fact is invented.

## Components

### `load_smard.py`

The loader only reads and parses one specified raw file. Inputs identify the path, country, actual/forecast type, and dataset timezone. It returns source fields, parsed fields, dataset metadata, the original-to-standardized column mapping, and parsing diagnostics needed by the audit.

It does not classify audit results, clean data, alter values, impute, remove rows, deduplicate, write files, or modify raw data. Do not add custom timestamp classes or other abstractions unless implementation demonstrates a concrete need.

Dataset timezones are stored once in metadata:

- Germany: `Europe/Berlin`
- Austria: `Europe/Vienna`

The standardized numeric mapping is:

| Exact original column | Standardized name |
|---|---|
| `grid load [MWh] Calculated resolutions` | `grid_load_mwh` |
| `Grid load incl. hydro pumped storage [MWh] Calculated resolutions` | `grid_load_including_pumped_storage_mwh` |
| `Hydro pumped storage [MWh] Calculated resolutions` | `pumped_storage_mwh` |
| `Residual load [MWh] Calculated resolutions` | `residual_load_mwh` |

The exact original mapping remains available in metadata and audit reports. Actual/forecast dataset metadata distinguishes columns that share a standardized name.

Each parsed observation retains only these timestamp fields:

- `interval_start_raw`
- `interval_end_raw`
- `interval_start_local`
- `interval_end_local`
- `interval_start_utc`
- `interval_end_utc`
- `is_repeated_autumn_hour`

The raw strings and naive local labels are preserved for traceability and calendar analysis. `interval_start_utc` is the unique key for integrity checks, alignment, and later modelling. Timezone-aware local timestamps can be derived later from UTC and dataset metadata; they are not stored.

### DST Disambiguation

SMARD source intervals are civil-time labels. Timestamp parsing therefore follows these rules:

1. Parse and preserve raw start and end labels as naive local timestamps.
2. Localize interval starts in chronological source-row order using the dataset timezone.
3. On an autumn transition, assign the summer-time offset to the first repeated hour and the standard-time offset to the second.
4. Derive `interval_start_utc` from that localized start.
5. Derive `interval_end_utc` as exactly one elapsed hour after the UTC start.
6. Preserve the separately parsed raw local end label even when a DST boundary prevents it from independently defining an unambiguous elapsed-time endpoint.

Do not shift nonexistent labels, remove autumn repetitions, or synthesize spring observations. Validate that UTC starts are unique, strictly increasing, and exactly one hour apart, and that every UTC interval lasts one hour. A failed or ambiguous interpretation is an audit `ERROR`, not a silent resolution.

### `audit_data.py`

The audit uses an explicit manifest for all four expected files and performs the complete integrity audit. Each check has a stable ID, scope, status, summary, and supporting evidence. Every check is `PASS`, `WARNING`, or `ERROR`.

## Audit Procedure

### 1. Provenance

For each file, report file name, byte size, SHA-256 hash, country, dataset type, timezone, and audit time. A missing or unreadable required file prevents trustworthy execution and produces exit code 2.

### 2. Physical Format

- Validate strict UTF-8 compatibility and report whether a BOM is present.
- Do not claim a uniquely detectable encoding when ASCII-only bytes are compatible with multiple encodings.
- Confirm the semicolon delimiter and consistent field counts.
- Confirm decimal-point and comma-thousands-separator handling.
- Preserve source strings so missing or failed numeric parsing remains observable.

A readable file with an unexpected delimiter, incompatible encoding, or inconsistent field counts produces an audit `ERROR` when enough evidence can be collected for trustworthy reports. A file that cannot be read sufficiently to produce trustworthy reports is an execution failure with exit code 2.

### 3. Schema and Target

- Report exact original headers, order, parsed names, and mapping.
- Verify expected actual and forecast schemas.
- Report exact row and column counts.
- Identify actual `grid_load_mwh` as the primary forecasting target and forecast `grid_load_mwh` as the external benchmark.
- Retain and document the included-pumping and residual-load series.

An unexpected or missing required source column is an audit `ERROR` when trustworthy reports can still be produced.

The target represents national electricity demand drawn from the public grid under the SMARD definition. It does not represent all behind-the-meter consumption and excludes pumping work according to SMARD's grid-load calculation.

### 4. Intervals and Coverage

- Report first and last raw, local, and UTC intervals.
- Check chronological source order, raw interval labels, UTC uniqueness, one-hour interval duration, and one-hour UTC continuity.
- Compare observed row count with the count implied by elapsed UTC coverage.
- Derive expected spring and autumn transitions from the relevant timezone rules.
- Classify correctly represented spring omissions and autumn repeated labels as `PASS`.
- Classify invalid intervals, unexpected duplicate UTC keys, unexplained local gaps, or broken actual UTC continuity as `ERROR`.

### 5. Completeness and Duplication

- Count missing, empty, non-numeric, and non-finite values by column.
- Classify such values in the primary actual target as `ERROR`.
- List invalid benchmark observations individually and classify them as `WARNING`.
- Classify missing or invalid auxiliary-column observations as `WARNING`.
- Detect exact duplicate source rows and distinguish expected repeated autumn labels from unexplained duplication.

An expected repeated autumn interval that resolves to two unique UTC keys is not a duplicate observation error, even if its numeric values happen to match. An unexplained exact duplicate outside that convention is an `ERROR`.

### 6. Numerical Integrity

For each numeric column, report valid count, minimum, maximum, mean, and percentiles 1, 5, 25, 50, 75, 95, and 99.

- Actual grid load less than or equal to zero: `ERROR`.
- Forecast grid load that is missing, non-numeric, non-finite, zero, or negative: invalid benchmark observation and `WARNING`.
- Zero and negative residual load are allowed.
- Zero pumped-storage consumption is allowed.
- Valid minima, maxima, percentiles, and largest one-hour changes are descriptive evidence under `PASS`, not warnings.
- Do not use IQR rules, z-scores, fixed country bounds, automated outlier classification, capping, replacement, or imputation.

### 7. Pumped-Storage Relationship

Test the numerical relationship between grid load, pumping consumption, and grid load including pumping over every complete actual row. Test candidate sign conventions rather than assuming one.

Use a 0.02 MWh arithmetic tolerance only after confirming that it is consistent with the source columns' displayed precision and possible rounding error, and document that justification. Do not introduce a more complex tolerance model. If the displayed precision does not justify this tolerance, do not invent an alternative: report the convention check as `WARNING` with descriptive discrepancies.

Report the fitted convention, mismatch count, affected timestamps, maximum absolute discrepancy, and discrepancy percentiles. If a positive-consumption convention is confirmed, negative pumping consumption is `ERROR`. If no convention can be established, report `WARNING` and do not reinterpret values.

### 8. Actual/Forecast Alignment

Within each country, compare actual and forecast intervals by `interval_start_utc`. Verify UTC starts and ends, coverage, row order, and one-to-one cardinality. Missing or invalid benchmark intervals are `WARNING`; missing actual-target intervals or broken actual continuity are `ERROR`. Never pool countries.

## Status and Exit Codes

The overall report status is:

- `ERROR` when any check is `ERROR`;
- `WARNING` when there are warnings but no errors; or
- `PASS` when every check passes.

Process exit codes are:

- `0`: successful execution with only `PASS` and `WARNING` results;
- `1`: successful execution with one or more audit `ERROR` results; or
- `2`: execution failure, such as a missing or unreadable file, invalid manifest/configuration, report-generation failure, wrong Python interpreter, or unexpected exception.

Warnings never cause a non-zero exit code. A successful program execution includes an audit whose status is `ERROR`: write valid reports, then exit 1. If trustworthy reports cannot be produced, exit 2 with a clear terminal explanation and preserve reports from the previous successful execution.

## Reports

Create only:

- `results/tables/smard_data_audit.md`
- `results/tables/smard_data_audit.json`
- `results/tables/smard_column_summary.csv`

Do not create logs, evidence files, per-check reports, figures, or other outputs.

### Markdown

The concise readable report contains dynamic overall/check counts, environment and provenance, source format, column mapping, interval coverage, DST findings, missing/invalid/duplicate findings, descriptive numerical summaries, pumped-storage findings, country-specific alignment, and target/benchmark identification. It includes important affected timestamps and concise representative examples when issue lists are long.

### JSON

The complete machine-readable report contains a report-schema version, run metadata, exact software versions, dynamic overall status, provenance, source and parsed schemas, every structured check, complete affected-observation lists, numerical summaries, arithmetic diagnostics, and country-level alignment. It does not embed complete source datasets.

### CSV

The compact summary has one row per source data column and includes file, country, dataset type, exact original and standardized names, role, unit, source and parsed dtypes, row count, missing/non-numeric/non-finite counts, valid numeric count, numerical summaries where applicable, and aggregated column status.

### Safe Replacement

Build and serialize all three reports successfully before replacing prior reports. Stage transient files, replace the report set safely, restore the prior set if replacement fails, and remove transient artifacts. An execution failure before a trustworthy new set exists must not overwrite the previous successful set. Audit status `ERROR` does not block report replacement.

### Terminal Output

Print only a short, dynamically calculated summary containing overall status, file count, `PASS`/`WARNING`/`ERROR` check counts, report paths, and a clear execution-failure explanation when applicable. Do not hard-code displayed counts or statuses.

## Execution Flow

1. Resolve project paths from `audit_data.py`, independently of the current working directory.
2. Verify 64-bit CPython 3.12; otherwise exit 2.
3. Validate the four-file manifest and output location.
4. Load and parse all files without changing them.
5. Run and aggregate every audit check.
6. Build and validate all three reports in memory.
7. Replace reports safely.
8. Print the dynamic summary and exit 0 or 1.

## Tests

Keep tests focused on approved requirements and use temporary synthetic CSV fixtures and temporary report directories. Tests must not alter the real raw files.

Test:

- exact source-string preservation and original/standardized mapping;
- semicolon, encoding, and numeric-convention parsing;
- missing, non-numeric, and non-finite values;
- spring omission and autumn repeated-hour disambiguation in both timezones;
- unique, strictly increasing hourly UTC keys;
- invalid timestamps, intervals, duplicates, and continuity failures;
- numerical rules for actual, forecast, residual, and pumping columns;
- pumped-storage convention and tolerance;
- country-specific actual/forecast alignment;
- Markdown, JSON, and CSV structures;
- safe report replacement and preservation after execution failure;
- absence of unrequested output files; and
- exit-code outcomes 0, 1, and 2.

Do not add coverage tools, mocking frameworks, property-based testing, or other test infrastructure.

Document these commands:

```powershell
.venv\Scripts\python.exe -m pytest code\01_data_audit
.venv\Scripts\python.exe code\01_data_audit\audit_data.py
```

## Out of Scope

- Data cleaning, removal, imputation, capping, aggregation, or preprocessing
- Processed datasets
- Exploratory plots and other figures
- Stationarity tests and decomposition
- Holiday and temperature data
- Feature engineering
- Model fitting, selection, validation, comparison, or forecasts
- Git initialization or any remote repository operation
