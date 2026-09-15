# Temperature Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate population-weighted historical ERA5-Land temperature series aligned exactly to the processed SMARD UTC keys.

**Architecture:** One ordinary Python pipeline script will derive required keys, download and hash immutable source files, parse the Eurostat grid and ERA5 NetCDF files, calculate fixed national weights, and write one validated CSV per country. Small unit tests will exercise alignment, Kelvin conversion, weighting and validation using synthetic data without network access.

**Tech Stack:** Python 3.12, pandas, NumPy, xarray, netCDF4, pyproj, cdsapi, pytest.

---

### Task 1: Add the required scientific I/O dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] Add exact installed versions for `cdsapi==0.7.7`, `xarray`, `netCDF4`, and `pyproj` after installation.
- [ ] Verify imports with `.venv\Scripts\python.exe`.

### Task 2: Download and inspect the official source files

**Files:**
- Create: `data/raw/era5_land/`
- Create: `data/raw/eurostat_population/`
- Create: `data/raw/era5_land/metadata.json`
- Create: `data/raw/eurostat_population/metadata.json`

- [ ] Read the exact UTC keys from both SMARD processed files and require equality.
- [ ] Derive the distinct required years, months, days and hours from those keys.
- [ ] Download one ERA5-Land NetCDF per required year with `2m_temperature` and the exact required date/time selections.
- [ ] Download `Eurostat_Census-GRID_2021_V3.zip` from the approved official URL.
- [ ] Hash every raw source file with SHA-256 and record URL, DOI, licence, version, access date, local filename and hash without storing credentials.

### Task 3: Implement weighted aggregation and alignment

**Files:**
- Create: `code/02_preprocessing/temperature_pipeline.py`

- [ ] Parse the Eurostat V3 total resident population field in EPSG:3035.
- [ ] Transform population-cell centroids to WGS84 with `pyproj`.
- [ ] Assign each positive-population cell to the ERA5 grid cell containing its centroid and normalize national weights separately.
- [ ] Read ERA5 `2m_temperature` in Kelvin, calculate weighted means and convert to Celsius with `K - 273.15`.
- [ ] Align output to exact SMARD UTC keys, preserving UTC and local timestamps.
- [ ] Write only `temperature_germany_hourly.csv` and `temperature_austria_hourly.csv` under `data/processed/` after all validation succeeds.

### Task 4: Add validation tests and documentation

**Files:**
- Create: `code/02_preprocessing/test_temperature_pipeline.py`
- Create: `docs/temperature_data.md`

- [ ] Test exact key alignment, local timestamp conversion, Kelvin conversion, weighted averaging, invalid-value handling and duplicate rejection.
- [ ] Document the sources, provenance fields, population variable, fixed-weight rule, historical-reanalysis interpretation and leakage restriction.

### Task 5: Run and validate the real pipeline

- [ ] Run the pipeline with the project interpreter.
- [ ] Validate expected row counts, missing values, duplicates, continuity, temperature ranges, weight totals and SMARD alignment.
- [ ] Run all project tests.
- [ ] Verify raw APG/SMARD hashes and confirm those files were not modified.
