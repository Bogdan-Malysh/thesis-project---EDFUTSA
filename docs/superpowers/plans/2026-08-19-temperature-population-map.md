# Temperature-Population Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate one reproducible 300-dpi PNG map of the historical ERA5-Land temperature field and final population weights for Germany and Austria.

**Architecture:** Keep the existing temperature pipeline unchanged and import its `build_weights()` and ERA5 helpers from a new visualization script. Store one official GISCO boundary GeoJSON with standalone provenance metadata, use Shapely only for country geometry masking, and write a single PNG under `results/figures`.

**Tech Stack:** Python, pandas, xarray, NumPy, Matplotlib, pyproj, Shapely, existing ERA5-Land and Eurostat Census Grid inputs.

---

### Task 1: Add the official GISCO boundary input

**Files:**
- Create: `data/raw/gisco_boundaries/CNTR_RG_01M_2024_4326.geojson`
- Create: `data/raw/gisco_boundaries/metadata.json`
- Modify: `requirements.txt`

- [ ] **Step 1: Download the approved official source**

Run:

```powershell
Test-Path -LiteralPath "data/raw/gisco_boundaries"
```

Create the directory only after confirming its parent exists, then download:

```powershell
New-Item -ItemType Directory -Path "data/raw/gisco_boundaries" -Force
Invoke-WebRequest -Uri "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_01M_2024_4326.geojson" -OutFile "data/raw/gisco_boundaries/CNTR_RG_01M_2024_4326.geojson"
```

If the URL fails or the response is not valid GeoJSON containing both `DE` and `AT`, stop rather than choosing another source.

- [ ] **Step 2: Record the downloaded file hash and provenance**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -c "from pathlib import Path; import hashlib; p=Path('data/raw/gisco_boundaries/CNTR_RG_01M_2024_4326.geojson'); h=hashlib.sha256(p.read_bytes()).hexdigest(); print(h)"
```

Write `metadata.json` with the actual hash, access date `2026-08-19`, source URL, GISCO Countries 2024 version, 1:1M scale, EPSG:4326, licence conditions, required attribution, and `DE`/`AT` coverage.

- [ ] **Step 3: Add the minimum dependency**

Install and inspect the package:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install shapely
& ".\.venv\Scripts\python.exe" -m pip show shapely
```

Pin the installed version in `requirements.txt` without changing unrelated entries.

- [ ] **Step 4: Verify the source before proceeding**

Run a Python check that loads the GeoJSON with the standard library, verifies `type=FeatureCollection`, finds properties with `CNTR_ID` values `DE` and `AT`, and confirms the file hash equals `metadata.json`.

### Task 2: Specify visualization behaviour with failing tests

**Files:**
- Create: `code/03_visualization/test_temperature_population_map.py`

- [ ] **Step 1: Write timestamp-selection tests**

Test that `select_cold_timestamp()` inner-joins Germany and Austria by UTC timestamp, restricts the rows inclusively to 2020-01-01 through 2025-12-31 23:00 UTC, selects the smallest joint mean, and honours a supplied UTC override.

- [ ] **Step 2: Write masking and validation tests**

Test that `mask_grid_to_geometry()` masks an outside grid cell while retaining an inside cell, and that `validate_plot_data()` rejects non-finite displayed temperatures or country weights whose sums are not approximately one.

- [ ] **Step 3: Run the new tests and confirm the expected RED state**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/03_visualization/test_temperature_population_map.py -q
```

Expected result: collection or test failures because the new module and functions do not yet exist.

### Task 3: Implement the reproducible map generator

**Files:**
- Create: `code/03_visualization/temperature_population_map.py`

- [ ] **Step 1: Add project paths and imports**

Import `json`, `argparse`, `Path`, `numpy`, `pandas`, `xarray`, `matplotlib`, `matplotlib.pyplot`, `matplotlib.colors.Normalize`, `matplotlib.lines.Line2D`, `shapely.geometry.shape`, and `shapely.ops.unary_union`. Load the existing preprocessing module by file path so the script works when run directly.

- [ ] **Step 2: Implement timestamp selection and override parsing**

Implement `select_cold_timestamp(germany, austria, override=None)` using the exact UTC key and date range described in Task 2. Parse overrides with `pd.Timestamp`, require timezone-aware UTC values, and return the selected timestamp plus Germany/Austria temperatures.

- [ ] **Step 3: Rebuild exact final weights**

Open `era5_land_2019_boundary.nc` through the existing `_open_era5_temperature()` helper, use its first time slice finite mask, and call `build_weights()` independently for Germany and Austria. Do not save or alter a mapping file. Use the returned latitude/longitude indices and normalized weights for bubble positions and sizes.

- [ ] **Step 4: Read and mask official boundaries**

Load the GISCO GeoJSON, filter `CNTR_ID` `DE` and `AT`, convert each feature to Shapely geometry, union multipart features where needed, and create a grid-centre mask using the selected ERA5 latitude/longitude arrays. Mask values outside each geometry before plotting.

- [ ] **Step 5: Render the two-panel figure**

Use a shared `Normalize` and `RdBu_r`-style cold-to-warm colormap across both panels. Draw masked `pcolormesh` fields, boundary outlines, and weight circles with `s = bubble_scale * weight`. Add country labels, one shared colorbar, one shared population-share legend, title, UTC/local-time subtitle, and source note with GISCO attribution.

- [ ] **Step 6: Enforce the output contract**

Save only `results/figures/temperature_population_cold_hour.png` with `dpi=300`, `facecolor="white"`, and `bbox_inches="tight"`. Do not save vector files or modify existing figure outputs.

- [ ] **Step 7: Run the new tests and confirm GREEN**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/03_visualization/test_temperature_population_map.py -q
```

Expected result: all visualization tests pass.

### Task 4: Render and validate the final artifact

**Files:**
- Create: `results/figures/temperature_population_cold_hour.png`

- [ ] **Step 1: Generate the figure**

Run:

```powershell
& ".\.venv\Scripts\python.exe" code/03_visualization/temperature_population_map.py
```

Record the selected UTC timestamp, Germany/Austria local timestamps, weighted national temperatures, plotted cell count, and country weight sums from the script output.

- [ ] **Step 2: Inspect the PNG properties and pixels**

Run a Python check that the file exists, has 300 dpi metadata, is RGB/RGBA, has non-trivial dimensions, and contains non-white pixels. Open the PNG with the file reader for visual inspection.

- [ ] **Step 3: Run the complete project test suite and metadata checks**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code -q
& ".\.venv\Scripts\python.exe" -m pip check
```

Confirm that existing source hashes and processed CSV modification times/content remain unchanged, and that only the approved boundary source, metadata, dependency line, visualization code/tests, and final PNG were added or modified.
