# SMARD Data Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the approved local CPython 3.12 project and produce a tested, read-only integrity audit of the four SMARD CSV files.

**Architecture:** Keep two standalone audit modules beside their tests. `load_smard.py` preserves and parses source data; `audit_data.py` classifies checks, aligns datasets by UTC interval start, and safely writes exactly three reports. No Git operation is part of this plan.

**Tech Stack:** 64-bit CPython 3.12, venv, pip, pandas, NumPy, SciPy, statsmodels, Matplotlib, seaborn, pytest, tzdata, standard-library `zoneinfo`, `hashlib`, `json`, `csv`, and `pathlib`.

---

## File Map

- Create `README.md`: setup, commands, outputs, and exit-code documentation.
- Create `AGENTS.md`: permanent local-only, data, modelling, and leakage constraints.
- Create `requirements.txt`: exact tested environment from the CPython 3.12 venv.
- Create `data/raw/README.md`: source metadata, immutable-data rule, series definitions, and audit-confirmed coverage.
- Create `code/01_data_audit/load_smard.py`: source-preserving CSV and DST-aware timestamp parser.
- Create `code/01_data_audit/audit_data.py`: manifest, audit checks, reports, safe replacement, terminal summary, and exit codes.
- Create `code/01_data_audit/test_load_smard.py`: focused loader and timestamp tests.
- Create `code/01_data_audit/test_audit_data.py`: focused audit, report, replacement, and exit-code tests.
- Create the approved empty research-stage, data, and results directories; generate only the three approved report files.

## Task 1: Install and Freeze the Approved Environment

- [ ] Confirm the current launcher state with `py -0p`; do not remove or modify Python 3.14.
- [ ] Install 64-bit CPython 3.12 for the current user with `winget install --id Python.Python.3.12 -e --scope user`. If installation fails, stop and report it instead of using 3.14.
- [ ] Verify detection with `py -3.12 -c "import platform, struct, sys; print(sys.version); print(platform.python_implementation()); print(struct.calcsize('P') * 8)"`.

Expected: CPython 3.12 and `64`.

- [ ] Create `.venv` with `py -3.12 -m venv .venv`.
- [ ] Verify `.venv\Scripts\python.exe` reports CPython 3.12, 64-bit.
- [ ] Install only the approved direct libraries:

```powershell
.venv\Scripts\python.exe -m pip install pandas numpy scipy statsmodels matplotlib seaborn pytest tzdata
```

- [ ] Record the complete exact environment with `.venv\Scripts\python.exe -m pip freeze`, then create `requirements.txt` from that exact output.
- [ ] Reinstall-check with `.venv\Scripts\python.exe -m pip check`; expected output is `No broken requirements found.`
- [ ] Run `.venv\Scripts\python.exe -m pip show jupyter prophet pmdarima scikit-learn`; expected: none of these distributions is installed.

## Task 2: Scaffold the Approved Local Structure and Policies

- [ ] Create the approved directories under `code/`, `data/`, `results/`, `docs/`, and no root `tests/` directory.
- [ ] Create `AGENTS.md` with all approved local-only, immutable-raw-data, classical-method, no-leakage, country-separation, and benchmark-only restrictions.
- [ ] Create the initial `README.md` with the exact environment, dependency, test, and audit commands plus the three output paths and exit-code meanings.
- [ ] Do not initialize Git and do not create a remote.

## Task 3: Build the Source-Preserving Loader with TDD

**Files:** `code/01_data_audit/test_load_smard.py`, `code/01_data_audit/load_smard.py`

- [ ] Write failing tests using temporary semicolon CSVs for exact raw string preservation, original headers, standardized mappings, decimal/thousands parsing, and missing/non-numeric/non-finite diagnostics.
- [ ] Run `.venv\Scripts\python.exe -m pytest code\01_data_audit\test_load_smard.py -v`; expected: failures because the loader does not exist.
- [ ] Implement constants for the exact actual/forecast schemas and a `load_smard_file(path, country, dataset_type, timezone_name)` function returning a plain dictionary with `source`, `data`, `metadata`, `column_map`, and `diagnostics`.
- [ ] Keep the source DataFrame as strings. Add only the approved standardized numeric and timestamp fields to the parsed DataFrame. Use explicit `sep=";"`, strict UTF-8-compatible decoding, comma removal for thousands, decimal-point parsing, and coercion diagnostics without cleaning source values.
- [ ] Run the loader tests; expected: all parsing and preservation tests pass.

## Task 4: Implement DST Disambiguation with TDD

**Files:** `code/01_data_audit/test_load_smard.py`, `code/01_data_audit/load_smard.py`

- [ ] Add failing tests for normal hours, spring local-label omission, two autumn `02:00` starts, first-occurrence summer offset, second-occurrence standard offset, both approved timezones, and malformed/triple-repeat cases.
- [ ] Assert retained fields are exactly the approved raw/local/UTC timestamps plus `is_repeated_autumn_hour`, with timezone stored once in metadata.
- [ ] Run the focused DST tests; expected: failures before DST parsing exists.
- [ ] Implement chronological localization of valid interval starts, derive UTC ends as start plus one elapsed hour, and retain naive source end labels independently.
- [ ] Record diagnostics instead of shifting nonexistent labels or silently resolving invalid repetition.
- [ ] Run `test_load_smard.py`; expected: UTC keys are unique, strictly increasing, and hourly for valid fixtures, while invalid fixtures retain actionable diagnostics.

## Task 5: Implement Core Audit Checks with TDD

**Files:** `code/01_data_audit/test_audit_data.py`, `code/01_data_audit/audit_data.py`

- [ ] Add failing tests for the fixed four-file manifest, file size/SHA-256, encoding/BOM, delimiter/field counts, exact schema, row/column counts, timestamp coverage, interval duration, UTC continuity, expected DST effects, missing values, exact duplicates, and unexplained gaps.
- [ ] Run `.venv\Scripts\python.exe -m pytest code\01_data_audit\test_audit_data.py -v`; expected: failures because audit functions do not exist.
- [ ] Implement project-root resolution from `__file__`, runtime verification for 64-bit CPython 3.12, and plain-dictionary check records containing `check_id`, `scope`, `status`, `summary`, and `evidence`.
- [ ] Implement provenance, physical-format, schema, interval, continuity, completeness, and duplicate checks with the approved severity rules.
- [ ] Ensure expected DST local-label effects pass, unexpected UTC duplicates and broken actual continuity error, invalid forecast observations warn, and invalid actual target observations error.
- [ ] Run the core audit tests; expected: all new tests pass.

## Task 6: Implement Numerical and Alignment Checks with TDD

**Files:** `code/01_data_audit/test_audit_data.py`, `code/01_data_audit/audit_data.py`

- [ ] Add failing tests for descriptive statistics, actual and forecast grid-load validity, allowed residual-load signs, zero pumping, candidate pumping sign conventions, conditional 0.02 MWh tolerance, negative pumping after convention confirmation, and country-specific actual/forecast alignment.
- [ ] Run the focused tests; expected: failures before these checks exist.
- [ ] Implement valid count, minimum, maximum, mean, and percentiles 1/5/25/50/75/95/99 without automated outlier classification.
- [ ] Implement the two candidate pumped-storage relationships. Use 0.02 MWh only when source display precision justifies it; otherwise return a descriptive `WARNING` without inventing a tolerance.
- [ ] Implement one-to-one actual/forecast comparison on `interval_start_utc`, separately by country, with actual continuity errors and benchmark availability warnings.
- [ ] Run the numerical and alignment tests; expected: all pass without data mutation.

## Task 7: Implement Reports, Safe Replacement, and Exit Codes with TDD

**Files:** `code/01_data_audit/test_audit_data.py`, `code/01_data_audit/audit_data.py`

- [ ] Add failing tests for concise Markdown, complete structured JSON, column-level CSV, dynamic status/check counts, and exactly the three approved report names.
- [ ] Add failing tests proving prior reports survive a pre-publication serialization failure and proving successful PASS, WARNING, and ERROR audits replace the complete report set.
- [ ] Add failing tests for exit 0 with PASS/WARNING, exit 1 with audit ERROR after report publication, and exit 2 for execution failure without untrustworthy report replacement.
- [ ] Implement report builders, dynamic overall status, environment provenance, short terminal output, and staged report-set replacement with cleanup/rollback of transient files.
- [ ] Implement `run_audit()` to return the process code and `main()` to call `sys.exit(run_audit())`.
- [ ] Run both audit test files; expected: all tests pass and temporary output directories contain no extra files.

## Task 8: Verify Locally and Complete Documentation

- [ ] Run the full focused suite:

```powershell
.venv\Scripts\python.exe -m pytest code\01_data_audit -v
```

Expected: all tests pass.

- [ ] Capture the four SMARD raw SHA-256 hashes with `Get-FileHash data\raw\smard\*.csv -Algorithm SHA256` before the real audit.
- [ ] Run `.venv\Scripts\python.exe code\01_data_audit\audit_data.py`.

Expected: exit 0 or 1 according to actual findings, never exit 2 for valid readable inputs; terminal output is short and reports exactly the three approved paths.

- [ ] Capture the raw hashes again and verify they are unchanged.
- [ ] Inspect Markdown for concise findings, JSON for complete check evidence, and CSV for one row per source column.
- [ ] Create `data/raw/README.md` using authoritative SMARD definitions and the validated report values for first interval start, final interval end, exact observations, and covered calendar years. Record the download date as not recorded unless authoritative local evidence exists.
- [ ] Re-run the full tests and audit after documentation changes; report the observed test result, audit status, exit code, and report paths without claiming PASS if evidence differs.

## Implementation Gate

Do not execute any task in this plan until the user explicitly approves it. No task authorizes Git initialization, commits, remotes, uploads, raw-data modification, preprocessing, figures, or modelling.
