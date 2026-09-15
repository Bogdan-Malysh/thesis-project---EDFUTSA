# Forecasting Family Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the existing forecasting implementation and result files by model family while preserving behavior and completed baseline CSV bytes.

**Architecture:** Keep shared forecasting logic in `code/04_models/common`, move the deterministic baseline implementation into `code/04_models/baselines`, and move Holt-Winters implementation/tests into the existing `code/04_models/exponential_smoothing` folder. Move only the corresponding result files into family-specific result directories; update imports and path constants without changing model logic.

**Tech Stack:** Python 3.12, pandas, NumPy, statsmodels, pytest, project `.venv`.

---

### Task 1: Move shared and family-specific source files

**Files:**
- Move: `code/07_forecasting_results/forecasting_framework.py` -> `code/04_models/common/forecasting_framework.py`
- Move: `code/07_forecasting_results/test_forecasting_framework.py` -> `code/04_models/common/test_forecasting_framework.py`
- Move: `code/07_forecasting_results/baseline_validation_2024.py` -> `code/04_models/baselines/baseline_validation_2024.py`
- Move: `code/07_forecasting_results/test_baseline_validation.py` -> `code/04_models/baselines/test_baseline_validation.py`
- Move: `code/07_forecasting_results/holt_winters_models.py` -> `code/04_models/exponential_smoothing/holt_winters_models.py`
- Move: `code/07_forecasting_results/holt_winters_phase2a.py` -> `code/04_models/exponential_smoothing/holt_winters_phase2a.py`
- Move: `code/07_forecasting_results/test_holt_winters_models.py` -> `code/04_models/exponential_smoothing/test_holt_winters_models.py`
- Move: `code/07_forecasting_results/test_holt_winters_phase2a.py` -> `code/04_models/exponential_smoothing/test_holt_winters_phase2a.py`

- [ ] **Step 1: Apply only the source-file moves**

Use the approved destinations above. Do not edit source contents in the move operation.

- [ ] **Step 2: Confirm the old forecasting source paths are absent**

Check that `code/07_forecasting_results` no longer contains the moved Python files and that all eight destination files exist.

### Task 2: Update imports and output path constants

**Files:**
- Modify: `code/04_models/common/forecasting_framework.py`
- Modify: `code/04_models/baselines/baseline_validation_2024.py`
- Modify: `code/04_models/exponential_smoothing/holt_winters_models.py`
- Modify: `code/04_models/exponential_smoothing/holt_winters_phase2a.py`

- [ ] **Step 1: Update project-root resolution**

Change the shared framework root from `Path(__file__).resolve().parents[2]` to `parents[3]` because it is now under `code/04_models/common`. Change baseline and Holt-Winters runner roots to `parents[3]` because they are now under a model-family subdirectory.

- [ ] **Step 2: Update imports without changing behavior**

At the top of baseline and Holt-Winters production modules, add `code/04_models` to `sys.path` if absent, then import shared functions from `common.forecasting_framework`. Keep model-family-local imports unchanged. The common test continues importing its colocated framework directly; family tests continue importing their colocated family modules.

- [ ] **Step 3: Update result directories**

Set baseline output constants to `results/baselines/tables` and `results/baselines/forecasts`. Set Holt-Winters table output to `results/exponential_smoothing/holt_winters/tables` and define its future forecast output directory as the sibling `forecasts` directory. Do not invoke either runner.

### Task 3: Move existing result files without rewriting

**Files:**
- Move: `results/tables/baseline_validation_2024.csv` -> `results/baselines/tables/baseline_validation_2024.csv`
- Move: `results/tables/validation_failures.csv` -> `results/baselines/tables/validation_failures.csv`
- Move: `results/forecasts/baseline_forecasts_2024.csv` -> `results/baselines/forecasts/baseline_forecasts_2024.csv`
- Move: `results/tables/holt_winters_screening_2024.csv` -> `results/exponential_smoothing/holt_winters/tables/holt_winters_screening_2024.csv`
- Move: `results/tables/holt_winters_shortlists_2024.csv` -> `results/exponential_smoothing/holt_winters/tables/holt_winters_shortlists_2024.csv`
- Move: `results/tables/holt_winters_optimizer_attempts_2024.csv` -> `results/exponential_smoothing/holt_winters/tables/holt_winters_optimizer_attempts_2024.csv`

- [ ] **Step 1: Create destination directories and move files**

Move bytes without loading, recomputing, or rewriting CSVs.

- [ ] **Step 2: Verify baseline hashes**

Compare destination hashes with the recorded pre-move hashes for all three baseline CSVs. Confirm old baseline paths no longer exist and new paths do.

### Task 4: Run the requested verification

**Files:**
- Verify: all files under `code/04_models/`
- Verify: `results/baselines/` and `results/exponential_smoothing/holt_winters/`

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
& ".venv\Scripts\python.exe" -m pytest
```

- [ ] **Step 2: Confirm no forecasting execution occurred**

Do not run baseline validation, Holt-Winters screening, smoke tests, full 2024 validation, or any 2025 command during verification.

- [ ] **Step 3: Report the reorganization**

Report the before/after structure, every moved and modified file, test counts, baseline hash equality, and that no methodology or production behavior changed.
