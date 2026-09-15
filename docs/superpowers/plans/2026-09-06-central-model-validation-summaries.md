# Central Model Validation Summaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reusable, source-driven central CSV summaries for completed 2024 forecasting families without rerunning forecasts or modifying family-specific result files.

**Architecture:** Add one generator under `code/04_models/common/` with explicit family source configuration. It normalizes authoritative family summaries, derives baseline job counts from baseline forecast/failure outputs, applies family-specific selection semantics, validates duplicate keys, and writes the all-models table, selected overview, and 2025 registry in deterministic order.

**Tech Stack:** Python 3.12, pandas, pytest, project interpreter `.venv\Scripts\python.exe`.

---

### Task 1: Add failing normalization and selection tests

**Files:**
- Create: `code/04_models/common/test_build_model_validation_summaries.py`

- [ ] **Step 1: Test stable machine-friendly family labels and source-value preservation**

  Build temporary authoritative CSV fixtures for one baseline, Holt–Winters, and ARIMA family. Assert normalized rows use `baselines`, `holt_winters`, and `arima`, retain source metric values exactly, and contain the required columns.

- [ ] **Step 2: Test baseline job-count derivation**

  Fixture baseline forecast rows with two target dates and one failure row. Assert completed jobs are the distinct successful target dates and failed jobs are the distinct failure keys, without recalculating metrics.

- [ ] **Step 3: Test selection flags and registry statuses**

  Fixture two baseline specifications and two statistical specifications per country. Assert the lowest-MAE baseline is selected in the all-models table and selected overview, while the registry marks statistical selections `frozen` and the baseline representative `benchmark`.

- [ ] **Step 4: Test deterministic ordering and duplicate prevention**

  Provide shuffled source rows and assert output order is country order, family order, and specification order. Assert duplicate normalized keys raise `ValueError` before output writing.

- [ ] **Step 5: Run the focused tests and confirm they fail before implementation**

  Run:

  ```powershell
  & ".\.venv\Scripts\python.exe" -m pytest "code/04_models/common/test_build_model_validation_summaries.py" -q
  ```

  Expected: collection or assertion failures because the generator module does not yet exist.

### Task 2: Implement the reusable central-summary generator

**Files:**
- Create: `code/04_models/common/build_model_validation_summaries.py`

- [ ] **Step 1: Define source configuration and output schemas**

  Define machine-friendly family labels and explicit source paths for `baselines`, `holt_winters`, and `arima`. Keep future family labels (`sarima`, `regression_arima_errors`, `sarimax`) available for later configuration. Define the required all-models, selected-overview, and registry column orders.

- [ ] **Step 2: Normalize authoritative family summaries**

  Implement readers that copy metrics and validation counts from existing summary CSVs. Map source columns without recomputing forecasts. Set `validation_year=2024`, `source_result_file`, and concise family-specific notes.

- [ ] **Step 3: Handle baseline-specific evidence and semantics**

  Read `baseline_forecasts_2024.csv` and `validation_failures.csv` only to derive completed and failed job counts. Preserve all four baseline specifications in the all-models table. Select the minimum-MAE baseline per country with deterministic RMSE, MAPE, and source-order tie-breaks.

- [ ] **Step 4: Apply statistical-family selections and registry semantics**

  Read the authoritative Holt–Winters and ARIMA selected-specification files. Set `selected_within_family` for their selected rows. Build the selected overview from one baseline representative plus one selected row per completed statistical family. Build the registry with `status=frozen` only for Holt–Winters and ARIMA, and `status=benchmark` for the representative baseline.

- [ ] **Step 5: Validate keys, sort deterministically, and write outputs**

  Reject duplicate `(country, model_family, specification_id)` keys. Sort using the project country order, configured family order, and specification order/ID. Write:

  - `results/model_validation_2024_all_models.csv`
  - `results/model_validation_2024_selected_overview.csv`
  - `results/selected_models_registry.csv`

  Expose a callable update function and a CLI entry point so future source configurations can regenerate the same master files.

### Task 3: Verify generated artifacts against authoritative files

**Files:**
- Modify: none

- [ ] **Step 1: Run the focused tests**

  Run the focused test command and require all tests to pass.

- [ ] **Step 2: Run the generator using the project interpreter**

  Run:

  ```powershell
  & ".\.venv\Scripts\python.exe" "code/04_models/common/build_model_validation_summaries.py"
  ```

  This reads existing result CSVs only and writes the three central artifacts.

- [ ] **Step 3: Run targeted artifact verification**

  Verify required columns, expected row counts, no duplicate keys, family/country ordering, selected flags, registry statuses, and equality of every metric/count value against the authoritative baseline, Holt–Winters, and ARIMA source tables.

- [ ] **Step 4: Confirm scope boundaries**

  Confirm no family-specific result CSV changed, no 2025 data was read, and no SARIMA process or output was created.
