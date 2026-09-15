# SARIMA Screening and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify bounded daily-seasonal SARIMA screening, frozen 2024 mini-validation and runtime benchmarking without starting full 2024 validation or accessing 2025.

**Architecture:** Keep SARIMA isolated under `code/04_models/sarima/`. Reuse the shared forecasting framework for information sets, origins, forecast paths, DST handling and metrics. Separate model fit validity from training residual adequacy rejection, and make 2024 validation resumable with parent-owned checkpoints.

**Tech Stack:** Python 3.12, pandas, NumPy, statsmodels 0.14.6, SciPy, pytest, Windows `spawn`, project interpreter `.venv\Scripts\python.exe`.

---

### Task 1: Model validity and diagnostic tests

**Files:**
- Create: `code/04_models/sarima/test_sarima_models.py`
- Create: `code/04_models/sarima/sarima_models.py`

- [ ] **Step 1:** Write failing tests for SARIMA IDs, `(d,D)` deterministic terms, exact 14-candidate interaction pool, combined-root validity, reconstructed component roots, retry behavior, and residual burn-in.
- [ ] **Step 2:** Run the model tests and confirm the expected import/function failures.
- [ ] **Step 3:** Implement `SarimaOrder`, deterministic-term helpers, fitted-record metadata, `SARIMAX` fitting, one `maxiter=1000` retry, combined statsmodels root gates, reconstructed component root diagnostics, AICc, and residual diagnostics using the larger state-space/conservative burn-in.
- [ ] **Step 4:** Add tests proving residual autocorrelation is diagnostic-only for a frozen 2024 fit record while training screening can retain a hard adequacy rejection flag.
- [ ] **Step 5:** Run the model tests and require all to pass.

### Task 2: Training-only screening

**Files:**
- Create: `code/04_models/sarima/test_sarima_screening.py`
- Create: `code/04_models/sarima/sarima_screening.py`

- [ ] **Step 1:** Write failing tests for seasonal-path viability, smallest supported `d`, ACF/PACF-guided bounded candidates, same-`(d,D)` IC narrowing, duplicate prevention and country separation.
- [ ] **Step 2:** Run the screening tests and confirm failure before implementation.
- [ ] **Step 3:** Implement 2020-2023-only extraction, `D=0/1` seasonal-dependence checks, ordinary ADF/KPSS `d` selection, ACF/PACF evidence flags, fixed candidate construction capped at 14 per viable path, training residual adequacy rejection, and within-`(d,D)` AIC/AICc/BIC narrowing.
- [ ] **Step 4:** Write screening outputs for differencing diagnostics, candidate screening, residual diagnostics and frozen shortlists under `results/arima_sarima/tables/`.
- [ ] **Step 5:** Run the screening tests and require all to pass.
- [ ] **Step 6:** Run the screening CLI and verify no 2024 or 2025 rows are used, both countries are separate, and each shortlist contains only valid fixed-pool specifications.

### Task 3: Forecast path and resumable mini-validation

**Files:**
- Create: `code/04_models/sarima/test_sarima_validation_2024.py`
- Create: `code/04_models/sarima/sarima_validation_2024.py`

- [ ] **Step 1:** Write failing tests for manifest keys, Germany/Austria origins, information leakage, bridge paths, 24/23/25 target lengths, target-only metrics, residual diagnostic retention, checkpoint retries and duplicate-free upserts.
- [ ] **Step 2:** Run validation tests and confirm failure before implementation.
- [ ] **Step 3:** Implement daily origin-by-origin SARIMA refitting using only the shared information set and continuous bridge forecasts.
- [ ] **Step 4:** Implement parent-owned atomic jobs/forecasts/diagnostics/summary persistence, completed-job skipping, failed-job retries and deterministic result ordering.
- [ ] **Step 5:** Keep the CLI mini-only; expose the full 366-date API only for later approved work.
- [ ] **Step 6:** Run validation tests and require all to pass.

### Task 4: Run the approved SARIMA development workload

**Files:**
- Create outputs under `results/arima_sarima/validation_2024/mini_validation/`.

- [ ] **Step 1:** Run ordinary/DST mini-validation for all frozen SARIMA shortlist candidates on `2024-02-15`, `2024-03-31` and `2024-10-27` for both countries.
- [ ] **Step 2:** Verify target coverage, no duplicate keys, origin timestamps, no leakage, bridge rows and residual diagnostics.
- [ ] **Step 3:** Do not create final SARIMA selection output until full 2024 validation is later completed.

### Task 5: Benchmark workers and project workload

**Files:**
- Create outputs under `results/arima_sarima/validation_2024/benchmark_workers_1/` through `_4/`.
- Modify: `results/computational_runtime_log.md` only after verified benchmark results.

- [ ] **Step 1:** Inspect numerical backend threading using the project interpreter.
- [ ] **Step 2:** Benchmark the actual SARIMA mini manifest with 1, 2, 3 and 4 workers under one numerical thread per worker.
- [ ] **Step 3:** Run a four-worker default-thread comparison when useful for oversubscription diagnosis.
- [ ] **Step 4:** Calculate `366*(K_Germany+K_Austria)` full jobs and `3*(K_Germany+K_Austria)` mini jobs from actual shortlist sizes.
- [ ] **Step 5:** Project full-year runtime by worker count using the measured `366/3` multiplier and estimate theoretical country-split benefit from Germany/Austria timings.
- [ ] **Step 6:** Stop without full 2024 SARIMA validation, 2025 access or second-machine execution.
