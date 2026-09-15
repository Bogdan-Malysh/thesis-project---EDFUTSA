# Computational Runtime Log

Purpose: persistent record of the computational requirements of the forecasting experiments for thesis reproducibility and presentation.

Last updated: 2026-09-08

## Environment

These values were measured in the current project environment. They are retained as the available environment record; historical runs did not capture start/finish timestamps or a separate environment snapshot.

| Item | Value |
|---|---|
| Interpreter | `.venv\Scripts\python.exe` |
| Python | 3.12.10 |
| pandas | 3.0.5 |
| NumPy | 2.5.2 |
| statsmodels | 0.14.6 |
| CPU | Intel(R) Core(TM) Ultra 7 155H |
| Physical cores | 16 |
| Logical processors | 22 |
| Installed RAM | 15.5 GB |

## Interpretation

- A Holt–Winters validation job is one country + target date + specification fit.
- The complete 2024 Holt–Winters validation will contain `2 x 366 x 4 = 2,928` jobs.
- Cumulative fitting time and wall-clock time are different measures. Cumulative fitting time is useful for sequential-work estimates; wall-clock time is used for worker benchmarks.
- AIC/AICc/BIC runtime belongs to screening only. Final forecasting-performance selection uses target-day validation metrics.
- Start and finish timestamps were not captured for completed historical runs and are recorded as unavailable rather than inferred.

## Recorded Forecasting Stages

### Baseline 2024 Validation

| Field | Value |
|---|---|
| Model family | Deterministic baselines |
| Stage | Full rolling-origin validation |
| Countries | Germany, Austria |
| Target period | 2024 |
| Candidate specifications | 4 baseline model families per country |
| Forecast jobs | 2,928 |
| Workers | Sequential; no parallel worker setting |
| Start / finish | Not captured |
| Wall-clock runtime | Not captured |
| Successful / failed jobs | 2,928 / 0 |
| Retries | 0 |
| Notes | All country/model combinations had complete 8,784-observation coverage. Evidence: `results\baselines\tables\baseline_validation_2024.csv` and empty `validation_failures.csv`. |

### Holt–Winters Phase 2A Screening

| Field | Value |
|---|---|
| Model family | Holt–Winters |
| Stage | First-origin specification screening |
| Countries | Germany, Austria |
| Target period | First 2024 target date; origins on 2023-12-31 at the country-specific forecast origin |
| Candidate specifications | 8 per country; 16 total fits |
| Model-fit jobs | 16 |
| Workers | Sequential |
| Start / finish | Not captured |
| Wall-clock runtime | Not captured |
| Successful / failed jobs | 16 / 0 |
| Optimizer retries | 8 additional optimizer attempts across the screening fits |
| Notes | AICc/AIC/BIC were used only for screening and shortlist construction. Persisted evidence: `results\exponential_smoothing\holt_winters\tables\holt_winters_screening_2024.csv`. |

### Holt–Winters Phase 2A Smoke Estimate

| Field | Value |
|---|---|
| Model family | Holt–Winters |
| Stage | Smoke validation and runtime estimation |
| Countries | Germany, Austria |
| Target dates | Germany: 2024-07-15, 2024-03-31, 2024-10-27; Austria: 2024-07-15 |
| Candidate specifications | 4 shortlisted specifications per country |
| Model-fit jobs | 16 smoke fits |
| Workers | Sequential |
| Start / finish | Not captured |
| Observed wall-clock runtime | Not captured |
| Earlier projected full-year runtime | Approximately 10.66 hours sequential |
| Successful / failed jobs | 16 / 0 reported at the time |
| Retries | Not persistently recorded |
| Notes | This was an estimate from a small, uneven smoke sample, not the full 2024 validation. The smoke result was not persisted as a per-job runtime CSV. |

### Holt–Winters 2024 Mini-Validation

| Field | Value |
|---|---|
| Model family | Holt–Winters |
| Stage | Resumable rolling-validation mini-run |
| Countries | Germany, Austria |
| Target dates | 2024-01-15, 2024-03-31, 2024-10-27 |
| Candidate specifications | 4 shortlisted specifications per country |
| Validation jobs | 24 |
| Workers | 2 |
| Start / finish | Not captured |
| Wall-clock runtime | 426.24 seconds for the 22 jobs executed during the resumable continuation; 2 jobs were already checkpointed |
| Successful / failed jobs | 24 / 0 |
| Retries | 0; completed jobs were resumed from checkpoints, not recomputed |
| Cumulative fitting time | 930.05 seconds for the original 24-job sample |
| Projected full-year sequential work | 31.52 hours |
| Projected full-year work at 2 workers | 15.76 hours by simple division of cumulative fitting work |
| Notes | This mini-run was not used to freeze final specifications. Outputs: `results\exponential_smoothing\holt_winters\mini_validation_2024\`. |

## Holt–Winters Worker Benchmark

The same 24 mini-validation jobs were explicitly recomputed into separate output directories. Forecasting methodology, specifications, optimizer settings, target dates, and information sets were unchanged.

Full-year wall-clock projections scale each observed benchmark by `366 / 3 = 122` target-date sets.

| Workers | Wall-clock runtime | Jobs | Failures | Projected full-2024 wall time | Output |
|---:|---:|---:|---:|---:|---|
| 1 | 574.24 s (9.57 min) | 24 | 0 | 19.46 h | `benchmark_2024_workers_1` |
| 2 | 298.83 s (4.98 min) | 24 | 0 | 10.13 h | `benchmark_2024_workers_2` |
| 4 | 1,326.07 s (22.10 min) | 24 | 0 | 44.94 h | `benchmark_2024_workers_4` |

Consistency checks:

- All three runs produced 24 completed jobs and zero failures.
- Each produced 840 forecast rows with no duplicate job or forecast keys.
- Forecast values matched exactly across worker counts; maximum absolute difference was `0.0`.
- Aggregate MAE, RMSE, MAPE, coverage, and target-day counts matched exactly.
- The two-worker run was approximately 1.92 times faster than the one-worker run.
- Four workers were slower than one worker, indicating CPU/numerical-library contention for this workload.

Decision: use **2 workers** for the full 2024 Holt–Winters validation.

### Full Holt–Winters 2024 Validation

| Field | Value |
|---|---|
| Model family | Holt–Winters |
| Stage | Complete full rolling-origin validation |
| Countries | Germany, Austria |
| Target period | Every valid local target day in 2024 |
| Candidate specifications | 4 frozen shortlisted specifications per country |
| Total validation jobs | 2,928 |
| Worker count | 2 |
| Initial checkpoint before final resumed segment | 2,390 completed jobs |
| Resumed segment start | 2026-09-06T15:35:18+02:00 |
| Resumed segment finish | 2026-09-06T16:51:43+02:00 |
| Resumed segment wall-clock runtime | 4,585.52 seconds (1:16:25.52) |
| Resumed segment jobs | 538 completed, 0 failed |
| Total completed / failed jobs | 2,928 / 0 |
| Optimizer retries | 5 additional optimizer attempts across 5 jobs; no job-level retries |
| Duplicate job / forecast keys | 0 / 0 |
| Forecast rows | 102,480 |
| Coverage | 100% for all eight country/specification combinations |
| Final Germany specification | `hw_mul_s168_damped` |
| Final Austria specification | `hw_mul_s168_damped` |
| Total wall-clock runtime | Not exactly reconstructable; external timestamps were not captured for the earlier 2,390-job segment |
| Notes | The resumed process was monitored with 2 workers; progress continued throughout and the 20-minute stall condition was not triggered. Detailed outputs are in `results\exponential_smoothing\holt_winters\validation_2024\`. |

### Non-seasonal ARIMA Development Stage

| Field | Value |
|---|---|
| Model family | Non-seasonal ARIMA |
| Training period | 2020-2023 only |
| Validation stage | Ordinary/DST 2024 mini-validation only |
| Countries | Germany, Austria |
| Selected differencing order | `d=1` for Germany; `d=1` for Austria |
| Training candidate fits | 22 total; 11 per country |
| Model-valid candidate fits | 22 total; all converged with finite criteria and valid roots |
| Residual adequacy flags | 22 total; retained diagnostically and not used for hard rejection |
| Frozen shortlist | Germany: `arima_p2_d1_q0`, `arima_p2_d1_q2`; Austria: `arima_p2_d1_q1`, `arima_p2_d1_q2` |
| Mini-validation jobs | 12 |
| Mini-validation failures | 0 |
| Mini-validation forecast rows | 420 |
| Mini-validation coverage | 100% for all four country/specification combinations |
| Worker benchmark workload | 12 jobs, 3 target dates per country, 2 shortlisted candidates per country |
| Runtime projection scale | `366 / 3 = 122` target-date sets per country |
| Numerical backend | scipy-openblas, configured with `MAX_THREADS=24` |
| Recommendation | 4 process workers with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` |
| Second-laptop estimate | Theoretical split could reduce the projected ~0.80 h to roughly ~0.40 h on an equivalent machine, but no second machine was used and no project data were copied |
| Full 2024 validation | Complete: 1,464/1,464 jobs, 0 failures, 100% target-day coverage |
| 2025 evaluation | Not started |

| Workers | Internal numerical threads | Observed mini wall-clock | Projected full-2024 wall time | Jobs | Failures |
|---:|---:|---:|---:|---:|---:|
| 1 | Default | 140.65 s (2:20.65) | 4.77 h | 12 | 0 |
| 2 | Default | 91.87 s (1:31.87) | 3.11 h | 12 | 0 |
| 3 | Default | 76.33 s (1:16.33) | 2.59 h | 12 | 0 |
| 4 | Default | 64.60 s (1:04.60) | 2.19 h | 12 | 0 |
| 4 | Controlled at 1 thread/process | 23.54 s (0:23.54) | 0.80 h | 12 | 0 |

All benchmark runs produced identical forecasts, 420 forecast rows, zero duplicate keys, and complete target-day coverage. The worker/thread benchmark outputs are under `results\arima\validation_2024\`.

### Full ARIMA 2024 Validation

| Field | Value |
|---|---|
| Model family | Non-seasonal ARIMA |
| Stage | Complete full rolling-origin validation |
| Countries | Germany, Austria |
| Target period | Every valid local target day in 2024 |
| Candidate specifications | 2 frozen shortlisted specifications per country |
| Total validation jobs | 1,464 |
| Worker count | 4 |
| Numerical threads per worker | 1 for OpenMP, MKL, OpenBLAS, and NumExpr |
| Initial pass | 1,459 completed, 5 Austria `arima_p2_d1_q2` convergence failures; 2,774.97 seconds |
| First checkpoint retry | 5 jobs retried with the original optimizer; all 5 reproduced the convergence failure; 28.81 seconds |
| Final checkpoint retry | 5 jobs retried; all completed after one `maxiter=1000` optimizer fallback; 52.69 seconds |
| Total completed / failed jobs | 1,464 / 0 |
| Optimizer fallback retries | 5 additional optimizer attempts across 5 jobs |
| Forecast rows | 51,240, including 35,136 evaluated target-day rows |
| Duplicate job / forecast keys | 0 / 0 |
| Coverage | 100% for all four country/specification combinations; 8,784 target observations each |
| Final Germany specification | `arima_p2_d1_q2` |
| Final Austria specification | `arima_p2_d1_q2` |
| Selection rule | Aggregate target-day MAE, then RMSE, MAPE, and frozen specification order |
| Aggregate output | `results\arima\validation_2024\full_validation\arima_validation_2024_summary.csv` |
| Selection output | `results\arima\tables\arima_selected_2024.csv` |
| Notes | Training residual diagnostic output remains under `results\arima\tables\arima_residual_diagnostics_training_2024.csv`. No 2025 evaluation or SARIMA run was started. |

### SARIMA Development Stage

| Field | Value |
|---|---|
| Model family | Daily-seasonal SARIMA `[24]` |
| Training period | 2020-2023 only |
| Countries | Germany, Austria |
| Candidate fits | 56 total; 28 per country across both viable `(d,D)` paths |
| Numerically valid candidates | 36 total; Germany 17, Austria 19 |
| Residual diagnostic flags | 56 total; retained diagnostically and not used for hard exclusion during narrowing |
| Frozen shortlist | Germany: `sarima_p1_d1_q1_P0_D0_Q1_s24`, `sarima_p2_d0_q0_P0_D1_Q0_s24`; Austria: `sarima_p1_d1_q1_P0_D0_Q1_s24`, `sarima_p1_d0_q1_P1_D1_Q0_s24` |
| Screening reuse | Eligibility and shortlisting were recomputed from persisted candidate records; no screening fits were rerun |
| Mini-validation dates | 2024-02-15 ordinary, 2024-03-31 spring-DST, 2024-10-27 autumn-DST |
| Mini-validation jobs | 12 |
| Mini-validation failures | 0 |
| Mini-validation forecast rows | 420 |
| Mini-validation coverage | 100% for all four country/specification combinations |
| Full 2024 validation | Complete: 1,464/1,464 jobs |
| Full-validation failures | 0 |
| Full-validation forecast rows | 51,240 |
| Full-validation coverage | 100% for all four country/specification combinations |
| Full-validation worker count | 3 |
| Numerical threads per worker | 1 for OpenMP, MKL, OpenBLAS and NumExpr |
| Optimizer retries | 0 |
| Final Germany specification | `sarima_p2_d0_q0_P0_D1_Q0_s24` |
| Final Austria specification | `sarima_p1_d0_q1_P1_D1_Q0_s24` |
| Initial segment | 1,268 / 1,464 jobs completed; reliable external wall-clock start and finish timestamps are unavailable |
| Resumed segment wall-clock | 3,982.9047135 seconds for the remaining 196 jobs, as reported by the validator |
| End-to-end wall-clock | Unavailable; existing filesystem timestamps cover output-directory creation and final output writes but do not reliably delimit the initial segment, so no total is estimated |
| Full-validation output | `results\arima_sarima\validation_2024\full_validation\` |
| 2025 evaluation | Not started |

### SARIMA Worker Benchmark

The same 12-job frozen SARIMA mini workload was recomputed into separate output directories with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`.

Full-year projections scale each observed mini workload by `366 / 3 = 122` target-date sets. The frozen shortlist implies `366 * (2 + 2) = 1,464` full-year jobs.

| Workers | Internal numerical threads | Observed mini wall-clock | Projected full-2024 wall time | Jobs | Failures | Output |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 248.27 s (4:08.27) | 8.41 h | 12 | 0 | `validation_2024\benchmark_workers_1` |
| 2 | 1 | 163.69 s (2:43.69) | 5.55 h | 12 | 0 | `validation_2024\benchmark_workers_2` |
| 3 | 1 | 124.82 s (2:04.82) | 4.23 h | 12 | 0 | `validation_2024\benchmark_workers_3` |
| 4 | 1 | 138.16 s (2:18.16) | 4.68 h | 12 | 0 | `validation_2024\benchmark_workers_4` |

Consistency checks:

- All four runs produced 12 completed jobs and zero failures.
- Each produced 420 forecast rows with no duplicate job or forecast keys.
- Forecast values matched exactly across worker counts; maximum absolute difference was `0.0`.
- Each run retained complete ordinary/DST target-day coverage and no 2025 rows.
- Three workers was fastest for this workload; four workers was slower because of process overhead/contention.

Decision: **3 workers** were used for the full SARIMA validation.

### Regression with ARIMA Errors 2024 Full Validation

| Field | Value |
|---|---|
| Model family | Regression with non-seasonal ARIMA errors |
| Stage | Complete full rolling-origin validation |
| Countries | Germany, Austria |
| Target period | Every valid local target day in 2024 |
| Candidate specifications | 4 frozen specifications per country |
| Total validation jobs | 2,928 |
| Worker count | 8 |
| Numerical threads per worker | 1 for OpenMP, MKL, OpenBLAS, and NumExpr |
| Start | `2026-09-07T11:02:02.2214291Z` (`2026-09-07 13:02:02.2214291 +02:00`) |
| Finish | `2026-09-07T20:23:17.0140523Z` (`2026-09-07 22:23:17.0140523 +02:00`) |
| Wall-clock runtime | `33,674.7926232` seconds (`9:21:14.7926232`) |
| Validator command elapsed | `33,674.5736682` seconds |
| Completed / failed jobs | 2,928 / 0 |
| Optimizer retries | 1 additional optimizer attempt across 1 job |
| Forecast rows | 102,480 |
| Duplicate job / forecast keys | 0 / 0 |
| Coverage | 100% for all eight country/specification combinations; 8,784 target observations each |
| Final Germany specification | `reg_arima_h_d_m_hol` |
| Final Austria specification | `reg_arima_h_d_m_hol` |
| Selection rule | Aggregate target-day MAE, then RMSE, MAPE, and frozen specification order |
| Residual diagnostic status | Warning across all 2,928 jobs: daily residual autocorrelation remains and Ljung–Box diagnostics are significant |
| Runtime segment record | `results\regression_arima_errors\validation_2024\full_validation_runtime_segments.jsonl` |
| Full-validation output | `results\regression_arima_errors\validation_2024\full_validation\` |
| Notes | The family is retained and frozen as a forecasting comparator. The residual warning is an adequacy limitation of the non-seasonal error structure, not a numerical forecast failure. No 2025 evaluation was started. |

### Regression with ARIMA Errors Intervention Mini Validation

| Field | Value |
|---|---|
| Model family | Regression with non-seasonal ARIMA errors intervention extension |
| Stage | Six-job ordinary/DST mini-validation |
| Countries | Germany, Austria |
| Target dates | 2024-02-15 ordinary, 2024-03-31 spring DST, 2024-10-27 autumn DST |
| Candidate specification | `reg_arima_h_d_m_hol_crisis` only |
| Total validation jobs | 6 |
| Workers | 1 |
| Numerical threads per worker | 1 for OpenMP, MKL, OpenBLAS, and NumExpr |
| First segment | Started `2026-09-07T21:23:03.404937Z`; interrupted after 2 jobs; finish timestamp and elapsed runtime unavailable |
| Resumed segment | `2026-09-07T21:25:40.106190Z` to `2026-09-07T21:28:31.922845Z` |
| Resumed segment wall-clock runtime | `171.8355598` seconds |
| Completed / failed jobs | 6 / 0 |
| Forecast rows | 210 |
| Runtime record | `results\regression_arima_errors_intervention\validation_2024\intervention_runtime_segments.jsonl` |
| Notes | The first segment's missing elapsed time was not estimated; the resumed checkpoint completed the four remaining jobs. |

### Regression with ARIMA Errors Intervention 2024 Full Validation

| Field | Value |
|---|---|
| Model family | Regression with non-seasonal ARIMA errors intervention extension |
| Stage | Complete full rolling-origin validation |
| Countries | Germany, Austria |
| Target period | Every valid local target day in 2024 |
| Candidate specification | `reg_arima_h_d_m_hol_crisis` only |
| Total validation jobs | 732 |
| Worker count | 8 |
| Numerical threads per worker | 1 for OpenMP, MKL, OpenBLAS, and NumExpr |
| Preflight segment | `2026-09-07T21:29:35.122608Z` to `2026-09-07T21:29:35.401731Z`; 0 jobs; exit code 1; no model fits |
| Production segment start | `2026-09-07T21:30:24.904147Z` (`2026-09-07 23:30:24.904147 +02:00`) |
| Production segment finish | `2026-09-07T23:49:05.544363Z` (`2026-09-08 01:49:05.544363 +02:00`) |
| Production segment wall-clock runtime | `8,320.5336928` seconds (`2:18:40.5336928`) |
| Completed / failed jobs | 732 / 0 |
| Forecast rows | 25,620 |
| Duplicate job / forecast keys | 0 / 0 |
| Coverage | 100% for both country/specification combinations; 8,784 target observations each |
| Final Germany specification | `reg_arima_h_d_m_hol_crisis` |
| Final Austria specification | `reg_arima_h_d_m_hol_crisis` |
| Full-validation output | `results\regression_arima_errors_intervention\validation_2024\full_validation\` |
| Runtime record | `results\regression_arima_errors_intervention\validation_2024\full_validation_runtime_segments.jsonl` |
| Notes | The intervention family remains separate from the calendar-only `regression_arima_errors` family. No 2025 evaluation or SARIMAX run was started. |

## Pending Runs

### 2025 Holt–Winters Final Test

This run has not started. Record it as a separate computational stage after the 2024 specification is frozen. Do not combine its runtime with the 2024 validation runtime.
