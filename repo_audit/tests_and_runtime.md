# Tests and Runtime

## Test Suites

The repository contains 35 test modules. This audit did not execute pytest, so it did not create test caches or alter project files.

| Test module |
|---|
| `code/01_data_audit/test_audit_data.py` |
| `code/01_data_audit/test_load_smard.py` |
| `code/02_preprocessing/test_apg_hourly.py` |
| `code/02_preprocessing/test_calendar_features.py` |
| `code/02_preprocessing/test_compare_apg_smard.py` |
| `code/02_preprocessing/test_crisis_features.py` |
| `code/02_preprocessing/test_modelling_datasets.py` |
| `code/02_preprocessing/test_smard_hourly.py` |
| `code/02_preprocessing/test_temperature_pipeline.py` |
| `code/03_exploratory_analysis/test_section_5_1.py` |
| `code/03_exploratory_analysis/test_section_5_2.py` |
| `code/03_exploratory_analysis/test_section_5_2_daily_weekly.py` |
| `code/03_exploratory_analysis/test_section_5_3_monthly_annual.py` |
| `code/03_exploratory_analysis/test_section_5_4_temperature_public_holidays.py` |
| `code/03_exploratory_analysis/test_section_5_5_crisis_policy_periods.py` |
| `code/03_visualization/test_dwd_station_validation.py` |
| `code/03_visualization/test_temperature_population_map.py` |
| `code/04_models/arima/test_arima_models.py` |
| `code/04_models/arima/test_arima_screening.py` |
| `code/04_models/arima/test_arima_validation_2024.py` |
| `code/04_models/baselines/test_baseline_validation.py` |
| `code/04_models/common/test_build_model_validation_summaries.py` |
| `code/04_models/common/test_forecasting_framework.py` |
| `code/04_models/exponential_smoothing/test_holt_winters_models.py` |
| `code/04_models/exponential_smoothing/test_holt_winters_phase2a.py` |
| `code/04_models/exponential_smoothing/test_holt_winters_validation_2024.py` |
| `code/04_models/regression_arima_errors/test_regression_arima_errors_intervention_validation_2024.py` |
| `code/04_models/regression_arima_errors/test_regression_arima_errors_models.py` |
| `code/04_models/regression_arima_errors/test_regression_arima_errors_validation_2024.py` |
| `code/04_models/sarima/test_sarima_models.py` |
| `code/04_models/sarima/test_sarima_screening.py` |
| `code/04_models/sarima/test_sarima_validation_2024.py` |
| `code/04_models/sarima/test_sarimax_models.py` |
| `code/04_models/sarima/test_sarimax_state_update_2024.py` |
| `code/04_models/sarima/test_sarimax_validation_2024.py` |

Recorded pass/fail counts:

- No complete machine-readable test-run report was found in the repository. Historical plan/spec documents mention expected or prior test results, but those are not treated as a current test execution.
- The current audit therefore reports test execution status as **not run by this audit**, not as pass or fail.

## Runtime Artifacts

- `results/arima_sarimax/validation_2024/sarimax_fixed_state_runtime_segments.jsonl`
- `results/arima_sarimax/validation_2024/sarimax_full_fit_benchmark.json`
- `results/arima_sarimax/validation_2024/sarimax_full_fit_continuation_benchmark.json`
- `results/arima_sarimax/validation_2024/sarimax_full_fit_final_benchmark.json`
- `results/arima_sarimax/validation_2024/sarimax_full_fit_final_parameter_vector.json`
- `results/arima_sarimax/validation_2024/sarimax_full_fit_progress.json`
- `results/arima_sarimax/validation_2024/sarimax_runtime_segments.jsonl`
- `results/regression_arima_errors/validation_2024/full_validation_runtime_segments.jsonl`
- `results/regression_arima_errors_intervention/validation_2024/full_validation_runtime_segments.jsonl`
- `results/regression_arima_errors_intervention/validation_2024/intervention_runtime_segments.jsonl`

## Recorded Runtime Log Extract

- # Computational Runtime Log
- Cumulative fitting time and wall-clock time are different measures. Cumulative fitting time is useful for sequential-work estimates; wall-clock time is used for worker benchmarks.
- AIC/AICc/BIC runtime belongs to screening only. Final forecasting-performance selection uses target-day validation metrics.
- | Workers | Sequential; no parallel worker setting |
- | Wall-clock runtime | Not captured |
- | Successful / failed jobs | 2,928 / 0 |
- | Workers | Sequential |
- | Wall-clock runtime | Not captured |
- | Successful / failed jobs | 16 / 0 |
- | Stage | Smoke validation and runtime estimation |
- | Workers | Sequential |
- | Observed wall-clock runtime | Not captured |
- | Earlier projected full-year runtime | Approximately 10.66 hours sequential |
- | Successful / failed jobs | 16 / 0 reported at the time |
- | Notes | This was an estimate from a small, uneven smoke sample, not the full 2024 validation. The smoke result was not persisted as a per-job runtime CSV. |
- | Workers | 2 |
- | Wall-clock runtime | 426.24 seconds for the 22 jobs executed during the resumable continuation; 2 jobs were already checkpointed |
- | Successful / failed jobs | 24 / 0 |
- | Projected full-year sequential work | 31.52 hours |
- | Projected full-year work at 2 workers | 15.76 hours by simple division of cumulative fitting work |
- Full-year wall-clock projections scale each observed benchmark by `366 / 3 = 122` target-date sets.
- | Workers | Wall-clock runtime | Jobs | Failures | Projected full-2024 wall time | Output |
- | 1 | 574.24 s (9.57 min) | 24 | 0 | 19.46 h | `benchmark_2024_workers_1` |
- | 2 | 298.83 s (4.98 min) | 24 | 0 | 10.13 h | `benchmark_2024_workers_2` |
- | 4 | 1,326.07 s (22.10 min) | 24 | 0 | 44.94 h | `benchmark_2024_workers_4` |
- Four workers were slower than one worker, indicating CPU/numerical-library contention for this workload.
- Decision: use **2 workers** for the full 2024 Holt–Winters validation.
- | Resumed segment wall-clock runtime | 4,585.52 seconds (1:16:25.52) |
- | Total completed / failed jobs | 2,928 / 0 |
- | Total wall-clock runtime | Not exactly reconstructable; external timestamps were not captured for the earlier 2,390-job segment |
- | Notes | The resumed process was monitored with 2 workers; progress continued throughout and the 20-minute stall condition was not triggered. Detailed outputs are in `results\exponential_smoothing\holt_winters\validation_2024\`. |
- | Runtime projection scale | `366 / 3 = 122` target-date sets per country |
- | Recommendation | 4 process workers with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` |
- | Second-laptop estimate | Theoretical split could reduce the projected ~0.80 h to roughly ~0.40 h on an equivalent machine, but no second machine was used and no project data were copied |
- | Workers | Internal numerical threads | Observed mini wall-clock | Projected full-2024 wall time | Jobs | Failures |
- | Initial pass | 1,459 completed, 5 Austria `arima_p2_d1_q2` convergence failures; 2,774.97 seconds |
- | First checkpoint retry | 5 jobs retried with the original optimizer; all 5 reproduced the convergence failure; 28.81 seconds |
- | Total completed / failed jobs | 1,464 / 0 |
- | Initial segment | 1,268 / 1,464 jobs completed; reliable external wall-clock start and finish timestamps are unavailable |
- | Resumed segment wall-clock | 3,982.9047135 seconds for the remaining 196 jobs, as reported by the validator |
- | End-to-end wall-clock | Unavailable; existing filesystem timestamps cover output-directory creation and final output writes but do not reliably delimit the initial segment, so no total is estimated |
- | Workers | Internal numerical threads | Observed mini wall-clock | Projected full-2024 wall time | Jobs | Failures | Output |
- | 1 | 1 | 248.27 s (4:08.27) | 8.41 h | 12 | 0 | `validation_2024\benchmark_workers_1` |
- | 2 | 1 | 163.69 s (2:43.69) | 5.55 h | 12 | 0 | `validation_2024\benchmark_workers_2` |
- | 3 | 1 | 124.82 s (2:04.82) | 4.23 h | 12 | 0 | `validation_2024\benchmark_workers_3` |
- | 4 | 1 | 138.16 s (2:18.16) | 4.68 h | 12 | 0 | `validation_2024\benchmark_workers_4` |
- Three workers was fastest for this workload; four workers was slower because of process overhead/contention.
- Decision: **3 workers** were used for the full SARIMA validation.
- | Wall-clock runtime | `33,674.7926232` seconds (`9:21:14.7926232`) |
- | Completed / failed jobs | 2,928 / 0 |
- | Runtime segment record | `results\regression_arima_errors\validation_2024\full_validation_runtime_segments.jsonl` |
- | Workers | 1 |
- | First segment | Started `2026-09-07T21:23:03.404937Z`; interrupted after 2 jobs; finish timestamp and elapsed runtime unavailable |
- | Resumed segment wall-clock runtime | `171.8355598` seconds |
- | Completed / failed jobs | 6 / 0 |
- | Runtime record | `results\regression_arima_errors_intervention\validation_2024\intervention_runtime_segments.jsonl` |
- | Production segment wall-clock runtime | `8,320.5336928` seconds (`2:18:40.5336928`) |
- | Completed / failed jobs | 732 / 0 |
- | Runtime record | `results\regression_arima_errors_intervention\validation_2024\full_validation_runtime_segments.jsonl` |
- This run has not started. Record it as a separate computational stage after the 2024 specification is frozen. Do not combine its runtime with the 2024 validation runtime.

## Known Computational Problems

- Runtime records document worker/thread benchmarks, optimizer retries, interrupted segments, and full-validation wall-clock measurements.
- ARIMA records include five Austria convergence failures in an initial pass that completed after a `maxiter=1000` fallback; the authoritative result ended with 1,464 completed and zero failed jobs.
- SARIMA full validation was run with three workers according to the runtime log. SARIMAX final Germany fitting converged in the final benchmark, while the 150-iteration continuation did not converge and later maximum-likelihood mini attempts were interrupted/aborted before authoritative validation output existed.
- Runtime records specify single numerical threads in several runs: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`.
