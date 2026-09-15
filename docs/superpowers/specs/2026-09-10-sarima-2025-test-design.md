# 2025 SARIMA Test

## Scope

Run the final untouched 2025 test set for exactly one frozen SARIMA
specification per country. Do not search orders, tune specifications, select on
2025 performance, or run another model family.

Before building jobs, verify the authoritative 2024 SARIMA selection,
selected-model registry, and frozen shortlist agree on these specifications:

| Country | Specification | `order` | `seasonal_order` | Trend |
|---|---|---|---|---|
| Germany | `sarima_p2_d0_q0_P0_D1_Q0_s24` | `(2, 0, 0)` | `(0, 1, 0, 24)` | `n` |
| Austria | `sarima_p1_d0_q1_P1_D1_Q0_s24` | `(1, 0, 1)` | `(1, 1, 0, 24)` | `n` |

The 2024 selected validation must be complete and failure-free for both
countries. The selected-model registry must remain unchanged.

## Implementation

Add `code/04_models/sarima/sarima_test_2025.py` as an isolated runner. Do not
modify `sarima_validation_2024.py`, `sarima_models.py`, or any 2024 runner.
Reuse the proven 2024 validation functions exactly:

- `sarima_validation_2024.execute_job` for fitting, bridge construction,
  target-day scoring, and fit metadata;
- `sarima_validation_2024.build_sarima_forecast_path` and
  `sarima_models.forecast_values` for forecast paths;
- `sarima_models.fit_sarima` for statsmodels SARIMAX-backed SARIMA fitting;
- the existing trend, convergence, retry, warning, stationarity,
  invertibility, forecast-validity, root, residual-diagnostic, and eligibility
  rules;
- `evaluate_target_day` and `calculate_metrics` for target-only metrics;
- the existing diagnostic-artifact completion checks and summary calculations.

The new runner must load the complete processed country frames so genuinely
available 2025 observations can enter later training sets. Each fit must call
`information_set(frame, context.cutoff_utc)` through the existing 2024 job
execution path, where the exact country-specific forecast origin is the upper
bound. No future 2025 observations may enter a fit.

The runner will:

- build exactly 365 local dates per country and 730 jobs total;
- create one manifest row per country/date with the frozen country-specific
  SARIMA order and `trend="n"`;
- use Germany's 18:00 previous-day origin and Austria's 08:00 previous-day
  origin;
- refit the model independently at every target-date origin, as in 2024;
- preserve bridge forecasts, exact UTC timestamps, target-only scoring, and
  23/24/25-hour DST behavior;
- record all job, optimizer, root, warning, residual, and forecast metadata
  produced by the existing `SarimaFitRecord` path;
- validate training-observation counts against the exact origin-bounded
  information set;
- reject duplicate target timestamps, non-finite forecasts, future local-date
  rows, incomplete bridge artifacts, or missing diagnostics.

## Outputs and Checkpointing

Use the separate final-test directory:

`results/arima_sarima/test_2025/`

Write these authoritative files there:

- `sarima_test_2025_jobs.csv`
- `sarima_test_2025_forecasts.csv`
- `sarima_test_2025_diagnostics.csv`
- `sarima_test_2025_summary.csv`

Checkpoint jobs, forecasts, diagnostics, and summaries after each completed
result. A job is resumably complete only when its job row is completed, its
diagnostic row exists, and its forecast artifact contains every expected bridge
and target timestamp with finite values. Failed or incomplete jobs must be
retried on the next invocation.

Require exactly three process workers for the full run. Launch with:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Do not benchmark worker counts again.

## 2025 Tables

Append one row per country to both existing 2025 model tables:

- `results/model_test_2025_all_models.csv`
- `results/model_test_2025_overview.csv`

Incoming rows use:

- `model_family = "sarima"`;
- the country-specific frozen `specification_id`;
- `n_observations = 8760`;
- the 2025 summary MAE, RMSE, MAPE, and coverage.

Merge by `(country, model_family, specification_id)`, replacing only matching
SARIMA rows on a rerun. Preserve the existing Weekly seasonal naive,
Holt-Winters, and ARIMA 2025 rows exactly. Sort deterministically by country,
model family, and specification identifier. Do not update the selected-model
registry or any 2024 result.

## Tests and Verification

Add focused tests for:

- authoritative selection, registry, shortlist, order, seasonal order, and
  trend verification;
- 2025 manifest construction and country-specific frozen specifications;
- exact Germany/Austria origins and expanding training counts;
- delegation to the existing 2024 `execute_job` path and retry behavior;
- bridge, target-only metric, UTC timestamp, finite-forecast, and DST rules;
- future-information rejection and job metadata integrity;
- checkpoint completion requiring jobs, diagnostics, and full forecast artifacts;
- retrying failed/incomplete jobs;
- deterministic, merge-safe table updates.

Run focused tests first. If they pass, run the full 730-job test with exactly
three workers and checkpoint/resume enabled.

Require:

- 730/730 completed jobs and zero unresolved failures;
- 8,760/8,760 evaluated target observations per country and 100% coverage;
- 23 intervals on `2025-03-30` and 25 intervals on `2025-10-26` for each
  country;
- no duplicate target timestamps or non-finite forecasts;
- no future local-date rows and exact origin-bounded training counts;
- convergence, retry, warning, root, and residual metadata consistent with the
  proven 2024 SARIMA methodology;
- no 2024 outputs, existing 2025 outputs, or registry changes.

Report Germany and Austria MAE, RMSE, MAPE, observations, coverage,
convergence count, retry count, and runtime beside the authoritative 2024
metrics. Do not use 2025 performance to alter the frozen specifications.
