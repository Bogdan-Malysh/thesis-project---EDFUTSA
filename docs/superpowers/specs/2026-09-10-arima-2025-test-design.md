# 2025 ARIMA Test

## Scope

Run the final 2025 test set for exactly one frozen ARIMA specification:
`arima_p2_d1_q2`, equivalent to ARIMA(2,1,2), separately for Germany and
Austria. Do not search orders, tune specifications, select on 2025 performance,
or run another model family.

Before building jobs, verify the authoritative 2024 selected specification and
selected-model registry both identify `arima_p2_d1_q2` for Germany and Austria,
and verify the 2024 selected validation is complete and failure-free.

## Implementation

Add `code/04_models/arima/arima_test_2025.py` as an isolated runner. It will
reuse the existing ARIMA model and validation functions without changing the
2024 runner:

- `fit_arima` for statsmodels ARIMA fitting;
- `trend_for_d`, which gives `trend="n"` for `d=1`;
- `build_arima_forecast_path` and `forecast_values` for bridge construction;
- `evaluate_target_day` and `calculate_metrics` for scoring;
- the existing stationarity/invertibility checks, convergence criteria, warning
  handling, and eligibility rules.

The exact 2024 retry policy is frozen and must not be changed: perform the
normal fit first, retry only when the first fit is not explicitly converged,
use exactly `method_kwargs={"maxiter": 1000}` for that one retry, and reject
the job if the resulting fit remains ineligible. Do not add optimizer settings,
fallbacks, or acceptance exceptions in the 2025 runner.

The runner will:

- build exactly 365 dates per country and 730 jobs total;
- use only order `(2, 1, 2)` and `trend="n"`;
- use the expanding `information_set` bounded by each exact origin, including
  genuinely available 2025 observations and no future observations;
- preserve Germany 18:00 and Austria 08:00 previous-day origins;
- preserve exact UTC timestamps, bridge forecasts, target-day scoring, and
  23/24/25-hour DST behavior;
- record convergence, retry, root, warning, and fit metadata from the existing
  `ArimaFitRecord` path;
- validate training-observation counts against the origin-bounded information
  set and reject duplicate target timestamps or non-finite forecasts.

## Outputs and Checkpointing

Use a separate test directory:

`results/arima/test_2025/`

with these authoritative files:

- `arima_test_2025_jobs.csv`
- `arima_test_2025_forecasts.csv`
- `arima_test_2025_summary.csv`

The runner will checkpoint completed jobs and resume only pending or previously
failed jobs. It will require exactly four workers for the full run and will be
launched with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
and `NUMEXPR_NUM_THREADS=1`.

## 2025 Tables

Append two rows to each existing 2025 model table:

- `results/model_test_2025_all_models.csv`
- `results/model_test_2025_overview.csv`

Incoming rows use:

- `model_family = "arima"`;
- `specification_id = "arima_p2_d1_q2"`.

Merge by `(country, model_family, specification_id)`, replacing only matching
ARIMA rows on a rerun and preserving the existing Weekly seasonal naive and
Holt-Winters rows. Sort deterministically by country, model family, and
specification identifier. Do not update the selected-model registry.

## Tests and Verification

Add focused tests for authoritative selection verification, frozen order/trend
and retry settings, 2025 job construction, expanding training counts, DST and
origin rules, target timestamp uniqueness, finite forecasts, metric delegation,
checkpoint resume, and merge-safe table updates.

Run the focused tests, then the full 730-job test with four workers. Verify
730/730 jobs, zero failures, 8,760 target observations per country, complete
coverage, 23/25 DST counts, no future leakage, convergence/retry metadata, and
runtime. Compare with authoritative 2024 ARIMA metrics. Hash-verify that all
2024 outputs, existing baseline/Holt-Winters 2025 outputs, and the selected
registry remain unchanged.
