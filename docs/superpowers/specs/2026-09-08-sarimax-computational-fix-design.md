# SARIMAX Computational Fix

## Goal

Keep the exact selected SARIMAX specifications and full 2020-2023 training
history while removing avoidable estimation and inference costs from production
forecast fits. Validate the change with exactly one full Germany
`sarimax_calendar` fit before any mini or full validation run.

## Statistical Contract

- Preserve the Germany order `(2, 0, 0)` with seasonal order `(0, 1, 0, 24)`.
- Preserve the Austria order `(1, 0, 1)` with seasonal order `(1, 1, 0, 24)`.
- Preserve all frozen calendar and crisis regressors.
- Preserve `trend="n"`, stationarity enforcement, invertibility enforcement,
  the full available 2020-2023 training history, and the existing forecasting
  information set.
- Do not replace SARIMAX with an approximate estimator.

## Forecast Fit Path

`fit_sarimax` will use an explicit forecast-fit configuration passed through the
shared SARIMA fitter:

- optimizer method: L-BFGS;
- maximum optimizer iterations: 50;
- maximum objective-function evaluations: 1,000;
- explicit `cov_type="none"`;
- `low_memory=True`;
- no automatic long retry after a bounded non-converged fit.

This configuration avoids covariance, Hessian, standard-error, and unnecessary
state-result storage during rolling forecast estimation. A non-converged fit
will be recorded as failed and will not silently produce forecasts.

Final reporting fits may opt into covariance and standard errors separately.
Forecast-fit records will retain `standard_errors_finite=False` as an explicit
unavailable value, but this will not invalidate a fit whose required forecast
checks pass.

## Fit Metadata

`SarimaFitRecord` and validation metadata will include optimizer iterations and
objective-function calls in addition to existing convergence, status, warning,
likelihood, parameter-finiteness, root, and forecast-validity fields.

The fitter will avoid repeating exogenous rank validation already performed by
the SARIMAX wrapper. The existing ordinary SARIMA path will retain its current
full-inference and retry defaults unless explicitly configured otherwise.

## Tests

Add focused tests covering:

- explicit forecast optimizer and covariance kwargs;
- no SARIMAX long retry;
- optional standard errors and forecast eligibility;
- optimizer iteration/function-call metadata;
- rejection of bounded non-convergence;
- preservation of the exact SARIMAX orders and exogenous columns;
- low-memory fitted-result forecasting and sequential state updates.

## Benchmark Gate

Run one representative full-data fit only:

- country: Germany;
- specification: `sarimax_calendar`;
- order: `(2, 0, 0)`;
- seasonal order: `(0, 1, 0, 24)`;
- training: all valid observations through `2023-12-31`.

The benchmark overrides the forecast profile's function-evaluation limit with
`maxfun=3000`. This higher bound is diagnostic only and is not adopted for
rolling validation unless the benchmark converges and passes all validity
checks.

Record observations, parameter count, state dimension, optimizer method,
function calls, iterations, convergence, log likelihood, wall-clock fit time,
RAM, and whether covariance/inference was skipped. Start the 12-job fixed-state
mini only if this benchmark is practically useful and valid. Do not start
full-year validation or 2025 validation at this stage.
