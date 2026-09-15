# ARIMA Screening and 2024 Validation Design

**Status:** Approved in principle; pending written-spec review

**Authoritative methodology:** `docs\chapter6.pdf`

## Goal

Implement the non-seasonal ARIMA stage described in Chapter 6 for Germany and
Austria. Identify and narrow low-order ARIMA candidates using only the
2020-2023 training period, evaluate the frozen shortlists using rolling
day-ahead forecasts on a small 2024 mini-validation, benchmark the actual
ARIMA workload, and stop before the full 2024 validation and before 2025.

## Non-goals

- Do not modify `docs\chapter6.pdf`.
- Do not modify the outdated ARIMA wording in
  `results\tables\table_6_2_candidate_models.csv` or
  `code\06_methodology\chapter6_visuals.py` during this stage.
- Do not search ARIMA orders using 2024 results.
- Do not implement or run SARIMA, regression with ARIMA errors, or SARIMAX.
- Do not run the full 2024 ARIMA validation.
- Do not access or evaluate 2025 data.
- Do not use temperature, SMARD forecasts, APG forecasts, or calendar
  regressors as ARIMA inputs.
- Do not use machine-learning models, Prophet, `pmdarima`, or a large brute-
  force order grid.

## Chapter 6 Alignment

- Germany and Austria are modeled separately.
- The objective is a fixed next-day hourly load forecast at the country-
  specific forecast origin.
- Germany uses 18:00 on the preceding local day; Austria uses 08:00 on the
  preceding local day.
- Only observations whose interval end is no later than the origin are
  available to the model.
- 2020-2023 is the initial identification and estimation period.
- 2024 is the rolling validation period and is used only after the order
  shortlist is frozen.
- 2025 remains the later fixed-specification out-of-sample test period.
- Each forecast path is frozen at its origin. Bridge forecasts are used for
  unavailable intermediate values and are not evaluated.
- Complete target timestamps are retained, including 23-hour and 25-hour
  daylight-saving target days.
- Aggregate target-day MAE is the primary 2024 selection criterion. RMSE and
  MAPE are secondary measures. Countries are selected separately.
- The official SMARD forecast remains an external benchmark and is never a
  model input or ARIMA selection variable.

## Shared Framework

The existing shared framework remains the only owner of:

- Country data loading and preparation.
- Country-specific forecast origins.
- Information-set cutoffs.
- UTC ordering and local-date handling.
- Complete DST target-day extraction.
- Target-day-only MAE, RMSE, MAPE and coverage calculations.

ARIMA code will call these functions rather than reproduce their logic.

## Differencing Identification

Differencing is determined separately for each country using only the complete
2020-2023 hourly training series. Candidate values are `d = 0, 1, 2`.

The exact tests are:

| Test | Series | Deterministic component | Lag rule |
|---|---|---|---|
| ADF | Level for `d=0`; first or second difference for `d=1` or `d=2` | `regression="ct"` for the level; `regression="c"` for differences | `maxlag=24`, `autolag="AIC"` |
| KPSS | Level for `d=0`; first or second difference for `d=1` or `d=2` | `regression="ct"` for the level; `regression="c"` for differences | `nlags="auto"` using the statsmodels 0.14.6 automatic bandwidth rule |

The decision rule at the predefined 5% level is:

```text
stationary_supported(d) = (ADF p-value < 0.05) and (KPSS p-value >= 0.05)
```

Select the smallest `d` for which `stationary_supported(d)` is true. This
automatically prefers no differencing or first differencing over unnecessary
additional differencing. If no value of `d` satisfies both tests, the screen
does not silently choose an order; it records the contradictory diagnostics
and stops without producing an ARIMA shortlist for that country.

The screening output records the tested series, deterministic components,
ADF lag selected by AIC, KPSS bandwidth returned by the automatic rule,
statistics, p-values, warnings and the selected `d`.

## Fixed Low-Order Candidate Set

After selecting the country-specific `d`, fit exactly these 11 non-seasonal
candidates:

```text
(0,d,0)
(1,d,0)
(0,d,1)
(1,d,1)
(2,d,0)
(0,d,2)
(2,d,1)
(1,d,2)
(2,d,2)
(3,d,0)
(0,d,3)
```

Constraints:

- `p <= 3` and `q <= 3`.
- ACF/PACF through lag 24 are calculated only as diagnostic guidance for
  interpreting the low-order candidates.
- ACF/PACF prominence never adds orders 6, 12, 24 or any other high order.
- ARIMA(24,d,0) and ARIMA(0,d,24) are never constructed.
- Daily seasonal dependence is reserved for the later SARIMA family.

## Deterministic-Term Convention

The deterministic term is fixed as a function of the selected `d` and is not
a candidate dimension:

| Selected `d` | statsmodels convention |
|---:|---|
| `0` | `trend="c"`, an intercept |
| `1` | `trend="n"`, no intercept or drift |
| `2` | `trend="n"`, no intercept, drift or trend |

All candidates within a country use the same convention, observations,
stationarity/invertibility enforcement and fitting settings.

## Training-Stage Information Criteria

For every candidate fit on the same 2020-2023 observations, retain AIC, AICc,
BIC, log likelihood, effective observations, parameter count, convergence
metadata and warnings.

Information criteria are compared only within the selected `d`. They narrow
the set but do not determine the final model. After diagnostic rejection, the
shortlist is the union of the top two eligible candidates under each of AIC,
AICc and BIC. The resulting country-specific shortlist is expected to contain
approximately 3-6 candidates and is frozen before 2024 validation.

## Objective Candidate Rejection Rules

### Convergence and fit validity

Hard reject a candidate if any condition holds:

- Fitting raises an exception.
- `mle_retvals["converged"]` is not explicitly `True`.
- `mle_retvals["success"]` exists and is not `True`.
- `mle_retvals["warnflag"]` exists and is not `0`.
- Any fitted parameter, standard error, log likelihood, AIC, BIC or AICc is
  non-finite.
- The forecast produced by the fitted model is non-finite.
- A warning demonstrably indicates an invalid or failed fit, including
  `ConvergenceWarning`, `RuntimeWarning`, or warning text containing
  `converg`, `overflow`, `underflow`, `invalid value`, `nan` or `infinite`,
  except when the warning is specifically about Hessian inversion or
  covariance estimation.

Hessian inversion and covariance-estimation warnings are retained as
diagnostic warnings, including messages referring to singular covariance,
non-positive-definite Hessians or failed covariance inversion. They are not
hard rejection reasons by themselves, even when emitted as a RuntimeWarning.
They cause rejection only when they also result in non-finite estimates,
invalid likelihood/information criteria, failed convergence, or unusable
forecasts.

### AR stationarity and MA invertibility

Fit with statsmodels stationarity and invertibility enforcement enabled, then
independently inspect the fitted roots. Hard reject when:

```text
min(abs(ar_roots)) <= 1.01
```

for a model with AR terms, or:

```text
min(abs(ma_roots)) <= 1.01
```

for a model with MA terms. The strict 1.01 threshold provides a fixed 1%
distance from the unit circle. A model with no corresponding terms passes that
check vacuously.

### Residual ACF and Ljung-Box Adequacy Diagnostics

Use residuals after dropping the first:

```text
max(24, p + q + d)
```

observations. Calculate residual ACF for lags 1-48. Store the 1% pointwise
threshold and apply the practical magnitude floor:

```text
acf_threshold = max(2.576 / sqrt(n_effective), 0.05)
```

A residual ACF lag is flagged when its absolute value exceeds this threshold.

Run Ljung-Box tests with:

```text
lags = [24, 48]
model_df = p + q
alpha = 0.01
```

The following joint condition sets a residual-adequacy flag:

- Both Ljung-Box p-values are below `0.01`.
- At least two residual ACF lags among 1-48 exceed `acf_threshold`.

A residual-adequacy flag is diagnostic information for the non-seasonal ARIMA
reference family and does not make the candidate ineligible. A candidate is
never rejected solely because one or both Ljung-Box tests are significant.
All residual ACF values, flagged lags, Ljung-Box statistics and p-values are
stored for every candidate. Non-finite residual diagnostics are stored as
unavailable diagnostic results and do not override the model-validity gates.
Eligibility is determined only by fit validity, convergence, finite estimates
and information criteria, AR stationarity, MA invertibility and usable
forecast output.

## Rolling ARIMA Forecasting

Use origin-by-origin re-estimation for the mini-validation and eventual 2024
rolling validation. There is no refit-frequency search and no separate
refit-frequency hyperparameter.

At each origin, for each frozen candidate:

1. Build the existing country-specific information set.
2. Fit the ARIMA model using only observations available at that origin.
3. Generate one continuous forecast from the first unavailable interval to the
   final target-day interval.
4. Preserve bridge forecasts but evaluate only complete target-day timestamps.
5. Freeze and store the path before later actual observations are available.

If daily refitting is retained, the current Chapter 6 sentence stating that
the re-estimation frequency is selected during validation will need a minor
wording update later. Chapter 6 is not modified during this stage.

## Resumable Validation Architecture

Implement the ARIMA family under:

```text
code\04_models\arima\
    arima_models.py
    arima_screening.py
    arima_validation_2024.py
    test_arima_models.py
    test_arima_screening.py
    test_arima_validation_2024.py
```

The eventual full-year validator will be resumable before use:

- Deterministic manifest keyed by `(country, target_date, specification_id)`.
- Atomic checkpoint writes.
- Completed jobs skipped; incomplete jobs retried.
- Forecast rows upserted by job key and timestamp.
- Separate job, forecast, summary, selection and diagnostic outputs.
- Windows-safe spawn-based workers.
- No final selection output for incomplete validation.

## Mini-Validation and Benchmarks

Use a separate mini-validation directory with these dates for both countries:

- `2024-02-15` ordinary day.
- `2024-03-31` spring DST day.
- `2024-10-27` autumn DST day.

Verify origins, target lengths, information leakage, complete paths,
target-only metrics, coverage, convergence and root diagnostics. The
mini-validation does not produce a final selection.

Benchmark this actual ARIMA workload with exactly 1, 2, 3 and 4 workers using
Windows `spawn`. Inspect the installed numerical backend first. If internal
numerical-library threading is active, compare controlled single-threaded
settings with the default setting at the recommended worker count. Otherwise
record that internal threading is not material.

Report observed runtimes, job counts, failures, retries, projected full-2024
runtime by worker count, recommended worker configuration and the theoretical
benefit of splitting Germany and Austria across a second machine. Do not copy
project data or execute anything on another machine during this stage.

## Targeted TDD and Execution Boundary

Before any mini-validation, tests will cover:

- Exact ADF/KPSS configuration and smallest-adequate-`d` decisions.
- Bounded candidate construction and duplicate prevention.
- Same-`d` information-criterion narrowing.
- Deterministic-term selection.
- Convergence, warning, root and residual rejection rules.
- Information-set leakage and bridge forecasting.
- Ordinary and DST target lengths and target-only metrics.
- Checkpoint resume, retries and duplicate-free upserts.
- Deterministic serialized results across worker counts.

The stage ends after targeted tests, training-only screening, ordinary/DST
mini-validation, worker/thread benchmarks and runtime projection. The full
2024 ARIMA validation and all 2025 evaluation remain explicitly out of scope.
