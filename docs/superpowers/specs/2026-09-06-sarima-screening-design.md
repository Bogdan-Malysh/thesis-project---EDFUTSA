# SARIMA Screening and 2024 Validation Design

**Status:** Approved in principle with implementation corrections recorded

**Authoritative methodology:** `docs\chapter6.pdf`

## Scope

Implement a bounded daily-seasonal SARIMA family separately for Germany and Austria. Use 2020-2023 only for candidate identification and screening, 2024 only for rolling validation and SARIMA selection, and leave 2025 untouched for the later fixed-specification test.

The completed non-seasonal ARIMA family remains unchanged. SARIMA uses a separate implementation area and specification identity:

```text
SARIMA(p,d,q)(P,D,Q)[24]
```

## Shared Forecasting Rules

- Germany and Austria are estimated and selected separately.
- Germany uses an 18:00 origin on the preceding local day.
- Austria uses an 08:00 origin on the preceding local day.
- Only observations whose interval end is no later than the origin are available.
- Forecast paths bridge unavailable intervals and evaluate only target-day timestamps.
- 23-, 24- and 25-hour target days are retained by complete timestamp.
- 2024 selection ranks aggregate target-day MAE first, then RMSE, MAPE and stable specification order.
- No official forecast, observed future temperature, or 2025 observation is a SARIMA input.

## Differencing

### Ordinary differencing

For each country and each viable seasonal path, test `d=0,1,2` using the established ADF/KPSS procedure:

- ADF: `maxlag=24`, `autolag="AIC"`;
- KPSS: `nlags="auto"`;
- level path uses `regression="ct"`;
- differenced paths use `regression="c"`;
- a path is stationary-supported when ADF p-value `<0.05` and KPSS p-value `>=0.05`;
- select the smallest supported `d` within that seasonal path.

### Seasonal differencing

Restrict `D` to `{0,1}` and do not describe ADF/KPSS as seasonal-unit-root tests.

For each country:

1. Assess seasonal dependence at lags 24, 48 and 72 on the training series.
2. Apply `(1-L^24)` for the `D=1` path.
3. Reassess the change in seasonal persistence at those lags.
4. Apply ordinary ADF/KPSS only to the resulting transformed path to determine its smallest adequate `d`.
5. Retain `D=0` and/or `D=1` when the seasonal evidence and transformed-path stationarity support that path.

Both seasonal paths may survive training. Their AIC/AICc/BIC values are never compared directly across different `(d,D)` combinations. If both survive, 2024 rolling MAE determines the final `D` and specification.

## Bounded Candidate Construction

Bounds are fixed at:

```text
p,q in {0,1,2}
P,Q in {0,1}
seasonal_period = 24
```

The core candidate pool is:

```text
(p,q,P,Q) =
(0,0,0,0), (1,0,0,0), (0,1,0,0), (1,1,0,0),
(2,0,0,0), (0,2,0,0), (0,0,1,0), (0,0,0,1)
```

Evidence-supported interactions may add only:

```text
(1,0,1,0), (0,1,0,1), (1,1,1,0), (1,1,0,1),
(1,0,0,1), (0,1,1,0)
```

The last two represent non-seasonal AR plus seasonal MA and non-seasonal MA plus seasonal AR. The maximum is 14 candidates per viable `(d,D)` path. Ordinary ACF/PACF through lag 24 and seasonal ACF/PACF at lags 24, 48 and 72 guide which interaction candidates are retained; they never create additional orders.

## Deterministic Terms

Use one fixed convention, not a candidate dimension:

| Differencing | Statsmodels trend |
|---|---|
| `d=0, D=0` | `trend="c"` |
| `d>=1, D=0` | `trend="n"` |
| `d=0, D=1` | `trend="n"` |
| `d>=1, D=1` | `trend="n"` |

No drift or trend candidates are added.

## Fitting and Validity

Use `statsmodels.tsa.statespace.sarimax.SARIMAX` with `order=(p,d,q)`, `seasonal_order=(P,D,Q,24)`, stationarity enforcement and invertibility enforcement enabled.

Fit once with the standard optimizer. If it explicitly fails to converge, retry once with `maxiter=1000`. Record all attempts and warnings. Hard model-invalidity conditions are:

- fit exception;
- final optimizer non-convergence, unsuccessful status or nonzero warning flag;
- non-finite parameters, standard errors, likelihood, AIC, AICc or BIC;
- non-finite or unusable forecast output;
- invalid AR or MA root conditions.

Hessian, covariance and covariance-inversion warnings remain diagnostic-only unless they also produce one of the hard invalidity conditions.

## Root Diagnostics

Primary stationarity/invertibility checks use the fitted combined roots reported by statsmodels:

- `result.arroots` for the combined AR polynomial;
- `result.maroots` for the combined MA polynomial.

Every applicable combined root must have modulus `>1.01`.

If separate component diagnostics are reported, they are calculated independently rather than relabelled from the combined arrays. The implementation reconstructs:

- the non-seasonal AR polynomial from fitted `ar.L*` coefficients;
- the non-seasonal MA polynomial from fitted `ma.L*` coefficients;
- the seasonal AR polynomial from fitted `ar.S.L24*` coefficients;
- the seasonal MA polynomial from fitted `ma.S.L24*` coefficients.

Roots are calculated from these lag-polynomial coefficient vectors with `numpy.roots`. Reported component minima are explicitly labelled as reconstructed component roots. The combined statsmodels root minima remain the primary validity gates; any applicable reconstructed component root at or below `1.01` also causes hard rejection.

## Residual Diagnostics and Burn-in

The fitted results are inspected for the state-space likelihood burn-in, using the reliable `loglikelihood_burn` value exposed by the fitted result or its state-space model. The diagnostic burn-in is:

```text
max(48, p + q + d + 24 * (P + Q + D))
```

The actual residual burn-in is the larger of those two values. This exact value is stored for every fit and covered by tests.

After burn-in:

- calculate residual ACF through lag 48;
- calculate Ljung–Box statistics and p-values at lags 24 and 48;
- use `model_df = p + q + P + Q` and alpha `0.01`;
- retain all residual ACF values, flags, Ljung–Box values and warnings.

During training screening, a materially clear residual pattern may reject a candidate. The training rejection rule is both Ljung–Box p-values below `0.01` plus at least two residual ACF exceedances of the practical threshold.

Once a specification enters the frozen 2024 shortlist, residual ACF/Ljung–Box outcomes are diagnostic flags only. A 2024 job is not discarded for residual autocorrelation. A 2024 job fails only for genuine fit or forecast invalidity listed above, preserving comparable target coverage between candidates.

## Training Information-Criterion Narrowing

Within each identical `(d,D)` path, use valid candidates only and retain the union of the best candidate by AIC, AICc and BIC. Do not compare information criteria across different differencing combinations. The expected country-specific shortlist is 3-6 specifications, with no artificial candidate added when fewer valid candidates remain.

## Validation Architecture

Create a separate SARIMA area:

```text
code\04_models\sarima\
    sarima_models.py
    sarima_screening.py
    sarima_validation_2024.py
    test_sarima_models.py
    test_sarima_screening.py
    test_sarima_validation_2024.py
```

The 2024 validator uses a deterministic manifest keyed by `(country,target_date,specification_id)`, Windows `spawn` workers, parent-owned atomic persistence, completed-job skipping, failed-job retry, job upserts, forecast upserts by timestamp, separate diagnostics and summaries, and no final selection output until complete.

Mini-validation dates for both countries are:

- `2024-02-15` ordinary day;
- `2024-03-31` spring-DST day;
- `2024-10-27` autumn-DST day.

## Runtime Benchmark and Projection

Benchmark the actual frozen SARIMA mini workload with 1, 2, 3 and 4 workers. The primary comparison controls `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` at one per worker. A four-worker default-thread comparison may quantify oversubscription against the completed ARIMA finding.

Use the actual country-specific shortlist sizes:

```text
full_2024_jobs = 366 * (K_Germany + K_Austria)
mini_jobs = 3 * (K_Germany + K_Austria)
```

For each worker count, project full-year wall time from the measured mini workload using `366/3=122`, preserving separate Germany/Austria candidate counts in the calculation. Estimate the theoretical second-machine benefit from the measured country workloads only. Do not copy data or run another machine during this stage.

## Explicit Stop Boundary

This stage ends after tests, 2020-2023 screening, frozen shortlists, ordinary/DST mini-validation, worker/thread benchmarks and runtime projections. It does not run the full 2024 SARIMA validation and does not access 2025.
