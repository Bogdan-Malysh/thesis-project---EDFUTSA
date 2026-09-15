# Phase 2A Holt-Winters Design

**Status:** Approved for implementation

**Goal:** Add and test a reproducible Holt-Winters specification screen and rolling-origin smoke runner without changing the approved Phase 1 baseline framework or outputs.

## Scope

Phase 2A will use `statsmodels.tsa.holtwinters.ExponentialSmoothing` for eight specifications per country. The specification grid contains:

| Seasonal form | Seasonal period | Trend | Damping |
|---|---:|---|---|
| additive | 24 | additive | undamped |
| additive | 24 | additive | damped |
| additive | 168 | additive | undamped |
| additive | 168 | additive | damped |
| multiplicative | 24 | additive | undamped |
| multiplicative | 24 | additive | damped |
| multiplicative | 168 | additive | undamped |
| multiplicative | 168 | additive | damped |

Every fit will use estimated initialization, optimized parameters, disabled Box-Cox transformation, and disabled bias correction. Multiplicative seasonality will be rejected before fitting when any training load is non-positive.

Phase 2A includes first-origin screening, frozen shortlist generation, and in-memory smoke testing. It does not include complete 2024 rolling validation, 2025 data, statistical model comparison against baselines, Chapter 7 outputs, or thesis text.

## Framework Contract

The implementation will call the existing Phase 1 public functions for country-specific origins, interval-end information sets, target-day extraction, and target-day metrics. It will not modify `forecasting_framework.py`, `baseline_validation_2024.py`, or any approved baseline CSV.

For each target date, training data will be the rows returned by `information_set(frame, forecast_origin_utc)`, where `interval_end_utc <= forecast_origin_utc`. The forecast path will contain all rows with `timestamp_utc >= forecast_origin_utc` through the requested local target date, including bridge intervals and 23-, 24-, or 25-hour target days. The fitted model will forecast the entire path continuously in one call. Only target-day rows will be evaluated.

No future actual load, observed temperature, or official forecast will enter a fit or forecast.

## Components

`code/07_forecasting_results/holt_winters_models.py` will contain the specification dataclass, deterministic grid construction, model fitting, AICc handling, screening, shortlist selection, continuous path forecasting, and failure records.

`code/07_forecasting_results/holt_winters_phase2a.py` will load Germany and Austria inputs, screen the first 2024 origin, write screening and shortlist CSVs, run the requested in-memory smoke tests, and print fitting-time and runtime-estimate summaries.

`code/07_forecasting_results/test_holt_winters_models.py` will cover specification construction, screening eligibility and selection, multiplicative failures, information-set use, path continuity and counts, and finite predictions.

`requirements.txt` and `requirements.lock` will include `statsmodels==0.14.6` and its resolved dependencies for reproducibility.

## Screening

The first local validation date in 2024 will be used for each country. All eight specifications will be fitted using only that origin's information set. Each screening row will record:

- country and deterministic specification identity
- seasonal form, seasonal period, additive trend, and damping flag
- first forecast origin in local and UTC time
- training observation count
- AIC, AICc, and BIC
- convergence status
- fitting time in seconds
- fit status and clear failure error, when applicable
- selection criterion and selection value

A candidate is eligible for shortlisting when its fit succeeds, its optimizer has not explicitly failed, and it has a finite AICc or, when AICc is unavailable, a finite AIC. The `selection_criterion` is `AICc` when the candidate has finite AICc and otherwise `AIC`; `selection_value` contains the corresponding finite value. An explicit optimizer failure makes the candidate ineligible. If convergence metadata are unavailable but the fit succeeds and the selected criterion is finite, convergence is recorded as `unknown` and the candidate remains eligible. Other non-finite criteria and failed fits are ineligible.

Within each seasonal-form and seasonal-period group, the eligible candidate with the lowest `selection_value` is selected, whether one or both candidates are eligible. Ties use deterministic specification order. If neither candidate in a group is eligible, Phase 2A stops before smoke testing. The group, both specification identities, statuses, and errors will be reported clearly.

The screening output will be `results/tables/holt_winters_screening_2024.csv`. The frozen identities will be saved to `results/tables/holt_winters_shortlists_2024.csv`.

## Rolling Smoke Tests

The smoke runner will load the frozen shortlist identities and refit each selected specification from scratch at every requested origin. Parameters and states will never be reused between origins.

Smoke cases:

- Germany ordinary day: 2024-07-15, four shortlisted specifications
- Austria ordinary day: 2024-07-15, four shortlisted specifications
- Germany spring DST day: 2024-03-31, four shortlisted specifications
- Germany autumn DST day: 2024-10-27, four shortlisted specifications

Smoke forecasts remain in memory and are not written as final results. Each case will report target interval count, continuous path count, missing predictions, failures, and fitting time by specification.

The complete 2024 runtime estimate will use measured smoke fitting times for the four selected specifications in each country, scaled to 366 target origins per country. The estimate will state the measured sample basis and projected number of fits.

## Acceptance Criteria

- All eight specification identities are constructed deterministically.
- Screening uses only the first 2024 origin's information set.
- Eligible shortlisting uses finite AICc, with finite AIC fallback only when AICc is unavailable.
- Explicit optimizer failures are ineligible; unavailable convergence metadata are recorded as `unknown` without automatic rejection.
- The lowest selection value is selected from all eligible candidates in each group, including when only one is eligible.
- A group with no eligible candidates stops Phase 2A before smoke testing and reports both failures.
- Four specifications are shortlisted per country when all groups have eligible candidates.
- Smoke refits selected specifications at every requested origin.
- Germany smoke paths preserve 6 bridge intervals plus the 23-, 24-, or 25-hour target day.
- Austria ordinary smoke paths preserve 16 bridge intervals plus the 24-hour target day.
- Smoke predictions contain no missing values and no failures.
- Focused tests and the complete test suite pass.
- Approved Phase 1 files and baseline CSVs remain unchanged.

## Phase 2A Corrections Before Full Validation

The screening optimizer sequence remains unchanged: default L-BFGS-B, the higher-limit L-BFGS-B retry when the iteration limit is reached, and least_squares as the final fallback when required. Each eligible screening row persists `optimizer_used`, `optimizer_attempt_count`, and the complete `optimizer_attempts_json`; the final successful attempt supplies the exact optimizer settings frozen for that shortlist row.

Smoke and future rolling validation will pass the persisted optimizer configuration into the fitter. The frozen configuration is attempted first, then the fixed fallback configurations are considered only after failure. Exact `(optimizer, minimize_kwargs)` pairs are deduplicated within each fit. Thus the currently shortlisted `hw_mul_s168_damped` rows reuse their persisted least_squares configuration directly, while the non-shortlisted `hw_mul_s168_undamped` row retains the normal screening sequence.

Frozen shortlist loading will validate the approved countries, exactly four unique seasonal-form/period groups per country, approved specification IDs, explicit boolean values, eligible rows, finite selection values, consistent specification identity fields, and a valid final successful optimizer attempt. Forecasting will reject ineligible fit records explicitly. Smoke and future forecast records will include optimizer metadata, and runtime estimation will stop instead of estimating from partial results when any smoke fit fails.
