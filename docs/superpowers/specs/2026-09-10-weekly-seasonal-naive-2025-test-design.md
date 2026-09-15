# 2025 Weekly Seasonal Naive Test

## Scope

Run the final 2025 test set for exactly one model: the existing
`Weekly seasonal naive` baseline selected and implemented in 2024. Germany and
Austria are evaluated separately. No model selection, tuning, crisis variant,
benchmark, or other model family is included.

## Implementation

Add `code/04_models/baselines/baseline_test_2025.py` as a separate runner. It
will reuse `common.forecasting_framework` without changing the 2024 runner or
its outputs. The runner will:

- load the existing processed country data;
- enumerate every local date in 2025 for both countries;
- call `forecast_path` only with `"Weekly seasonal naive"`;
- preserve the existing Germany 18:00 and Austria 08:00 previous-day origins;
- retain exact UTC timestamps, DST path construction, target-day-only scoring,
  and the existing 168-hour source-lag implementation;
- calculate MAE, RMSE, and MAPE through the existing
  `evaluate_target_day`/`calculate_metrics` functions, without reimplementing
  metric formulas;
- reject missing dates, duplicate UTC timestamps, non-finite forecasts, or
  source timestamps that violate the information cutoff;
- write authoritative full forecast paths to
  `results/baselines/forecasts/baseline_forecasts_2025.csv`;
- derive the country-level authoritative summary at
  `results/baselines/tables/baseline_validation_2025.csv`.

The 2025 runner will fail validation unless it produces 365 jobs per country,
8,760 scored observations per country, and complete coverage. It will explicitly
check the 2025 spring DST date for 23 target intervals and the autumn DST date
for 25 target intervals.

## 2025 Tables

Create the requested tables deterministically from authoritative 2025 model
outputs:

- `results/model_test_2025_all_models.csv`
- `results/model_test_2025_overview.csv`

Both tables initially contain only two rows, one per country, with
`model_family = "baselines"` and
`specification_id = "Weekly seasonal naive"`. They include at least
`country`, `model_family`, `specification_id`, `mae`, `rmse`, `mape`,
`n_observations`, and `coverage`.

The table builder will preserve existing completed 2025 rows and merge new
model-family rows by the deterministic key
`(country, model_family, specification_id)`. Re-running a completed family
replaces only its same-key rows with values derived from its authoritative
output; rows for other completed families are retained. Output ordering will
be deterministic by country, model family, and specification identifier.

The 2024 comparison values will be read from the existing
`results/baselines/tables/baseline_validation_2024.csv`. No 2024 output or
selection registry will be written.

## Tests

Add focused tests for the new runner and table builder. Tests will cover the
single-model manifest, 2025 date/job counts, Germany and Austria origin rules,
spring/autumn DST lengths, target-only evaluation, UTC uniqueness, finite
forecasts, no future-information leakage, required result-table schema, and
deterministic table generation from authoritative outputs.

## Verification

Run the focused test suite, then execute the full 730-job 2025 test using the
project interpreter. Verify both country summaries, forecast-path integrity,
and the two 2025 tables. Do not run any other model.
