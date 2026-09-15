# 2025 Holt-Winters Test

## Scope

Run the final 2025 test set for exactly one frozen Holt-Winters specification:
`hw_mul_s168_damped`, separately for Germany and Austria. Do not retune, screen,
compare, or select any 2025 specification. Do not run another model family.

Before any job is built, verify the authoritative 2024 selected-specification
file contains exactly `hw_mul_s168_damped` for both countries with complete,
failure-free 2024 validation. Load the corresponding frozen-shortlist rows to
reuse their exact optimizer configuration and model settings.

## Implementation

Add `code/04_models/exponential_smoothing/holt_winters_test_2025.py` as a
separate runner. It will reuse the existing Holt-Winters classes, frozen
shortlist parsing, `forecast_holt_winters`, worker execution, and
`evaluate_target_day`/`calculate_metrics` functions without changing the 2024
runner.

The runner will:

- construct exactly 365 target dates per country for 2025;
- construct exactly 730 jobs for `hw_mul_s168_damped`;
- use the existing Germany 18:00 and Austria 08:00 previous-day origins;
- fit with the expanding information set returned by `information_set` at each
  exact origin cutoff, including all observed 2020--2024 history and genuinely
  available 2025 observations;
- never truncate history to a new fixed four-year window;
- preserve the existing multiplicative seasonal period 168, damped trend,
  initialization, optimizer settings, bridge construction, UTC timestamps, and
  DST behavior;
- score target-day rows only through the existing metric functions;
- reject duplicate target timestamps, non-finite forecasts, invalid DST counts,
  origin/cutoff mismatches, and training-observation counts inconsistent with
  the origin-bounded information set;
- write separate authoritative 2025 Holt-Winters test outputs under
  `results/exponential_smoothing/holt_winters/test_2025/`:
  `holt_winters_test_2025_jobs.csv`,
  `holt_winters_test_2025_forecasts.csv`, and
  `holt_winters_test_2025_summary.csv`.

The full run will use two process workers with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`.

## 2025 Tables

Append the completed Holt-Winters rows to:

- `results/model_test_2025_all_models.csv`
- `results/model_test_2025_overview.csv`

The incoming rows use the existing 2024 schema:

- `model_family = "holt_winters"`
- `specification_id = "hw_mul_s168_damped"`

The table update will merge by
`(country, model_family, specification_id)`, replace only same-key rows on a
rerun, retain Weekly seasonal naive rows and any other completed families, and
sort deterministically by country, model family, and specification identifier.

## Tests

Add focused tests for authoritative 2024 selection verification, the frozen
specification/settings, 2025 job construction, expanding-history training
counts, Germany/Austria origin rules, spring/autumn DST target lengths,
target-only metric delegation, target timestamp uniqueness, finite forecasts,
and merge-safe 2025 table updates.

## Verification

Run the focused Holt-Winters tests, then the full 730-job 2025 test with two
workers. Verify 730/730 jobs, 8,760 target observations per country, 100%
coverage, DST counts of 23 and 25, no future-information leakage, runtime,
2024 comparison metrics, preserved 2024 files, preserved baseline 2025 files,
and an unchanged selected-model registry. Do not run any other model.
