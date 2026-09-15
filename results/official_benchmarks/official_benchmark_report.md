# Official Forecast Benchmark Report

External official forecasts only. No statistical model was fitted, simulated, or modified. Existing SARIMAX and other model artifacts were not written.

## SMARD 2024

- Calendar-year selection uses the prepared local target labels for 2024, then aligns the exact unique UTC target-timestamp set represented by the authoritative completed 2024 model forecast artifacts: 8,784 timestamps per country.
- Actual target values are `actual_grid_load_mwh`; official forecasts are `forecasted_grid_load_mwh` from the prepared SMARD tables.
- Missing official forecasts are excluded from MAE/RMSE/MAPE without interpolation or imputation; coverage is matched observations divided by the 8,784-timestamp target set.
- The model-validation actual target definition matches prepared SMARD actual load (`actual_load_mwh` in model forecast artifacts equals `actual_grid_load_mwh`). Details are in `official_benchmark_data_quality.csv`.

## APG 2020-2022

- APG is evaluated over its full prepared overlap only, with separate 2020, 2021, and 2022 rows.
- APG is an historical Austrian operational benchmark, not a 2024 benchmark. It is not directly period-comparable to the 2024 model validations.

## Source Consistency

- `austria_smard_apg_consistency.csv` compares actual and official forecast series on common hourly UTC timestamps for 2020-2022. It is descriptive source validation only.

## Outputs

- `official_benchmark_summary.csv` contains SMARD, APG, and source-consistency summary rows.
- `smard_2024_benchmark.csv` and `apg_2020_2022_benchmark.csv` contain official-forecast accuracy metrics.
- `official_benchmark_data_quality.csv` contains availability, duplicate, coverage, DST, and model-target-definition checks.
- Timestamp-level aligned files are retained for external review.
