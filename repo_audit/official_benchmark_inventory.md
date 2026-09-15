# Official Benchmark Inventory

This is a read-only audit of source and prepared benchmark files; no forecasting model was calculated.

## Files
| File | Country | Series | Rows | Minimum timestamp | Maximum timestamp | Resolution | Missing counts | Duplicate timestamps |
|---|---|---|---:|---|---|---|---|---:|
| data/processed/apg_austria_hourly.csv | Austria | Actual load + official load forecast (combined) | 26304 | 2019-12-31T23:00:00+00:00 | 2022-12-31T22:00:00+00:00 | 1 hour | `{"actual_load_mwh": 0, "forecasted_load_mwh": 0, "interval_end_local": 0, "interval_end_utc": 0, "interval_start_local": 0, "interval_start_utc": 0}` | 0 |
| data/processed/smard_austria_hourly.csv | Austria | Actual load + official load forecast (combined) | 52608 | 2019-12-31T23:00:00+00:00 | 2025-12-31T22:00:00+00:00 | 1 hour | `{"actual_grid_load_including_pumped_storage_mwh": 0, "actual_grid_load_including_pumped_storage_valid": 0, "actual_grid_load_mwh": 0, "actual_grid_load_valid": 0, "actual_pumped_storage_mwh": 0, "actual_pumped_storage_valid": 0, "actual_residual_load_mwh": 1, "actual_residual_load_valid": 0, "forecasted_grid_load_mwh": 0, "forecasted_grid_load_valid": 0, "forecasted_residual_load_mwh": 72, "forecasted_residual_load_valid": 0, "interval_end_local": 0, "interval_end_raw": 0, "interval_end_utc": 0, "interval_start_local": 0, "interval_start_raw": 0, "interval_start_utc": 0, "is_repeated_autumn_hour": 0}` | 0 |
| data/processed/smard_germany_hourly.csv | Germany | Actual load + official load forecast (combined) | 52608 | 2019-12-31T23:00:00+00:00 | 2025-12-31T22:00:00+00:00 | 1 hour | `{"actual_grid_load_including_pumped_storage_mwh": 0, "actual_grid_load_including_pumped_storage_valid": 0, "actual_grid_load_mwh": 0, "actual_grid_load_valid": 0, "actual_pumped_storage_mwh": 0, "actual_pumped_storage_valid": 0, "actual_residual_load_mwh": 0, "actual_residual_load_valid": 0, "forecasted_grid_load_mwh": 24, "forecasted_grid_load_valid": 0, "forecasted_residual_load_mwh": 24, "forecasted_residual_load_valid": 0, "interval_end_local": 0, "interval_end_raw": 0, "interval_end_utc": 0, "interval_start_local": 0, "interval_start_raw": 0, "interval_start_utc": 0, "is_repeated_autumn_hour": 0}` | 0 |
| data/raw/apg/apg_austria_actual_2020_15min.csv | Austria | Actual electricity load (actual) | 35136 | 2020-01-01T00:00:00 | 2020-12-31T23:45:00 | 15 minutes | `{"Power [MW]": 0, "Time from [CET/CEST]": 0, "Time to [CET/CEST]": 0}` | 4 |
| data/raw/apg/apg_austria_actual_2021_15min.csv | Austria | Actual electricity load (actual) | 35040 | 2021-01-01T00:00:00 | 2021-12-31T23:45:00 | 15 minutes | `{"Power [MW]": 0, "Time from [CET/CEST]": 0, "Time to [CET/CEST]": 0}` | 4 |
| data/raw/apg/apg_austria_actual_2022_15min.csv | Austria | Actual electricity load (actual) | 35040 | 2022-01-01T00:00:00 | 2022-12-31T23:45:00 | 15 minutes | `{"Power [MW]": 0, "Time from [CET/CEST]": 0, "Time to [CET/CEST]": 0}` | 4 |
| data/raw/apg/apg_austria_forecasted_2020_15min.csv | Austria | Official load forecast (forecast) | 35136 | 2020-01-01T00:00:00 | 2020-12-31T23:45:00 | 15 minutes | `{"Load [MW]": 0, "Time from [CET/CEST]": 0, "Time to [CET/CEST]": 0}` | 4 |
| data/raw/apg/apg_austria_forecasted_2021_15min.csv | Austria | Official load forecast (forecast) | 35040 | 2021-01-01T00:00:00 | 2021-12-31T23:45:00 | 15 minutes | `{"Load [MW]": 0, "Time from [CET/CEST]": 0, "Time to [CET/CEST]": 0}` | 4 |
| data/raw/apg/apg_austria_forecasted_2022_15min.csv | Austria | Official load forecast (forecast) | 35040 | 2022-01-01T00:00:00 | 2022-12-31T23:45:00 | 15 minutes | `{"Load [MW]": 0, "Time from [CET/CEST]": 0, "Time to [CET/CEST]": 0}` | 4 |
| data/raw/smard/smard_austria_actual.csv | Austria | Actual electricity load (actual) | 52608 | 2020-01-01T00:00:00 | 2025-12-31T23:00:00 | 1 hour | `{"End date": 0, "Grid load incl. hydro pumped storage [MWh] Calculated resolutions": 0, "Hydro pumped storage [MWh] Calculated resolutions": 0, "Residual load [MWh] Calculated resolutions": 0, "Start date": 0, "grid load [MWh] Calculated resolutions": 0}` | 6 |
| data/raw/smard/smard_austria_forecasted.csv | Austria | Official load forecast (forecast) | 52608 | 2020-01-01T00:00:00 | 2025-12-31T23:00:00 | 1 hour | `{"End date": 0, "Residual load [MWh] Calculated resolutions": 0, "Start date": 0, "grid load [MWh] Calculated resolutions": 0}` | 6 |
| data/raw/smard/smard_germany_actual.csv | Germany | Actual electricity load (actual) | 52608 | 2020-01-01T00:00:00 | 2025-12-31T23:00:00 | 1 hour | `{"End date": 0, "Grid load incl. hydro pumped storage [MWh] Calculated resolutions": 0, "Hydro pumped storage [MWh] Calculated resolutions": 0, "Residual load [MWh] Calculated resolutions": 0, "Start date": 0, "grid load [MWh] Calculated resolutions": 0}` | 6 |
| data/raw/smard/smard_germany_forecasted.csv | Germany | Official load forecast (forecast) | 52608 | 2020-01-01T00:00:00 | 2025-12-31T23:00:00 | 1 hour | `{"End date": 0, "Residual load [MWh] Calculated resolutions": 0, "Start date": 0, "grid load [MWh] Calculated resolutions": 0}` | 6 |

## Column Definitions

- SMARD actual files use `Start date`, `End date`, `grid load [MWh] Calculated resolutions`, `Grid load incl. hydro pumped storage [MWh] Calculated resolutions`, `Hydro pumped storage [MWh] Calculated resolutions`, and `Residual load [MWh] Calculated resolutions`.
- SMARD forecast files use `Start date`, `End date`, `grid load [MWh] Calculated resolutions`, and `Residual load [MWh] Calculated resolutions`.
- APG actual files use `Time from [CET/CEST]`, `Time to [CET/CEST]`, and `Power [MW]`.
- APG forecast files use `Time from [CET/CEST]`, `Time to [CET/CEST]`, and `Load [MW]`.
- Prepared SMARD files expose `actual_grid_load_mwh` and `forecasted_grid_load_mwh`; prepared APG exposes `actual_load_mwh` and `forecasted_load_mwh`.

## Period Comparison

- SMARD Germany and Austria actual/official-forecast series cover the model data window through 2025, so they overlap both the 2024 validation period and the 2025 test-design period.
- APG raw and hourly-prepared files cover 2020-2022 only. They do not overlap 2024 or 2025 and therefore cannot be directly compared with the 2024/2025 SMARD/model validation periods without changing the period or source definition.
- APG is natively 15-minute CET/CEST data and is aggregated to UTC hourly energy before comparison. SMARD is hourly MWh data. The sources are therefore not directly resolution-equivalent before APG aggregation.
- Missing and duplicate counts above are source-file observations. Repeated local autumn labels are expected where the UTC representation remains unique.
