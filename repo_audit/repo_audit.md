# Repository Audit

Read-only snapshot of the current electricity-demand-thesis workspace. Existing code, datasets, results, registries, and tests were not modified by this audit.

## Scope and Current State

- The project forecasts hourly national electricity demand separately for Germany and Austria using classical time-series methods.
- Raw inputs are retained under `data/raw/`; prepared hourly inputs and feature tables are under `data/processed/`; validation metadata is under `data/validation/`.
- The authoritative completed 2024 comparison is represented by `results/model_validation_2024_all_models.csv`, with selected-family rows mirrored in `results/model_validation_2024_selected_overview.csv` and the registry.
- SARIMAX code and benchmark artifacts are present, but the current SARIMAX mini/full validation is not an authoritative completed 2024 model result and is not included in the cross-family selected registry.
- `results/model_validation_2024_all_models.csv` and the selected overview report 2024 only. No authoritative 2025 evaluation result was found.

## Pipeline

1. **Raw source audit.** `code/01_data_audit/load_smard.py` and `audit_data.py` parse the four SMARD semicolon CSVs, preserve raw strings, validate numeric fields, UTC/local interval logic, DST transitions, duplicate rows, and source schemas; reports are written under `results/tables/`.
2. **SMARD preparation.** `code/02_preprocessing/smard_hourly.py` loads separate actual and official forecast files for Germany and Austria, aligns intervals by UTC start, preserves local labels and repeated autumn observations, and writes two prepared SMARD hourly tables.
3. **APG preparation.** `apg_hourly.py` parses CET/CEST and explicit 2A/2B repeated-hour labels, converts to UTC, validates 15-minute continuity, aggregates four quarter-hours to hourly MWh, and writes `data/processed/apg_austria_hourly.csv`.
4. **Weather preparation.** `temperature_pipeline.py` reads ERA5-Land 2 m temperature, applies population-grid weights from the Eurostat Census Grid, joins to SMARD UTC keys, converts Kelvin to Celsius, and writes country-specific hourly temperature tables. It is historical reanalysis, not an archived day-ahead forecast input.
5. **Calendar features.** `calendar_features.py` converts UTC to `Europe/Berlin` or `Europe/Vienna`, creates local hour, day-of-week, month, weekend, and public-holiday flags using the `holidays` package and documented official sources.
6. **Interventions.** `crisis_features.py` creates inclusive local-date flags for COVID (`2020-03-11` through `2023-05-05`) and post-invasion (`2022-02-24` onward), with source metadata and validation summaries.
7. **Modelling table.** `modelling_datasets.py` requires identical UTC keys across SMARD, temperature, calendar, and crisis tables, validates hourly continuity, offset-aware local timestamps, binary flags, and finite values, then writes the two modelling datasets.
8. **Forecast origin and information set.** `code/04_models/common/forecasting_framework.py` uses Germany’s preceding-day 18:00 local origin and Austria’s preceding-day 08:00 local origin. Training information is limited to rows whose interval end is no later than the UTC cutoff. Target extraction preserves 23-, 24-, and 25-observation DST days.
9. **Model validation.** Baselines, Holt-Winters, ARIMA, SARIMA, and regression-with-ARIMA-errors modules use chronological 2024 validation with bridge paths from the origin to the target day. Metrics are MAE, RMSE, MAPE, evaluated observations, and coverage. Forecast paths retain UTC/local timestamps and information cutoffs.
10. **Summary generation.** `code/04_models/common/build_model_validation_summaries.py` normalizes authoritative family summaries, checks Germany/Austria coverage and unique keys, applies selected specifications, and writes the cross-family all-model and selected-overview CSVs plus the selected registry.

## Official Forecast Benchmarks

- SMARD official load forecasts are retained as external benchmark columns in the prepared modelling tables; they are not model inputs to the classical candidate fits.
- APG actual and forecast files are kept as an Austria historical comparison source. APG covers 2020-2022 at 15-minute resolution and is aggregated before comparison; it does not cover 2024 or 2025.
- The exact benchmark audit is in `official_benchmark_inventory.md`.

## Timestamp and DST Handling

- UTC is the join and ordering key. Local timestamps retain the country’s IANA timezone offset.
- Spring-forward days contain 23 local observations; autumn-repeat days contain 25. No synthetic local hour is added and no repeated autumn observation is dropped.
- Raw local duplicate labels are expected at the autumn transition when their UTC intervals remain unique. Audit inventories distinguish timestamp duplicates from expected local-label repetition.

## Important Python Modules

| File | Purpose | Inputs | Outputs | Model family |
|---|---|---|---|---|
| `code/01_data_audit/audit_data.py` | Audit raw SMARD schemas, values, timestamps, DST, duplicates, and environment. | raw SMARD CSV | audit reports | data audit |
| `code/01_data_audit/load_smard.py` | Audit raw SMARD schemas, values, timestamps, DST, duplicates, and environment. | raw SMARD CSV | audit reports | data audit |
| `code/01_data_audit/test_audit_data.py` | Automated tests for audit data | raw SMARD CSV | pytest pass/fail result | tests |
| `code/01_data_audit/test_load_smard.py` | Automated tests for load smard | raw SMARD CSV | pytest pass/fail result | tests |
| `code/02_preprocessing/apg_hourly.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/calendar_features.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/chapter4_figures.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/compare_apg_smard.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/crisis_features.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/modelling_datasets.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/smard_hourly.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/temperature_pipeline.py` | Create hourly SMARD/APG, weather, calendar, crisis, and modelling tables. | raw and prepared data | prepared data tables | preprocessing |
| `code/02_preprocessing/test_apg_hourly.py` | Automated tests for apg hourly | raw and prepared data | pytest pass/fail result | tests |
| `code/02_preprocessing/test_calendar_features.py` | Automated tests for calendar features | raw and prepared data | pytest pass/fail result | tests |
| `code/02_preprocessing/test_compare_apg_smard.py` | Automated tests for compare apg smard | raw and prepared data | pytest pass/fail result | tests |
| `code/02_preprocessing/test_crisis_features.py` | Automated tests for crisis features | raw and prepared data | pytest pass/fail result | tests |
| `code/02_preprocessing/test_modelling_datasets.py` | Automated tests for modelling datasets | raw and prepared data | pytest pass/fail result | tests |
| `code/02_preprocessing/test_smard_hourly.py` | Automated tests for smard hourly | raw and prepared data | pytest pass/fail result | tests |
| `code/02_preprocessing/test_temperature_pipeline.py` | Automated tests for temperature pipeline | raw and prepared data | pytest pass/fail result | tests |
| `code/03_exploratory_analysis/section_5_1.py` | Produce thesis descriptive, temperature, holiday, and crisis analyses. | prepared modelling data | chapter 5 tables/figures | exploratory analysis |
| `code/03_exploratory_analysis/section_5_2.py` | Produce thesis descriptive, temperature, holiday, and crisis analyses. | prepared modelling data | chapter 5 tables/figures | exploratory analysis |
| `code/03_exploratory_analysis/section_5_2_daily_weekly.py` | Produce thesis descriptive, temperature, holiday, and crisis analyses. | prepared modelling data | chapter 5 tables/figures | exploratory analysis |
| `code/03_exploratory_analysis/section_5_3_monthly_annual.py` | Produce thesis descriptive, temperature, holiday, and crisis analyses. | prepared modelling data | chapter 5 tables/figures | exploratory analysis |
| `code/03_exploratory_analysis/section_5_4_temperature_public_holidays.py` | Produce thesis descriptive, temperature, holiday, and crisis analyses. | prepared modelling data | chapter 5 tables/figures | exploratory analysis |
| `code/03_exploratory_analysis/section_5_5_crisis_policy_periods.py` | Produce thesis descriptive, temperature, holiday, and crisis analyses. | prepared modelling data | chapter 5 tables/figures | exploratory analysis |
| `code/03_exploratory_analysis/test_section_5_1.py` | Automated tests for section 5 1 | prepared modelling data | pytest pass/fail result | tests |
| `code/03_exploratory_analysis/test_section_5_2.py` | Automated tests for section 5 2 | prepared modelling data | pytest pass/fail result | tests |
| `code/03_exploratory_analysis/test_section_5_2_daily_weekly.py` | Automated tests for section 5 2 daily weekly | prepared modelling data | pytest pass/fail result | tests |
| `code/03_exploratory_analysis/test_section_5_3_monthly_annual.py` | Automated tests for section 5 3 monthly annual | prepared modelling data | pytest pass/fail result | tests |
| `code/03_exploratory_analysis/test_section_5_4_temperature_public_holidays.py` | Automated tests for section 5 4 temperature public holidays | prepared modelling data | pytest pass/fail result | tests |
| `code/03_exploratory_analysis/test_section_5_5_crisis_policy_periods.py` | Automated tests for section 5 5 crisis policy periods | prepared modelling data | pytest pass/fail result | tests |
| `code/03_visualization/dwd_station_validation.py` | Validate weather/geospatial visual inputs and create visual outputs. | processed weather/geospatial data | figures and validation tables | visualization |
| `code/03_visualization/temperature_population_map.py` | Validate weather/geospatial visual inputs and create visual outputs. | processed weather/geospatial data | figures and validation tables | visualization |
| `code/03_visualization/test_dwd_station_validation.py` | Automated tests for dwd station validation | processed weather/geospatial data | pytest pass/fail result | tests |
| `code/03_visualization/test_temperature_population_map.py` | Automated tests for temperature population map | processed weather/geospatial data | pytest pass/fail result | tests |
| `code/04_models/arima/arima_models.py` | Screen ARIMA orders, fit selected specifications, and validate 2024. | prepared modelling data and shortlist | ARIMA results | ARIMA |
| `code/04_models/arima/arima_screening.py` | Screen ARIMA orders, fit selected specifications, and validate 2024. | prepared modelling data and shortlist | ARIMA results | ARIMA |
| `code/04_models/arima/arima_validation_2024.py` | Screen ARIMA orders, fit selected specifications, and validate 2024. | prepared modelling data and shortlist | ARIMA results | ARIMA |
| `code/04_models/arima/test_arima_models.py` | Automated tests for arima models | prepared modelling data and shortlist | pytest pass/fail result | tests |
| `code/04_models/arima/test_arima_screening.py` | Automated tests for arima screening | prepared modelling data and shortlist | pytest pass/fail result | tests |
| `code/04_models/arima/test_arima_validation_2024.py` | Automated tests for arima validation 2024 | prepared modelling data and shortlist | pytest pass/fail result | tests |
| `code/04_models/baselines/baseline_validation_2024.py` | Generate deterministic forecast baselines and 2024 validation outputs. | prepared modelling data | baseline forecasts and summaries | baselines |
| `code/04_models/baselines/test_baseline_validation.py` | Automated tests for baseline validation | prepared modelling data | pytest pass/fail result | tests |
| `code/04_models/common/build_model_validation_summaries.py` | Provide shared forecast-origin, information-set, metric, and cross-family summary logic. | prepared modelling data and family result files | normalised summaries and registries | shared framework |
| `code/04_models/common/forecasting_framework.py` | Provide shared forecast-origin, information-set, metric, and cross-family summary logic. | prepared modelling data and family result files | normalised summaries and registries | shared framework |
| `code/04_models/common/test_build_model_validation_summaries.py` | Automated tests for build model validation summaries | prepared modelling data and family result files | pytest pass/fail result | tests |
| `code/04_models/common/test_forecasting_framework.py` | Automated tests for forecasting framework | prepared modelling data and family result files | pytest pass/fail result | tests |
| `code/04_models/exponential_smoothing/holt_winters_models.py` | Screen, shortlist, fit, and validate Holt-Winters variants. | prepared modelling data and shortlists | Holt-Winters results | Holt-Winters |
| `code/04_models/exponential_smoothing/holt_winters_phase2a.py` | Screen, shortlist, fit, and validate Holt-Winters variants. | prepared modelling data and shortlists | Holt-Winters results | Holt-Winters |
| `code/04_models/exponential_smoothing/holt_winters_validation_2024.py` | Screen, shortlist, fit, and validate Holt-Winters variants. | prepared modelling data and shortlists | Holt-Winters results | Holt-Winters |
| `code/04_models/exponential_smoothing/test_holt_winters_models.py` | Automated tests for holt winters models | prepared modelling data and shortlists | pytest pass/fail result | tests |
| `code/04_models/exponential_smoothing/test_holt_winters_phase2a.py` | Automated tests for holt winters phase2a | prepared modelling data and shortlists | pytest pass/fail result | tests |
| `code/04_models/exponential_smoothing/test_holt_winters_validation_2024.py` | Automated tests for holt winters validation 2024 | prepared modelling data and shortlists | pytest pass/fail result | tests |
| `code/04_models/regression_arima_errors/regression_arima_errors_intervention_validation_2024.py` | Build calendar/intervention regressors with ARIMA errors and validate them. | prepared modelling data and calendar/crisis features | regression-with-ARIMA-errors results | regression with ARIMA errors |
| `code/04_models/regression_arima_errors/regression_arima_errors_models.py` | Build calendar/intervention regressors with ARIMA errors and validate them. | prepared modelling data and calendar/crisis features | regression-with-ARIMA-errors results | regression with ARIMA errors |
| `code/04_models/regression_arima_errors/regression_arima_errors_validation_2024.py` | Build calendar/intervention regressors with ARIMA errors and validate them. | prepared modelling data and calendar/crisis features | regression-with-ARIMA-errors results | regression with ARIMA errors |
| `code/04_models/regression_arima_errors/test_regression_arima_errors_intervention_validation_2024.py` | Automated tests for regression arima errors intervention validation 2024 | prepared modelling data and calendar/crisis features | pytest pass/fail result | tests |
| `code/04_models/regression_arima_errors/test_regression_arima_errors_models.py` | Automated tests for regression arima errors models | prepared modelling data and calendar/crisis features | pytest pass/fail result | tests |
| `code/04_models/regression_arima_errors/test_regression_arima_errors_validation_2024.py` | Automated tests for regression arima errors validation 2024 | prepared modelling data and calendar/crisis features | pytest pass/fail result | tests |
| `code/04_models/sarima/benchmark_sarimax_fit_2024.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/benchmark_sarimax_fit_continuation_2024.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/benchmark_sarimax_fit_final_2024.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarima_models.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarima_screening.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarima_validation_2024.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarimax_models.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarimax_state_mini_four_workers_2024.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarimax_state_update.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarimax_state_validation_2024.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/sarimax_validation_2024.py` | Screen SARIMA orders, validate SARIMA, and contain current SARIMAX/state-update work. | prepared modelling data and registries | SARIMA/SARIMAX results | SARIMA/SARIMAX |
| `code/04_models/sarima/test_sarima_models.py` | Automated tests for sarima models | prepared modelling data and registries | pytest pass/fail result | tests |
| `code/04_models/sarima/test_sarima_screening.py` | Automated tests for sarima screening | prepared modelling data and registries | pytest pass/fail result | tests |
| `code/04_models/sarima/test_sarima_validation_2024.py` | Automated tests for sarima validation 2024 | prepared modelling data and registries | pytest pass/fail result | tests |
| `code/04_models/sarima/test_sarimax_models.py` | Automated tests for sarimax models | prepared modelling data and registries | pytest pass/fail result | tests |
| `code/04_models/sarima/test_sarimax_state_update_2024.py` | Automated tests for sarimax state update 2024 | prepared modelling data and registries | pytest pass/fail result | tests |
| `code/04_models/sarima/test_sarimax_validation_2024.py` | Automated tests for sarimax validation 2024 | prepared modelling data and registries | pytest pass/fail result | tests |
| `code/06_methodology/chapter6_visuals.py` | Create methodology/chapter visual material. | results and project metadata | methodology figures | documentation |

## Existing Result Artifacts

The complete CSV/JSON/JSONL result inventory, including authority classification, is in `results_inventory.csv`. `model_status.md` records completion only where authoritative result artifacts support it.

## Consistency Check

- PASS: selected overview keys equal rows selected in model_validation_2024_all_models.csv.
- PASS: selected registry country/family keys equal selected all-model country/family keys.
- PASS: registry selected MAE values match all-model selected rows.
- PASS: per-family authoritative MAE/RMSE/MAPE/coverage values match model_validation_2024_all_models.csv.
- PASS: every all-model source_result_file exists at audit time.
- OBSERVATION: SARIMAX code and mini/benchmark artifacts exist, but SARIMAX is absent from model_validation_2024_all_models.csv and selected_models_registry.csv.

## Audit Limitations

- This audit reads existing files and does not rerun preprocessing, model fitting, forecasting, or tests.
- Binary/container datasets are inventoried at file/container level; NetCDF variables and timestamp metadata are reported where readable.
- A result’s presence in code or a mini/benchmark directory is not treated as authoritative full 2024 validation.
