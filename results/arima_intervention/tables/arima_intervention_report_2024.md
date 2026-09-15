# 2024 ARIMA Crisis Experiment

Model family: `arima_intervention`.
Specification: `arima_p2_d1_q2_crisis`.
Base error model: ARIMA(2,1,2), trend=`n`, jointly estimated regression coefficients and ARIMA errors, with stationarity and invertibility enforcement enabled.

## Exogenous Variables

Only `is_covid_period` and `is_post_invasion` were used. COVID is inclusive from 2020-03-11 through 2023-05-05; post-invasion is inclusive from 2022-02-24 onward. No calendar variables, temperature, SMARD/APG forecasts, future load, future observed temperature, or 2025 data were used.

## Validation Design

Daily refitting used the established rolling information set, Germany's 18:00 previous-day origin, Austria's 08:00 previous-day origin, bridge forecasts through the target day, target-day-only scoring, and the existing DST-aware target extraction. Full validation completed 366 days per country and 8,784 target observations per country.

## Accuracy Comparison

### Germany

- Plain ARIMA(2,1,2): MAE 7,678.8906 MWh; RMSE 8,941.3201 MWh; MAPE 15.4199%; coverage 1.0000.
- Crisis ARIMA: MAE 7,679.0705 MWh; RMSE 8,941.4542 MWh; MAPE 15.4203%; coverage 1.0000.
- Change: MAE 0.1798 MWh (0.0023%); RMSE 0.1341 MWh (0.0015%); MAPE 0.0003 percentage points.

### Austria

- Plain ARIMA(2,1,2): MAE 1,037.6423 MWh; RMSE 1,196.2127 MWh; MAPE 16.0730%; coverage 1.0000.
- Crisis ARIMA: MAE 1,037.6380 MWh; RMSE 1,196.2213 MWh; MAPE 16.0730%; coverage 1.0000.
- Change: MAE -0.0043 MWh (-0.0004%); RMSE 0.0086 MWh (0.0007%); MAPE -0.0000 percentage points.

## Fit and Residual Diagnostics

Detailed convergence/retry counts, crisis coefficient summaries, and residual diagnostics are in the companion CSV tables. A coefficient's `absolute_t_ge_1_96_jobs` count uses the absolute estimate divided by its finite standard error.

## Central Summary Status

The experiment is retained as an unselected comparator. It does not replace the already selected plain ARIMA specification and is not added to the frozen model registry.
