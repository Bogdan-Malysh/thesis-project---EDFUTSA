# Holt-Winters 2024 Resumable Validation Design

## Goal

Add resumable, progressively persisted, optionally parallel full-year 2024 validation infrastructure for the existing shortlisted Holt-Winters specifications, without running the full validation or implementing the 2025 test.

## Scope

The validation evaluates every existing shortlisted specification for Germany and Austria at every local target date in 2024. Each job uses the existing `forecast_holt_winters` implementation, country-specific forecast origin, information-set cutoff, continuous forecast path, target-day metrics, DST handling, and frozen optimizer metadata from the 2024 shortlist.

The implementation also provides deterministic aggregate 2024 metrics and country-specific selection using:

1. Eligible coverage only; materially incomplete specifications are excluded.
2. Lowest aggregate target-day MAE.
3. Lowest aggregate target-day RMSE.
4. Lowest aggregate target-day MAPE.
5. Existing stable specification order.

Coverage is reported, not optimized. A configurable minimum coverage defaults to complete coverage (`1.0`) for final selection. A mini-validation never produces a final selection.

## Non-goals

- Running the full 2024 validation during this change.
- Any 2025 forecasting or evaluation.
- Changing the shortlist, Holt-Winters specification grid, optimizer fallback/reuse logic, forecast origins, information-set rules, metrics, or DST treatment.
- Using AIC, AICc, or BIC for forecasting-performance selection.

## Architecture

Create `code/04_models/exponential_smoothing/holt_winters_validation_2024.py` as the orchestration boundary. It will contain deterministic job construction, worker execution, checkpoint persistence, aggregate reporting, selection, and a CLI. Existing model fitting remains in `holt_winters_models.py`; existing Phase 2A screening and smoke behavior remains unchanged.

Each job is identified by `(country, target_date, specification_id)`. The manifest is sorted by country configuration order, target date, and existing specification order. The parent process owns all persistence. Worker processes load processed data once through a spawn-safe initializer and return only serializable job results, including forecast rows, target-day metrics, fit metadata, and errors.

The checkpoint store uses atomically replaced CSV files. A successful forecast is written before its job is marked completed. If a process stops between those writes, the job is retried and forecast rows are upserted by job key plus timestamp, so reruns cannot create duplicate rows. Failed jobs remain recorded diagnostically but are eligible for retry. Only completed jobs are skipped.

## Outputs

For a supplied output directory, write:

- `holt_winters_validation_2024_jobs.csv`: one checkpoint row per job, including status, metrics, fit metadata, optimizer metadata, and error details.
- `holt_winters_validation_2024_forecasts.csv`: progressively persisted forecast paths, including bridge and target-day rows.
- `holt_winters_validation_2024_summary.csv`: progressively recomputed country/specification aggregates over completed jobs.
- `holt_winters_selected_specifications_2024.csv`: written only when every one of the 366 local 2024 target dates is completed and the coverage-eligible winner is determinable for both countries.

The mini-validation uses an explicitly separate output directory so it cannot be mistaken for complete-year results or accidentally make a partial run appear complete.

## Parallelism and reproducibility

Use `ProcessPoolExecutor` with the `spawn` multiprocessing context for worker isolation on Windows. `--workers` and the Python API accept the worker count; one worker uses the same worker function sequentially. Results are persisted by the parent in deterministic key order, and CSV outputs are sorted by their natural keys. Parallelism changes scheduling only; each job calls the same forecasting function with the same frozen optimizer configuration and inputs.

## Testing

Tests will cover:

- Stable manifest keys and no duplicate jobs.
- Loading completed checkpoints and skipping only completed jobs.
- Retrying failed jobs.
- Forecast and job upserts without duplicate rows.
- Recovery from a partial checkpoint/output state.
- Identical serialized results for sequential and parallel execution using a deterministic test worker.
- Target-day-only metrics, DST target lengths, aggregate coverage, and the specified selection tie-breakers.
- No selection from an incomplete mini-validation.

## Execution boundary

After implementation, run the existing test suite and a small 2024 mini-validation containing ordinary, spring-DST, and autumn-DST target dates for both countries. Report job counts, failures, metric summaries, measured runtime, and a full-year runtime estimate. Stop before the full 2024 run.
