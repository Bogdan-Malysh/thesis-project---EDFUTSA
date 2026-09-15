# Forecasting Family Reorganization Design

**Status:** Approved for implementation

**Goal:** Separate shared forecasting utilities, baseline code, Holt-Winters code/tests, and their result files without changing forecasting behavior or recomputing existing outputs.

## Approved Structure

Use the existing `code/04_models/` model area:

```text
code/04_models/
    common/
        forecasting_framework.py
        test_forecasting_framework.py
    baselines/
        baseline_validation_2024.py
        test_baseline_validation.py
    exponential_smoothing/
        holt_winters_models.py
        holt_winters_phase2a.py
        test_holt_winters_models.py
        test_holt_winters_phase2a.py
    arima/
    sarima/
    sarimax/
```

The existing empty ARIMA-family directories remain placeholders. No new parallel `holt_winters` code directory is created.

Results move to:

```text
results/
    baselines/
        tables/
        forecasts/
    exponential_smoothing/
        holt_winters/
            tables/
            forecasts/
```

## Import and Path Handling

The shared framework remains the only owner of country loading, forecast origins, information sets, target-day extraction, DST-aware paths, and common metrics. Model-family modules add the existing `code/04_models` directory to their import path before importing `common.forecasting_framework`; no model logic is copied or changed.

The shared framework resolves the project root from its new location. Baseline output constants point to `results/baselines/tables` and `results/baselines/forecasts`. Holt-Winters table constants point to `results/exponential_smoothing/holt_winters/tables`, with the sibling forecasts directory reserved for future validation outputs.

## Preservation and Verification

The three completed baseline CSVs are moved without rewriting:

- `baseline_forecasts_2024.csv`
- `baseline_validation_2024.csv`
- `validation_failures.csv`

Their SHA-256 hashes will be compared before and after the move. The complete test suite will run after restructuring. No screening, smoke tests, full Holt-Winters validation, 2025 data, resumability, or parallelization will be run.
