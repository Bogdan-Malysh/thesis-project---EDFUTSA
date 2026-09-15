from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import (
    BASELINE_MODELS,
    COUNTRY_CONFIG,
    PROCESSED,
    build_information_context,
    calculate_metrics,
    evaluate_target_day,
    extract_target_day,
    forecast_path,
    load_country_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "baselines" / "tables"
FORECAST_DIRECTORY = PROJECT_ROOT / "results" / "baselines" / "forecasts"
VALIDATION_YEAR = 2024
DIAGNOSTIC_NOTE = (
    "Deterministic baseline; convergence, stationarity/invertibility and "
    "residual diagnostics are not applicable."
)


def _validation_dates(frame: pd.DataFrame, year: int) -> list[date]:
    dates = sorted(
        {
            date.fromisoformat(value)
            for value in frame.loc[
                frame["local_date"].str.startswith(f"{year:04d}-"), "local_date"
            ]
        }
    )
    if not dates:
        raise ValueError(f"no validation dates found for {year}")
    return dates


def _write_outputs(
    candidate_rows: list[dict[str, object]],
    forecast_rows: list[pd.DataFrame],
    failure_rows: list[dict[str, object]],
) -> None:
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FORECAST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(
        TABLE_DIRECTORY / "baseline_validation_2024.csv", index=False
    )
    if forecast_rows:
        forecasts = pd.concat(forecast_rows, ignore_index=True)
    else:
        forecasts = pd.DataFrame(
            columns=[
                "country",
                "model_family",
                "target_date",
                "forecast_origin_local",
                "forecast_origin_utc",
                "information_cutoff_utc",
                "timestamp_utc",
                "interval_end_utc",
                "timestamp_local",
                "local_date",
                "actual_load_mwh",
                "forecast_mwh",
                "bridge_used",
                "source_timestamp_utc",
                "source_kind",
                "is_target_day",
                "evaluated",
            ]
        )
    forecasts.to_csv(
        FORECAST_DIRECTORY / "baseline_forecasts_2024.csv", index=False
    )
    failure_columns = [
        "country",
        "target_date",
        "model_family",
        "stage",
        "error_type",
        "message",
    ]
    pd.DataFrame(failure_rows, columns=failure_columns).to_csv(
        TABLE_DIRECTORY / "validation_failures.csv", index=False
    )


def run_validation(
    processed_directory: str | Path = PROCESSED,
    validation_year: int = VALIDATION_YEAR,
) -> dict[str, object]:
    candidate_rows: list[dict[str, object]] = []
    forecast_rows: list[pd.DataFrame] = []
    failure_rows: list[dict[str, object]] = []
    country_summaries: dict[str, dict[str, int]] = {}

    for country in COUNTRY_CONFIG:
        frame = load_country_data(country, processed_directory)
        validation_dates = _validation_dates(frame, validation_year)
        expected_observations = sum(
            len(extract_target_day(frame, target_date)) for target_date in validation_dates
        )
        by_model: dict[str, list[pd.DataFrame]] = {model: [] for model in BASELINE_MODELS}
        for target_date in validation_dates:
            context = build_information_context(frame, country, target_date)
            for model_family in BASELINE_MODELS:
                try:
                    path = forecast_path(
                        frame,
                        country,
                        target_date,
                        model_family,
                        context=context,
                    )
                except Exception as error:  # keep one failure from hiding other candidates
                    failure_rows.append(
                        {
                            "country": country,
                            "target_date": target_date.isoformat(),
                            "model_family": model_family,
                            "stage": "forecast",
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    continue
                by_model[model_family].append(path)
                forecast_rows.append(path)

        country_summaries[country] = {
            "validation_days": len(validation_dates),
            "expected_observations": expected_observations,
        }
        for model_family in BASELINE_MODELS:
            model_paths = by_model[model_family]
            if model_paths:
                model_forecasts = pd.concat(model_paths, ignore_index=True)
                metrics = evaluate_target_day(
                    model_forecasts,
                    expected_observations=expected_observations,
                )
                bridge_count = int(model_forecasts["bridge_used"].sum())
            else:
                metrics = calculate_metrics([], [], expected_observations)
                bridge_count = 0
            diagnostic_note = DIAGNOSTIC_NOTE
            country_failures = sum(
                row["country"] == country and row["model_family"] == model_family
                for row in failure_rows
            )
            if country_failures:
                diagnostic_note += f" {country_failures} forecast failure(s) recorded."
            candidate_rows.append(
                {
                    "country": country,
                    "model_family": model_family,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "evaluated_observations": metrics["evaluated_observations"],
                    "expected_observations": expected_observations,
                    "coverage": metrics["coverage"],
                    "bridge_forecast_observations": bridge_count,
                    "diagnostic_note": diagnostic_note,
                }
            )

    _write_outputs(candidate_rows, forecast_rows, failure_rows)
    return {
        "validation_year": validation_year,
        "countries": country_summaries,
        "candidate_rows": len(candidate_rows),
        "forecast_rows": sum(len(frame) for frame in forecast_rows),
        "failure_rows": len(failure_rows),
    }


def run_smoke_test(processed_directory: str | Path = PROCESSED) -> dict[str, object]:
    ordinary = date(2024, 2, 15)
    spring_dst = date(2024, 3, 31)
    autumn_dst = date(2024, 10, 27)
    summaries: dict[str, object] = {}
    day_names = {
        ordinary: "ordinary",
        spring_dst: "spring_dst",
        autumn_dst: "autumn_dst",
    }
    for country in COUNTRY_CONFIG:
        frame = load_country_data(country, processed_directory)
        counts = {
            name: len(extract_target_day(frame, target_date))
            for target_date, name in day_names.items()
        }
        for target_date in day_names:
            context = build_information_context(frame, country, target_date)
            for model_family in BASELINE_MODELS:
                path = forecast_path(
                    frame, country, target_date, model_family, context=context
                )
                target = path.loc[path["is_target_day"]]
                if len(target) != counts[day_names[target_date]]:
                    raise AssertionError("smoke target-day extraction is inconsistent")
                if target["forecast_mwh"].isna().any():
                    raise AssertionError("smoke forecast contains missing values")
        summaries[country] = counts
    return {
        "countries": list(COUNTRY_CONFIG),
        "ordinary_target_intervals": summaries["Germany"]["ordinary"],
        "spring_dst_target_intervals": summaries["Germany"]["spring_dst"],
        "autumn_dst_target_intervals": summaries["Germany"]["autumn_dst"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1 baselines on 2024 data")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run ordinary and DST smoke checks without writing validation outputs",
    )
    args = parser.parse_args()
    result = run_smoke_test() if args.smoke else run_validation()
    print(pd.Series(result).to_json(indent=2))


if __name__ == "__main__":
    main()
