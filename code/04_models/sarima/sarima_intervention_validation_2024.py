from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import sarimax_validation_2024 as sarimax_validation
from common.forecasting_framework import COUNTRY_CONFIG, PROCESSED, extract_target_day
from sarimax_models import (
    SARIMA_INTERVENTION_SPECIFICATION_IDS,
    specification_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2024
MODEL_FAMILY = "sarima_intervention"
MINI_DATES = sarimax_validation.MINI_DATES
FULL_YEAR_DATE_STRINGS = sarimax_validation.FULL_YEAR_DATE_STRINGS
FULL_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "arima_sarima"
    / "validation_2024"
    / "full_validation_crisis"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "arima_sarima"
    / "validation_2024"
    / "mini_validation_crisis"
)
JOB_KEY_COLUMNS = sarimax_validation.JOB_KEY_COLUMNS
FORECAST_KEY_COLUMNS = sarimax_validation.FORECAST_KEY_COLUMNS
EXOG_COLUMNS = sarimax_validation.EXOG_COLUMNS
JOB_COLUMNS = sarimax_validation.JOB_COLUMNS
FORECAST_COLUMNS = sarimax_validation.FORECAST_COLUMNS
DIAGNOSTIC_COLUMNS = sarimax_validation.DIAGNOSTIC_COLUMNS
DIAGNOSTIC_KEY_COLUMNS = sarimax_validation.DIAGNOSTIC_KEY_COLUMNS
JOB_FILENAME = "sarima_intervention_validation_2024_jobs.csv"
FORECAST_FILENAME = "sarima_intervention_validation_2024_forecasts.csv"
DIAGNOSTIC_FILENAME = "sarima_intervention_validation_2024_diagnostics.csv"
SUMMARY_FILENAME = "sarima_intervention_validation_2024_summary.csv"
SPECIFICATION_FILENAME = "sarima_intervention_specifications_2024.csv"

load_validation_country_data = sarimax_validation.load_validation_country_data
upsert_frame = sarimax_validation.upsert_frame


def _coerce_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _specification_for_country(country: str) -> str:
    try:
        return SARIMA_INTERVENTION_SPECIFICATION_IDS[country]
    except KeyError as error:
        raise ValueError(f"unsupported intervention country: {country}") from error


def _validate_selected_sarima_orders() -> None:
    sarimax_validation._validate_selected_sarima_orders()


def build_manifest(
    target_dates: Iterable[date | str | pd.Timestamp]
    | Mapping[str, Iterable[date | str | pd.Timestamp]] = MINI_DATES,
    *,
    full_year: bool = False,
) -> pd.DataFrame:
    _validate_selected_sarima_orders()
    if isinstance(target_dates, Mapping):
        dates_by_country = {
            country: sorted(
                {_coerce_date(value).isoformat() for value in target_dates.get(country, ())}
            )
            for country in COUNTRY_CONFIG
        }
    else:
        values = sorted({_coerce_date(value).isoformat() for value in target_dates})
        dates_by_country = {country: values for country in COUNTRY_CONFIG}

    for country, values in dates_by_country.items():
        if full_year:
            if set(values) != FULL_YEAR_DATE_STRINGS or len(values) != len(set(values)):
                raise ValueError(
                    "full 2024 SARIMA intervention validation requires every 2024 date"
                )
        elif set(values) != {value.isoformat() for value in MINI_DATES} or len(values) != 3:
            raise ValueError(
                "SARIMA intervention mini-validation requires the approved ordinary/DST dates"
            )

    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        specification_id = _specification_for_country(country)
        order = sarimax_validation.order_for_country(country)
        for target_date in dates_by_country[country]:
            rows.append(
                {
                    "country": country,
                    "target_date": target_date,
                    "specification_id": specification_id,
                    "specification_order": 1,
                    "p": order.p,
                    "d": order.d,
                    "q": order.q,
                    "P": order.P,
                    "D": order.D,
                    "Q": order.Q,
                    "seasonal_period": order.seasonal_period,
                    "trend": "n",
                }
            )
    manifest = pd.DataFrame(rows)
    if manifest.duplicated(list(JOB_KEY_COLUMNS)).any():
        raise ValueError("SARIMA intervention manifest contains duplicate job keys")
    return manifest


def build_intervention_forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
    specification_id: str,
    fitted_result: object,
    context: object | None = None,
) -> pd.DataFrame:
    if specification_id != _specification_for_country(country):
        raise ValueError("intervention specification does not match country")
    path = sarimax_validation.build_sarimax_forecast_path(
        frame,
        country,
        target_date,
        specification_id,
        fitted_result,
        context=context,
        order=sarimax_validation.order_for_country(country),
    )
    path["model_family"] = MODEL_FAMILY
    path["source_kind"] = "sarima_intervention_forecast"
    return path


def _execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    country = str(job["country"])
    specification_id = str(job["specification_id"])
    if specification_id != _specification_for_country(country):
        raise ValueError("intervention specification does not match country")
    result = sarimax_validation.execute_job(job, frame)
    forecasts = result.get("forecasts", [])
    for forecast in forecasts:
        forecast["model_family"] = MODEL_FAMILY
        forecast["source_kind"] = "sarima_intervention_forecast"
    result["forecasts"] = forecasts
    return result


def execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    return _execute_job(job, frame)


_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def _initialize_worker(processed_directory: str) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: load_validation_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _worker_execute(job: Mapping[str, object]) -> dict[str, object]:
    return execute_job(job, _WORKER_FRAMES[str(job["country"])])


def _summary_frame(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    expected_forecast_timestamps: Mapping[tuple[str, str, str], set[str]],
) -> pd.DataFrame:
    summary = sarimax_validation._summary_frame(
        jobs,
        manifest,
        forecasts,
        diagnostics,
        expected_forecast_timestamps,
    )
    summary["model_family"] = MODEL_FAMILY
    return summary


def _write_summary(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "country",
        "model_family",
        "specification_id",
        "specification_order",
        "expected_jobs",
        "completed_jobs",
        "failed_jobs",
        "expected_observations",
        "evaluated_observations",
        "coverage",
        "coverage_eligible",
        "mae",
        "rmse",
        "mape",
    ]
    sarimax_validation._atomic_write(summary, path, columns)


def run_validation(
    target_dates: Iterable[date | str | pd.Timestamp]
    | Mapping[str, Iterable[date | str | pd.Timestamp]]
    | None = None,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
    *,
    full_year: bool = False,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        country: load_validation_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    if full_year:
        dates_by_country = {
            country: sorted(
                value
                for value in frames[country]["local_date"].astype(str).unique()
                if value in FULL_YEAR_DATE_STRINGS
            )
            for country in COUNTRY_CONFIG
        }
    else:
        dates = MINI_DATES if target_dates is None else target_dates
        if isinstance(dates, Mapping):
            dates_by_country = {
                country: [_coerce_date(value).isoformat() for value in dates.get(country, ())]
                for country in COUNTRY_CONFIG
            }
        else:
            values = [_coerce_date(value).isoformat() for value in dates]
            dates_by_country = {country: values for country in COUNTRY_CONFIG}
    manifest = build_manifest(dates_by_country, full_year=full_year)
    manifest["expected_observations"] = [
        len(extract_target_day(frames[country], target_date))
        for country, target_date in zip(manifest["country"], manifest["target_date"])
    ]
    expected_timestamps = {
        (
            str(row["country"]),
            str(row["target_date"]),
            str(row["specification_id"]),
        ): sarimax_validation.expected_forecast_timestamps(
            frames[str(row["country"])],
            str(row["country"]),
            str(row["target_date"]),
        )
        for row in manifest.to_dict("records")
    }
    jobs_path = output / JOB_FILENAME
    forecasts_path = output / FORECAST_FILENAME
    diagnostics_path = output / DIAGNOSTIC_FILENAME
    summary_path = output / SUMMARY_FILENAME
    specifications_path = output / SPECIFICATION_FILENAME
    sarimax_validation._atomic_write(
        pd.DataFrame(
            [
                {
                    "country": country,
                    "specification_order": 1,
                    "specification_id": specification_id,
                    "exog_columns": json.dumps(list(specification_columns(specification_id))),
                }
                for country, specification_id in SARIMA_INTERVENTION_SPECIFICATION_IDS.items()
            ]
        ),
        specifications_path,
        ["country", "specification_order", "specification_id", "exog_columns"],
    )
    jobs = sarimax_validation._load_csv(jobs_path, JOB_COLUMNS)
    forecasts = sarimax_validation._load_csv(forecasts_path, FORECAST_COLUMNS)
    diagnostics = sarimax_validation._load_csv(diagnostics_path, DIAGNOSTIC_COLUMNS)
    completed = sarimax_validation.load_completed_job_keys(
        jobs, forecasts, diagnostics, expected_timestamps
    )
    pending = [
        row
        for row in manifest.to_dict("records")
        if tuple(str(row[column]) for column in JOB_KEY_COLUMNS) not in completed
    ]
    _write_summary(
        _summary_frame(jobs, manifest, forecasts, diagnostics, expected_timestamps),
        summary_path,
    )

    def persist(result: Mapping[str, object]) -> None:
        nonlocal jobs, forecasts, diagnostics
        incoming_job = pd.DataFrame([result["job"]])
        incoming_forecasts = pd.DataFrame(result.get("forecasts", []))
        incoming_diagnostic = pd.DataFrame(
            [result.get("diagnostics") or sarimax_validation._diagnostic_row(result["job"])]
        )
        jobs = upsert_frame(jobs, incoming_job, JOB_KEY_COLUMNS)
        forecasts = sarimax_validation._without_job_rows(forecasts, result["job"])
        forecasts = upsert_frame(forecasts, incoming_forecasts, FORECAST_KEY_COLUMNS)
        diagnostics = upsert_frame(
            diagnostics, incoming_diagnostic, DIAGNOSTIC_KEY_COLUMNS
        )
        sarimax_validation._atomic_write(forecasts, forecasts_path, FORECAST_COLUMNS)
        sarimax_validation._atomic_write(diagnostics, diagnostics_path, DIAGNOSTIC_COLUMNS)
        sarimax_validation._atomic_write(jobs, jobs_path, JOB_COLUMNS)
        _write_summary(
            _summary_frame(jobs, manifest, forecasts, diagnostics, expected_timestamps),
            summary_path,
        )

    started = perf_counter()
    if workers == 1:
        for job in pending:
            persist(execute_job(job, frames[str(job["country"])]))
    elif pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(str(processed_directory),),
        ) as executor:
            for result in executor.map(_worker_execute, pending):
                persist(result)
    elapsed = perf_counter() - started
    summary = _summary_frame(jobs, manifest, forecasts, diagnostics, expected_timestamps)
    _write_summary(summary, summary_path)
    completed = sarimax_validation.load_completed_job_keys(
        jobs, forecasts, diagnostics, expected_timestamps
    )
    manifest_keys = {
        tuple(str(row[column]) for column in JOB_KEY_COLUMNS)
        for row in manifest.to_dict("records")
    }
    return {
        "total_jobs": len(manifest),
        "completed_jobs": len(completed.intersection(manifest_keys)),
        "failed_jobs": int(
            jobs.loc[
                jobs["status"].astype(str).eq("failed")
                & jobs["country"].astype(str).isin(manifest["country"].astype(str))
            ].shape[0]
        ),
        "pending_jobs": len(manifest) - len(completed.intersection(manifest_keys)),
        "forecast_rows": len(forecasts),
        "diagnostic_rows": len(diagnostics),
        "elapsed_seconds": elapsed,
        "summary": summary.to_dict("records"),
        "output_directory": str(output),
    }


def run_mini_validation(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return run_validation(
        MINI_DATES,
        output_directory,
        workers=workers,
        processed_directory=processed_directory,
    )


def run_full_validation(
    output_directory: str | Path = FULL_OUTPUT_DIRECTORY,
    workers: int = 3,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return run_validation(
        output_directory=output_directory,
        workers=workers,
        processed_directory=processed_directory,
        full_year=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bounded 2024 SARIMA crisis-intervention validation"
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--mini", action="store_true")
    stage.add_argument("--full-year", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = (
        run_full_validation(
            output_directory=FULL_OUTPUT_DIRECTORY,
            workers=args.workers,
            processed_directory=args.processed_directory,
        )
        if args.full_year
        else run_mini_validation(
            output_directory=args.output_directory,
            workers=args.workers,
            processed_directory=args.processed_directory,
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
