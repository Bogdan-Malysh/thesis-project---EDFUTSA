from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import sarima_validation_2024 as sarima_validation
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    PROCESSED,
    build_information_context,
    calculate_metrics,
    evaluate_target_day,
    extract_target_day,
    information_set,
)
from sarima_models import SarimaOrder, forecast_values
from sarimax_models import (
    SARIMAX_SPECIFICATION_IDS,
    build_exog,
    fit_sarimax,
    order_for_country,
    specification_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2024
MODEL_FAMILY = "SARIMAX"
MINI_DATES = sarima_validation.MINI_DATES
FULL_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "arima_sarimax" / "validation_2024" / "full_validation"
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "arima_sarimax" / "validation_2024" / "mini_validation"
)
SELECTED_SARIMA = PROJECT_ROOT / "results" / "arima_sarima" / "tables" / "sarima_selected_2024.csv"
FULL_YEAR_DATE_STRINGS = sarima_validation.FULL_YEAR_DATE_STRINGS
JOB_KEY_COLUMNS = ("country", "target_date", "specification_id")
FORECAST_KEY_COLUMNS = JOB_KEY_COLUMNS + ("timestamp_utc",)
EXOG_COLUMNS = (
    "exog_columns",
    "exog_rank",
    "exog_condition_number",
    "transformed_exog_rank",
    "parameter_estimates",
    "standard_errors",
)
JOB_COLUMNS = sarima_validation.JOB_COLUMNS + list(EXOG_COLUMNS)
FORECAST_COLUMNS = sarima_validation.FORECAST_COLUMNS
DIAGNOSTIC_COLUMNS = sarima_validation.DIAGNOSTIC_COLUMNS + list(EXOG_COLUMNS)
JOB_FILENAME = "sarimax_validation_2024_jobs.csv"
FORECAST_FILENAME = "sarimax_validation_2024_forecasts.csv"
DIAGNOSTIC_FILENAME = "sarimax_validation_2024_diagnostics.csv"
SUMMARY_FILENAME = "sarimax_validation_2024_summary.csv"
SPECIFICATION_FILENAME = "sarimax_specifications_2024.csv"
_WORKER_FRAMES: dict[str, pd.DataFrame] = {}
load_validation_country_data = sarima_validation.load_validation_country_data
upsert_frame = sarima_validation.upsert_frame
_without_job_rows = sarima_validation._without_job_rows
_atomic_write = sarima_validation._atomic_write
_load_csv = sarima_validation._load_csv
load_completed_job_keys = sarima_validation.load_completed_job_keys
expected_forecast_timestamps = sarima_validation.expected_forecast_timestamps
_forecast_timestamps_by_job = sarima_validation._forecast_timestamps_by_job
_validate_hourly_forecast_window = sarima_validation._validate_hourly_forecast_window
DIAGNOSTIC_KEY_COLUMNS = sarima_validation.DIAGNOSTIC_KEY_COLUMNS


def _json_value(value: object) -> str:
    return json.dumps(value, default=str, allow_nan=True, sort_keys=True)


def _coerce_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _validate_selected_sarima_orders() -> None:
    selected = pd.read_csv(SELECTED_SARIMA)
    required = {"country", "selected_specification_id"}
    if not required.issubset(selected.columns):
        raise ValueError("selected SARIMA registry is missing required columns")
    actual = dict(
        zip(
            selected["country"].astype(str),
            selected["selected_specification_id"].astype(str),
            strict=True,
        )
    )
    expected = {
        country: order_for_country(country).specification_id for country in COUNTRY_CONFIG
    }
    if actual != expected:
        raise ValueError("selected SARIMA orders do not match the frozen SARIMAX orders")


def build_manifest(
    target_dates: Iterable[date | str | pd.Timestamp]
    | Mapping[str, Iterable[date | str | pd.Timestamp]] = MINI_DATES,
    *,
    full_year: bool = False,
) -> pd.DataFrame:
    _validate_selected_sarima_orders()
    if isinstance(target_dates, Mapping):
        dates_by_country: dict[str, list[str]] = {}
        for country in COUNTRY_CONFIG:
            raw = target_dates.get(country, ())
            values = [_coerce_date(value).isoformat() for value in raw]
            dates_by_country[country] = sorted(set(values))
    else:
        values = [_coerce_date(value).isoformat() for value in target_dates]
        dates_by_country = {country: sorted(set(values)) for country in COUNTRY_CONFIG}
    for country, values in dates_by_country.items():
        if full_year:
            if set(values) != FULL_YEAR_DATE_STRINGS or len(values) != len(set(values)):
                raise ValueError("full 2024 SARIMAX validation requires every 2024 date")
        elif set(values) != {value.isoformat() for value in MINI_DATES} or len(values) != 3:
            raise ValueError("SARIMAX mini-validation requires the approved ordinary/DST dates")

    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        order = order_for_country(country)
        for target_date in dates_by_country[country]:
            for specification_order, specification_id in enumerate(SARIMAX_SPECIFICATION_IDS, 1):
                rows.append(
                    {
                        "country": country,
                        "target_date": target_date,
                        "specification_id": specification_id,
                        "specification_order": specification_order,
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
        raise ValueError("SARIMAX manifest contains duplicate job keys")
    return manifest


def _fit_parameter_metadata(result: object) -> tuple[str, str]:
    if result is None:
        return _json_value({}), _json_value({})
    params = np.asarray(getattr(result, "params", []), dtype=float).reshape(-1)
    bse = np.asarray(getattr(result, "bse", []), dtype=float).reshape(-1)
    names = [str(name) for name in getattr(result, "param_names", [])]
    if len(names) != len(params):
        names = [f"parameter_{index}" for index in range(len(params))]
    estimates = {name: float(value) for name, value in zip(names, params)}
    errors = {
        name: float(value) for name, value in zip(names[: len(bse)], bse)
    }
    return _json_value(estimates), _json_value(errors)


def _fit_metadata(fit: object) -> dict[str, object]:
    base_fit = fit.sarima_fit
    metadata = sarima_validation._fit_metadata(base_fit)
    estimates, errors = _fit_parameter_metadata(fit.fitted_result)
    metadata.update(
        {
            "exog_columns": _json_value(list(fit.exog_columns)),
            "exog_rank": fit.exog_rank,
            "exog_condition_number": fit.exog_condition_number,
            "transformed_exog_rank": fit.transformed_exog_rank,
            "parameter_estimates": estimates,
            "standard_errors": errors,
        }
    )
    return metadata


def _diagnostic_row(job_row: Mapping[str, object]) -> dict[str, object]:
    row = sarima_validation._diagnostic_row(job_row)
    for column in EXOG_COLUMNS:
        row[column] = job_row.get(column)
    return row


def _failed_result(
    row: dict[str, object],
    diagnostics: dict[str, object],
    error: Exception | str,
) -> dict[str, object]:
    row.update(
        {
            "status": "failed",
            "error_message": str(error),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "target_intervals": 0,
            "path_intervals": 0,
            "bridge_intervals": 0,
            "missing_predictions": 0,
            "evaluated_observations": 0,
            "expected_observations": row.get("expected_observations") or 0,
            "coverage": 0.0,
        }
    )
    diagnostics.update(_diagnostic_row(row))
    return {"job": row, "forecasts": [], "diagnostics": diagnostics}


def build_sarimax_forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
    specification_id: str,
    fitted_result: object,
    context: object | None = None,
    order: SarimaOrder | None = None,
) -> pd.DataFrame:
    prepared = sarima_validation._as_prepared(frame)
    local_date = _coerce_date(target_date).isoformat()
    context = (
        build_information_context(prepared, country, local_date)
        if context is None
        else context
    )
    target = extract_target_day(prepared, local_date)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(context.cutoff_utc)
        & prepared["local_date"].le(local_date)
    ].copy()
    if path.empty or not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError(f"no complete SARIMAX forecast path exists for {country} {local_date}")
    sarima_validation._validate_hourly_forecast_window(
        path, target, context.cutoff_utc, local_date
    )
    exog = build_exog(path["timestamp_utc"], country, specification_id)
    if len(exog) != len(path) or tuple(exog.columns) != specification_columns(specification_id):
        raise ValueError("SARIMAX forecast exog is not aligned with the forecast path")
    forecasts = forecast_values(fitted_result, len(path), exog=exog)
    path["country"] = country
    path["model_family"] = MODEL_FAMILY
    path["target_date"] = local_date
    path["specification_id"] = specification_id
    if order is not None:
        path["p"] = order.p
        path["d"] = order.d
        path["q"] = order.q
        path["P"] = order.P
        path["D"] = order.D
        path["Q"] = order.Q
        path["seasonal_period"] = order.seasonal_period
    else:
        for column in ("p", "d", "q", "P", "D", "Q", "seasonal_period"):
            path[column] = pd.NA
    path["trend"] = "n"
    path["forecast_origin_local"] = context.origin_local
    path["forecast_origin_utc"] = context.cutoff_utc.isoformat()
    path["information_cutoff_utc"] = context.cutoff_utc.isoformat()
    path["forecast_mwh"] = forecasts
    path["bridge_used"] = ~path["local_date"].eq(local_date)
    path["source_kind"] = "sarimax_forecast"
    path["is_target_day"] = path["local_date"].eq(local_date)
    path["evaluated"] = path["is_target_day"]
    return path[FORECAST_COLUMNS]


def _execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    row = {column: None for column in JOB_COLUMNS}
    row.update(dict(job))
    diagnostics: dict[str, object] = {}
    fit = None
    try:
        country = str(job["country"])
        target_date = str(job["target_date"])
        specification_id = str(job["specification_id"])
        order = order_for_country(country)
        context = build_information_context(frame, country, target_date)
        available = information_set(frame, context.cutoff_utc)
        target = extract_target_day(frame, target_date)
        train_exog = build_exog(available["timestamp_utc"], country, specification_id)
        row.update(
            {
                "p": order.p,
                "d": order.d,
                "q": order.q,
                "P": order.P,
                "D": order.D,
                "Q": order.Q,
                "seasonal_period": order.seasonal_period,
                "trend": "n",
                "forecast_origin_local": context.origin_local,
                "forecast_origin_utc": context.cutoff_utc.isoformat(),
                "information_cutoff_utc": context.cutoff_utc.isoformat(),
                "training_observations": len(available),
                "expected_observations": len(target),
            }
        )
        fit = fit_sarimax(
            available["actual_load_mwh"].to_numpy(dtype=float),
            train_exog,
            order,
        )
        row.update(_fit_metadata(fit))
        row["error_message"] = fit.error_message
        if not fit.eligible or fit.fitted_result is None:
            return _failed_result(row, _diagnostic_row(row), fit.error_message or "SARIMAX fit failed")
        path = build_sarimax_forecast_path(
            frame,
            country,
            target_date,
            specification_id,
            fit.fitted_result,
            context=context,
            order=order,
        )
        metrics = evaluate_target_day(path, expected_observations=len(target))
        row.update(
            {
                "status": "completed",
                "error_message": None,
                "target_intervals": int(path["is_target_day"].sum()),
                "path_intervals": len(path),
                "bridge_intervals": int(path["bridge_used"].sum()),
                "missing_predictions": int(path["forecast_mwh"].isna().sum()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "evaluated_observations": metrics["evaluated_observations"],
                "coverage": metrics["coverage"],
            }
        )
        diagnostics = _diagnostic_row(row)
        return {"job": row, "forecasts": path.to_dict("records"), "diagnostics": diagnostics}
    except Exception as error:
        if fit is not None:
            row.update(_fit_metadata(fit))
        return _failed_result(row, diagnostics, error)


def execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    return _execute_job(job, frame)


def _initialize_worker(processed_directory: str) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: sarima_validation.load_validation_country_data(country, processed_directory)
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
    summary = sarima_validation._summary_frame(
        jobs,
        manifest,
        forecasts,
        diagnostics=diagnostics,
        expected_forecast_timestamps=expected_forecast_timestamps,
    )
    summary.insert(1, "model_family", MODEL_FAMILY)
    return summary


def _write_summary(
    summary: pd.DataFrame,
    path: Path,
) -> None:
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
    sarima_validation._atomic_write(summary, path, columns)


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
    _validate_selected_sarima_orders()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        country: sarima_validation.load_validation_country_data(country, processed_directory)
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
        ): sarima_validation.expected_forecast_timestamps(
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
    specification_rows = []
    for order, specification_id in enumerate(SARIMAX_SPECIFICATION_IDS, 1):
        specification_rows.append(
            {
                "specification_order": order,
                "specification_id": specification_id,
                "exog_columns": _json_value(list(specification_columns(specification_id))),
            }
        )
    sarima_validation._atomic_write(
        pd.DataFrame(specification_rows),
        specifications_path,
        ["specification_order", "specification_id", "exog_columns"],
    )
    jobs = sarima_validation._load_csv(jobs_path, JOB_COLUMNS)
    forecasts = sarima_validation._load_csv(forecasts_path, FORECAST_COLUMNS)
    diagnostics = sarima_validation._load_csv(diagnostics_path, DIAGNOSTIC_COLUMNS)
    completed = sarima_validation.load_completed_job_keys(
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
            [result.get("diagnostics") or _diagnostic_row(result["job"])]
        )
        jobs = sarima_validation.upsert_frame(jobs, incoming_job, JOB_KEY_COLUMNS)
        forecasts = sarima_validation._without_job_rows(forecasts, result["job"])
        forecasts = sarima_validation.upsert_frame(
            forecasts, incoming_forecasts, FORECAST_KEY_COLUMNS
        )
        diagnostics = sarima_validation.upsert_frame(
            diagnostics, incoming_diagnostic, sarima_validation.DIAGNOSTIC_KEY_COLUMNS
        )
        sarima_validation._atomic_write(forecasts, forecasts_path, FORECAST_COLUMNS)
        sarima_validation._atomic_write(diagnostics, diagnostics_path, DIAGNOSTIC_COLUMNS)
        sarima_validation._atomic_write(jobs, jobs_path, JOB_COLUMNS)
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
    completed = sarima_validation.load_completed_job_keys(
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
    workers: int = 8,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return run_validation(
        output_directory=output_directory,
        workers=workers,
        processed_directory=processed_directory,
        full_year=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded SARIMAX 2024 validation")
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--mini", action="store_true")
    stage.add_argument("--full-year", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = (
        run_full_validation(
            output_directory=args.output_directory,
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
