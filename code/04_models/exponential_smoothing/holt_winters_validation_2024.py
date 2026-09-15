from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import (
    COUNTRY_CONFIG,
    PROCESSED,
    calculate_metrics,
    country_forecast_origin,
    evaluate_target_day,
    extract_target_day,
    load_country_data,
)
from holt_winters_models import (
    HoltWintersForecastError,
    HoltWintersSpecification,
    OptimizerConfiguration,
    forecast_holt_winters,
    fit_record_to_row,
)
from holt_winters_phase2a import (
    SHORTLIST_OUTPUT,
    _optimizer_configuration_from_row,
    _specification_from_row,
    load_frozen_shortlists,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "exponential_smoothing" / "holt_winters" / "validation_2024"
)
JOB_FILENAME = "holt_winters_validation_2024_jobs.csv"
FORECAST_FILENAME = "holt_winters_validation_2024_forecasts.csv"
SUMMARY_FILENAME = "holt_winters_validation_2024_summary.csv"
SELECTION_FILENAME = "holt_winters_selected_specifications_2024.csv"

JOB_KEY_COLUMNS = ("country", "target_date", "specification_id")
FORECAST_KEY_COLUMNS = JOB_KEY_COLUMNS + ("timestamp_utc",)
JOB_COLUMNS = (
    "country",
    "target_date",
    "specification_id",
    "seasonal_form",
    "seasonal_periods",
    "damped_trend",
    "specification_order",
    "forecast_origin_local",
    "forecast_origin_utc",
    "training_observations",
    "aic",
    "aicc",
    "bic",
    "convergence_status",
    "fit_status",
    "fitting_time_seconds",
    "selection_criterion",
    "selection_value",
    "eligible",
    "error_message",
    "error_type",
    "optimizer_used",
    "optimizer_attempt_count",
    "optimizer_attempts_json",
    "status",
    "target_intervals",
    "path_intervals",
    "bridge_intervals",
    "missing_predictions",
    "mae",
    "rmse",
    "mape",
    "evaluated_observations",
    "expected_observations",
    "coverage",
)


@dataclass(frozen=True)
class ValidationJob:
    country: str
    target_date: str
    specification_id: str
    seasonal_form: str
    seasonal_periods: int
    damped_trend: bool
    specification_order: int
    optimizer_used: str
    minimize_kwargs: dict[str, object]

    @property
    def job_key(self) -> tuple[str, str, str]:
        return self.country, self.target_date, self.specification_id

    @property
    def specification(self) -> HoltWintersSpecification:
        return HoltWintersSpecification(
            specification_id=self.specification_id,
            seasonal_form=self.seasonal_form,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
        )

    @property
    def optimizer_configuration(self) -> OptimizerConfiguration:
        return OptimizerConfiguration(self.optimizer_used, dict(self.minimize_kwargs))


@dataclass(frozen=True)
class ValidationResult:
    job_row: dict[str, object]
    forecast: pd.DataFrame


def _coerce_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def _target_date_strings(target_dates: Iterable[date | str | pd.Timestamp]) -> list[str]:
    dates = sorted({_coerce_date(value).isoformat() for value in target_dates})
    invalid = [value for value in dates if not value.startswith("2024-")]
    if invalid:
        raise ValueError(f"2024 validation dates required, received: {invalid}")
    return dates


def build_validation_jobs(
    shortlists: pd.DataFrame,
    target_dates: Iterable[date | str | pd.Timestamp],
) -> list[ValidationJob]:
    required = {
        "country",
        "specification_id",
        "seasonal_form",
        "seasonal_periods",
        "damped_trend",
        "specification_order",
        "optimizer_used",
        "optimizer_attempts_json",
    }
    missing = sorted(required.difference(shortlists.columns))
    if missing:
        raise ValueError(f"shortlist is missing required columns: {missing}")

    dates = _target_date_strings(target_dates)
    rows_by_country = {
        country: shortlists.loc[shortlists["country"].eq(country)].sort_values(
            "specification_order", kind="mergesort"
        )
        for country in COUNTRY_CONFIG
    }
    jobs: list[ValidationJob] = []
    seen: set[tuple[str, str, str]] = set()
    for country in COUNTRY_CONFIG:
        for target_date in dates:
            for _, row in rows_by_country[country].iterrows():
                specification = _specification_from_row(row)
                optimizer = _optimizer_configuration_from_row(row)
                job = ValidationJob(
                    country=country,
                    target_date=target_date,
                    specification_id=specification.specification_id,
                    seasonal_form=specification.seasonal_form,
                    seasonal_periods=specification.seasonal_periods,
                    damped_trend=specification.damped_trend,
                    specification_order=int(row["specification_order"]),
                    optimizer_used=optimizer.optimizer,
                    minimize_kwargs=dict(optimizer.minimize_kwargs),
                )
                if job.job_key in seen:
                    raise ValueError(f"duplicate validation job: {job.job_key}")
                seen.add(job.job_key)
                jobs.append(job)
    return jobs


def _read_csv(path: Path, columns: Sequence[str] = ()) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path)
    if not columns:
        return frame
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, list(columns)]


def _atomic_write_csv(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = pd.NA
    output = output.loc[:, list(columns)]
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            output.to_csv(temporary, index=False)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def _normalize_job_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in JOB_KEY_COLUMNS:
        if column in normalized:
            normalized[column] = normalized[column].astype(str)
    return normalized


def _normalize_forecast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        return normalized
    for column in JOB_KEY_COLUMNS:
        normalized[column] = normalized[column].astype(str)
    normalized["timestamp_utc"] = pd.to_datetime(
        normalized["timestamp_utc"], utc=True
    ).map(pd.Timestamp.isoformat)
    return normalized


def _upsert(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    if existing.empty and incoming.empty:
        return existing.copy()
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    for column in key_columns:
        combined[column] = combined[column].astype(str)
    combined = combined.drop_duplicates(subset=list(key_columns), keep="last")
    return combined.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)


class ValidationStore:
    def __init__(self, output_directory: str | Path):
        self.output_directory = Path(output_directory)
        self.jobs_path = self.output_directory / JOB_FILENAME
        self.forecasts_path = self.output_directory / FORECAST_FILENAME
        self.summary_path = self.output_directory / SUMMARY_FILENAME
        self.selection_path = self.output_directory / SELECTION_FILENAME

    def read_jobs(self) -> pd.DataFrame:
        return _normalize_job_frame(_read_csv(self.jobs_path, JOB_COLUMNS))

    def read_forecasts(self) -> pd.DataFrame:
        return _normalize_forecast_frame(_read_csv(self.forecasts_path))

    def completed_keys(self) -> set[tuple[str, str, str]]:
        jobs = self.read_jobs()
        forecasts = self.read_forecasts()
        if jobs.empty or forecasts.empty:
            return set()
        completed = jobs.loc[jobs["status"].eq("completed"), list(JOB_KEY_COLUMNS)]
        forecast_keys = forecasts.loc[:, list(JOB_KEY_COLUMNS)].drop_duplicates()
        joined = completed.merge(forecast_keys, on=list(JOB_KEY_COLUMNS), how="inner")
        return {tuple(row) for row in joined.itertuples(index=False, name=None)}

    def persist_result(
        self,
        job_row: dict[str, object],
        forecast: pd.DataFrame,
    ) -> None:
        status = str(job_row.get("status", ""))
        if status == "completed" and forecast.empty:
            raise ValueError("completed validation jobs must include forecast rows")
        incoming_jobs = _normalize_job_frame(pd.DataFrame([job_row]))
        existing_jobs = self.read_jobs()
        jobs = _upsert(existing_jobs, incoming_jobs, JOB_KEY_COLUMNS)

        if not forecast.empty:
            incoming_forecasts = _normalize_forecast_frame(forecast)
            forecasts = _upsert(
                self.read_forecasts(), incoming_forecasts, FORECAST_KEY_COLUMNS
            )
            _atomic_write_csv(forecasts, self.forecasts_path, forecasts.columns)
        elif self.forecasts_path.exists():
            forecasts = self.read_forecasts()
        else:
            forecasts = pd.DataFrame()

        _atomic_write_csv(jobs, self.jobs_path, JOB_COLUMNS)

    def write_summary(self, summary: pd.DataFrame) -> None:
        _atomic_write_csv(summary, self.summary_path, summary.columns)

    def write_selection(self, selection: pd.DataFrame) -> None:
        _atomic_write_csv(selection, self.selection_path, selection.columns)

    def clear_selection(self) -> None:
        self.selection_path.unlink(missing_ok=True)


def _base_job_row(job: ValidationJob, status: str) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in JOB_COLUMNS}
    row.update(
        {
            "country": job.country,
            "target_date": job.target_date,
            "specification_id": job.specification_id,
            "seasonal_form": job.seasonal_form,
            "seasonal_periods": job.seasonal_periods,
            "damped_trend": job.damped_trend,
            "specification_order": job.specification_order,
            "optimizer_used": job.optimizer_used,
            "optimizer_attempt_count": None,
            "optimizer_attempts_json": "[]",
            "status": status,
            "error_type": None,
        }
    )
    return row


def _execute_validation_job(
    job: ValidationJob,
    frame: pd.DataFrame,
) -> ValidationResult:
    target = extract_target_day(frame, job.target_date)
    expected_observations = len(target)
    origin = country_forecast_origin(job.target_date, job.country)
    try:
        result = forecast_holt_winters(
            frame,
            job.country,
            job.target_date,
            job.specification,
            optimizer_configuration=job.optimizer_configuration,
        )
        forecast = result.forecast
        metrics = evaluate_target_day(
            forecast, expected_observations=expected_observations
        )
        row = _base_job_row(job, "completed")
        row.update(fit_record_to_row(result.fit_record, job.country, origin))
        row.update(
            {
                "status": "completed",
                "target_intervals": int(metrics["evaluated_observations"]),
                "path_intervals": len(forecast),
                "bridge_intervals": int((~forecast["is_target_day"]).sum()),
                "missing_predictions": int(forecast["forecast_mwh"].isna().sum()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "evaluated_observations": metrics["evaluated_observations"],
                "expected_observations": expected_observations,
                "coverage": metrics["coverage"],
                "error_type": None,
                "error_message": None,
            }
        )
        return ValidationResult(row, forecast.copy())
    except HoltWintersForecastError as error:
        row = _base_job_row(job, "failed")
        row.update(fit_record_to_row(error.record, job.country, origin))
        row.update(
            {
                "status": "failed",
                "expected_observations": expected_observations,
                "coverage": 0.0,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        return ValidationResult(row, pd.DataFrame())
    except Exception as error:
        row = _base_job_row(job, "failed")
        row.update(
            {
                "expected_observations": expected_observations,
                "coverage": 0.0,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        return ValidationResult(row, pd.DataFrame())


_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def _initialize_worker(processed_directory: str | Path) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _run_validation_job(job: ValidationJob) -> ValidationResult:
    if job.country not in _WORKER_FRAMES:
        raise RuntimeError("validation worker was not initialized")
    return _execute_validation_job(job, _WORKER_FRAMES[job.country])


def iter_job_results(
    jobs: Sequence[ValidationJob],
    processed_directory: str | Path = PROCESSED,
    workers: int = 1,
    worker_function: Callable[[ValidationJob], Any] | None = None,
) -> Iterator[Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    worker = _run_validation_job if worker_function is None else worker_function
    initializer = _initialize_worker if worker_function is None else None
    initargs = (str(processed_directory),) if initializer is not None else ()

    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        for job in jobs:
            yield worker(job)
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def execute_jobs(
    jobs: Sequence[ValidationJob],
    processed_directory: str | Path = PROCESSED,
    workers: int = 1,
    worker_function: Callable[[ValidationJob], Any] | None = None,
) -> list[Any]:
    return list(
        iter_job_results(
            jobs,
            processed_directory=processed_directory,
            workers=workers,
            worker_function=worker_function,
        )
    )


def build_summary(
    manifest: Sequence[ValidationJob],
    job_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    expected_observations_by_country: dict[str, int],
    minimum_coverage: float = 1.0,
) -> pd.DataFrame:
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")
    rows: list[dict[str, object]] = []
    normalized_jobs = _normalize_job_frame(job_frame)
    normalized_forecasts = _normalize_forecast_frame(forecast_frame)
    if not normalized_forecasts.empty:
        target_mask = normalized_forecasts["is_target_day"].astype(str).str.lower().eq("true")
        normalized_forecasts = normalized_forecasts.loc[target_mask].copy()

    for country in COUNTRY_CONFIG:
        country_jobs = [job for job in manifest if job.country == country]
        for specification_id in dict.fromkeys(
            job.specification_id for job in country_jobs
        ):
            specification = next(
                job for job in country_jobs if job.specification_id == specification_id
            )
            expected_jobs = sum(
                job.specification_id == specification_id for job in country_jobs
            )
            completed_jobs = int(
                (
                    normalized_jobs["country"].eq(country)
                    & normalized_jobs["specification_id"].eq(specification_id)
                    & normalized_jobs["status"].eq("completed")
                ).sum()
            ) if not normalized_jobs.empty else 0
            failed_jobs = int(
                (
                    normalized_jobs["country"].eq(country)
                    & normalized_jobs["specification_id"].eq(specification_id)
                    & normalized_jobs["status"].eq("failed")
                ).sum()
            ) if not normalized_jobs.empty else 0
            target = normalized_forecasts.loc[
                normalized_forecasts["country"].eq(country)
                & normalized_forecasts["specification_id"].eq(specification_id)
            ] if not normalized_forecasts.empty else pd.DataFrame()
            metrics = calculate_metrics(
                target["actual_load_mwh"] if not target.empty else [],
                target["forecast_mwh"] if not target.empty else [],
                expected_observations=expected_observations_by_country[country],
            )
            coverage = float(metrics["coverage"])
            rows.append(
                {
                    "country": country,
                    "specification_id": specification_id,
                    "specification_order": specification.specification_order,
                    "expected_jobs": expected_jobs,
                    "completed_jobs": completed_jobs,
                    "failed_jobs": failed_jobs,
                    "expected_observations": expected_observations_by_country[country],
                    "evaluated_observations": metrics["evaluated_observations"],
                    "coverage": coverage,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "mape": metrics["mape"],
                    "coverage_eligible": coverage >= minimum_coverage,
                }
            )
    return pd.DataFrame(rows)


def select_final_specifications(
    summary: pd.DataFrame,
    minimum_coverage: float = 1.0,
) -> pd.DataFrame:
    required = {
        "country",
        "specification_id",
        "specification_order",
        "coverage",
        "rmse",
        "mae",
        "mape",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")
    candidates = summary.copy()
    numeric_columns = ["coverage", "rmse", "mae", "mape", "specification_order"]
    for column in numeric_columns:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    eligible = (
        candidates["coverage"].ge(minimum_coverage)
        & np.isfinite(candidates["rmse"])
        & np.isfinite(candidates["mae"])
        & np.isfinite(candidates["mape"])
    )
    candidates = candidates.loc[eligible].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values(
            ["country", "mae", "rmse", "mape", "specification_order"],
            kind="mergesort",
        )
        .groupby("country", sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _all_validation_dates(frame: pd.DataFrame) -> list[str]:
    dates = sorted(
        value for value in frame["local_date"].astype(str).unique() if value.startswith("2024-")
    )
    if not dates:
        raise ValueError("no 2024 target dates found")
    return dates


def _runtime_estimate(
    jobs: Sequence[ValidationJob],
    job_frame: pd.DataFrame,
    full_job_count: int,
    full_dates_per_country: int,
    workers: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    fitting = _normalize_job_frame(job_frame)
    fitting["fitting_time_seconds"] = pd.to_numeric(
        fitting.get("fitting_time_seconds", pd.Series(dtype=float)), errors="coerce"
    )
    successful = fitting.loc[
        fitting.get("status", pd.Series(dtype=str)).eq("completed")
        & np.isfinite(fitting["fitting_time_seconds"])
    ] if not fitting.empty else pd.DataFrame()
    if successful.empty:
        projected_fit_seconds = None
        projected_parallel_seconds = None
    else:
        means = successful.groupby(["country", "specification_id"])[
            "fitting_time_seconds"
        ].mean()
        projected_fit_seconds = float(means.sum() * full_dates_per_country)
        projected_parallel_seconds = projected_fit_seconds / workers
    return {
        "measured_elapsed_seconds": elapsed_seconds,
        "measured_successful_fits": int(len(successful)),
        "projected_full_year_fits": full_job_count,
        "projected_full_year_sequential_fit_seconds": projected_fit_seconds,
        "projected_full_year_parallel_fit_seconds": projected_parallel_seconds,
        "workers": workers,
        "full_dates_per_country": full_dates_per_country,
        "basis": "mean successful fit time by country/specification scaled to all 2024 target dates",
    }


def run_validation(
    processed_directory: str | Path = PROCESSED,
    shortlist_path: str | Path = SHORTLIST_OUTPUT,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    target_dates: Iterable[date | str | pd.Timestamp] | None = None,
    workers: int = 1,
    minimum_coverage: float = 1.0,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    frames = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    shortlists = load_frozen_shortlists(shortlist_path)
    all_dates = _all_validation_dates(frames[next(iter(COUNTRY_CONFIG))])
    selected_dates = all_dates if target_dates is None else _target_date_strings(target_dates)
    manifest = build_validation_jobs(shortlists, selected_dates)
    full_manifest = build_validation_jobs(shortlists, all_dates)
    expected_observations = {
        country: sum(
            len(extract_target_day(frames[country], target_date))
            for target_date in selected_dates
        )
        for country in COUNTRY_CONFIG
    }
    store = ValidationStore(output_directory)
    initial_summary = build_summary(
        manifest,
        store.read_jobs(),
        store.read_forecasts(),
        expected_observations,
        minimum_coverage,
    )
    store.write_summary(initial_summary)
    completed_keys = store.completed_keys()
    pending = [job for job in manifest if job.job_key not in completed_keys]
    started = perf_counter()
    for result in iter_job_results(
        pending,
        processed_directory=processed_directory,
        workers=workers,
    ):
        if not isinstance(result, ValidationResult):
            raise TypeError("validation workers must return ValidationResult")
        store.persist_result(result.job_row, result.forecast)
        summary = build_summary(
            manifest,
            store.read_jobs(),
            store.read_forecasts(),
            expected_observations,
            minimum_coverage,
        )
        store.write_summary(summary)
    elapsed = perf_counter() - started

    job_frame = store.read_jobs()
    summary = build_summary(
        manifest,
        job_frame,
        store.read_forecasts(),
        expected_observations,
        minimum_coverage,
    )
    store.write_summary(summary)
    completed = store.completed_keys()
    all_complete = set(job.job_key for job in manifest) == completed
    full_year_complete = all_complete and selected_dates == all_dates and len(all_dates) == 366
    if full_year_complete:
        selection = select_final_specifications(summary, minimum_coverage)
        if set(selection.get("country", [])) == set(COUNTRY_CONFIG):
            store.write_selection(selection)
        else:
            store.clear_selection()
    else:
        store.clear_selection()

    failed_count = int(job_frame.get("status", pd.Series(dtype=str)).eq("failed").sum())
    completed_count = len(completed)
    runtime = _runtime_estimate(
        manifest,
        job_frame,
        full_job_count=len(full_manifest),
        full_dates_per_country=len(all_dates),
        workers=workers,
        elapsed_seconds=elapsed,
    )
    return {
        "requested_jobs": len(manifest),
        "completed_jobs": completed_count,
        "failed_jobs": failed_count,
        "pending_jobs": len(manifest) - completed_count,
        "complete": all_complete,
        "full_year_complete": full_year_complete,
        "summary": summary.to_dict(orient="records"),
        "runtime": runtime,
        "output_directory": str(Path(output_directory)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable 2024 Holt-Winters validation"
    )
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--shortlist", type=Path, default=SHORTLIST_OUTPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--target-date",
        action="append",
        dest="target_dates",
        help="2024 local target date; repeat for a mini-validation",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimum-coverage", type=float, default=1.0)
    args = parser.parse_args()
    result = run_validation(
        processed_directory=args.processed_directory,
        shortlist_path=args.shortlist,
        output_directory=args.output_directory,
        target_dates=args.target_dates,
        workers=args.workers,
        minimum_coverage=args.minimum_coverage,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
