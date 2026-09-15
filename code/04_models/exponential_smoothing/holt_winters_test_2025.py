from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import holt_winters_validation_2024 as validation_2024
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    PROCESSED,
    country_forecast_origin,
    information_set,
    load_country_data,
)
from holt_winters_phase2a import (
    SHORTLIST_OUTPUT,
    _optimizer_configuration_from_row,
    _specification_from_row,
    load_frozen_shortlists,
)
from holt_winters_validation_2024 import (
    ValidationJob,
    ValidationResult,
    build_summary,
    iter_job_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPECIFICATION_ID = "hw_mul_s168_damped"
VALIDATION_YEAR = 2025
EXPECTED_DAYS = 365
EXPECTED_OBSERVATIONS = 8760
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "exponential_smoothing" / "holt_winters" / "test_2025"
)
JOB_FILENAME = "holt_winters_test_2025_jobs.csv"
FORECAST_FILENAME = "holt_winters_test_2025_forecasts.csv"
SUMMARY_FILENAME = "holt_winters_test_2025_summary.csv"
AUTHORITATIVE_SELECTION_PATH = (
    validation_2024.DEFAULT_OUTPUT_DIRECTORY / validation_2024.SELECTION_FILENAME
)
TABLE_COLUMNS = [
    "country",
    "model_family",
    "specification_id",
    "mae",
    "rmse",
    "mape",
    "n_observations",
    "coverage",
]
TABLE_KEY_COLUMNS = ["country", "model_family", "specification_id"]
COUNTRY_ORDER = {country: index for index, country in enumerate(COUNTRY_CONFIG)}


class TestValidationStore(validation_2024.ValidationStore):
    def __init__(self, output_directory: str | Path = OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)
        self.jobs_path = self.output_directory / JOB_FILENAME
        self.forecasts_path = self.output_directory / FORECAST_FILENAME
        self.summary_path = self.output_directory / SUMMARY_FILENAME
        self.selection_path = self.output_directory / "holt_winters_test_2025_selection.csv"


def verify_authoritative_selection(
    selection_path: str | Path = AUTHORITATIVE_SELECTION_PATH,
) -> pd.DataFrame:
    selected = pd.read_csv(selection_path)
    required = {
        "country",
        "specification_id",
        "expected_jobs",
        "completed_jobs",
        "failed_jobs",
        "coverage",
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"authoritative Holt-Winters selection is missing columns: {missing}")
    selected = selected.loc[selected["specification_id"].eq(SPECIFICATION_ID)].copy()
    if set(selected["country"].astype(str)) != set(COUNTRY_CONFIG) or len(selected) != 2:
        raise ValueError("authoritative selection must contain hw_mul_s168_damped for both countries")
    if not selected["expected_jobs"].eq(366).all():
        raise ValueError("authoritative 2024 Holt-Winters selection must expect 366 jobs per country")
    if not selected["completed_jobs"].eq(366).all():
        raise ValueError("authoritative 2024 Holt-Winters selection is incomplete")
    if not selected["failed_jobs"].eq(0).all() or not selected["coverage"].eq(1.0).all():
        raise ValueError("authoritative 2024 Holt-Winters selection contains failures or incomplete coverage")
    return selected


def load_test_shortlists(shortlist_path: str | Path = SHORTLIST_OUTPUT) -> pd.DataFrame:
    verify_authoritative_selection()
    shortlists = load_frozen_shortlists(shortlist_path)
    selected = shortlists.loc[shortlists["specification_id"].eq(SPECIFICATION_ID)].copy()
    if len(selected) != len(COUNTRY_CONFIG) or set(selected["country"]) != set(COUNTRY_CONFIG):
        raise ValueError("frozen shortlist must contain one hw_mul_s168_damped row per country")
    for _, row in selected.iterrows():
        specification = _specification_from_row(row)
        if (
            specification.seasonal_form != "mul"
            or specification.seasonal_periods != 168
            or specification.damped_trend is not True
        ):
            raise ValueError("frozen shortlist settings do not match hw_mul_s168_damped")
        optimizer = _optimizer_configuration_from_row(row)
        if not optimizer.optimizer or not isinstance(optimizer.minimize_kwargs, dict):
            raise ValueError("frozen shortlist has invalid optimizer configuration")
    return selected.reset_index(drop=True)


def test_dates(frame: pd.DataFrame) -> list[date]:
    values = sorted(
        {
            date.fromisoformat(value)
            for value in frame.loc[
                frame["local_date"].astype(str).str.startswith("2025-"), "local_date"
            ]
        }
    )
    expected = [date(2025, 1, 1) + timedelta(days=offset) for offset in range(EXPECTED_DAYS)]
    if values != expected:
        raise ValueError("2025 Holt-Winters test requires every local date from 2025-01-01 through 2025-12-31")
    return values


def build_test_jobs(
    shortlists: pd.DataFrame,
    target_dates: Iterable[date | str | pd.Timestamp],
) -> list[ValidationJob]:
    dates = sorted({pd.Timestamp(value).date().isoformat() for value in target_dates})
    if not dates or any(not value.startswith("2025-") for value in dates):
        raise ValueError("Holt-Winters test jobs require 2025 target dates")
    rows = {
        country: shortlists.loc[shortlists["country"].eq(country)]
        for country in COUNTRY_CONFIG
    }
    jobs: list[ValidationJob] = []
    seen: set[tuple[str, str, str]] = set()
    for country in COUNTRY_CONFIG:
        if len(rows[country]) != 1:
            raise ValueError(f"frozen shortlist must contain one row for {country}")
        row = rows[country].iloc[0]
        specification = _specification_from_row(row)
        optimizer = _optimizer_configuration_from_row(row)
        for target_date in dates:
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
                raise ValueError(f"duplicate Holt-Winters test job: {job.job_key}")
            seen.add(job.job_key)
            jobs.append(job)
    return jobs


def expected_training_observations(job: ValidationJob, frame: pd.DataFrame) -> int:
    origin = country_forecast_origin(job.target_date, job.country)
    return len(information_set(frame, origin))


def validate_job_result(
    job_row: dict[str, object] | pd.Series,
    forecast: pd.DataFrame,
    frame: pd.DataFrame,
) -> None:
    if forecast.empty:
        raise ValueError("completed Holt-Winters job has no forecast rows")
    country = str(job_row["country"])
    target_date = str(job_row["target_date"])
    if str(job_row["specification_id"]) != SPECIFICATION_ID:
        raise ValueError("Holt-Winters result has the wrong specification")
    if not forecast["model_family"].eq("Holt-Winters").all():
        raise ValueError("Holt-Winters result has the wrong model family")
    if not forecast["specification_id"].eq(SPECIFICATION_ID).all():
        raise ValueError("Holt-Winters result has the wrong specification label")
    if not forecast["target_date"].eq(target_date).all():
        raise ValueError("Holt-Winters result has the wrong target date")
    timestamps = pd.to_datetime(forecast["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Holt-Winters result contains invalid or duplicate UTC timestamps")
    forecasts = pd.to_numeric(forecast["forecast_mwh"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(forecasts).all():
        raise ValueError("Holt-Winters result contains non-finite forecasts")
    if not forecast["forecast_origin_utc"].eq(forecast["information_cutoff_utc"]).all():
        raise ValueError("Holt-Winters forecast origin and information cutoff differ")
    origin = country_forecast_origin(target_date, country)
    if not forecast["forecast_origin_utc"].eq(origin.isoformat()).all():
        raise ValueError("Holt-Winters result has an incorrect forecast origin")
    expected_training = len(information_set(frame, origin))
    if int(job_row["training_observations"]) != expected_training:
        raise ValueError("training observations do not match the origin-bounded information set")
    target_mask = forecast["is_target_day"].astype(str).str.lower().eq("true")
    target = forecast.loc[target_mask]
    if target.empty or not target["local_date"].eq(target_date).all():
        raise ValueError("Holt-Winters result does not contain the complete target day")
    if len(target) not in (23, 24, 25):
        raise ValueError("Holt-Winters target day has an invalid DST interval count")
    if target["timestamp_utc"].duplicated().any():
        raise ValueError("Holt-Winters target day contains duplicate UTC timestamps")
    if not forecast["local_date"].astype(str).le(target_date).all():
        raise ValueError("Holt-Winters path contains a future local date")


def validate_aggregate(
    job_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> None:
    if len(job_frame) != 730 or not job_frame["status"].eq("completed").all():
        raise ValueError("Holt-Winters test does not contain 730 completed jobs")
    for country, frame in frames.items():
        country_jobs = job_frame.loc[job_frame["country"].eq(country)].copy()
        country_jobs["target_date"] = pd.to_datetime(country_jobs["target_date"])
        country_jobs = country_jobs.sort_values("target_date", kind="mergesort")
        training = pd.to_numeric(country_jobs["training_observations"], errors="coerce")
        if not training.is_monotonic_increasing:
            raise ValueError(f"Holt-Winters training history is not expanding for {country}")
        for row in country_jobs.to_dict("records"):
            expected = len(
                information_set(frame, country_forecast_origin(row["target_date"], country))
            )
            if int(row["training_observations"]) != expected:
                raise ValueError(f"Holt-Winters training count is not origin-bounded for {country}")
        target = forecast_frame.loc[
            forecast_frame["country"].eq(country)
            & forecast_frame["is_target_day"].astype(str).str.lower().eq("true")
        ]
        if len(target) != EXPECTED_OBSERVATIONS:
            raise ValueError(f"{country} does not have 8760 target observations")
        if target["timestamp_utc"].duplicated().any():
            raise ValueError(f"duplicate Holt-Winters target timestamps for {country}")
        for target_date, expected_count in (("2025-03-30", 23), ("2025-10-26", 25)):
            if int(target["local_date"].eq(target_date).sum()) != expected_count:
                raise ValueError(f"invalid Holt-Winters DST count for {country} {target_date}")


def build_test_summary(
    manifest: list[ValidationJob],
    job_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
) -> pd.DataFrame:
    return build_summary(
        manifest,
        job_frame,
        forecast_frame,
        {country: EXPECTED_OBSERVATIONS for country in COUNTRY_CONFIG},
    )


def _read_existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TABLE_COLUMNS)
    frame = pd.read_csv(path)
    missing = sorted(set(TABLE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"2025 model table is missing required columns: {missing}")
    return frame


def _merge_table(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    extra = sorted(set(existing.columns).difference(TABLE_COLUMNS))
    columns = TABLE_COLUMNS + extra
    merged = pd.concat(
        [existing.reindex(columns=columns), incoming.reindex(columns=columns)],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(TABLE_KEY_COLUMNS, keep="last")
    merged["_country_order"] = merged["country"].map(COUNTRY_ORDER).fillna(len(COUNTRY_ORDER))
    return (
        merged.sort_values(
            ["_country_order", "model_family", "specification_id"],
            kind="mergesort",
        )
        .drop(columns="_country_order")
        .reset_index(drop=True)
        .loc[:, columns]
    )


def update_test_tables(
    summary_path: str | Path,
    all_models_path: str | Path = PROJECT_ROOT / "results" / "model_test_2025_all_models.csv",
    overview_path: str | Path = PROJECT_ROOT / "results" / "model_test_2025_overview.csv",
) -> dict[str, Path]:
    summary = pd.read_csv(summary_path)
    required = {"country", "specification_id", "mae", "rmse", "mape", "evaluated_observations", "coverage"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Holt-Winters test summary is missing columns: {missing}")
    incoming = summary.loc[:, ["country", "mae", "rmse", "mape", "evaluated_observations", "coverage"]].copy()
    incoming["model_family"] = "holt_winters"
    incoming["specification_id"] = SPECIFICATION_ID
    incoming = incoming.rename(columns={"evaluated_observations": "n_observations"})
    incoming = incoming.loc[:, TABLE_COLUMNS]
    all_path = Path(all_models_path)
    overview_path = Path(overview_path)
    all_models = _merge_table(_read_existing_table(all_path), incoming)
    overview = _merge_table(_read_existing_table(overview_path), incoming)
    validation_2024._atomic_write_csv(all_models, all_path, all_models.columns)
    validation_2024._atomic_write_csv(overview, overview_path, overview.columns)
    return {"all_models": all_path, "overview": overview_path}


def _validate_existing_completed(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> None:
    if jobs.empty:
        return
    for job in jobs.loc[jobs["status"].eq("completed")].to_dict("records"):
        forecast = forecasts.loc[
            forecasts["country"].eq(str(job["country"]))
            & forecasts["target_date"].eq(str(job["target_date"]))
            & forecasts["specification_id"].eq(str(job["specification_id"]))
        ]
        validate_job_result(job, forecast, frames[str(job["country"])])


def run_test(
    processed_directory: str | Path = PROCESSED,
    shortlist_path: str | Path = SHORTLIST_OUTPUT,
    output_directory: str | Path = OUTPUT_DIRECTORY,
    workers: int = 2,
) -> dict[str, object]:
    if workers != 2:
        raise ValueError("the approved Holt-Winters 2025 test requires exactly 2 workers")
    shortlists = load_test_shortlists(shortlist_path)
    frames = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    dates = test_dates(frames["Germany"])
    for country in COUNTRY_CONFIG:
        if test_dates(frames[country]) != dates:
            raise ValueError(f"2025 target dates differ between countries for {country}")
    manifest = build_test_jobs(shortlists, dates)
    if len(manifest) != 730:
        raise ValueError(f"expected 730 Holt-Winters test jobs, found {len(manifest)}")
    store = TestValidationStore(output_directory)
    initial_jobs = store.read_jobs()
    initial_forecasts = store.read_forecasts()
    _validate_existing_completed(initial_jobs, initial_forecasts, frames)
    initial_summary = build_test_summary(manifest, initial_jobs, initial_forecasts)
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
            raise TypeError("Holt-Winters workers must return ValidationResult")
        if result.job_row.get("status") == "completed":
            validate_job_result(
                result.job_row,
                result.forecast,
                frames[str(result.job_row["country"])],
            )
        store.persist_result(result.job_row, result.forecast)
        store.write_summary(
            build_test_summary(manifest, store.read_jobs(), store.read_forecasts())
        )
    elapsed = perf_counter() - started
    jobs = store.read_jobs()
    forecasts = store.read_forecasts()
    _validate_existing_completed(jobs, forecasts, frames)
    validate_aggregate(jobs, forecasts, frames)
    summary = build_test_summary(manifest, jobs, forecasts)
    store.write_summary(summary)
    failed = int(jobs["status"].eq("failed").sum())
    completed = len(store.completed_keys())
    if completed != 730 or failed != 0:
        raise RuntimeError(f"Holt-Winters 2025 test incomplete: {completed}/730 completed, {failed} failed")
    table_paths = update_test_tables(store.summary_path)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "pending_jobs": len(manifest) - completed,
        "elapsed_seconds": elapsed,
        "summary": summary.to_dict(orient="records"),
        "output_directory": str(store.output_directory),
        "table_paths": {name: str(path) for name, path in table_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final 2025 Holt-Winters test only")
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--shortlist", type=Path, default=SHORTLIST_OUTPUT)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            run_test(
                processed_directory=args.processed_directory,
                shortlist_path=args.shortlist,
                output_directory=args.output_directory,
                workers=args.workers,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
