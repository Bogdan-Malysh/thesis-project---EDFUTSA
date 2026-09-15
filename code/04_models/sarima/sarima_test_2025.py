from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
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

import sarima_validation_2024 as validation_2024
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    PROCESSED,
    country_forecast_origin,
    extract_target_day,
    information_set,
    load_country_data,
)
from sarima_models import SarimaOrder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2025
EXPECTED_DAYS = 365
EXPECTED_OBSERVATIONS = 8760
MODEL_FAMILY = "SARIMA"
TABLE_MODEL_FAMILY = "sarima"
SPECIFICATION_BY_COUNTRY = {
    "Germany": "sarima_p2_d0_q0_P0_D1_Q0_s24",
    "Austria": "sarima_p1_d0_q1_P1_D1_Q0_s24",
}
ORDER_BY_COUNTRY = {
    "Germany": (2, 0, 0, 0, 1, 0, 24),
    "Austria": (1, 0, 1, 1, 1, 0, 24),
}
TREND = "n"
BENCHMARK_WORKERS = (2, 4, 6, 8)
DEFAULT_WORKERS = 4
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "arima_sarima" / "test_2025"
JOB_FILENAME = "sarima_test_2025_jobs.csv"
FORECAST_FILENAME = "sarima_test_2025_forecasts.csv"
DIAGNOSTIC_FILENAME = "sarima_test_2025_diagnostics.csv"
SUMMARY_FILENAME = "sarima_test_2025_summary.csv"
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "arima_sarima" / "tables"
AUTHORITATIVE_SELECTION_PATH = TABLE_DIRECTORY / "sarima_selected_2024.csv"
AUTHORITATIVE_SHORTLIST_PATH = TABLE_DIRECTORY / "sarima_shortlists_2024.csv"
AUTHORITATIVE_VALIDATION_PATH = (
    validation_2024.FULL_OUTPUT_DIRECTORY / validation_2024.SUMMARY_FILENAME
)
REGISTRY_PATH = PROJECT_ROOT / "results" / "selected_models_registry.csv"
ALL_MODELS_PATH = PROJECT_ROOT / "results" / "model_test_2025_all_models.csv"
OVERVIEW_PATH = PROJECT_ROOT / "results" / "model_test_2025_overview.csv"
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
JOB_KEY_COLUMNS = list(validation_2024.JOB_KEY_COLUMNS)
FORECAST_KEY_COLUMNS = list(validation_2024.FORECAST_KEY_COLUMNS)
DIAGNOSTIC_KEY_COLUMNS = list(validation_2024.DIAGNOSTIC_KEY_COLUMNS)
JOB_COLUMNS = list(validation_2024.JOB_COLUMNS)
FORECAST_COLUMNS = list(validation_2024.FORECAST_COLUMNS)
DIAGNOSTIC_COLUMNS = list(validation_2024.DIAGNOSTIC_COLUMNS)

_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def _order_for_country(country: str) -> SarimaOrder:
    try:
        return SarimaOrder(*ORDER_BY_COUNTRY[country])
    except KeyError as error:
        raise ValueError(f"unsupported country: {country}") from error


def _bool_values(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def _timestamp_values(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    return timestamps.map(lambda value: value.isoformat() if pd.notna(value) else "NaT")


def _target_date_values(
    target_dates: Iterable[date | str | pd.Timestamp],
) -> list[date]:
    values = [pd.Timestamp(value).date() for value in target_dates]
    if len(values) != len(set(values)):
        raise ValueError("duplicate SARIMA 2025 target dates are not allowed")
    values = sorted(values)
    if not values or any(
        value < date(VALIDATION_YEAR, 1, 1)
        or value > date(VALIDATION_YEAR, 12, 31)
        for value in values
    ):
        raise ValueError("SARIMA 2025 target dates must be within 2025")
    return values


def verify_authoritative_selection(
    selection_path: str | Path = AUTHORITATIVE_SELECTION_PATH,
    registry_path: str | Path = REGISTRY_PATH,
    validation_summary_path: str | Path = AUTHORITATIVE_VALIDATION_PATH,
) -> pd.DataFrame:
    selected = pd.read_csv(selection_path)
    required = {"country", "selected_specification_id"}
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"authoritative SARIMA selection is missing columns: {missing}")
    selected = selected.copy()
    selected["country"] = selected["country"].astype(str)
    selected["selected_specification_id"] = selected["selected_specification_id"].astype(str)
    observed_selection = dict(
        zip(
            selected["country"],
            selected["selected_specification_id"],
            strict=True,
        )
    )
    if observed_selection != SPECIFICATION_BY_COUNTRY:
        raise ValueError("authoritative SARIMA selection does not match the frozen specifications")

    validation_summary = pd.read_csv(validation_summary_path)
    summary_required = {
        "country",
        "specification_id",
        "expected_jobs",
        "completed_jobs",
        "failed_jobs",
        "expected_observations",
        "evaluated_observations",
        "coverage",
    }
    missing = sorted(summary_required.difference(validation_summary.columns))
    if missing:
        raise ValueError(f"authoritative SARIMA validation is missing columns: {missing}")
    for country, specification_id in SPECIFICATION_BY_COUNTRY.items():
        rows = validation_summary.loc[
            validation_summary["country"].astype(str).eq(country)
            & validation_summary["specification_id"].astype(str).eq(specification_id)
        ]
        if len(rows) != 1:
            raise ValueError(f"authoritative SARIMA validation must contain one row for {country}")
        row = rows.iloc[0]
        if int(row["expected_jobs"]) != 366 or int(row["completed_jobs"]) != 366:
            raise ValueError(f"authoritative 2024 SARIMA validation is incomplete for {country}")
        if int(row["failed_jobs"]) != 0 or float(row["coverage"]) != 1.0:
            raise ValueError(f"authoritative 2024 SARIMA validation has failures for {country}")
        if int(row["expected_observations"]) != 8784 or int(row["evaluated_observations"]) != 8784:
            raise ValueError(f"authoritative 2024 SARIMA observations are incomplete for {country}")
        if "coverage_eligible" in row and str(row["coverage_eligible"]).lower() != "true":
            raise ValueError(f"authoritative 2024 SARIMA coverage is ineligible for {country}")

    registry = pd.read_csv(registry_path)
    registry_required = {"country", "model_family", "selected_specification_id", "status"}
    missing = sorted(registry_required.difference(registry.columns))
    if missing:
        raise ValueError(f"selected-model registry is missing columns: {missing}")
    registry_rows = registry.loc[
        registry["model_family"].astype(str).eq(TABLE_MODEL_FAMILY)
    ].copy()
    observed_registry = dict(
        zip(
            registry_rows["country"].astype(str),
            registry_rows["selected_specification_id"].astype(str),
            strict=True,
        )
    )
    if observed_registry != SPECIFICATION_BY_COUNTRY:
        raise ValueError("selected-model registry does not match the frozen SARIMA specifications")
    if not registry_rows["status"].astype(str).str.lower().eq("frozen").all():
        raise ValueError("selected-model registry SARIMA rows are not frozen")

    country_rank = {country: index for index, country in enumerate(COUNTRY_CONFIG)}
    return selected.assign(_country_order=selected["country"].map(country_rank)).sort_values(
        "_country_order", kind="mergesort"
    ).drop(columns="_country_order").reset_index(drop=True)


def load_test_shortlists(
    shortlist_path: str | Path = AUTHORITATIVE_SHORTLIST_PATH,
) -> pd.DataFrame:
    shortlist = pd.read_csv(shortlist_path)
    required = {
        "country",
        "specification_id",
        "specification_order",
        "p",
        "d",
        "q",
        "P",
        "D",
        "Q",
        "seasonal_period",
        "trend",
    }
    missing = sorted(required.difference(shortlist.columns))
    if missing:
        raise ValueError(f"SARIMA shortlist is missing columns: {missing}")
    selected_rows: list[dict[str, object]] = []
    for country, specification_id in SPECIFICATION_BY_COUNTRY.items():
        rows = shortlist.loc[
            shortlist["country"].astype(str).eq(country)
            & shortlist["specification_id"].astype(str).eq(specification_id)
        ]
        if len(rows) != 1:
            raise ValueError(f"SARIMA shortlist must contain one frozen row for {country}")
        row = rows.iloc[0].to_dict()
        order = SarimaOrder(
            int(row["p"]),
            int(row["d"]),
            int(row["q"]),
            int(row["P"]),
            int(row["D"]),
            int(row["Q"]),
            int(row["seasonal_period"]),
        )
        if order.specification_id != specification_id or str(row["trend"]) != TREND:
            raise ValueError(f"SARIMA shortlist row has the wrong frozen order for {country}")
        if "model_valid" in row and str(row["model_valid"]).lower() != "true":
            raise ValueError(f"SARIMA shortlist row is not model-valid for {country}")
        if "eligible" in row and str(row["eligible"]).lower() != "true":
            raise ValueError(f"SARIMA shortlist row is not eligible for {country}")
        selected_rows.append(row)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def test_dates(frame: pd.DataFrame) -> list[date]:
    values = sorted(
        {
            date.fromisoformat(str(value))
            for value in frame.loc[
                frame["local_date"].astype(str).str.startswith(f"{VALIDATION_YEAR:04d}-"),
                "local_date",
            ]
        }
    )
    expected = [date(VALIDATION_YEAR, 1, 1) + timedelta(days=offset) for offset in range(EXPECTED_DAYS)]
    if values != expected:
        raise ValueError(
            "SARIMA 2025 test requires every local date from 2025-01-01 through 2025-12-31"
        )
    return values


def build_test_manifest(
    frames: Mapping[str, pd.DataFrame],
    target_dates: Iterable[date | str | pd.Timestamp],
    shortlists: pd.DataFrame | None = None,
) -> pd.DataFrame:
    dates = _target_date_values(target_dates)
    shortlists = load_test_shortlists() if shortlists is None else shortlists
    specification_orders = shortlists.set_index("country")["specification_order"].to_dict()
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for country in COUNTRY_CONFIG:
        if country not in frames:
            raise ValueError(f"missing processed frame for {country}")
        order = _order_for_country(country)
        for target_date in dates:
            key = (country, target_date.isoformat(), order.specification_id)
            if key in seen:
                raise ValueError(f"duplicate SARIMA 2025 job: {key}")
            seen.add(key)
            rows.append(
                {
                    "country": country,
                    "target_date": target_date.isoformat(),
                    "specification_id": order.specification_id,
                    "specification_order": int(specification_orders[country]),
                    "p": order.p,
                    "d": order.d,
                    "q": order.q,
                    "P": order.P,
                    "D": order.D,
                    "Q": order.Q,
                    "seasonal_period": order.seasonal_period,
                    "trend": TREND,
                    "expected_observations": len(extract_target_day(frames[country], target_date)),
                }
            )
    return pd.DataFrame(rows)


def expected_training_observations(
    job: pd.Series | Mapping[str, object], frame: pd.DataFrame
) -> int:
    origin = country_forecast_origin(job["target_date"], str(job["country"]))
    return len(information_set(frame, origin))


def execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    return validation_2024.execute_job(job, frame)


def validate_job_result(
    job_row: Mapping[str, object] | pd.Series,
    forecast: pd.DataFrame,
    frame: pd.DataFrame,
    diagnostic: Mapping[str, object] | pd.Series | None = None,
) -> None:
    if forecast.empty:
        raise ValueError("completed SARIMA job has no forecast rows")
    country = str(job_row["country"])
    target_date = str(job_row["target_date"])
    expected_specification = SPECIFICATION_BY_COUNTRY.get(country)
    if expected_specification is None:
        raise ValueError(f"unsupported country in SARIMA result: {country}")
    if str(job_row["specification_id"]) != expected_specification:
        raise ValueError("SARIMA result has the wrong country-specific specification")
    if not forecast["country"].astype(str).eq(country).all():
        raise ValueError("SARIMA result has the wrong country")
    if not forecast["model_family"].astype(str).eq(MODEL_FAMILY).all():
        raise ValueError("SARIMA result has the wrong model family")
    if not forecast["specification_id"].astype(str).eq(expected_specification).all():
        raise ValueError("SARIMA result has the wrong specification label")
    if not forecast["target_date"].astype(str).eq(target_date).all():
        raise ValueError("SARIMA result has the wrong target date")

    order = _order_for_country(country)
    for column, expected in {
        "p": order.p,
        "d": order.d,
        "q": order.q,
        "P": order.P,
        "D": order.D,
        "Q": order.Q,
        "seasonal_period": order.seasonal_period,
    }.items():
        values = pd.to_numeric(forecast[column], errors="coerce")
        if values.isna().any() or not values.eq(expected).all():
            raise ValueError(f"SARIMA result has the wrong {column} order value")
    if not forecast["trend"].astype(str).eq(TREND).all():
        raise ValueError("SARIMA result has the wrong trend")

    timestamps = pd.to_datetime(forecast["timestamp_utc"], utc=True, errors="coerce")
    interval_ends = pd.to_datetime(forecast["interval_end_utc"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("SARIMA result contains invalid or duplicate UTC timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("SARIMA result is not chronologically ordered")
    if not interval_ends.eq(timestamps + pd.Timedelta(hours=1)).all():
        raise ValueError("SARIMA result contains invalid interval ends")
    forecasts = pd.to_numeric(forecast["forecast_mwh"], errors="coerce").to_numpy(dtype=float)
    actuals = pd.to_numeric(forecast["actual_load_mwh"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(forecasts).all() or not np.isfinite(actuals).all():
        raise ValueError("SARIMA result contains non-finite forecast or actual values")

    origin = country_forecast_origin(target_date, country)
    origin_iso = origin.isoformat()
    context = validation_2024.build_information_context(frame, country, target_date)
    if not _timestamp_values(forecast["forecast_origin_utc"]).eq(origin_iso).all():
        raise ValueError("SARIMA result has an incorrect forecast origin")
    if not _timestamp_values(forecast["information_cutoff_utc"]).eq(origin_iso).all():
        raise ValueError("SARIMA result has an incorrect information cutoff")
    if not forecast["forecast_origin_local"].astype(str).eq(context.origin_local).all():
        raise ValueError("SARIMA result has an incorrect local forecast origin")
    if int(job_row["training_observations"]) != len(information_set(frame, origin)):
        raise ValueError("training observations do not match the origin-bounded information set")

    target_mask = _bool_values(forecast["is_target_day"])
    evaluated_mask = _bool_values(forecast["evaluated"])
    if not target_mask.equals(evaluated_mask):
        raise ValueError("SARIMA result does not restrict evaluation to the target day")
    target = forecast.loc[target_mask].copy()
    if target.empty or not target["local_date"].astype(str).eq(target_date).all():
        raise ValueError("SARIMA result does not contain the complete target day")
    if len(target) not in (23, 24, 25):
        raise ValueError("SARIMA target day has an invalid DST interval count")
    if target["timestamp_utc"].duplicated().any():
        raise ValueError("SARIMA target day contains duplicate UTC timestamps")
    if not forecast["local_date"].astype(str).le(target_date).all():
        raise ValueError("SARIMA path contains a future local date")
    expected_path_timestamps = validation_2024.expected_forecast_timestamps(
        frame, country, target_date
    )
    if {timestamp.isoformat() for timestamp in timestamps} != expected_path_timestamps:
        raise ValueError("SARIMA result does not contain the complete bridge and target path")
    path_for_validation = forecast.copy()
    path_for_validation["timestamp_utc"] = timestamps
    path_for_validation["interval_end_utc"] = interval_ends
    validation_2024._validate_hourly_forecast_window(
        path_for_validation,
        extract_target_day(frame, target_date),
        origin,
        target_date,
    )
    if "target_intervals" in job_row and int(job_row["target_intervals"]) != len(target):
        raise ValueError("SARIMA job target interval count is inconsistent")
    if "evaluated_observations" in job_row and int(job_row["evaluated_observations"]) != len(target):
        raise ValueError("SARIMA job evaluated observation count is inconsistent")
    if diagnostic is not None:
        diagnostic_key = tuple(str(diagnostic[column]) for column in DIAGNOSTIC_KEY_COLUMNS)
        expected_key = tuple(str(job_row[column]) for column in DIAGNOSTIC_KEY_COLUMNS)
        if diagnostic_key != expected_key:
            raise ValueError("SARIMA diagnostic key does not match the completed job")
        if str(diagnostic["diagnostic_type"]) != "sarima_fit_and_residual":
            raise ValueError("SARIMA diagnostic has the wrong diagnostic type")
        if str(diagnostic["status"]) != "completed":
            raise ValueError("SARIMA diagnostic is not complete")


def validate_aggregate(
    job_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    diagnostic_frame: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    target_dates: Iterable[date | str | pd.Timestamp],
) -> None:
    dates = _target_date_values(target_dates)
    expected_keys = {
        (country, target_date.isoformat(), SPECIFICATION_BY_COUNTRY[country])
        for country in COUNTRY_CONFIG
        for target_date in dates
    }
    observed_keys = {
        tuple(str(value) for value in row)
        for row in job_frame.loc[:, JOB_KEY_COLUMNS].itertuples(index=False, name=None)
    }
    if observed_keys != expected_keys or len(job_frame) != len(expected_keys):
        raise ValueError("SARIMA 2025 jobs do not match the exact target manifest")
    if not job_frame["status"].astype(str).eq("completed").all():
        raise ValueError("SARIMA 2025 test has unresolved job failures")
    diagnostic_keys = {
        tuple(str(value) for value in row)
        for row in diagnostic_frame.loc[:, DIAGNOSTIC_KEY_COLUMNS].itertuples(
            index=False, name=None
        )
    }
    if diagnostic_keys != expected_keys or len(diagnostic_frame) != len(expected_keys):
        raise ValueError("SARIMA 2025 diagnostics are incomplete")

    for country, frame in frames.items():
        country_jobs = job_frame.loc[job_frame["country"].astype(str).eq(country)].copy()
        if len(country_jobs) != len(dates):
            raise ValueError(f"{country} does not contain the expected SARIMA job count")
        country_jobs["target_date"] = pd.to_datetime(country_jobs["target_date"])
        country_jobs = country_jobs.sort_values("target_date", kind="mergesort")
        training = pd.to_numeric(country_jobs["training_observations"], errors="coerce")
        if not training.is_monotonic_increasing:
            raise ValueError(f"SARIMA training history is not expanding for {country}")
        for job in country_jobs.to_dict("records"):
            key = (country, str(job["target_date"].date()), SPECIFICATION_BY_COUNTRY[country])
            forecast = forecast_frame.loc[
                forecast_frame["country"].astype(str).eq(country)
                & forecast_frame["target_date"].astype(str).eq(key[1])
                & forecast_frame["specification_id"].astype(str).eq(key[2])
            ]
            diagnostic = diagnostic_frame.loc[
                diagnostic_frame["country"].astype(str).eq(country)
                & diagnostic_frame["target_date"].astype(str).eq(key[1])
                & diagnostic_frame["specification_id"].astype(str).eq(key[2])
            ]
            if len(diagnostic) != 1:
                raise ValueError(f"missing SARIMA diagnostic for {country} {key[1]}")
            job["target_date"] = key[1]
            validate_job_result(job, forecast, frame, diagnostic.iloc[0])
        target = forecast_frame.loc[
            forecast_frame["country"].astype(str).eq(country)
            & forecast_frame["specification_id"].astype(str).eq(SPECIFICATION_BY_COUNTRY[country])
            & _bool_values(forecast_frame["is_target_day"])
            & _bool_values(forecast_frame["evaluated"])
        ]
        expected_observations = sum(
            len(extract_target_day(frame, target_date)) for target_date in dates
        )
        if len(target) != expected_observations:
            raise ValueError(f"{country} does not have the expected target observations")
        if target["timestamp_utc"].duplicated().any():
            raise ValueError(f"duplicate SARIMA target timestamps for {country}")
        for target_date, expected_count in (("2025-03-30", 23), ("2025-10-26", 25)):
            if target_date in {value.isoformat() for value in dates}:
                if int(target["local_date"].astype(str).eq(target_date).sum()) != expected_count:
                    raise ValueError(f"invalid SARIMA DST count for {country} {target_date}")


def build_test_summary(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    expected_forecast_timestamps: Mapping[tuple[str, str, str], set[str]],
) -> pd.DataFrame:
    return validation_2024._summary_frame(
        jobs,
        manifest,
        forecasts,
        diagnostics=diagnostics,
        expected_forecast_timestamps=expected_forecast_timestamps,
    )


class TestValidationStore:
    def __init__(self, output_directory: str | Path = OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)
        self.jobs_path = self.output_directory / JOB_FILENAME
        self.forecasts_path = self.output_directory / FORECAST_FILENAME
        self.diagnostics_path = self.output_directory / DIAGNOSTIC_FILENAME
        self.summary_path = self.output_directory / SUMMARY_FILENAME

    def read_jobs(self) -> pd.DataFrame:
        return validation_2024._load_csv(self.jobs_path, JOB_COLUMNS)

    def read_forecasts(self) -> pd.DataFrame:
        return validation_2024._load_csv(self.forecasts_path, FORECAST_COLUMNS)

    def read_diagnostics(self) -> pd.DataFrame:
        return validation_2024._load_csv(self.diagnostics_path, DIAGNOSTIC_COLUMNS)

    def completed_keys(
        self,
        expected_forecast_timestamps: Mapping[tuple[str, str, str], set[str]],
    ) -> set[tuple[str, str, str]]:
        return validation_2024.load_completed_job_keys(
            self.read_jobs(),
            self.read_forecasts(),
            self.read_diagnostics(),
            expected_forecast_timestamps,
        )

    def persist_result(
        self,
        result: Mapping[str, object],
        jobs: pd.DataFrame | None = None,
        forecasts: pd.DataFrame | None = None,
        diagnostics: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        job = dict(result["job"])
        incoming_forecasts = pd.DataFrame(result.get("forecasts", []))
        if str(job.get("status", "")) == "completed" and incoming_forecasts.empty:
            raise ValueError("completed SARIMA jobs must include forecast rows")
        jobs = self.read_jobs() if jobs is None else jobs
        forecasts = self.read_forecasts() if forecasts is None else forecasts
        diagnostics = self.read_diagnostics() if diagnostics is None else diagnostics
        diagnostic_payload = result.get("diagnostics") or validation_2024._diagnostic_row(job)
        incoming_diagnostics = pd.DataFrame([diagnostic_payload])
        jobs = validation_2024.upsert_frame(
            jobs, pd.DataFrame([job]), JOB_KEY_COLUMNS
        )
        forecasts = validation_2024._without_job_rows(forecasts, job)
        forecasts = validation_2024.upsert_frame(
            forecasts, incoming_forecasts, FORECAST_KEY_COLUMNS
        )
        diagnostics = validation_2024.upsert_frame(
            diagnostics, incoming_diagnostics, DIAGNOSTIC_KEY_COLUMNS
        )
        validation_2024._atomic_write(forecasts, self.forecasts_path, FORECAST_COLUMNS)
        validation_2024._atomic_write(diagnostics, self.diagnostics_path, DIAGNOSTIC_COLUMNS)
        validation_2024._atomic_write(jobs, self.jobs_path, JOB_COLUMNS)
        return jobs, forecasts, diagnostics

    def write_summary(self, summary: pd.DataFrame) -> None:
        validation_2024._atomic_write(summary, self.summary_path, list(summary.columns))


def _read_existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TABLE_COLUMNS)
    frame = pd.read_csv(path)
    missing = sorted(set(TABLE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required 2025 table columns: {missing}")
    return frame


def _table_rows(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "country",
        "specification_id",
        "mae",
        "rmse",
        "mape",
        "evaluated_observations",
        "coverage",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"SARIMA test summary is missing columns: {missing}")
    rows = summary.loc[
        :, ["country", "specification_id", "mae", "rmse", "mape", "evaluated_observations", "coverage"]
    ].copy()
    if any(
        rows["specification_id"].astype(str).ne(
            rows["country"].astype(str).map(SPECIFICATION_BY_COUNTRY)
        )
    ):
        raise ValueError("SARIMA summary contains a non-frozen country specification")
    rows["model_family"] = TABLE_MODEL_FAMILY
    rows = rows.rename(columns={"evaluated_observations": "n_observations"})
    return rows.loc[:, TABLE_COLUMNS]


def _merge_table(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    extra_columns = sorted(set(existing.columns).difference(TABLE_COLUMNS))
    columns = TABLE_COLUMNS + extra_columns
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
    all_models_path: str | Path = ALL_MODELS_PATH,
    overview_path: str | Path = OVERVIEW_PATH,
) -> dict[str, Path]:
    incoming = _table_rows(pd.read_csv(summary_path))
    all_path = Path(all_models_path)
    overview_path = Path(overview_path)
    all_models = _merge_table(_read_existing_table(all_path), incoming)
    overview = _merge_table(_read_existing_table(overview_path), incoming)
    validation_2024._atomic_write(all_models, all_path, list(all_models.columns))
    validation_2024._atomic_write(overview, overview_path, list(overview.columns))
    return {"all_models": all_path, "overview": overview_path}


def _validate_existing_completed(
    store: TestValidationStore,
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    expected_forecast_timestamps: Mapping[tuple[str, str, str], set[str]],
) -> None:
    completed_keys = store.completed_keys(expected_forecast_timestamps)
    for job in jobs.loc[jobs["status"].astype(str).eq("completed")].to_dict("records"):
        key = tuple(str(job[column]) for column in JOB_KEY_COLUMNS)
        if key not in completed_keys:
            continue
        forecast = forecasts.loc[
            forecasts["country"].astype(str).eq(key[0])
            & forecasts["target_date"].astype(str).eq(key[1])
            & forecasts["specification_id"].astype(str).eq(key[2])
        ]
        diagnostic = diagnostics.loc[
            diagnostics["country"].astype(str).eq(key[0])
            & diagnostics["target_date"].astype(str).eq(key[1])
            & diagnostics["specification_id"].astype(str).eq(key[2])
        ]
        if len(diagnostic) != 1:
            raise ValueError(f"missing SARIMA diagnostic for existing job {key}")
        validate_job_result(job, forecast, frames[key[0]], diagnostic.iloc[0])


def _initialize_worker(processed_directory: str | Path) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _worker_execute(job: Mapping[str, object]) -> dict[str, object]:
    return execute_job(job, _WORKER_FRAMES[str(job["country"])])


def run_test(
    processed_directory: str | Path = PROCESSED,
    shortlist_path: str | Path = AUTHORITATIVE_SHORTLIST_PATH,
    output_directory: str | Path = OUTPUT_DIRECTORY,
    workers: int = DEFAULT_WORKERS,
    target_dates: Iterable[date | str | pd.Timestamp] | None = None,
    update_tables: bool | None = None,
) -> dict[str, object]:
    if workers not in BENCHMARK_WORKERS:
        raise ValueError("SARIMA 2025 test requires one of the approved worker counts: 2, 4, 6, or 8")
    verify_authoritative_selection()
    shortlists = load_test_shortlists(shortlist_path)
    frames = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    all_dates = test_dates(frames["Germany"])
    for country in COUNTRY_CONFIG:
        if test_dates(frames[country]) != all_dates:
            raise ValueError(f"2025 target dates differ between countries for {country}")
    dates = all_dates if target_dates is None else _target_date_values(target_dates)
    if not set(dates).issubset(set(all_dates)):
        raise ValueError("SARIMA 2025 target dates are not available in the processed frames")
    full_run = dates == all_dates
    if update_tables is None:
        update_tables = full_run
    if update_tables and not full_run:
        raise ValueError("2025 model tables can only be updated by the complete SARIMA test")

    manifest = build_test_manifest(frames, dates, shortlists)
    expected_forecast_timestamps = {
        tuple(str(row[column]) for column in JOB_KEY_COLUMNS): validation_2024.expected_forecast_timestamps(
            frames[str(row["country"])], str(row["country"]), str(row["target_date"])
        )
        for row in manifest.to_dict("records")
    }
    store = TestValidationStore(output_directory)
    jobs = store.read_jobs()
    forecasts = store.read_forecasts()
    diagnostics = store.read_diagnostics()
    _validate_existing_completed(
        store, jobs, forecasts, diagnostics, frames, expected_forecast_timestamps
    )
    store.write_summary(
        build_test_summary(jobs, manifest, forecasts, diagnostics, expected_forecast_timestamps)
    )
    completed_keys = store.completed_keys(expected_forecast_timestamps)
    pending = [
        row
        for row in manifest.to_dict("records")
        if tuple(str(row[column]) for column in JOB_KEY_COLUMNS) not in completed_keys
    ]

    started = perf_counter()
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(str(processed_directory),),
        ) as executor:
            for result in executor.map(_worker_execute, pending):
                job = result["job"]
                if str(job.get("status")) == "completed":
                    forecast = pd.DataFrame(result["forecasts"], columns=FORECAST_COLUMNS)
                    diagnostic = result.get("diagnostics")
                    validate_job_result(
                        job,
                        forecast,
                        frames[str(job["country"])],
                        diagnostic,
                    )
                jobs, forecasts, diagnostics = store.persist_result(
                    result, jobs, forecasts, diagnostics
                )
                store.write_summary(
                    build_test_summary(
                        jobs,
                        manifest,
                        forecasts,
                        diagnostics,
                        expected_forecast_timestamps,
                    )
                )
    elapsed = perf_counter() - started

    jobs = store.read_jobs()
    forecasts = store.read_forecasts()
    diagnostics = store.read_diagnostics()
    _validate_existing_completed(
        store, jobs, forecasts, diagnostics, frames, expected_forecast_timestamps
    )
    validate_aggregate(jobs, forecasts, diagnostics, frames, dates)
    summary = build_test_summary(
        jobs, manifest, forecasts, diagnostics, expected_forecast_timestamps
    )
    store.write_summary(summary)
    completed = len(store.completed_keys(expected_forecast_timestamps))
    failed = int(
        jobs.loc[
            jobs.apply(
                lambda row: tuple(str(row[column]) for column in JOB_KEY_COLUMNS)
                in set(expected_forecast_timestamps),
                axis=1,
            ),
            "status",
        ].astype(str).eq("failed").sum()
    )
    if completed != len(manifest) or failed != 0:
        raise RuntimeError(
            f"SARIMA 2025 test incomplete: {completed}/{len(manifest)} completed, {failed} failed"
        )
    table_paths = update_test_tables(store.summary_path) if update_tables else {}
    return {
        "total_jobs": len(manifest),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "pending_jobs": len(manifest) - completed,
        "elapsed_seconds": elapsed,
        "summary": summary.to_dict(orient="records"),
        "output_directory": str(store.output_directory),
        "table_paths": {name: str(path) for name, path in table_paths.items()},
        "converged_jobs": int(_bool_values(jobs["converged"]).sum()),
        "optimizer_retry_count": int(
            pd.to_numeric(jobs["optimizer_retry_count"], errors="coerce").fillna(0).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final 2025 SARIMA test only")
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--shortlist", type=Path, default=AUTHORITATIVE_SHORTLIST_PATH)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--target-date", action="append", dest="target_dates")
    args = parser.parse_args()
    print(
        json.dumps(
            run_test(
                processed_directory=args.processed_directory,
                shortlist_path=args.shortlist,
                output_directory=args.output_directory,
                workers=args.workers,
                target_dates=args.target_dates,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
