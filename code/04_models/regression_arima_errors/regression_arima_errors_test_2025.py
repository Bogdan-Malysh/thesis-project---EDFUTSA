from __future__ import annotations

import argparse
from collections.abc import Iterable
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

import regression_arima_errors_validation_2024 as validation_2024
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    PROCESSED,
    build_information_context,
    country_forecast_origin,
    extract_target_day,
    information_set,
    load_country_data,
)
from regression_arima_errors_models import (
    ERROR_ORDER,
    TREND,
    SPECIFICATION_IDS,
    specification_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPECIFICATION_ID = "reg_arima_h_d_m_hol"
SPECIFICATION_ORDER = 4
MODEL_FAMILY = "regression_arima_errors"
TABLE_MODEL_FAMILY = MODEL_FAMILY
VALIDATION_YEAR = 2025
EXPECTED_DAYS = 365
EXPECTED_OBSERVATIONS = 8760
WORKERS = 2
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "regression_arima_errors" / "test_2025"
JOB_FILENAME = "regression_arima_errors_test_2025_jobs.csv"
FORECAST_FILENAME = "regression_arima_errors_test_2025_forecasts.csv"
SUMMARY_FILENAME = "regression_arima_errors_test_2025_summary.csv"
AUTHORITATIVE_SELECTION_PATH = (
    PROJECT_ROOT / "results" / "regression_arima_errors" / "tables"
    / "regression_arima_errors_selected_2024.csv"
)
AUTHORITATIVE_SPECIFICATION_PATH = (
    PROJECT_ROOT / "results" / "regression_arima_errors" / "validation_2024"
    / "full_validation" / "regression_arima_errors_specifications_2024.csv"
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
JOB_KEY_COLUMNS = list(validation_2024.JOB_KEY)
FORECAST_KEY_COLUMNS = list(validation_2024.FORECAST_KEY)
JOB_COLUMNS = [
    *validation_2024.JOB_COLUMNS,
    "forecast_origin_local",
    "forecast_origin_utc",
    "information_cutoff_utc",
    "training_observations",
]
FORECAST_COLUMNS = list(validation_2024.FORECAST_COLUMNS)
_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def verify_authoritative_selection(
    selection_path: str | Path = AUTHORITATIVE_SELECTION_PATH,
    registry_path: str | Path = REGISTRY_PATH,
) -> pd.DataFrame:
    selected = pd.read_csv(selection_path)
    required = {
        "country",
        "selected_specification_id",
        "expected_jobs",
        "completed_jobs",
        "failed_jobs",
        "coverage",
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"authoritative selection is missing columns: {missing}")
    selected = selected.loc[
        selected["selected_specification_id"].eq(SPECIFICATION_ID)
    ].copy()
    if len(selected) != len(COUNTRY_CONFIG) or set(selected["country"]) != set(COUNTRY_CONFIG):
        raise ValueError(
            "authoritative selection must contain reg_arima_h_d_m_hol for both countries"
        )
    if not selected["expected_jobs"].eq(366).all():
        raise ValueError("authoritative selection must expect 366 jobs per country")
    if not selected["completed_jobs"].eq(366).all():
        raise ValueError("authoritative selection is incomplete")
    if not selected["failed_jobs"].eq(0).all() or not selected["coverage"].eq(1.0).all():
        raise ValueError("authoritative selection contains failures or incomplete coverage")

    registry = pd.read_csv(registry_path)
    registry_required = {"country", "model_family", "selected_specification_id", "status"}
    missing = sorted(registry_required.difference(registry.columns))
    if missing:
        raise ValueError(f"selected-model registry is missing columns: {missing}")
    registry_rows = registry.loc[
        registry["model_family"].eq(TABLE_MODEL_FAMILY)
    ].copy()
    if len(registry_rows) != len(COUNTRY_CONFIG) or set(registry_rows["country"]) != set(COUNTRY_CONFIG):
        raise ValueError("selected-model registry must contain one row per country")
    if not registry_rows["selected_specification_id"].eq(SPECIFICATION_ID).all():
        raise ValueError("selected-model registry has the wrong specification")
    if not registry_rows["status"].eq("frozen").all():
        raise ValueError("selected-model registry row is not frozen")
    return selected.sort_values("country", kind="mergesort").reset_index(drop=True)


def load_frozen_specification(
    specification_path: str | Path = AUTHORITATIVE_SPECIFICATION_PATH,
) -> pd.DataFrame:
    specification = pd.read_csv(specification_path)
    required = {
        "specification_order",
        "specification_id",
        "error_order",
        "trend",
        "exog_columns",
    }
    missing = sorted(required.difference(specification.columns))
    if missing:
        raise ValueError(f"authoritative specification is missing columns: {missing}")
    selected = specification.loc[
        specification["specification_id"].eq(SPECIFICATION_ID)
    ].copy()
    if len(selected) != 1:
        raise ValueError("authoritative specification must contain one selected row")
    row = selected.iloc[0]
    if int(row["specification_order"]) != SPECIFICATION_ORDER:
        raise ValueError("authoritative specification has the wrong specification order")
    if tuple(json.loads(row["error_order"])) != ERROR_ORDER:
        raise ValueError("authoritative specification has the wrong ARIMA error order")
    if str(row["trend"]) != TREND:
        raise ValueError("authoritative specification has the wrong trend")
    if tuple(json.loads(row["exog_columns"])) != specification_columns(SPECIFICATION_ID):
        raise ValueError("authoritative specification has the wrong exogenous columns")
    return selected.reset_index(drop=True)


def test_dates(frame: pd.DataFrame) -> list[date]:
    values = sorted(
        {
            date.fromisoformat(value)
            for value in frame.loc[
                frame["local_date"].astype(str).str.startswith(f"{VALIDATION_YEAR:04d}-"),
                "local_date",
            ]
        }
    )
    expected = [
        date(VALIDATION_YEAR, 1, 1) + timedelta(days=offset)
        for offset in range(EXPECTED_DAYS)
    ]
    if values != expected:
        raise ValueError(
            "2025 Regression + ARIMA errors test requires every local date from "
            "2025-01-01 through 2025-12-31"
        )
    return values


def build_test_manifest(
    frames: dict[str, pd.DataFrame],
    target_dates: Iterable[date | str | pd.Timestamp],
) -> pd.DataFrame:
    dates = sorted({pd.Timestamp(value).date().isoformat() for value in target_dates})
    if not dates or any(
        value < "2025-01-01" or value > "2025-12-31" for value in dates
    ):
        raise ValueError("2025 test manifest requires non-empty 2025 target dates")

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for country in COUNTRY_CONFIG:
        if country not in frames:
            raise ValueError(f"missing processed frame for {country}")
        for target_date in dates:
            key = (country, target_date, SPECIFICATION_ID)
            if key in seen:
                raise ValueError(f"duplicate test job: {key}")
            seen.add(key)
            rows.append(
                {
                    "country": country,
                    "target_date": target_date,
                    "specification_id": SPECIFICATION_ID,
                    "specification_order": SPECIFICATION_ORDER,
                    "error_order": json.dumps(list(ERROR_ORDER)),
                    "trend": TREND,
                    "expected_observations": len(
                        extract_target_day(frames[country], target_date)
                    ),
                }
            )
    return pd.DataFrame(rows)


def expected_training_observations(
    job: pd.Series | dict[str, object],
    frame: pd.DataFrame,
) -> int:
    origin = country_forecast_origin(job["target_date"], str(job["country"]))
    return len(information_set(frame, origin))


def _execute_job_2025(
    job: dict[str, object],
    frame: pd.DataFrame,
) -> dict[str, object]:
    context = build_information_context(frame, str(job["country"]), str(job["target_date"]))
    result = validation_2024._execute_job(job, frame, model_family=MODEL_FAMILY)
    result["job"].update(
        {
            "forecast_origin_local": context.origin_local,
            "forecast_origin_utc": context.cutoff_utc.isoformat(),
            "information_cutoff_utc": context.cutoff_utc.isoformat(),
            "training_observations": len(information_set(frame, context.cutoff_utc)),
            "expected_observations": len(
                extract_target_day(frame, str(job["target_date"]))
            ),
        }
    )
    return result


def validate_job_result(
    job_row: dict[str, object] | pd.Series,
    forecast: pd.DataFrame,
    frame: pd.DataFrame,
) -> None:
    if forecast.empty:
        raise ValueError("completed regression ARIMA job has no forecast rows")
    country = str(job_row["country"])
    target_date = str(job_row["target_date"])
    if str(job_row["specification_id"]) != SPECIFICATION_ID:
        raise ValueError("regression ARIMA result has the wrong specification")
    if not forecast["model_family"].eq(MODEL_FAMILY).all():
        raise ValueError("regression ARIMA result has the wrong model family")
    if not forecast["specification_id"].eq(SPECIFICATION_ID).all():
        raise ValueError("regression ARIMA result has the wrong specification label")
    if not forecast["country"].eq(country).all():
        raise ValueError("regression ARIMA result has the wrong country")
    if not forecast["target_date"].eq(target_date).all():
        raise ValueError("regression ARIMA result has the wrong target date")
    timestamps = pd.to_datetime(forecast["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("regression ARIMA result contains invalid or duplicate UTC timestamps")
    forecasts = pd.to_numeric(forecast["forecast_mwh"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(forecasts).all():
        raise ValueError("regression ARIMA result contains non-finite forecasts")
    if not forecast["forecast_origin_utc"].eq(forecast["information_cutoff_utc"]).all():
        raise ValueError("regression ARIMA forecast origin and information cutoff differ")
    origin = country_forecast_origin(target_date, country)
    if not forecast["forecast_origin_utc"].eq(origin.isoformat()).all():
        raise ValueError("regression ARIMA result has an incorrect forecast origin")
    expected_training = len(information_set(frame, origin))
    if int(job_row["training_observations"]) != expected_training:
        raise ValueError("training observations do not match the origin-bounded information set")
    target_mask = forecast["is_target_day"].astype(str).str.lower().eq("true")
    target = forecast.loc[target_mask]
    if target.empty or not target["local_date"].eq(target_date).all():
        raise ValueError("regression ARIMA result does not contain the complete target day")
    if len(target) not in (23, 24, 25):
        raise ValueError("regression ARIMA target day has an invalid DST interval count")
    if int(job_row["expected_observations"]) != len(target):
        raise ValueError("expected target observations do not match the forecast")
    if target["timestamp_utc"].duplicated().any():
        raise ValueError("regression ARIMA target day contains duplicate UTC timestamps")
    if not forecast["local_date"].astype(str).le(target_date).all():
        raise ValueError("regression ARIMA path contains a future local date")


def _read_csv(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame


def _normalize_keys(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column in normalized:
            normalized[column] = normalized[column].astype(str)
    return normalized


def _normalize_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_keys(frame, FORECAST_KEY_COLUMNS)
    if not normalized.empty and "timestamp_utc" in normalized:
        timestamps = pd.to_datetime(
            normalized["timestamp_utc"], utc=True, errors="coerce"
        )
        normalized["timestamp_utc"] = timestamps.map(
            lambda value: value.isoformat() if pd.notna(value) else value
        )
    return normalized


def _upsert(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    if existing.empty and incoming.empty:
        return existing.copy()
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    combined = _normalize_keys(combined, keys)
    combined = combined.drop_duplicates(keys, keep="last")
    return combined.sort_values(keys, kind="mergesort").reset_index(drop=True)


class TestValidationStore:
    def __init__(self, output_directory: str | Path = OUTPUT_DIRECTORY):
        self.output_directory = Path(output_directory)
        self.jobs_path = self.output_directory / JOB_FILENAME
        self.forecasts_path = self.output_directory / FORECAST_FILENAME
        self.summary_path = self.output_directory / SUMMARY_FILENAME

    def read_jobs(self) -> pd.DataFrame:
        return _normalize_keys(_read_csv(self.jobs_path, JOB_COLUMNS), JOB_KEY_COLUMNS)

    def read_forecasts(self) -> pd.DataFrame:
        return _normalize_forecasts(_read_csv(self.forecasts_path, FORECAST_COLUMNS))

    def completed_keys(self) -> set[tuple[str, str, str]]:
        jobs = self.read_jobs()
        forecasts = self.read_forecasts()
        if jobs.empty or forecasts.empty:
            return set()
        completed = jobs.loc[jobs["status"].eq("completed"), JOB_KEY_COLUMNS]
        forecast_keys = forecasts.loc[:, JOB_KEY_COLUMNS].drop_duplicates()
        joined = completed.merge(forecast_keys, on=JOB_KEY_COLUMNS, how="inner")
        return {tuple(row) for row in joined.itertuples(index=False, name=None)}

    def persist_result(self, job_row: dict[str, object], forecast: pd.DataFrame) -> None:
        if str(job_row.get("status", "")) == "completed" and forecast.empty:
            raise ValueError("completed regression ARIMA jobs must include forecast rows")
        jobs = _upsert(
            self.read_jobs(),
            _normalize_keys(pd.DataFrame([job_row]), JOB_KEY_COLUMNS),
            JOB_KEY_COLUMNS,
        )
        if not forecast.empty:
            forecasts = _upsert(
                self.read_forecasts(),
                _normalize_forecasts(forecast),
                FORECAST_KEY_COLUMNS,
            )
            _atomic_write(forecasts, self.forecasts_path, FORECAST_COLUMNS)
        _atomic_write(jobs, self.jobs_path, JOB_COLUMNS)

    def write_summary(self, summary: pd.DataFrame) -> None:
        _atomic_write(summary, self.summary_path, list(summary.columns))


def _summary_frame(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    return validation_2024._summary_frame(
        jobs,
        manifest,
        forecasts,
        model_family=MODEL_FAMILY,
    )


def build_test_summary(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    return _summary_frame(jobs, manifest, forecasts)


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
        "mae",
        "rmse",
        "mape",
        "evaluated_observations",
        "coverage",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"test summary is missing required columns: {missing}")
    rows = summary.loc[
        :, ["country", "mae", "rmse", "mape", "evaluated_observations", "coverage"]
    ].copy()
    rows["model_family"] = TABLE_MODEL_FAMILY
    rows["specification_id"] = SPECIFICATION_ID
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
    merged["_country_order"] = merged["country"].map(COUNTRY_ORDER).fillna(
        len(COUNTRY_ORDER)
    )
    return (
        merged.sort_values(
            ["_country_order", "model_family", "specification_id"],
            kind="mergesort",
        )
        .drop(columns="_country_order")
        .reset_index(drop=True)
        .loc[:, columns]
    )


def _atomic_write(
    frame: pd.DataFrame,
    path: Path,
    columns: list[str] | Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame if columns is None else frame.reindex(columns=list(columns))
    temporary = path.with_name(f".{path.name}.tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def update_test_tables(
    summary_path: str | Path,
    all_models_path: str | Path = ALL_MODELS_PATH,
    overview_path: str | Path = OVERVIEW_PATH,
) -> dict[str, Path]:
    summary = pd.read_csv(summary_path)
    incoming = _table_rows(summary)
    all_path = Path(all_models_path)
    overview_path = Path(overview_path)
    all_models = _merge_table(_read_existing_table(all_path), incoming)
    overview = _merge_table(_read_existing_table(overview_path), incoming)
    _atomic_write(all_models, all_path, list(all_models.columns))
    _atomic_write(overview, overview_path, list(overview.columns))
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


def validate_aggregate(
    job_frame: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> None:
    if len(job_frame) != 2 * EXPECTED_DAYS or not job_frame["status"].eq("completed").all():
        raise ValueError("2025 regression ARIMA test does not contain 730 completed jobs")
    if job_frame.duplicated(JOB_KEY_COLUMNS).any():
        raise ValueError("2025 regression ARIMA test contains duplicate job keys")
    if not job_frame["specification_id"].eq(SPECIFICATION_ID).all():
        raise ValueError("2025 regression ARIMA test contains the wrong specification")
    for country, frame in frames.items():
        country_jobs = job_frame.loc[job_frame["country"].eq(country)].copy()
        if len(country_jobs) != EXPECTED_DAYS:
            raise ValueError(f"{country} does not contain 365 regression ARIMA jobs")
        country_jobs["target_date"] = pd.to_datetime(country_jobs["target_date"])
        country_jobs = country_jobs.sort_values("target_date", kind="mergesort")
        training = pd.to_numeric(country_jobs["training_observations"], errors="coerce")
        if not training.is_monotonic_increasing:
            raise ValueError(f"regression ARIMA training history is not expanding for {country}")
        for row in country_jobs.to_dict("records"):
            expected = len(
                information_set(
                    frame,
                    country_forecast_origin(row["target_date"], country),
                )
            )
            if int(row["training_observations"]) != expected:
                raise ValueError(f"regression ARIMA training count is not origin-bounded for {country}")
        target = forecast_frame.loc[
            forecast_frame["country"].eq(country)
            & forecast_frame["specification_id"].eq(SPECIFICATION_ID)
            & forecast_frame["is_target_day"].astype(str).str.lower().eq("true")
        ]
        if len(target) != EXPECTED_OBSERVATIONS:
            raise ValueError(
                f"{country} does not have 8760 regression ARIMA target observations"
            )
        if target["timestamp_utc"].duplicated().any():
            raise ValueError(f"duplicate regression ARIMA target timestamps for {country}")
        for target_date, expected_count in (("2025-03-30", 23), ("2025-10-26", 25)):
            if int(target["local_date"].eq(target_date).sum()) != expected_count:
                raise ValueError(
                    f"invalid regression ARIMA DST count for {country} {target_date}"
                )


def _initialize_worker(processed_directory: str | Path) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _worker_execute(job: dict[str, object]) -> dict[str, object]:
    return _execute_job_2025(job, _WORKER_FRAMES[str(job["country"])])


def run_test(
    processed_directory: str | Path = PROCESSED,
    output_directory: str | Path = OUTPUT_DIRECTORY,
    workers: int = WORKERS,
    selection_path: str | Path = AUTHORITATIVE_SELECTION_PATH,
    specification_path: str | Path = AUTHORITATIVE_SPECIFICATION_PATH,
    registry_path: str | Path = REGISTRY_PATH,
    all_models_path: str | Path = ALL_MODELS_PATH,
    overview_path: str | Path = OVERVIEW_PATH,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    verify_authoritative_selection(selection_path, registry_path)
    load_frozen_specification(specification_path)
    frames = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    dates = test_dates(frames["Germany"])
    for country in COUNTRY_CONFIG:
        if test_dates(frames[country]) != dates:
            raise ValueError(f"2025 target dates differ between countries for {country}")
    manifest = build_test_manifest(frames, dates)
    if len(manifest) != 2 * EXPECTED_DAYS:
        raise ValueError(f"expected 730 regression ARIMA test jobs, found {len(manifest)}")
    store = TestValidationStore(output_directory)
    initial_jobs = store.read_jobs()
    initial_forecasts = store.read_forecasts()
    _validate_existing_completed(initial_jobs, initial_forecasts, frames)
    store.write_summary(build_test_summary(initial_jobs, manifest, initial_forecasts))
    completed_keys = store.completed_keys()
    pending = [
        row
        for row in manifest.to_dict("records")
        if (row["country"], row["target_date"], row["specification_id"])
        not in completed_keys
    ]

    started = perf_counter()

    def persist(result: dict[str, object]) -> None:
        if result["job"].get("status") == "completed":
            validate_job_result(
                result["job"],
                pd.DataFrame(result["forecasts"], columns=FORECAST_COLUMNS),
                frames[str(result["job"]["country"])],
            )
        store.persist_result(
            result["job"],
            pd.DataFrame(result["forecasts"], columns=FORECAST_COLUMNS),
        )
        store.write_summary(
            build_test_summary(store.read_jobs(), manifest, store.read_forecasts())
        )

    if workers == 1:
        for job in pending:
            persist(_execute_job_2025(job, frames[str(job["country"])]))
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
    jobs = store.read_jobs()
    forecasts = store.read_forecasts()
    _validate_existing_completed(jobs, forecasts, frames)
    validate_aggregate(jobs, forecasts, frames)
    summary = build_test_summary(jobs, manifest, forecasts)
    store.write_summary(summary)
    failed = int(jobs["status"].eq("failed").sum())
    completed = len(store.completed_keys())
    if completed != 2 * EXPECTED_DAYS or failed != 0:
        raise RuntimeError(
            f"2025 regression ARIMA test incomplete: {completed}/730 completed, {failed} failed"
        )
    table_paths = update_test_tables(store.summary_path, all_models_path, overview_path)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": completed,
        "failed_jobs": failed,
        "pending_jobs": len(manifest) - completed,
        "elapsed_seconds": elapsed,
        "summary": summary.to_dict(orient="records"),
        "output_directory": str(store.output_directory),
        "table_paths": {name: str(path) for name, path in table_paths.items()},
        "converged_jobs": int(jobs["converged"].astype(str).str.lower().eq("true").sum()),
        "optimizer_retry_count": int(
            pd.to_numeric(jobs["optimizer_retry_count"], errors="coerce")
            .fillna(0)
            .sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final 2025 Regression + ARIMA errors test only"
    )
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    print(
        json.dumps(
            run_test(
                processed_directory=args.processed_directory,
                output_directory=args.output_directory,
                workers=args.workers,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
