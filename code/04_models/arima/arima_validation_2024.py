from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import (
    COUNTRY_CONFIG,
    PROCESSED,
    _as_prepared,
    build_information_context,
    calculate_metrics,
    country_forecast_origin,
    evaluate_target_day,
    extract_target_day,
    information_set,
    load_country_data,
)
from arima_models import ArimaFitRecord, fit_arima, forecast_values, order_id
from arima_screening import TABLE_DIRECTORY


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2024
ARIMA_JOB_KEY = ("country", "target_date", "specification_id")
FORECAST_KEY = (
    "country",
    "target_date",
    "specification_id",
    "timestamp_utc",
)
MINI_DATES = {
    "Germany": (date(2024, 2, 15), date(2024, 3, 31), date(2024, 10, 27)),
    "Austria": (date(2024, 2, 15), date(2024, 3, 31), date(2024, 10, 27)),
}
JOB_COLUMNS = [
    "country",
    "target_date",
    "specification_id",
    "p",
    "d",
    "q",
    "trend",
    "status",
    "error_message",
    "fit_status",
    "convergence_status",
    "converged",
    "parameters_finite",
    "standard_errors_finite",
    "stationarity_ok",
    "invertibility_ok",
    "nobs",
    "n_params",
    "log_likelihood",
    "aic",
    "aicc",
    "bic",
    "ar_root_minimum",
    "ma_root_minimum",
    "warning_messages",
    "optimizer_retry_count",
    "residual_n_effective",
    "residual_rejected",
    "residual_rejection_reason",
    "residual_acf_values",
    "residual_acf_flagged_lags",
    "residual_ljung_box_statistics",
    "residual_ljung_box_pvalues",
    "mae",
    "rmse",
    "mape",
    "evaluated_observations",
    "coverage",
]
FORECAST_COLUMNS = [
    "country",
    "model_family",
    "target_date",
    "specification_id",
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
    "source_kind",
    "is_target_day",
    "evaluated",
]

_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def build_manifest(
    shortlists: pd.DataFrame,
    target_dates_by_country: dict[str, Iterable[date]],
) -> pd.DataFrame:
    required = {"country", "specification_id", "specification_order", "p", "d", "q", "trend"}
    missing = sorted(required.difference(shortlists.columns))
    if missing:
        raise ValueError(f"shortlist is missing required columns: {missing}")
    rows: list[dict[str, object]] = []
    country_rank = {country: index for index, country in enumerate(COUNTRY_CONFIG)}
    for country in COUNTRY_CONFIG:
        country_shortlist = shortlists.loc[shortlists["country"].eq(country)].sort_values(
            "specification_order"
        )
        for target_date in sorted(target_dates_by_country.get(country, ())):
            for row in country_shortlist.itertuples(index=False):
                rows.append(
                    {
                        "country": country,
                        "target_date": target_date.isoformat(),
                        "specification_id": str(row.specification_id),
                        "specification_order": int(row.specification_order),
                        "p": int(row.p),
                        "d": int(row.d),
                        "q": int(row.q),
                        "trend": str(row.trend),
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "target_date",
                "specification_id",
                "specification_order",
                "p",
                "d",
                "q",
                "trend",
            ]
        )
    manifest["_country_rank"] = manifest["country"].map(country_rank)
    manifest = manifest.sort_values(
        ["_country_rank", "target_date", "specification_order"],
        ignore_index=True,
    ).drop(columns="_country_rank")
    if manifest.duplicated(list(ARIMA_JOB_KEY)).any():
        raise ValueError("manifest contains duplicate ARIMA job keys")
    return manifest


def load_completed_job_keys(jobs: pd.DataFrame) -> set[tuple[str, str, str]]:
    if jobs.empty:
        return set()
    completed = jobs.loc[jobs["status"].eq("completed")]
    return {
        (str(row.country), str(row.target_date), str(row.specification_id))
        for row in completed.itertuples(index=False)
    }


def upsert_frame(
    existing: pd.DataFrame,
    replacement: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    if existing.empty and replacement.empty:
        return existing.copy()
    combined = pd.concat([existing, replacement], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(key_columns, keep="last")
    return combined.sort_values(key_columns, kind="stable", ignore_index=True)


def sort_job_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    country_rank = {country: index for index, country in enumerate(COUNTRY_CONFIG)}
    return sorted(
        rows,
        key=lambda row: (
            country_rank.get(str(row["country"]), len(country_rank)),
            str(row["target_date"]),
            str(row["specification_id"]),
        ),
    )


def build_arima_forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str,
    specification_id: str,
    fitted_result: object,
) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    local_date = pd.Timestamp(target_date).date().isoformat()
    context = build_information_context(prepared, country, local_date)
    target = extract_target_day(prepared, local_date)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(context.cutoff_utc)
        & prepared["local_date"].le(local_date)
    ].copy()
    if path.empty or not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError(f"no complete ARIMA forecast path exists for {country} {local_date}")
    forecast = forecast_values(fitted_result, len(path))
    path["country"] = country
    path["model_family"] = "ARIMA"
    path["target_date"] = local_date
    path["specification_id"] = specification_id
    path["forecast_origin_local"] = context.origin_local
    path["forecast_origin_utc"] = context.cutoff_utc.isoformat()
    path["information_cutoff_utc"] = context.cutoff_utc.isoformat()
    path["forecast_mwh"] = forecast
    path["bridge_used"] = ~path["local_date"].eq(local_date)
    path["source_kind"] = "arima_forecast"
    path["is_target_day"] = path["local_date"].eq(local_date)
    path["evaluated"] = path["is_target_day"]
    return path[
        [
            "country",
            "model_family",
            "target_date",
            "specification_id",
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
            "source_kind",
            "is_target_day",
            "evaluated",
        ]
    ]


def _fit_metadata(fit: ArimaFitRecord) -> dict[str, object]:
    residual = fit.residual_diagnostics
    return {
        "fit_status": fit.fit_status,
        "convergence_status": fit.convergence_status,
        "converged": fit.converged,
        "parameters_finite": fit.parameters_finite,
        "standard_errors_finite": fit.standard_errors_finite,
        "stationarity_ok": fit.stationarity_ok,
        "invertibility_ok": fit.invertibility_ok,
        "nobs": fit.nobs,
        "n_params": fit.n_params,
        "log_likelihood": fit.log_likelihood,
        "aic": fit.aic,
        "aicc": fit.aicc,
        "bic": fit.bic,
        "ar_root_minimum": fit.ar_root_minimum,
        "ma_root_minimum": fit.ma_root_minimum,
        "warning_messages": json.dumps(list(fit.warning_messages)),
        "optimizer_retry_count": fit.optimizer_retry_count,
        "residual_n_effective": residual.n_effective if residual else 0,
        "residual_rejected": residual.rejected if residual else True,
        "residual_rejection_reason": (
            residual.rejection_reason if residual else "residual diagnostics unavailable"
        ),
        "residual_acf_values": json.dumps(residual.acf_values if residual else {}),
        "residual_acf_flagged_lags": json.dumps(
            list(residual.flagged_acf_lags) if residual else []
        ),
        "residual_ljung_box_statistics": json.dumps(
            residual.ljung_box_statistics if residual else {}
        ),
        "residual_ljung_box_pvalues": json.dumps(
            residual.ljung_box_pvalues if residual else {}
        ),
    }


def _execute_job(job: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    country = str(job["country"])
    target_date = str(job["target_date"])
    specification_id = str(job["specification_id"])
    order = (int(job["p"]), int(job["d"]), int(job["q"]))
    context = build_information_context(frame, country, target_date)
    available = information_set(frame, context.cutoff_utc)
    fit = fit_arima(available["actual_load_mwh"].to_numpy(dtype=float), order)
    job_row: dict[str, object] = {
        "country": country,
        "target_date": target_date,
        "specification_id": specification_id,
        "p": order[0],
        "d": order[1],
        "q": order[2],
        "trend": fit.trend,
    }
    job_row.update(_fit_metadata(fit))
    job_row["error_message"] = fit.error_message
    if not fit.eligible or fit.fitted_result is None:
        job_row.update(
            {
                "status": "failed",
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "evaluated_observations": 0,
                "coverage": 0.0,
            }
        )
        return {"job": job_row, "forecasts": []}

    try:
        path = build_arima_forecast_path(
            frame, country, target_date, specification_id, fit.fitted_result
        )
        metrics = evaluate_target_day(path, expected_observations=int(path["is_target_day"].sum()))
        job_row.update(
            {
                "status": "completed",
                "error_message": None,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "evaluated_observations": metrics["evaluated_observations"],
                "coverage": metrics["coverage"],
            }
        )
        return {
            "job": job_row,
            "forecasts": path.to_dict("records"),
        }
    except Exception as error:
        job_row.update(
            {
                "status": "failed",
                "error_message": f"{type(error).__name__}: {error}",
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "evaluated_observations": 0,
                "coverage": 0.0,
            }
        )
        return {"job": job_row, "forecasts": []}


def _initialize_worker(processed_directory: str) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _worker_execute(job: dict[str, object]) -> dict[str, object]:
    return _execute_job(job, _WORKER_FRAMES[str(job["country"])])


def _atomic_write(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame if columns is None else frame.reindex(columns=columns)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def _summary_frame(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for (country, specification_id), expected in manifest.groupby(
        ["country", "specification_id"], sort=False
    ):
        completed = jobs.loc[
            jobs["country"].eq(country)
            & jobs["specification_id"].eq(specification_id)
            & jobs["status"].eq("completed")
        ]
        if forecasts is not None and not forecasts.empty:
            evaluated = forecasts.loc[
                forecasts["country"].eq(country)
                & forecasts["specification_id"].eq(specification_id)
                & forecasts["is_target_day"].eq(True)
                & forecasts["evaluated"].eq(True)
            ]
        else:
            evaluated = pd.DataFrame()
        if evaluated.empty:
            metrics = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "evaluated_observations": 0,
                "coverage": 0.0,
            }
        else:
            metrics = calculate_metrics(
                evaluated["actual_load_mwh"],
                evaluated["forecast_mwh"],
                expected_observations=int(expected["expected_observations"].sum()),
            )
        rows.append(
            {
                "country": country,
                "specification_id": specification_id,
                "expected_jobs": len(expected),
                "completed_jobs": len(completed),
                "failed_jobs": int(len(expected) - len(completed)),
                "expected_observations": int(expected["expected_observations"].sum()),
                "evaluated_observations": int(metrics["evaluated_observations"]),
                "coverage": float(metrics["coverage"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
            }
        )
    return pd.DataFrame(rows)


def run_validation(
    shortlists: pd.DataFrame,
    target_dates_by_country: dict[str, Iterable[date]],
    output_directory: str | Path,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(shortlists, target_dates_by_country)
    count_frames = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    manifest["expected_observations"] = [
        len(extract_target_day(count_frames[country], target_date))
        if not count_frames[country].empty
        else 24
        for country, target_date in zip(manifest["country"], manifest["target_date"])
    ]
    jobs_path = output / "arima_validation_2024_jobs.csv"
    forecasts_path = output / "arima_validation_2024_forecasts.csv"
    summary_path = output / "arima_validation_2024_summary.csv"
    jobs = _load_csv(jobs_path, JOB_COLUMNS)
    forecasts = _load_csv(forecasts_path, FORECAST_COLUMNS)
    completed_keys = load_completed_job_keys(jobs)
    pending = [
        dict(zip(manifest.columns, row))
        for row in manifest.itertuples(index=False, name=None)
        if (row[0], row[1], row[2]) not in completed_keys
    ]

    def persist(result: dict[str, object]) -> None:
        nonlocal jobs, forecasts
        job_frame = pd.DataFrame([result["job"]])
        forecast_frame = pd.DataFrame(result["forecasts"], columns=FORECAST_COLUMNS)
        jobs = upsert_frame(jobs, job_frame, list(ARIMA_JOB_KEY))
        forecasts = upsert_frame(forecasts, forecast_frame, list(FORECAST_KEY))
        _atomic_write(jobs, jobs_path, JOB_COLUMNS)
        _atomic_write(forecasts, forecasts_path, FORECAST_COLUMNS)
        _atomic_write(_summary_frame(jobs, manifest, forecasts), summary_path)

    if workers == 1:
        frames = count_frames
        for job in pending:
            persist(_execute_job(job, frames[str(job["country"])]))
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

    summary = _summary_frame(jobs, manifest, forecasts)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(jobs["status"].eq("completed").sum()) if not jobs.empty else 0,
        "failed_jobs": int(jobs["status"].eq("failed").sum()) if not jobs.empty else 0,
        "forecast_rows": len(forecasts),
        "summary": summary.to_dict("records"),
        "output_directory": str(output),
    }


def load_shortlists(path: str | Path = TABLE_DIRECTORY / "arima_shortlists_2024.csv") -> pd.DataFrame:
    shortlists = pd.read_csv(path)
    required = {"country", "specification_id", "specification_order", "p", "d", "q", "trend"}
    missing = sorted(required.difference(shortlists.columns))
    if missing:
        raise ValueError(f"ARIMA shortlist is missing required columns: {missing}")
    if set(shortlists["country"]) != set(COUNTRY_CONFIG):
        raise ValueError("ARIMA shortlist must contain exactly Germany and Austria")
    return shortlists


def run_mini_validation(
    output_directory: str | Path,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return run_validation(
        load_shortlists(),
        MINI_DATES,
        output_directory,
        workers=workers,
        processed_directory=processed_directory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARIMA ordinary/DST mini-validation")
    parser.add_argument("--mini", action="store_true", help="required development-stage mini-validation guard")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processed-directory", default=str(PROCESSED))
    args = parser.parse_args()
    if not args.mini:
        parser.error("--mini is required; full-year ARIMA validation is disabled in this stage")
    result = run_mini_validation(
        args.output_dir,
        workers=args.workers,
        processed_directory=args.processed_directory,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
