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
    _prepare_frame,
    build_information_context,
    calculate_metrics,
    evaluate_target_day,
    extract_target_day,
    information_set,
)
from regression_arima_errors_models import (
    ERROR_ORDER,
    SPECIFICATION_IDS,
    build_exog,
    fit_regression_arima,
    forecast_values,
    specification_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2024
MODEL_FAMILY = "regression_arima_errors"
JOB_KEY = ("country", "target_date", "specification_id")
FORECAST_KEY = JOB_KEY + ("timestamp_utc",)
MINI_DATES = {
    "Germany": (date(2024, 2, 15), date(2024, 3, 31), date(2024, 10, 27)),
    "Austria": (date(2024, 2, 15), date(2024, 3, 31), date(2024, 10, 27)),
}
VALIDATION_ROWS_THROUGH_2024 = 43_848

JOB_COLUMNS = [
    "country",
    "target_date",
    "specification_id",
    "specification_order",
    "error_order",
    "trend",
    "status",
    "error_message",
    "fit_status",
    "convergence_status",
    "converged",
    "parameters_finite",
    "standard_errors_finite",
    "nobs",
    "n_params",
    "log_likelihood",
    "aic",
    "aicc",
    "bic",
    "parameter_estimates",
    "standard_errors",
    "optimizer_attempts",
    "optimizer_retry_count",
    "warning_messages",
    "coefficient_warnings",
    "exog_columns",
    "exog_rank",
    "exog_condition_number",
    "transformed_exog_rank",
    "ar_root_minimum",
    "ma_root_minimum",
    "stationarity_ok",
    "invertibility_ok",
    "residual_n_effective",
    "residual_acf_values",
    "residual_acf_threshold",
    "residual_acf_flagged_lags",
    "residual_ljung_box_statistics",
    "residual_ljung_box_pvalues",
    "mae",
    "rmse",
    "mape",
    "evaluated_observations",
    "expected_observations",
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
_WORKER_MODEL_FAMILY = MODEL_FAMILY


def _validation_frame(country: str, processed_directory: str | Path) -> pd.DataFrame:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    path = Path(processed_directory) / COUNTRY_CONFIG[country]["filename"]
    raw = pd.read_csv(path, nrows=VALIDATION_ROWS_THROUGH_2024)
    frame = _prepare_frame(raw)
    if len(frame) != VALIDATION_ROWS_THROUGH_2024:
        raise ValueError("validation input does not contain the complete 2020-2024 period")
    if frame["local_date"].max() != "2024-12-31":
        raise ValueError("validation input is not bounded at 2024-12-31")
    return frame


def build_manifest(
    target_dates_by_country: dict[str, Iterable[date]],
    specification_ids: Iterable[str] = SPECIFICATION_IDS,
) -> pd.DataFrame:
    specification_ids = tuple(specification_ids)
    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        for target_date in sorted(target_dates_by_country.get(country, ())):
            if target_date.year != VALIDATION_YEAR:
                raise ValueError("regression ARIMA validation dates must be in 2024")
            for specification_order, specification_id in enumerate(specification_ids, 1):
                rows.append(
                    {
                        "country": country,
                        "target_date": target_date.isoformat(),
                        "specification_id": specification_id,
                        "specification_order": specification_order,
                        "p": ERROR_ORDER[0],
                        "d": ERROR_ORDER[1],
                        "q": ERROR_ORDER[2],
                        "trend": "n",
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
    if manifest.duplicated(list(JOB_KEY)).any():
        raise ValueError("manifest contains duplicate regression ARIMA job keys")
    return manifest.sort_values(
        ["country", "target_date", "specification_order"],
        ignore_index=True,
    )


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


def build_regression_forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str,
    specification_id: str,
    fitted_result: object,
    model_family: str = MODEL_FAMILY,
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
        raise ValueError(f"no complete regression ARIMA path exists for {country} {local_date}")
    exog = build_exog(path["timestamp_utc"], country, specification_id)
    if len(exog) != len(path) or list(exog.columns) != list(specification_columns(specification_id)):
        raise ValueError("forecast exog is not aligned with the complete forecast path")
    forecasts = forecast_values(fitted_result, exog)
    path["country"] = country
    path["model_family"] = model_family
    path["target_date"] = local_date
    path["specification_id"] = specification_id
    path["forecast_origin_local"] = context.origin_local
    path["forecast_origin_utc"] = context.cutoff_utc.isoformat()
    path["information_cutoff_utc"] = context.cutoff_utc.isoformat()
    path["forecast_mwh"] = forecasts
    path["bridge_used"] = ~path["local_date"].eq(local_date)
    path["source_kind"] = f"{model_family}_forecast"
    path["is_target_day"] = path["local_date"].eq(local_date)
    path["evaluated"] = path["is_target_day"]
    return path[FORECAST_COLUMNS]


def _json(value: object) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _fit_metadata(fit: object) -> dict[str, object]:
    residual = fit.residual_diagnostics
    return {
        "error_order": _json(list(fit.order)),
        "trend": fit.trend,
        "fit_status": fit.fit_status,
        "convergence_status": fit.convergence_status,
        "converged": fit.converged,
        "parameters_finite": fit.parameters_finite,
        "standard_errors_finite": fit.standard_errors_finite,
        "nobs": fit.nobs,
        "n_params": fit.n_params,
        "log_likelihood": fit.log_likelihood,
        "aic": fit.aic,
        "aicc": fit.aicc,
        "bic": fit.bic,
        "parameter_estimates": _json(fit.parameter_estimates),
        "standard_errors": _json(fit.standard_errors),
        "optimizer_attempts": _json(list(fit.optimizer_attempts)),
        "optimizer_retry_count": fit.optimizer_retry_count,
        "warning_messages": _json(list(fit.warning_messages)),
        "coefficient_warnings": _json(list(fit.coefficient_warnings)),
        "exog_columns": _json(list(fit.exog_columns)),
        "exog_rank": fit.exog_rank,
        "exog_condition_number": fit.exog_condition_number,
        "transformed_exog_rank": fit.transformed_exog_rank,
        "ar_root_minimum": fit.ar_root_minimum,
        "ma_root_minimum": fit.ma_root_minimum,
        "stationarity_ok": fit.stationarity_ok,
        "invertibility_ok": fit.invertibility_ok,
        "residual_n_effective": residual.n_effective if residual else 0,
        "residual_acf_values": _json(residual.acf_values if residual else {}),
        "residual_acf_threshold": residual.acf_threshold if residual else float("nan"),
        "residual_acf_flagged_lags": _json(
            list(residual.flagged_acf_lags) if residual else []
        ),
        "residual_ljung_box_statistics": _json(
            residual.ljung_box_statistics if residual else {}
        ),
        "residual_ljung_box_pvalues": _json(
            residual.ljung_box_pvalues if residual else {}
        ),
    }


def _execute_job(
    job: dict[str, object],
    frame: pd.DataFrame,
    model_family: str = MODEL_FAMILY,
) -> dict[str, object]:
    country = str(job["country"])
    target_date = str(job["target_date"])
    specification_id = str(job["specification_id"])
    context = build_information_context(frame, country, target_date)
    available = information_set(frame, context.cutoff_utc)
    train_exog = build_exog(available["timestamp_utc"], country, specification_id)
    fit = fit_regression_arima(
        available["actual_load_mwh"].to_numpy(dtype=float),
        train_exog,
        specification_id,
    )
    job_row: dict[str, object] = {
        "country": country,
        "target_date": target_date,
        "specification_id": specification_id,
        "specification_order": int(job["specification_order"]),
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
                "expected_observations": len(extract_target_day(frame, target_date)),
                "coverage": 0.0,
            }
        )
        return {"job": job_row, "forecasts": []}
    try:
        path = build_regression_forecast_path(
            frame,
            country,
            target_date,
            specification_id,
            fit.fitted_result,
            model_family=model_family,
        )
        expected = int(path["is_target_day"].sum())
        metrics = evaluate_target_day(path, expected_observations=expected)
        job_row.update(
            {
                "status": "completed",
                "error_message": None,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "evaluated_observations": metrics["evaluated_observations"],
                "expected_observations": expected,
                "coverage": metrics["coverage"],
            }
        )
        return {"job": job_row, "forecasts": path.to_dict("records")}
    except Exception as error:
        job_row.update(
            {
                "status": "failed",
                "error_message": f"{type(error).__name__}: {error}",
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "evaluated_observations": 0,
                "expected_observations": len(extract_target_day(frame, target_date)),
                "coverage": 0.0,
            }
        )
        return {"job": job_row, "forecasts": []}


def _initialize_worker(
    processed_directory: str,
    model_family: str = MODEL_FAMILY,
) -> None:
    global _WORKER_FRAMES, _WORKER_MODEL_FAMILY
    _WORKER_MODEL_FAMILY = model_family
    _WORKER_FRAMES = {
        country: _validation_frame(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _worker_execute(job: dict[str, object]) -> dict[str, object]:
    return _execute_job(
        job,
        _WORKER_FRAMES[str(job["country"])],
        model_family=_WORKER_MODEL_FAMILY,
    )


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
    forecasts: pd.DataFrame,
    model_family: str = MODEL_FAMILY,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country, specification_id in manifest[["country", "specification_id"]].drop_duplicates().itertuples(index=False):
        expected_jobs = manifest.loc[
            manifest["country"].eq(country)
            & manifest["specification_id"].eq(specification_id)
        ]
        completed = jobs.loc[
            jobs["country"].eq(country)
            & jobs["specification_id"].eq(specification_id)
            & jobs["status"].eq("completed")
        ]
        evaluated = forecasts.loc[
            forecasts["country"].eq(country)
            & forecasts["specification_id"].eq(specification_id)
            & forecasts["is_target_day"].eq(True)
            & forecasts["evaluated"].eq(True)
        ]
        expected_observations = int(expected_jobs["expected_observations"].sum())
        metrics = (
            calculate_metrics(
                evaluated["actual_load_mwh"],
                evaluated["forecast_mwh"],
                expected_observations=expected_observations,
            )
            if not evaluated.empty
            else {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "evaluated_observations": 0,
                "coverage": 0.0,
            }
        )
        rows.append(
            {
                "country": country,
                "model_family": model_family,
                "specification_id": specification_id,
                "expected_jobs": len(expected_jobs),
                "completed_jobs": len(completed),
                "failed_jobs": int(len(expected_jobs) - len(completed)),
                "expected_observations": expected_observations,
                "evaluated_observations": int(metrics["evaluated_observations"]),
                "coverage": float(metrics["coverage"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["country", "specification_id"], ignore_index=True
    )


def run_validation(
    target_dates_by_country: dict[str, Iterable[date]],
    output_directory: str | Path,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
    model_family: str = MODEL_FAMILY,
    specification_ids: Iterable[str] = SPECIFICATION_IDS,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    specification_ids = tuple(specification_ids)
    manifest = build_manifest(target_dates_by_country, specification_ids)
    frames = {
        country: _validation_frame(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    manifest["expected_observations"] = [
        len(extract_target_day(frames[country], target_date))
        for country, target_date in zip(manifest["country"], manifest["target_date"])
    ]
    jobs_path = output / f"{model_family}_validation_2024_jobs.csv"
    forecasts_path = output / f"{model_family}_validation_2024_forecasts.csv"
    summary_path = output / f"{model_family}_validation_2024_summary.csv"
    specifications_path = output / f"{model_family}_specifications_2024.csv"
    jobs = _load_csv(jobs_path, JOB_COLUMNS)
    forecasts = _load_csv(forecasts_path, FORECAST_COLUMNS)
    completed_keys = load_completed_job_keys(jobs)
    pending = [
        dict(zip(manifest.columns, row))
        for row in manifest.itertuples(index=False, name=None)
        if (row[0], row[1], row[2]) not in completed_keys
    ]
    specification_rows = []
    for specification_order, specification_id in enumerate(specification_ids, 1):
        specification_rows.append(
            {
                "specification_order": specification_order,
                "specification_id": specification_id,
                "error_order": _json(list(ERROR_ORDER)),
                "trend": "n",
                "exog_columns": _json(list(specification_columns(specification_id))),
            }
        )
    _atomic_write(pd.DataFrame(specification_rows), specifications_path)

    def persist(result: dict[str, object]) -> None:
        nonlocal jobs, forecasts
        jobs = upsert_frame(
            jobs,
            pd.DataFrame([result["job"]]),
            list(JOB_KEY),
        )
        forecasts = upsert_frame(
            forecasts,
            pd.DataFrame(result["forecasts"], columns=FORECAST_COLUMNS),
            list(FORECAST_KEY),
        )
        _atomic_write(jobs, jobs_path, JOB_COLUMNS)
        _atomic_write(forecasts, forecasts_path, FORECAST_COLUMNS)
        _atomic_write(
            _summary_frame(jobs, manifest, forecasts, model_family=model_family),
            summary_path,
        )

    if workers == 1:
        for job in pending:
            persist(
                _execute_job(
                    job,
                    frames[str(job["country"])],
                    model_family=model_family,
                )
            )
    elif pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(str(processed_directory), model_family),
        ) as executor:
            for result in executor.map(_worker_execute, pending):
                persist(result)

    summary = _summary_frame(jobs, manifest, forecasts, model_family=model_family)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(jobs["status"].eq("completed").sum()) if not jobs.empty else 0,
        "failed_jobs": int(jobs["status"].eq("failed").sum()) if not jobs.empty else 0,
        "forecast_rows": len(forecasts),
        "summary": summary.to_dict("records"),
        "output_directory": str(output),
    }


def run_mini_validation(
    output_directory: str | Path,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return run_validation(
        MINI_DATES,
        output_directory,
        workers=workers,
        processed_directory=processed_directory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the regression-with-ARIMA-errors ordinary/DST mini-validation"
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="required development-stage mini-validation guard",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processed-directory", default=str(PROCESSED))
    args = parser.parse_args()
    if not args.mini:
        parser.error("--mini is required; full-year validation is disabled in this stage")
    result = run_mini_validation(
        args.output_dir,
        workers=args.workers,
        processed_directory=args.processed_directory,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
