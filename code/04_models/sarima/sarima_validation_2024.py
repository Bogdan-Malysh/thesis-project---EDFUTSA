from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

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
)
from sarima_models import (
    SarimaFitRecord,
    SarimaOrder,
    fit_sarima,
    forecast_values,
    trend_for_d_D,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2024
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "arima_sarima" / "tables"
DEFAULT_SHORTLIST = TABLE_DIRECTORY / "sarima_shortlists_2024.csv"
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "arima_sarima" / "validation_2024" / "mini_validation"
)
FULL_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "arima_sarima" / "validation_2024" / "full_validation"
)
MINI_DATES = (date(2024, 2, 15), date(2024, 3, 31), date(2024, 10, 27))
APPROVED_TARGET_DATE_STRINGS = frozenset(value.isoformat() for value in MINI_DATES)
FULL_YEAR_DATE_STRINGS = frozenset(
    value.date().isoformat()
    for value in pd.date_range("2024-01-01", "2024-12-31", freq="D")
)
VALIDATION_END_DATE = date(2024, 12, 31)

JOB_KEY_COLUMNS = ("country", "target_date", "specification_id")
FORECAST_KEY_COLUMNS = JOB_KEY_COLUMNS + ("timestamp_utc",)
DIAGNOSTIC_KEY_COLUMNS = JOB_KEY_COLUMNS
COMPLETION_FORECAST_COLUMNS = {
    *FORECAST_KEY_COLUMNS,
    "interval_end_utc",
    "actual_load_mwh",
    "forecast_mwh",
    "is_target_day",
    "evaluated",
}
SARIMA_JOB_KEY = JOB_KEY_COLUMNS
FORECAST_KEY = FORECAST_KEY_COLUMNS
JOB_FILENAME = "sarima_validation_2024_jobs.csv"
FORECAST_FILENAME = "sarima_validation_2024_forecasts.csv"
DIAGNOSTIC_FILENAME = "sarima_validation_2024_diagnostics.csv"
SUMMARY_FILENAME = "sarima_validation_2024_summary.csv"

JOB_COLUMNS = [
    "country",
    "target_date",
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
    "forecast_origin_local",
    "forecast_origin_utc",
    "information_cutoff_utc",
    "training_observations",
    "fit_status",
    "convergence_status",
    "optimizer_success",
    "optimizer_status",
    "optimizer_warnflag",
    "optimizer_iterations",
    "optimizer_function_calls",
    "optimizer_message",
    "optimizer_gradient_norm",
    "converged",
    "parameters_finite",
    "standard_errors_finite",
    "forecast_valid",
    "forecast_error_message",
    "nobs",
    "effective_nobs",
    "n_params",
    "log_likelihood",
    "aic",
    "aicc",
    "bic",
    "ar_root_minimum",
    "ma_root_minimum",
    "stationarity_ok",
    "invertibility_ok",
    "warning_messages",
    "hard_warning_messages",
    "optimizer_retry_count",
    "residual_burn_in",
    "residual_state_space_loglikelihood_burn",
    "residual_conservative_burn_in",
    "residual_n_effective",
    "residual_rejected",
    "residual_adequacy_rejected",
    "residual_adequacy_reason",
    "residual_rejection_reason",
    "residual_diagnostic_warnings",
    "residual_acf_values",
    "residual_acf_threshold",
    "residual_acf_flagged_lags",
    "residual_ljung_box_statistics",
    "residual_ljung_box_pvalues",
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
    "status",
    "error_message",
]

FORECAST_COLUMNS = [
    "country",
    "model_family",
    "target_date",
    "specification_id",
    "p",
    "d",
    "q",
    "P",
    "D",
    "Q",
    "seasonal_period",
    "trend",
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

DIAGNOSTIC_COLUMNS = [
    "country",
    "target_date",
    "specification_id",
    "diagnostic_type",
    "fit_status",
    "convergence_status",
    "converged",
    "optimizer_iterations",
    "optimizer_function_calls",
    "optimizer_message",
    "optimizer_gradient_norm",
    "parameters_finite",
    "standard_errors_finite",
    "forecast_valid",
    "forecast_error_message",
    "nobs",
    "effective_nobs",
    "n_params",
    "log_likelihood",
    "aic",
    "aicc",
    "bic",
    "ar_root_minimum",
    "ma_root_minimum",
    "stationarity_ok",
    "invertibility_ok",
    "warning_messages",
    "hard_warning_messages",
    "optimizer_retry_count",
    "residual_burn_in",
    "residual_state_space_loglikelihood_burn",
    "residual_conservative_burn_in",
    "residual_n_effective",
    "residual_adequacy_rejected",
    "residual_rejected",
    "residual_adequacy_reason",
    "residual_rejection_reason",
    "residual_diagnostic_warnings",
    "residual_acf_values",
    "residual_acf_threshold",
    "residual_acf_flagged_lags",
    "residual_ljung_box_statistics",
    "residual_ljung_box_pvalues",
    "status",
    "error_message",
]

_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def _coerce_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def load_validation_country_data(
    country: str,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    path = Path(processed_directory) / COUNTRY_CONFIG[country]["filename"]
    raw = pd.read_csv(path)
    if "timestamp_local" not in raw:
        raise ValueError("validation frame is missing timestamp_local")
    local_dates = pd.to_datetime(
        raw["timestamp_local"].astype(str).str[:10], errors="coerce"
    )
    if local_dates.isna().any():
        raise ValueError("validation frame contains invalid local dates")
    raw = raw.loc[local_dates.dt.date.le(VALIDATION_END_DATE)].copy()
    return _as_prepared(raw)


def _target_date_strings(
    target_dates: Iterable[date | str | pd.Timestamp],
) -> list[str]:
    dates = [_coerce_date(value).isoformat() for value in target_dates]
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate target dates are not allowed")
    if set(dates) != APPROVED_TARGET_DATE_STRINGS:
        raise ValueError(
            "only the approved mini-validation dates are allowed: "
            f"{sorted(APPROVED_TARGET_DATE_STRINGS)}"
        )
    return sorted(dates)


def _full_year_target_date_strings(
    target_dates: Iterable[date | str | pd.Timestamp],
) -> list[str]:
    dates = [_coerce_date(value).isoformat() for value in target_dates]
    if set(dates) != FULL_YEAR_DATE_STRINGS or len(dates) != len(set(dates)):
        raise ValueError("full 2024 validation requires every date from 2024-01-01 through 2024-12-31")
    return sorted(dates)


def _required_shortlist_columns() -> set[str]:
    return {
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


def _order_from_row(row: Mapping[str, object]) -> SarimaOrder:
    order = SarimaOrder(
        int(row["p"]),
        int(row["d"]),
        int(row["q"]),
        int(row["P"]),
        int(row["D"]),
        int(row["Q"]),
        int(row["seasonal_period"]),
    )
    if str(row["specification_id"]) != order.specification_id:
        raise ValueError(
            f"specification_id does not match SARIMA order: {row['specification_id']}"
        )
    expected_trend = trend_for_d_D(order.d, order.D)
    if str(row["trend"]) != expected_trend:
        raise ValueError(
            f"trend does not match SARIMA order {order.specification_id}: {row['trend']}"
        )
    return order


def _validate_shortlist(shortlists: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_required_shortlist_columns().difference(shortlists.columns))
    if missing:
        raise ValueError(f"SARIMA shortlist is missing required columns: {missing}")
    if set(shortlists["country"].astype(str)) != set(COUNTRY_CONFIG):
        raise ValueError("SARIMA shortlist must contain exactly Germany and Austria")

    normalized = shortlists.copy()
    for row in normalized.to_dict("records"):
        _order_from_row(row)
    normalized["country"] = normalized["country"].astype(str)
    normalized["specification_id"] = normalized["specification_id"].astype(str)
    normalized["specification_order"] = pd.to_numeric(
        normalized["specification_order"], errors="raise"
    ).astype(int)
    if normalized.duplicated(["country", "specification_id"]).any():
        raise ValueError("SARIMA shortlist contains duplicate country/specification_id rows")
    return normalized


def load_shortlists(path: str | Path | pd.DataFrame = DEFAULT_SHORTLIST) -> pd.DataFrame:
    shortlists = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    return _validate_shortlist(shortlists)


def build_manifest(
    shortlists: pd.DataFrame,
    target_dates: Iterable[date | str | pd.Timestamp]
    | Mapping[str, Iterable[date | str | pd.Timestamp]],
    *,
    allow_full_year: bool = False,
) -> pd.DataFrame:
    required = _required_shortlist_columns()
    missing = sorted(required.difference(shortlists.columns))
    if missing:
        raise ValueError(f"shortlist is missing required columns: {missing}")

    if isinstance(target_dates, Mapping):
        dates_by_country = {
            country: (
                _full_year_target_date_strings(target_dates.get(country, ()))
                if allow_full_year
                else _target_date_strings(target_dates.get(country, ()))
            )
            for country in COUNTRY_CONFIG
        }
    else:
        dates = (
            _full_year_target_date_strings(target_dates)
            if allow_full_year
            else _target_date_strings(target_dates)
        )
        dates_by_country = {country: dates for country in COUNTRY_CONFIG}

    rows: list[dict[str, object]] = []
    country_rank = {country: index for index, country in enumerate(COUNTRY_CONFIG)}
    for country in COUNTRY_CONFIG:
        country_shortlist = shortlists.loc[
            shortlists["country"].astype(str).eq(country)
        ].sort_values(["specification_order", "specification_id"], kind="mergesort")
        for target_date in dates_by_country[country]:
            for row in country_shortlist.to_dict("records"):
                order = _order_from_row(row)
                rows.append(
                    {
                        "country": country,
                        "target_date": target_date,
                        "specification_id": str(row["specification_id"]),
                        "specification_order": int(row["specification_order"]),
                        "p": order.p,
                        "d": order.d,
                        "q": order.q,
                        "P": order.P,
                        "D": order.D,
                        "Q": order.Q,
                        "seasonal_period": order.seasonal_period,
                        "trend": str(row["trend"]),
                    }
                )

    columns = [
        "country",
        "target_date",
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
    ]
    manifest = pd.DataFrame(rows, columns=columns)
    if manifest.empty:
        return manifest
    manifest["_country_rank"] = manifest["country"].map(country_rank)
    manifest = manifest.sort_values(
        ["_country_rank", "target_date", "specification_order", "specification_id"],
        kind="mergesort",
        ignore_index=True,
    ).drop(columns="_country_rank")
    if manifest.duplicated(list(JOB_KEY_COLUMNS)).any():
        raise ValueError("manifest contains duplicate SARIMA job keys")
    return manifest


def _job_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    if frame.empty or not set(JOB_KEY_COLUMNS).issubset(frame.columns):
        return set()
    return {
        tuple(str(value) for value in row)
        for row in frame.loc[:, list(JOB_KEY_COLUMNS)].itertuples(
            index=False, name=None
        )
    }


def _forecast_timestamps_by_job(
    forecasts: pd.DataFrame,
) -> dict[tuple[str, str, str], set[str]]:
    timestamps: dict[tuple[str, str, str], set[str]] = {}
    if forecasts.empty or not set(FORECAST_KEY_COLUMNS).issubset(forecasts.columns):
        return timestamps
    for row in forecasts.itertuples(index=False):
        raw_timestamp = getattr(row, "timestamp_utc")
        if pd.isna(raw_timestamp):
            continue
        timestamp = pd.Timestamp(raw_timestamp)
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        key = (str(row.country), str(row.target_date), str(row.specification_id))
        timestamps.setdefault(key, set()).add(timestamp.isoformat())
    return timestamps


def load_completed_job_keys(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame | None = None,
    diagnostics: pd.DataFrame | None = None,
    expected_forecast_timestamps: Mapping[
        tuple[str, str, str], set[str]
    ] | None = None,
) -> set[tuple[str, str, str]]:
    if (
        jobs.empty
        or "status" not in jobs
        or forecasts is None
        or diagnostics is None
        or expected_forecast_timestamps is None
        or "diagnostic_type" not in diagnostics
    ):
        return set()
    completed = jobs.loc[jobs["status"].astype(str).eq("completed")]
    completed_keys = _job_keys(completed)
    diagnostic_keys = _job_keys(diagnostics)
    forecast_timestamps = _forecast_timestamps_by_job(forecasts)
    return {
        key
        for key in completed_keys.intersection(diagnostic_keys)
        if _complete_forecast_artifact(
            forecasts,
            key,
            set(expected_forecast_timestamps.get(key, set())),
            forecast_timestamps,
        )
    }


def _complete_forecast_artifact(
    forecasts: pd.DataFrame,
    key: tuple[str, str, str],
    expected_timestamps: set[str],
    forecast_timestamps: Mapping[tuple[str, str, str], set[str]],
) -> bool:
    if not expected_timestamps or not COMPLETION_FORECAST_COLUMNS.issubset(
        forecasts.columns
    ):
        return False
    matching = forecasts.loc[
        forecasts["country"].astype(str).eq(key[0])
        & forecasts["target_date"].astype(str).eq(key[1])
        & forecasts["specification_id"].astype(str).eq(key[2])
    ].copy()
    if matching.empty or matching[list(COMPLETION_FORECAST_COLUMNS)].isna().any().any():
        return False
    timestamps = pd.to_datetime(matching["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any():
        return False
    numeric = matching[["actual_load_mwh", "forecast_mwh"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return False
    return forecast_timestamps.get(key, set()) == expected_timestamps


def _normalise_key_values(frame: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in key_columns:
        if column not in normalized:
            normalized[column] = pd.NA
        if column == "timestamp_utc":
            normalized[column] = pd.to_datetime(
                normalized[column], utc=True, errors="coerce"
            ).map(lambda value: value.isoformat() if pd.notna(value) else "NaT")
        else:
            normalized[column] = normalized[column].astype(str)
    return normalized


def upsert_frame(
    existing: pd.DataFrame,
    replacement: pd.DataFrame,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    if existing.empty and replacement.empty:
        return existing.copy()
    combined = pd.concat([existing, replacement], ignore_index=True, sort=False)
    combined = _normalise_key_values(combined, key_columns)
    combined = combined.drop_duplicates(list(key_columns), keep="last")
    return combined.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)


def _without_job_rows(
    frame: pd.DataFrame,
    job: Mapping[str, object],
) -> pd.DataFrame:
    if frame.empty or not set(JOB_KEY_COLUMNS).issubset(frame.columns):
        return frame.copy()
    same_job = np.ones(len(frame), dtype=bool)
    for column in JOB_KEY_COLUMNS:
        same_job &= frame[column].astype(str).to_numpy() == str(job[column])
    return frame.loc[~same_job].copy()


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


def _validate_hourly_forecast_window(
    path: pd.DataFrame,
    target: pd.DataFrame,
    cutoff_utc: pd.Timestamp,
    local_date: str,
) -> None:
    if path.empty or path["timestamp_utc"].iloc[0] != cutoff_utc:
        raise ValueError("forecast path fails hourly continuity at the forecast origin")
    timestamp_gaps = path["timestamp_utc"].diff().dropna()
    if not timestamp_gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("forecast path fails hourly continuity between origin and target day")
    expected_interval_end = path["timestamp_utc"] + pd.Timedelta(hours=1)
    if not path["interval_end_utc"].eq(expected_interval_end).all():
        raise ValueError("forecast path fails hourly continuity at interval ends")
    target_path = path.loc[path["local_date"].eq(local_date)]
    if len(target_path) != len(target) or not target["timestamp_utc"].isin(
        target_path["timestamp_utc"]
    ).all():
        raise ValueError("forecast path fails hourly continuity on the target day")


def expected_forecast_timestamps(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
) -> set[str]:
    prepared = _as_prepared(frame)
    local_date = _coerce_date(target_date).isoformat()
    context = build_information_context(prepared, country, local_date)
    target = extract_target_day(prepared, local_date)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(context.cutoff_utc)
        & prepared["local_date"].le(local_date)
    ].copy()
    if path.empty or not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError(f"no complete SARIMA forecast path exists for {country} {local_date}")
    _validate_hourly_forecast_window(path, target, context.cutoff_utc, local_date)
    return {timestamp.isoformat() for timestamp in path["timestamp_utc"]}


def build_sarima_forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
    specification_id: str,
    fitted_result: object,
    context: object | None = None,
    order: SarimaOrder | None = None,
    trend: str | None = None,
) -> pd.DataFrame:
    prepared = _as_prepared(frame)
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
        raise ValueError(f"no complete SARIMA forecast path exists for {country} {local_date}")
    _validate_hourly_forecast_window(path, target, context.cutoff_utc, local_date)

    forecasts = forecast_values(fitted_result, len(path))
    path["country"] = country
    path["model_family"] = "SARIMA"
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
    path["trend"] = trend
    path["forecast_origin_local"] = context.origin_local
    path["forecast_origin_utc"] = context.cutoff_utc.isoformat()
    path["information_cutoff_utc"] = context.cutoff_utc.isoformat()
    path["forecast_mwh"] = forecasts
    path["bridge_used"] = ~path["local_date"].eq(local_date)
    path["source_kind"] = "sarima_forecast"
    path["is_target_day"] = path["local_date"].eq(local_date)
    path["evaluated"] = path["is_target_day"]
    return path[FORECAST_COLUMNS]


def _json_value(value: object) -> str:
    return json.dumps(value, default=str, allow_nan=True)


def _fit_metadata(fit: SarimaFitRecord) -> dict[str, object]:
    residual = fit.residual_diagnostics
    return {
        "fit_status": fit.fit_status,
        "convergence_status": fit.convergence_status,
        "optimizer_success": fit.optimizer_success,
        "optimizer_status": fit.optimizer_status,
        "optimizer_warnflag": fit.optimizer_warnflag,
        "optimizer_iterations": fit.optimizer_iterations,
        "optimizer_function_calls": fit.optimizer_function_calls,
        "optimizer_message": fit.optimizer_message,
        "optimizer_gradient_norm": fit.optimizer_gradient_norm,
        "converged": fit.converged,
        "parameters_finite": fit.parameters_finite,
        "standard_errors_finite": fit.standard_errors_finite,
        "forecast_valid": fit.forecast_valid,
        "forecast_error_message": fit.forecast_error_message,
        "nobs": fit.nobs,
        "effective_nobs": fit.effective_nobs,
        "n_params": fit.n_params,
        "log_likelihood": fit.log_likelihood,
        "aic": fit.aic,
        "aicc": fit.aicc,
        "bic": fit.bic,
        "ar_root_minimum": fit.ar_root_minimum,
        "ma_root_minimum": fit.ma_root_minimum,
        "stationarity_ok": fit.stationarity_ok,
        "invertibility_ok": fit.invertibility_ok,
        "warning_messages": _json_value(list(fit.warning_messages)),
        "hard_warning_messages": _json_value(list(fit.hard_warning_messages)),
        "optimizer_retry_count": fit.optimizer_retry_count,
        "residual_burn_in": residual.burn_in,
        "residual_state_space_loglikelihood_burn": residual.state_space_loglikelihood_burn,
        "residual_conservative_burn_in": residual.conservative_burn_in,
        "residual_n_effective": residual.n_effective,
        "residual_rejected": residual.training_adequacy_rejected,
        "residual_adequacy_rejected": residual.training_adequacy_rejected,
        "residual_adequacy_reason": residual.training_adequacy_reason,
        "residual_rejection_reason": residual.training_adequacy_reason,
        "residual_diagnostic_warnings": _json_value(list(residual.warnings)),
        "residual_acf_values": _json_value(residual.acf_values),
        "residual_acf_threshold": residual.acf_threshold,
        "residual_acf_flagged_lags": _json_value(list(residual.flagged_acf_lags)),
        "residual_ljung_box_statistics": _json_value(residual.ljung_box_statistics),
        "residual_ljung_box_pvalues": _json_value(residual.ljung_box_pvalues),
        "error_message": fit.error_message,
    }


def _base_job_row(job: Mapping[str, object], status: str = "failed") -> dict[str, object]:
    row = {column: None for column in JOB_COLUMNS}
    row.update(dict(job))
    row["status"] = status
    row["error_message"] = None
    return row


def _diagnostic_row(job_row: Mapping[str, object]) -> dict[str, object]:
    row = {column: None for column in DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "country": job_row.get("country"),
            "target_date": job_row.get("target_date"),
            "specification_id": job_row.get("specification_id"),
            "diagnostic_type": "sarima_fit_and_residual",
            "fit_status": job_row.get("fit_status"),
            "convergence_status": job_row.get("convergence_status"),
            "converged": job_row.get("converged"),
            "parameters_finite": job_row.get("parameters_finite"),
            "standard_errors_finite": job_row.get("standard_errors_finite"),
            "forecast_valid": job_row.get("forecast_valid"),
            "forecast_error_message": job_row.get("forecast_error_message"),
            "nobs": job_row.get("nobs"),
            "effective_nobs": job_row.get("effective_nobs"),
            "n_params": job_row.get("n_params"),
            "log_likelihood": job_row.get("log_likelihood"),
            "aic": job_row.get("aic"),
            "aicc": job_row.get("aicc"),
            "bic": job_row.get("bic"),
            "ar_root_minimum": job_row.get("ar_root_minimum"),
            "ma_root_minimum": job_row.get("ma_root_minimum"),
            "stationarity_ok": job_row.get("stationarity_ok"),
            "invertibility_ok": job_row.get("invertibility_ok"),
            "warning_messages": job_row.get("warning_messages"),
            "hard_warning_messages": job_row.get("hard_warning_messages"),
            "optimizer_retry_count": job_row.get("optimizer_retry_count"),
            "residual_burn_in": job_row.get("residual_burn_in"),
            "residual_state_space_loglikelihood_burn": job_row.get(
                "residual_state_space_loglikelihood_burn"
            ),
            "residual_conservative_burn_in": job_row.get("residual_conservative_burn_in"),
            "residual_n_effective": job_row.get("residual_n_effective"),
            "residual_adequacy_rejected": job_row.get("residual_rejected"),
            "residual_rejected": job_row.get("residual_rejected"),
            "residual_adequacy_reason": job_row.get("residual_rejection_reason"),
            "residual_rejection_reason": job_row.get("residual_rejection_reason"),
            "residual_diagnostic_warnings": job_row.get("residual_diagnostic_warnings"),
            "residual_acf_values": job_row.get("residual_acf_values"),
            "residual_acf_threshold": job_row.get("residual_acf_threshold"),
            "residual_acf_flagged_lags": job_row.get("residual_acf_flagged_lags"),
            "residual_ljung_box_statistics": job_row.get(
                "residual_ljung_box_statistics"
            ),
            "residual_ljung_box_pvalues": job_row.get("residual_ljung_box_pvalues"),
            "status": job_row.get("status"),
            "error_message": job_row.get("error_message"),
        }
    )
    return row


def _failed_result(
    row: dict[str, object],
    diagnostics: dict[str, object],
    error: Exception | str,
) -> dict[str, object]:
    message = str(error)
    row.update(
        {
            "status": "failed",
            "error_message": message,
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


def _execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    row = _base_job_row(job)
    diagnostics: dict[str, object] = {}
    fit: SarimaFitRecord | None = None
    try:
        country = str(job["country"])
        target_date = str(job["target_date"])
        order = _order_from_row(job)
        context = build_information_context(frame, country, target_date)
        available = information_set(frame, context.cutoff_utc)
        target = extract_target_day(frame, target_date)
        row.update(
            {
                "p": order.p,
                "d": order.d,
                "q": order.q,
                "P": order.P,
                "D": order.D,
                "Q": order.Q,
                "seasonal_period": order.seasonal_period,
                "trend": str(job["trend"]),
                "forecast_origin_local": context.origin_local,
                "forecast_origin_utc": context.cutoff_utc.isoformat(),
                "information_cutoff_utc": context.cutoff_utc.isoformat(),
                "training_observations": len(available),
                "expected_observations": len(target),
            }
        )
        fit = fit_sarima(available["actual_load_mwh"].to_numpy(dtype=float), order)
        row.update(_fit_metadata(fit))
        row["trend"] = fit.trend
        if not fit.eligible or fit.fitted_result is None:
            return _failed_result(row, _diagnostic_row(row), fit.error_message or "SARIMA fit failed")

        path = build_sarima_forecast_path(
            frame,
            country,
            target_date,
            str(job["specification_id"]),
            fit.fitted_result,
            context=context,
            order=order,
            trend=fit.trend,
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
        return {
            "job": row,
            "forecasts": path.to_dict("records"),
            "diagnostics": diagnostics,
        }
    except Exception as error:
        if fit is not None:
            row.update(_fit_metadata(fit))
        return _failed_result(row, diagnostics, error)


def execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    return _execute_job(job, frame)


def _initialize_worker(processed_directory: str) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: load_validation_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _worker_execute(job: Mapping[str, object]) -> dict[str, object]:
    return execute_job(job, _WORKER_FRAMES[str(job["country"])])


def _atomic_write(frame: pd.DataFrame, path: Path, columns: Sequence[str]) -> None:
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


def _load_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame


def _summary_frame(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame,
    minimum_coverage: float = 1.0,
    diagnostics: pd.DataFrame | None = None,
    expected_forecast_timestamps: Mapping[
        tuple[str, str, str], set[str]
    ] | None = None,
) -> pd.DataFrame:
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")
    rows: list[dict[str, object]] = []
    completed_keys = load_completed_job_keys(
        jobs, forecasts, diagnostics, expected_forecast_timestamps
    )
    manifest_keys = {
        tuple(str(value) for value in row)
        for row in manifest.loc[:, list(JOB_KEY_COLUMNS)].itertuples(
            index=False, name=None
        )
    }
    completed_keys = completed_keys.intersection(manifest_keys)
    for country in COUNTRY_CONFIG:
        country_manifest = manifest.loc[manifest["country"].eq(country)]
        for specification_id, expected in country_manifest.groupby(
            "specification_id", sort=False
        ):
            expected_keys = {
                tuple(str(value) for value in row)
                for row in expected.loc[:, list(JOB_KEY_COLUMNS)].itertuples(
                    index=False, name=None
                )
            }
            completed = len(expected_keys.intersection(completed_keys))
            failed = 0
            if not jobs.empty and set(JOB_KEY_COLUMNS + ("status",)).issubset(jobs.columns):
                job_statuses = jobs.loc[
                    jobs["country"].astype(str).eq(country)
                    & jobs["target_date"].astype(str).isin(expected["target_date"])
                    & jobs["specification_id"].astype(str).eq(str(specification_id))
                ]
                failed = int(
                    job_statuses["status"].astype(str).str.lower().eq("failed").sum()
                )
            expected_observations = int(expected["expected_observations"].sum())
            if forecasts.empty:
                target = pd.DataFrame()
            else:
                target = forecasts.loc[
                    forecasts["country"].astype(str).eq(country)
                    & forecasts["target_date"].astype(str).isin(expected["target_date"])
                    & forecasts["specification_id"].astype(str).eq(str(specification_id))
                    & forecasts["is_target_day"].astype(str).str.lower().eq("true")
                    & forecasts["evaluated"].astype(str).str.lower().eq("true")
                ]
                target_keys = list(
                    zip(
                        target["country"].astype(str),
                        target["target_date"].astype(str),
                        target["specification_id"].astype(str),
                        strict=True,
                    )
                )
                target = target.loc[
                    [key in completed_keys for key in target_keys]
                ]
            metrics = calculate_metrics(
                target["actual_load_mwh"] if not target.empty else [],
                target["forecast_mwh"] if not target.empty else [],
                expected_observations=expected_observations,
            )
            specification_order = int(expected["specification_order"].iloc[0])
            rows.append(
                {
                    "country": country,
                    "specification_id": str(specification_id),
                    "specification_order": specification_order,
                    "expected_jobs": len(expected),
                    "completed_jobs": completed,
                    "failed_jobs": failed,
                    "expected_observations": expected_observations,
                    "evaluated_observations": metrics["evaluated_observations"],
                    "coverage": metrics["coverage"],
                    "coverage_eligible": float(metrics["coverage"]) >= minimum_coverage,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                }
            )
    return pd.DataFrame(rows)


def run_validation(
    shortlists: pd.DataFrame | str | Path = DEFAULT_SHORTLIST,
    target_dates: Iterable[date | str | pd.Timestamp] | None = None,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
    minimum_coverage: float = 1.0,
    *,
    shortlist_path: str | Path | None = None,
    full_year: bool = False,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between 0 and 1")
    if shortlist_path is not None:
        shortlists = shortlist_path
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    shortlist_frame = load_shortlists(shortlists)
    frames = {
        country: load_validation_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    if full_year:
        if target_dates is not None:
            raise ValueError("full-year validation cannot receive explicit target dates")
        dates_by_country = {
            country: sorted(
                {
                    value
                    for value in frames[country]["local_date"].astype(str)
                    if value in FULL_YEAR_DATE_STRINGS
                }
            )
            for country in COUNTRY_CONFIG
        }
        manifest = build_manifest(
            shortlist_frame,
            dates_by_country,
            allow_full_year=True,
        )
    else:
        selected_dates = _target_date_strings(
            MINI_DATES if target_dates is None else target_dates
        )
        manifest = build_manifest(shortlist_frame, selected_dates)
    manifest["expected_observations"] = [
        len(extract_target_day(frames[country], target_date))
        for country, target_date in zip(manifest["country"], manifest["target_date"])
    ]
    expected_forecast_timestamps_by_job = {
        (
            str(row["country"]),
            str(row["target_date"]),
            str(row["specification_id"]),
        ): expected_forecast_timestamps(
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
    jobs = _load_csv(jobs_path, JOB_COLUMNS)
    forecasts = _load_csv(forecasts_path, FORECAST_COLUMNS)
    diagnostics = _load_csv(diagnostics_path, DIAGNOSTIC_COLUMNS)
    _atomic_write(
        _summary_frame(
            jobs,
            manifest,
            forecasts,
            minimum_coverage,
            diagnostics=diagnostics,
            expected_forecast_timestamps=expected_forecast_timestamps_by_job,
        ),
        summary_path,
        [
            "country",
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
        ],
    )
    completed_keys = load_completed_job_keys(
        jobs, forecasts, diagnostics, expected_forecast_timestamps_by_job
    )
    pending = [
        row
        for row in manifest.to_dict("records")
        if (
            str(row["country"]),
            str(row["target_date"]),
            str(row["specification_id"]),
        )
        not in completed_keys
    ]

    def persist(result: Mapping[str, object]) -> None:
        nonlocal jobs, forecasts, diagnostics
        incoming_job = pd.DataFrame([result["job"]])
        incoming_forecasts = pd.DataFrame(result.get("forecasts", []))
        diagnostic_payload = result.get("diagnostics") or _diagnostic_row(result["job"])
        incoming_diagnostics = pd.DataFrame([diagnostic_payload])
        jobs = upsert_frame(jobs, incoming_job, JOB_KEY_COLUMNS)
        forecasts = _without_job_rows(forecasts, result["job"])
        forecasts = upsert_frame(forecasts, incoming_forecasts, FORECAST_KEY_COLUMNS)
        diagnostics = upsert_frame(
            diagnostics, incoming_diagnostics, DIAGNOSTIC_KEY_COLUMNS
        )
        _atomic_write(forecasts, forecasts_path, FORECAST_COLUMNS)
        _atomic_write(diagnostics, diagnostics_path, DIAGNOSTIC_COLUMNS)
        _atomic_write(jobs, jobs_path, JOB_COLUMNS)
        _atomic_write(
            _summary_frame(
                jobs,
                manifest,
                forecasts,
                minimum_coverage,
                diagnostics=diagnostics,
                expected_forecast_timestamps=expected_forecast_timestamps_by_job,
            ),
            summary_path,
            [
                "country",
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
            ],
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

    summary = _summary_frame(
        jobs,
        manifest,
        forecasts,
        minimum_coverage,
        diagnostics=diagnostics,
        expected_forecast_timestamps=expected_forecast_timestamps_by_job,
    )
    _atomic_write(
        summary,
        summary_path,
        [
            "country",
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
        ],
    )
    manifest_keys = {
        tuple(str(value) for value in row)
        for row in manifest.loc[:, list(JOB_KEY_COLUMNS)].itertuples(index=False, name=None)
    }
    completed_count = len(
        load_completed_job_keys(
            jobs,
            forecasts,
            diagnostics,
            expected_forecast_timestamps_by_job,
        ).intersection(manifest_keys)
    )
    failed_count = 0
    if not jobs.empty and "status" in jobs:
        for row in jobs.itertuples(index=False):
            key = (str(row.country), str(row.target_date), str(row.specification_id))
            if key in manifest_keys and str(row.status) == "failed":
                failed_count += 1
    return {
        "total_jobs": len(manifest),
        "completed_jobs": completed_count,
        "failed_jobs": failed_count,
        "pending_jobs": len(manifest) - completed_count,
        "forecast_rows": len(forecasts),
        "elapsed_seconds": elapsed,
        "summary": summary.to_dict("records"),
        "output_directory": str(output),
    }


def run_mini_validation(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
    shortlist: str | Path = DEFAULT_SHORTLIST,
    minimum_coverage: float = 1.0,
) -> dict[str, object]:
    return run_validation(
        shortlist,
        MINI_DATES,
        output_directory,
        workers=workers,
        processed_directory=processed_directory,
        minimum_coverage=minimum_coverage,
    )


def run_full_validation(
    output_directory: str | Path = FULL_OUTPUT_DIRECTORY,
    workers: int = 3,
    processed_directory: str | Path = PROCESSED,
    shortlist: str | Path = DEFAULT_SHORTLIST,
    minimum_coverage: float = 1.0,
) -> dict[str, object]:
    return run_validation(
        shortlist,
        target_dates=None,
        output_directory=output_directory,
        workers=workers,
        processed_directory=processed_directory,
        minimum_coverage=minimum_coverage,
        full_year=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded SARIMA 2024 ordinary/DST mini-validation"
    )
    parser.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--target-date",
        action="append",
        dest="target_dates",
        help="explicit 2024 local target date; repeat for a mini workload",
    )
    parser.add_argument(
        "--full-year",
        action="store_true",
        help="run the complete 2024 manifest; requires explicit approval",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimum-coverage", type=float, default=1.0)
    args = parser.parse_args()
    if args.full_year and args.target_dates is not None:
        parser.error("--full-year cannot be combined with --target-date")
    result = run_validation(
        args.shortlist,
        None if args.full_year else (MINI_DATES if args.target_dates is None else args.target_dates),
        FULL_OUTPUT_DIRECTORY if args.full_year else args.output_directory,
        workers=args.workers,
        processed_directory=args.processed_directory,
        minimum_coverage=args.minimum_coverage,
        full_year=args.full_year,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
