from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import threading
from time import perf_counter
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(MODEL_ROOT))

import sarima_validation_2024 as base
from sarima_models import fit_sarima


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "enhanced_sarima"
AUTHORITATIVE_ROOT = PROJECT_ROOT / "results" / "arima_sarima"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
ANALYSIS_LABEL = "enhanced/post-hoc SARIMA robustness analysis"
VALIDATION_YEAR = 2024
TEST_YEAR = 2025
FULL_DATES_2024 = tuple(
    (date(2024, 1, 1) + timedelta(days=offset)).isoformat() for offset in range(366)
)
FULL_DATES_2025 = tuple(
    (date(2025, 1, 1) + timedelta(days=offset)).isoformat() for offset in range(365)
)
SCREENING_DATES = (
    "2024-01-15",
    "2024-02-17",
    "2024-03-10",
    "2024-03-31",
    "2024-04-15",
    "2024-05-18",
    "2024-06-15",
    "2024-07-14",
    "2024-08-12",
    "2024-09-21",
    "2024-10-27",
    "2024-11-11",
    "2024-11-23",
    "2024-12-08",
    "2024-12-16",
    "2024-12-27",
)
BENCHMARK_DATES = ("2024-01-15", "2024-03-31", "2024-06-15", "2024-10-27")
FEASIBILITY_DATE = "2024-06-15"
WORKER_COUNTS = tuple(range(1, 9))
JOB_KEY = ("country", "target_date", "specification_id")
FORECAST_KEY = JOB_KEY + ("timestamp_utc",)

EXTRA_COLUMNS = [
    "analysis_label",
    "runtime_seconds",
    "reused_authoritative",
    "reuse_source",
    "residual_acf_168",
    "residual_acf_168_flagged",
    "worker_pid",
]
JOB_COLUMNS = list(base.JOB_COLUMNS) + EXTRA_COLUMNS
FORECAST_COLUMNS = list(base.FORECAST_COLUMNS) + [
    "analysis_label",
    "reused_authoritative",
    "reuse_source",
]
DIAGNOSTIC_COLUMNS = list(base.DIAGNOSTIC_COLUMNS) + EXTRA_COLUMNS

_WORKER_FRAMES: dict[str, pd.DataFrame] = {}
_WORKER_INCLUDE_2025 = False


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


@dataclass(frozen=True)
class ExtendedSarimaOrder:
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    seasonal_period: int

    @property
    def nonseasonal_order(self) -> tuple[int, int, int]:
        return self.p, self.d, self.q

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return self.P, self.D, self.Q, self.seasonal_period

    @property
    def specification_id(self) -> str:
        return (
            f"sarima_p{self.p}_d{self.d}_q{self.q}_"
            f"P{self.P}_D{self.D}_Q{self.Q}_s{self.seasonal_period}"
        )


def _order_tuple(row: Mapping[str, object]) -> tuple[int, int, int, int, int, int]:
    return tuple(int(row[column]) for column in ("p", "d", "q", "P", "D", "Q"))


def _order_from_row(row: Mapping[str, object]) -> ExtendedSarimaOrder:
    order = ExtendedSarimaOrder(*_order_tuple(row), int(row["seasonal_period"]))
    if str(row["specification_id"]) != order.specification_id:
        raise ValueError("specification_id does not match SARIMA order")
    if str(row.get("trend", "n")) != "n":
        raise ValueError("enhanced SARIMA robustness requires trend=n")
    return order


def candidate_definitions() -> pd.DataFrame:
    orders = (
        (1, 0, 0, 0, 1, 0, 24),
        (2, 0, 0, 0, 1, 0, 24),
        (3, 0, 0, 0, 1, 0, 24),
        (1, 0, 1, 0, 1, 0, 24),
        (2, 0, 1, 0, 1, 0, 24),
        (1, 0, 2, 0, 1, 0, 24),
        (2, 0, 2, 0, 1, 0, 24),
        (1, 0, 0, 1, 1, 0, 24),
        (1, 0, 1, 1, 1, 0, 24),
        (2, 0, 0, 1, 1, 0, 24),
        (2, 0, 1, 1, 1, 0, 24),
        (1, 0, 1, 0, 1, 1, 24),
        (2, 0, 0, 0, 1, 0, 168),
        (1, 0, 1, 0, 1, 0, 168),
    )
    rows: list[dict[str, object]] = []
    for specification_order, values in enumerate(orders, 1):
        p, d, q, P, D, Q, seasonal_period = values
        order = ExtendedSarimaOrder(*values)
        rows.append(
            {
                "specification_order": specification_order,
                "specification_id": order.specification_id,
                "branch": "s24" if seasonal_period == 24 else "s168_feasibility",
                "p": p,
                "d": d,
                "q": q,
                "P": P,
                "D": D,
                "Q": Q,
                "seasonal_period": seasonal_period,
                "trend": "n",
                "order": (p, d, q, P, D, Q),
                "analysis_label": ANALYSIS_LABEL,
            }
        )
    return pd.DataFrame(rows)


def build_screening_calendar() -> pd.DataFrame:
    dst_dates = {"2024-03-31": "spring DST", "2024-10-27": "autumn DST"}
    rows = []
    for target_date in SCREENING_DATES:
        timestamp = pd.Timestamp(target_date)
        rows.append(
            {
                "target_date": target_date,
                "season": (
                    "winter"
                    if timestamp.month in (1, 2, 12)
                    else "spring"
                    if timestamp.month in (3, 4, 5)
                    else "summer"
                    if timestamp.month in (6, 7, 8)
                    else "autumn"
                ),
                "weekday": timestamp.day_name(),
                "is_weekend": timestamp.dayofweek >= 5,
                "dst_period": dst_dates.get(target_date, "none"),
                "analysis_label": ANALYSIS_LABEL,
            }
        )
    return pd.DataFrame(rows)


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _job_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return str(row["country"]), _date_text(row["target_date"]), str(row["specification_id"])


def _write(frame: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    base._atomic_write(frame, path, list(columns))


def _load_all_country_data(country: str, processed_directory: str | Path) -> pd.DataFrame:
    path = Path(processed_directory) / base.COUNTRY_CONFIG[country]["filename"]
    return base._as_prepared(pd.read_csv(path))


def _load_frames(processed_directory: str | Path, *, include_2025: bool = False) -> dict[str, pd.DataFrame]:
    loader = _load_all_country_data if include_2025 else base.load_validation_country_data
    return {
        country: loader(country, processed_directory) for country in base.COUNTRY_CONFIG
    }


def _manifest_from_rows(
    rows: pd.DataFrame,
    target_dates: Iterable[str],
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for country in base.COUNTRY_CONFIG:
        country_rows = (
            rows.loc[rows["country"].astype(str).eq(country)]
            if "country" in rows
            else rows
        )
        for target_date in target_dates:
            target = base.extract_target_day(frames[country], target_date)
            context = base.build_information_context(frames[country], country, target_date)
            for candidate in country_rows.to_dict("records"):
                records.append(
                    {
                        "country": country,
                        "target_date": target_date,
                        "specification_id": str(candidate["specification_id"]),
                        "specification_order": int(candidate["specification_order"]),
                        "p": int(candidate["p"]),
                        "d": int(candidate["d"]),
                        "q": int(candidate["q"]),
                        "P": int(candidate["P"]),
                        "D": int(candidate["D"]),
                        "Q": int(candidate["Q"]),
                        "seasonal_period": int(candidate["seasonal_period"]),
                        "trend": "n",
                        "forecast_origin_local": context.origin_local,
                        "forecast_origin_utc": context.cutoff_utc.isoformat(),
                        "information_cutoff_utc": context.cutoff_utc.isoformat(),
                        "expected_observations": len(target),
                        "analysis_label": ANALYSIS_LABEL,
                    }
                )
    return pd.DataFrame(records)


def _candidate_rows_for_country(candidates: pd.DataFrame, country: str) -> pd.DataFrame:
    return candidates.loc[
        candidates["country"].astype(str).eq(country)
    ] if "country" in candidates else candidates


def _add_extras(
    row: dict[str, object],
    *,
    runtime_seconds: float | None = None,
    reused: bool = False,
    reuse_source: str = "",
    residual_acf_168: float = float("nan"),
    residual_acf_168_flagged: bool = False,
    worker_pid: int | None = None,
) -> None:
    row.update(
        {
            "analysis_label": ANALYSIS_LABEL,
            "runtime_seconds": runtime_seconds,
            "reused_authoritative": reused,
            "reuse_source": reuse_source,
            "residual_acf_168": residual_acf_168,
            "residual_acf_168_flagged": residual_acf_168_flagged,
            "worker_pid": worker_pid,
        }
    )


def _residual_acf_168(fit: object, order: ExtendedSarimaOrder) -> tuple[float, bool]:
    result = getattr(fit, "fitted_result", None)
    try:
        residuals = np.asarray(result.resid, dtype=float).reshape(-1)
        burn_in = max(48, 24 * (order.P + order.Q + order.D) + order.p + order.q)
        residuals = residuals[burn_in:]
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) <= 168:
            return float("nan"), False
        value = float(acf(residuals, nlags=168, fft=True, adjusted=False)[168])
        threshold = 1.96 / np.sqrt(len(residuals))
        return value, bool(np.isfinite(value) and abs(value) > threshold)
    except (AttributeError, TypeError, ValueError, IndexError):
        return float("nan"), False


def _execute_s24_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    started = perf_counter()
    row = {column: None for column in base.JOB_COLUMNS}
    row.update(dict(job))
    diagnostics: dict[str, object] = {}
    fit = None
    acf_168 = float("nan")
    acf_168_flagged = False
    try:
        country = str(job["country"])
        target_date = str(job["target_date"])
        order = _order_from_row(job)
        if order.seasonal_period != 24:
            raise ValueError("only s=24 candidates are valid for origin-by-origin validation")
        context = base.build_information_context(frame, country, target_date)
        available = base.information_set(frame, context.cutoff_utc)
        target = base.extract_target_day(frame, target_date)
        row.update(
            {
                "forecast_origin_local": context.origin_local,
                "forecast_origin_utc": context.cutoff_utc.isoformat(),
                "information_cutoff_utc": context.cutoff_utc.isoformat(),
                "training_observations": len(available),
                "expected_observations": len(target),
            }
        )
        fit = fit_sarima(available["actual_load_mwh"].to_numpy(dtype=float), order)
        row.update(base._fit_metadata(fit))
        row["error_message"] = fit.error_message
        if not fit.eligible or fit.fitted_result is None:
            result = base._failed_result(
                row, base._diagnostic_row(row), fit.error_message or "SARIMA fit failed"
            )
        else:
            acf_168, acf_168_flagged = _residual_acf_168(fit, order)
            path = base.build_sarima_forecast_path(
                frame,
                country,
                target_date,
                str(job["specification_id"]),
                fit.fitted_result,
                context=context,
                order=order,
                trend=fit.trend,
            )
            metrics = base.evaluate_target_day(path, expected_observations=len(target))
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
            diagnostics = base._diagnostic_row(row)
            for forecast in path.to_dict("records"):
                forecast.update(
                    {
                        "analysis_label": ANALYSIS_LABEL,
                        "reused_authoritative": False,
                        "reuse_source": "",
                    }
                )
            result = {
                "job": row,
                "forecasts": path.to_dict("records"),
                "diagnostics": [diagnostics],
            }
        runtime = perf_counter() - started
        _add_extras(
            result["job"],
            runtime_seconds=runtime,
            residual_acf_168=acf_168,
            residual_acf_168_flagged=acf_168_flagged,
            worker_pid=os.getpid(),
        )
        if isinstance(result.get("diagnostics"), dict):
            result["diagnostics"] = [result["diagnostics"]]
        for item in result.get("forecasts", []):
            item.update(
                {
                    "analysis_label": ANALYSIS_LABEL,
                    "reused_authoritative": False,
                    "reuse_source": "",
                }
            )
        for item in result.get("diagnostics", []):
            _add_extras(
                item,
                runtime_seconds=runtime,
                residual_acf_168=acf_168,
                residual_acf_168_flagged=acf_168_flagged,
                worker_pid=os.getpid(),
            )
        return result
    except Exception as error:
        if fit is not None:
            row.update(base._fit_metadata(fit))
        result = base._failed_result(row, diagnostics, error)
        runtime = perf_counter() - started
        _add_extras(
            result["job"],
            runtime_seconds=runtime,
            residual_acf_168=acf_168,
            residual_acf_168_flagged=acf_168_flagged,
            worker_pid=os.getpid(),
        )
        diagnostic_rows = result.get("diagnostics", [])
        if isinstance(diagnostic_rows, dict):
            diagnostic_rows = [diagnostic_rows]
            result["diagnostics"] = diagnostic_rows
        for item in diagnostic_rows:
            _add_extras(item, runtime_seconds=runtime, worker_pid=os.getpid())
        return result


def _authoritative_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = AUTHORITATIVE_ROOT / "validation_2024" / "full_validation"
    return (
        pd.read_csv(directory / "sarima_validation_2024_jobs.csv"),
        pd.read_csv(directory / "sarima_validation_2024_forecasts.csv"),
        pd.read_csv(directory / "sarima_validation_2024_diagnostics.csv"),
    )


def audit_authoritative_reuse(candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    candidates = candidate_definitions() if candidates is None else candidates
    jobs, forecasts, diagnostics = _authoritative_frames()
    rows: list[dict[str, object]] = []
    for country in base.COUNTRY_CONFIG:
        for candidate in candidates.loc[candidates["branch"].eq("s24")].to_dict("records"):
            specification_id = str(candidate["specification_id"])
            matching_jobs = jobs.loc[
                jobs["country"].astype(str).eq(country)
                & jobs["specification_id"].astype(str).eq(specification_id)
            ]
            matching_forecasts = forecasts.loc[
                forecasts["country"].astype(str).eq(country)
                & forecasts["specification_id"].astype(str).eq(specification_id)
                & forecasts["evaluated"].astype(str).str.lower().eq("true")
            ]
            matching_diagnostics = diagnostics.loc[
                diagnostics["country"].astype(str).eq(country)
                & diagnostics["specification_id"].astype(str).eq(specification_id)
            ]
            complete = bool(
                len(matching_jobs) == 366
                and matching_jobs["status"].astype(str).eq("completed").all()
                and matching_forecasts["target_date"].nunique() == 366
                and matching_diagnostics["target_date"].nunique() == 366
            )
            rows.append(
                {
                    "country": country,
                    "specification_id": specification_id,
                    "seasonal_period": int(candidate["seasonal_period"]),
                    "authoritative_jobs": len(matching_jobs),
                    "authoritative_forecast_dates": matching_forecasts["target_date"].nunique(),
                    "authoritative_diagnostic_dates": matching_diagnostics["target_date"].nunique(),
                    "complete_reusable": complete,
                    "reuse_source": str(
                        AUTHORITATIVE_ROOT
                        / "validation_2024"
                        / "full_validation"
                        / "sarima_validation_2024_jobs.csv"
                    ),
                    "analysis_label": ANALYSIS_LABEL,
                }
            )
    audit = pd.DataFrame(rows)
    _write(audit, OUTPUT_ROOT / "audit" / "authoritative_reuse_audit.csv", audit.columns)
    return audit


def _reused_result(
    job: Mapping[str, object],
    authoritative: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> dict[str, object] | None:
    auth_jobs, auth_forecasts, auth_diagnostics = authoritative
    country, target_date, specification_id = _job_key(job)
    matching_job = auth_jobs.loc[
        auth_jobs["country"].astype(str).eq(country)
        & auth_jobs["target_date"].map(_date_text).eq(target_date)
        & auth_jobs["specification_id"].astype(str).eq(specification_id)
    ]
    if len(matching_job) != 1 or str(matching_job.iloc[0]["status"]) != "completed":
        return None
    matching_forecasts = auth_forecasts.loc[
        auth_forecasts["country"].astype(str).eq(country)
        & auth_forecasts["target_date"].map(_date_text).eq(target_date)
        & auth_forecasts["specification_id"].astype(str).eq(specification_id)
    ].copy()
    matching_diagnostics = auth_diagnostics.loc[
        auth_diagnostics["country"].astype(str).eq(country)
        & auth_diagnostics["target_date"].map(_date_text).eq(target_date)
        & auth_diagnostics["specification_id"].astype(str).eq(specification_id)
    ].copy()
    if matching_forecasts.empty or matching_diagnostics.empty:
        return None
    result_job = matching_job.iloc[0].to_dict()
    result_job.update(dict(job))
    result_job.update(matching_job.iloc[0].to_dict())
    result_job.update(dict(job))
    runtime = result_job.get("runtime_seconds")
    _add_extras(
        result_job,
        runtime_seconds=runtime if runtime not in (None, "nan") else float("nan"),
        reused=True,
        reuse_source=str(AUTHORITATIVE_ROOT / "validation_2024" / "full_validation"),
    )
    forecast_rows = matching_forecasts.to_dict("records")
    for item in forecast_rows:
        item.update(
            {
                "analysis_label": ANALYSIS_LABEL,
                "reused_authoritative": True,
                "reuse_source": str(AUTHORITATIVE_ROOT / "validation_2024" / "full_validation"),
            }
        )
    diagnostic_rows = matching_diagnostics.to_dict("records")
    for item in diagnostic_rows:
        _add_extras(item, reused=True, reuse_source=str(AUTHORITATIVE_ROOT / "validation_2024" / "full_validation"))
    return {"job": result_job, "forecasts": forecast_rows, "diagnostics": diagnostic_rows}


def _persist_result(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    result: Mapping[str, object],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    job = dict(result["job"])
    incoming_job = pd.DataFrame([job]).reindex(columns=JOB_COLUMNS)
    incoming_forecasts = pd.DataFrame(result.get("forecasts", [])).reindex(columns=FORECAST_COLUMNS)
    incoming_diagnostics = pd.DataFrame(result.get("diagnostics", [])).reindex(columns=DIAGNOSTIC_COLUMNS)
    jobs = base.upsert_frame(jobs, incoming_job, JOB_KEY)
    forecasts = base._without_job_rows(forecasts, job)
    if not incoming_forecasts.empty:
        forecasts = base.upsert_frame(forecasts, incoming_forecasts, FORECAST_KEY)
    diagnostics = base._without_job_rows(diagnostics, job)
    if not incoming_diagnostics.empty:
        diagnostics = base.upsert_frame(diagnostics, incoming_diagnostics, JOB_KEY)
    _write(jobs, output / "jobs.csv", JOB_COLUMNS)
    _write(forecasts, output / "forecasts.csv", FORECAST_COLUMNS)
    _write(diagnostics, output / "diagnostics.csv", DIAGNOSTIC_COLUMNS)
    return jobs, forecasts, diagnostics


def _load_checkpoint(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        base._load_csv(output / "jobs.csv", JOB_COLUMNS),
        base._load_csv(output / "forecasts.csv", FORECAST_COLUMNS),
        base._load_csv(output / "diagnostics.csv", DIAGNOSTIC_COLUMNS),
    )


def _terminal_keys(jobs: pd.DataFrame) -> set[tuple[str, str, str]]:
    if jobs.empty or not set(JOB_KEY).issubset(jobs.columns):
        return set()
    return {
        _job_key(row)
        for row in jobs.loc[jobs["status"].astype(str).eq("completed")].to_dict("records")
    }


def _summary_from_jobs(jobs: pd.DataFrame, forecasts: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if jobs.empty:
        return pd.DataFrame()
    for (country, specification_id), group in jobs.groupby(
        ["country", "specification_id"], sort=False
    ):
        completed = group.loc[group["status"].astype(str).eq("completed")]
        target_forecasts = forecasts.loc[
            forecasts["country"].astype(str).eq(str(country))
            & forecasts["specification_id"].astype(str).eq(str(specification_id))
            & forecasts["evaluated"].astype(str).str.lower().eq("true")
        ].copy()
        expected = int(pd.to_numeric(group["expected_observations"], errors="coerce").fillna(0).sum())
        numeric = target_forecasts[["actual_load_mwh", "forecast_mwh"]].apply(
            pd.to_numeric, errors="coerce"
        ) if not target_forecasts.empty else pd.DataFrame()
        if not numeric.empty and np.isfinite(numeric.to_numpy(dtype=float)).all():
            metrics = base.calculate_metrics(
                numeric["actual_load_mwh"],
                numeric["forecast_mwh"],
                expected_observations=expected,
            )
        else:
            metrics = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "evaluated_observations": 0,
                "coverage": 0.0,
            }
        first = group.iloc[0]
        rows.append(
            {
                "country": str(country),
                "specification_id": str(specification_id),
                "specification_order": int(first["specification_order"]),
                "p": int(first["p"]),
                "d": int(first["d"]),
                "q": int(first["q"]),
                "P": int(first["P"]),
                "D": int(first["D"]),
                "Q": int(first["Q"]),
                "seasonal_period": int(first["seasonal_period"]),
                "trend": str(first["trend"]),
                "expected_jobs": len(group),
                "completed_jobs": len(completed),
                "failed_jobs": int((~group["status"].astype(str).eq("completed")).sum()),
                "expected_observations": expected,
                "evaluated_observations": int(metrics["evaluated_observations"]),
                "coverage": float(metrics["coverage"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
                "valid_dates": int(completed["target_date"].nunique()),
                "invalid_dates": int(len(group) - completed["target_date"].nunique()),
                "mean_runtime_seconds": float(
                    pd.to_numeric(group["runtime_seconds"], errors="coerce").mean()
                ),
                "median_runtime_seconds": float(
                    pd.to_numeric(group["runtime_seconds"], errors="coerce").median()
                ),
                "converged_jobs": int(group["converged"].map(_bool_value).sum())
                if "converged" in group
                else 0,
                "residual_rejected_jobs": int(
                    group["residual_rejected"].map(_bool_value).sum()
                )
                if "residual_rejected" in group
                else 0,
                "residual_acf_168_mean": float(
                    pd.to_numeric(group["residual_acf_168"], errors="coerce").mean()
                ),
                "eligible": bool(
                    len(completed) == len(group)
                    and np.isfinite(float(metrics["mae"]))
                    and np.isfinite(float(metrics["rmse"]))
                    and np.isfinite(float(metrics["mape"]))
                    and float(metrics["coverage"]) >= 1.0
                ),
                "stage": stage,
                "analysis_label": ANALYSIS_LABEL,
            }
        )
    return pd.DataFrame(rows)


def select_shortlist(screening: pd.DataFrame) -> pd.DataFrame:
    required = {"country", "specification_id", "mae", "rmse", "mape", "eligible"}
    missing = sorted(required.difference(screening.columns))
    if missing:
        raise ValueError(f"screening frame is missing required columns: {missing}")
    rows: list[pd.Series] = []
    for country, group in screening.groupby("country", sort=False):
        eligible = group.loc[group["eligible"].map(_bool_value)].copy()
        if eligible.empty:
            continue
        for column in ("mae", "rmse", "mape"):
            eligible[f"_{column}"] = pd.to_numeric(eligible[column], errors="coerce")
        eligible["_order"] = pd.to_numeric(
            eligible.get("specification_order", pd.Series(999999, index=eligible.index)),
            errors="coerce",
        ).fillna(999999)
        selected = eligible.sort_values(
            ["_mae", "_rmse", "_mape", "_order", "specification_id"], kind="stable"
        ).head(3)
        rows.extend(selected.drop(columns=["_mae", "_rmse", "_mape", "_order"]).itertuples(index=False, name=None))
    if not rows:
        return screening.iloc[0:0].copy()
    return pd.DataFrame(rows, columns=screening.columns)


def run_screening(
    processed_directory: str | Path = PROCESSED_ROOT,
    workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output = OUTPUT_ROOT / "screening_2024"
    output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_definitions()
    _write(candidates, OUTPUT_ROOT / "specifications" / "candidate_definitions.csv", candidates.columns)
    calendar = build_screening_calendar()
    _write(calendar, OUTPUT_ROOT / "specifications" / "screening_calendar_2024.csv", calendar.columns)
    audit = audit_authoritative_reuse(candidates)
    frames = _load_frames(processed_directory)
    manifest = _manifest_from_rows(
        candidates.loc[candidates["branch"].eq("s24")], SCREENING_DATES, frames
    )
    _write(manifest, output / "manifest.csv", manifest.columns)
    jobs, forecasts, diagnostics = _load_checkpoint(output)
    authoritative = _authoritative_frames()
    terminal = _terminal_keys(jobs)
    pending: list[dict[str, object]] = []
    for job in manifest.to_dict("records"):
        key = _job_key(job)
        if key in terminal:
            continue
        result = _reused_result(job, authoritative)
        if result is not None:
            jobs, forecasts, diagnostics = _persist_result(
                jobs, forecasts, diagnostics, result, output
            )
            terminal.add(key)
        else:
            pending.append(job)
    worker_pids: set[int] = set()
    if workers == 1:
        results = [
            _execute_s24_job(job, frames[str(job["country"])]) for job in pending
        ]
    else:
        pending_frame = pd.DataFrame(pending) if pending else manifest.iloc[0:0].copy()
        results = _run_parallel(pending_frame, workers, processed_directory)
    for result in results:
        jobs, forecasts, diagnostics = _persist_result(
            jobs, forecasts, diagnostics, result, output
        )
        worker_pid = result.get("job", {}).get("worker_pid")
        if worker_pid is not None:
            worker_pids.add(int(worker_pid))
    summary = _summary_from_jobs(jobs, forecasts, stage="screening_2024")
    _write(summary, output / "summary.csv", summary.columns)
    shortlist = select_shortlist(summary)
    shortlist["analysis_label"] = ANALYSIS_LABEL
    _write(shortlist, output / "shortlist_2024.csv", shortlist.columns)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(jobs["status"].astype(str).eq("completed").sum()),
        "failed_jobs": int(jobs["status"].astype(str).eq("failed").sum()),
        "reused_jobs": int(jobs["reused_authoritative"].map(_bool_value).sum()),
        "shortlist_rows": len(shortlist),
        "authoritative_reuse_rows": int(audit["complete_reusable"].sum()),
        "requested_workers": workers,
        "active_worker_processes": len(worker_pids) if workers > 1 else 1,
        "worker_pids": sorted(worker_pids),
    }


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0.0


def _process_tree_rss(root_pid: int) -> float:
    parents: dict[int, int] = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
            with (entry / "status").open(encoding="utf-8") as handle:
                parent = None
                for line in handle:
                    if line.startswith("PPid:"):
                        parent = int(line.split()[1])
                        break
            if parent is not None:
                parents[pid] = parent
        except (FileNotFoundError, OSError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(_rss_mb(pid) for pid in descendants)


def _swap_used_mb() -> float:
    try:
        values = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("SwapTotal:", "SwapFree:")):
                    values[line.split(":", 1)[0]] = float(line.split()[1])
        return max(0.0, (values.get("SwapTotal", 0.0) - values.get("SwapFree", 0.0)) / 1024.0)
    except (FileNotFoundError, OSError, ValueError):
        return float("nan")


class _RssMonitor:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, _process_tree_rss(os.getpid()))
            self.stop.wait(0.2)

    def __enter__(self) -> _RssMonitor:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        self.peak = max(self.peak, _process_tree_rss(os.getpid()))


def _initialize_worker(processed_directory: str, include_2025: bool) -> None:
    global _WORKER_FRAMES, _WORKER_INCLUDE_2025
    _WORKER_INCLUDE_2025 = include_2025
    _WORKER_FRAMES = _load_frames(processed_directory, include_2025=include_2025)


def _worker_execute(job: Mapping[str, object]) -> dict[str, object]:
    result = _execute_s24_job(job, _WORKER_FRAMES[str(job["country"])])
    result["job"]["worker_pid"] = os.getpid()
    for item in result.get("diagnostics", []):
        item["worker_pid"] = os.getpid()
    return result


def _run_parallel(
    manifest: pd.DataFrame,
    workers: int,
    processed_directory: str | Path,
    *,
    include_2025: bool = False,
    on_result: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    context = mp.get_context("spawn")
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(processed_directory), include_2025),
    ) as executor:
        for result in executor.map(
            _worker_execute, manifest.to_dict("records"), chunksize=1
        ):
            if on_result is None:
                results.append(result)
            else:
                on_result(result)
    return results


def _forecast_map(forecasts: pd.DataFrame) -> dict[tuple[str, str, str, str], float]:
    result: dict[tuple[str, str, str, str], float] = {}
    if forecasts.empty:
        return result
    for row in forecasts.to_dict("records"):
        key = (
            str(row["country"]),
            _date_text(row["target_date"]),
            str(row["specification_id"]),
            pd.Timestamp(row["timestamp_utc"]).tz_convert("UTC").isoformat(),
        )
        result[key] = float(row["forecast_mwh"])
    return result


def run_worker_benchmark(
    processed_directory: str | Path = PROCESSED_ROOT,
) -> dict[str, object]:
    screening_dir = OUTPUT_ROOT / "screening_2024"
    shortlist = pd.read_csv(screening_dir / "shortlist_2024.csv")
    frames = _load_frames(processed_directory)
    manifest = _manifest_from_rows(shortlist, BENCHMARK_DATES, frames)
    benchmark = OUTPUT_ROOT / "benchmark"
    _write(manifest, benchmark / "workload.csv", manifest.columns)
    records: list[dict[str, object]] = []
    baseline_map: dict[tuple[str, str, str, str], float] | None = None
    for worker_count in WORKER_COUNTS:
        worker_output = benchmark / f"workers_{worker_count}"
        started = perf_counter()
        with _RssMonitor() as monitor:
            results = _run_parallel(manifest, worker_count, processed_directory)
        elapsed = perf_counter() - started
        jobs = pd.DataFrame([result["job"] for result in results])
        forecasts = pd.DataFrame(
            [item for result in results for item in result.get("forecasts", [])]
        )
        diagnostics = pd.DataFrame(
            [item for result in results for item in result.get("diagnostics", [])]
        )
        if baseline_map is None:
            baseline_map = _forecast_map(forecasts)
            equivalent = True
        else:
            current = _forecast_map(forecasts)
            equivalent = set(current) == set(baseline_map) and all(
                np.isclose(current[key], baseline_map[key], rtol=1e-10, atol=1e-12)
                for key in baseline_map
            )
        active_workers = int(jobs["worker_pid"].nunique()) if not jobs.empty else 0
        failures = int(jobs["status"].astype(str).eq("failed").sum()) if not jobs.empty else len(manifest)
        peak = float(monitor.peak)
        safe = bool(
            failures == 0
            and equivalent
            and active_workers == worker_count
            and (not np.isfinite(_swap_used_mb()) or _swap_used_mb() <= 0.0)
            and peak < 24_000.0
        )
        _write(jobs, worker_output / "jobs.csv", JOB_COLUMNS)
        _write(forecasts, worker_output / "forecasts.csv", FORECAST_COLUMNS)
        _write(diagnostics, worker_output / "diagnostics.csv", DIAGNOSTIC_COLUMNS)
        summary = _summary_from_jobs(jobs, forecasts, stage=f"benchmark_workers_{worker_count}")
        _write(summary, worker_output / "summary.csv", summary.columns)
        records.append(
            {
                "requested_workers": worker_count,
                "actually_active_workers": active_workers,
                "elapsed_seconds": elapsed,
                "jobs_per_minute": len(manifest) / elapsed * 60.0 if elapsed else float("nan"),
                "failures": failures,
                "peak_total_rss_mb": peak,
                "swap_mb": _swap_used_mb(),
                "forecast_equivalence": equivalent,
                "safe": safe,
                "analysis_label": ANALYSIS_LABEL,
            }
        )
    benchmark_summary = pd.DataFrame(records)
    _write(benchmark_summary, benchmark / "worker_benchmark_summary.csv", benchmark_summary.columns)
    safe_rows = benchmark_summary.loc[benchmark_summary["safe"]]
    if safe_rows.empty:
        raise RuntimeError("no safe worker count completed the enhanced SARIMA benchmark")
    selected = safe_rows.sort_values("elapsed_seconds", kind="stable").iloc[0]
    selection = pd.DataFrame(
        [
            {
                "selected_workers": int(selected["requested_workers"]),
                "selection_rule": "fastest safe worker count by elapsed_seconds",
                "analysis_label": ANALYSIS_LABEL,
            }
        ]
    )
    _write(selection, benchmark / "worker_selection.csv", selection.columns)
    return {
        "workload_jobs": len(manifest),
        "selected_workers": int(selected["requested_workers"]),
        "benchmark_rows": len(benchmark_summary),
    }


def run_full_validation(
    processed_directory: str | Path = PROCESSED_ROOT,
    workers: int | None = None,
) -> dict[str, object]:
    screening_dir = OUTPUT_ROOT / "screening_2024"
    shortlist = pd.read_csv(screening_dir / "shortlist_2024.csv")
    frames = _load_frames(processed_directory)
    manifest = _manifest_from_rows(shortlist, FULL_DATES_2024, frames)
    output = OUTPUT_ROOT / "validation_2024"
    output.mkdir(parents=True, exist_ok=True)
    _write(manifest, output / "manifest.csv", manifest.columns)
    jobs, forecasts, diagnostics = _load_checkpoint(output)
    authoritative = _authoritative_frames()
    terminal = _terminal_keys(jobs)
    pending: list[dict[str, object]] = []
    for job in manifest.to_dict("records"):
        key = _job_key(job)
        if key in terminal:
            continue
        reused = _reused_result(job, authoritative)
        if reused is not None:
            jobs, forecasts, diagnostics = _persist_result(
                jobs, forecasts, diagnostics, reused, output
            )
            terminal.add(key)
        else:
            pending.append(job)
    if pending:
        selected_workers = workers
        if selected_workers is None:
            selection = pd.read_csv(OUTPUT_ROOT / "benchmark" / "worker_selection.csv")
            selected_workers = int(selection.iloc[0]["selected_workers"])

        def persist(result: dict[str, object]) -> None:
            nonlocal jobs, forecasts, diagnostics
            jobs, forecasts, diagnostics = _persist_result(
                jobs, forecasts, diagnostics, result, output
            )

        _run_parallel(
            pd.DataFrame(pending),
            selected_workers,
            processed_directory,
            on_result=persist,
        )
    summary = _summary_from_jobs(jobs, forecasts, stage="validation_2024")
    _write(summary, output / "summary.csv", summary.columns)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(jobs["status"].astype(str).eq("completed").sum()),
        "failed_jobs": int(jobs["status"].astype(str).eq("failed").sum()),
        "reused_jobs": int(jobs["reused_authoritative"].map(_bool_value).sum()),
        "forecast_rows": len(forecasts),
        "diagnostic_rows": len(diagnostics),
    }


def freeze_2024() -> dict[str, object]:
    output = OUTPUT_ROOT / "validation_2024"
    jobs, forecasts, diagnostics = _load_checkpoint(output)
    summary = pd.read_csv(output / "summary.csv")
    rows: list[dict[str, object]] = []
    for (country, specification_id), group in jobs.groupby(
        ["country", "specification_id"], sort=False
    ):
        completed = group["status"].astype(str).eq("completed")
        finite_metrics = group[["mae", "rmse", "mape"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        metadata_ok = group[["converged", "parameters_finite", "stationarity_ok", "invertibility_ok"]].apply(
            lambda col: col.map(_bool_value)
        ).all(axis=1)
        diagnostic_rows = diagnostics.loc[
            diagnostics["country"].astype(str).eq(str(country))
            & diagnostics["specification_id"].astype(str).eq(str(specification_id))
        ]
        candidate_summary = summary.loc[
            summary["country"].astype(str).eq(str(country))
            & summary["specification_id"].astype(str).eq(str(specification_id))
        ].iloc[0]
        reason_parts = []
        if not completed.all():
            reason_parts.append("incomplete jobs")
        if not finite_metrics.all():
            reason_parts.append("non-finite target metrics")
        if not metadata_ok.all():
            reason_parts.append("convergence or root metadata failed")
        if len(diagnostic_rows) != len(group):
            reason_parts.append("incomplete diagnostic review")
        rows.append(
            {
                "country": str(country),
                "specification_id": str(specification_id),
                "specification_order": int(group.iloc[0]["specification_order"]),
                "p": int(group.iloc[0]["p"]),
                "d": int(group.iloc[0]["d"]),
                "q": int(group.iloc[0]["q"]),
                "P": int(group.iloc[0]["P"]),
                "D": int(group.iloc[0]["D"]),
                "Q": int(group.iloc[0]["Q"]),
                "seasonal_period": int(group.iloc[0]["seasonal_period"]),
                "trend": "n",
                "mae": float(candidate_summary["mae"]),
                "rmse": float(candidate_summary["rmse"]),
                "mape": float(candidate_summary["mape"]),
                "valid_jobs": int(completed.sum()),
                "invalid_jobs": int((~completed).sum()),
                "diagnostics_reviewed": len(diagnostic_rows) == len(group),
                "convergence_adequate": bool(metadata_ok.all()),
                "residual_rejected_jobs": int(group["residual_rejected"].map(_bool_value).sum()),
                "adequate": not reason_parts,
                "inadequacy_reasons": "; ".join(reason_parts),
                "analysis_label": ANALYSIS_LABEL,
            }
        )
    adequacy = pd.DataFrame(rows)
    frozen_rows: list[dict[str, object]] = []
    for country, group in adequacy.groupby("country", sort=False):
        eligible = group.loc[group["adequate"].map(_bool_value)].sort_values(
            ["mae", "rmse", "mape", "specification_order"], kind="stable"
        )
        if eligible.empty:
            raise RuntimeError(f"no adequate enhanced SARIMA candidate for {country}")
        chosen = eligible.iloc[0].to_dict()
        chosen.update(
            {
                "selection_status": "frozen",
                "selection_rule": "2024 MAE, then RMSE, MAPE, specification order",
                "frozen_for_year": 2025,
                "analysis_label": ANALYSIS_LABEL,
            }
        )
        frozen_rows.append(chosen)
    freeze = pd.DataFrame(frozen_rows)
    _write(adequacy, OUTPUT_ROOT / "freeze_2024" / "adequacy_review_2024.csv", adequacy.columns)
    _write(freeze, OUTPUT_ROOT / "freeze_2024" / "frozen_specifications_2024.csv", freeze.columns)
    return {
        "adequacy_rows": len(adequacy),
        "frozen": freeze[["country", "specification_id"]].to_dict("records"),
    }


def _build_test_2025_manifest(
    freeze: pd.DataFrame, frames: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = freeze.copy()
    return _manifest_from_rows(rows, FULL_DATES_2025, frames)


def run_test_2025(
    processed_directory: str | Path = PROCESSED_ROOT,
    workers: int | None = None,
) -> dict[str, object]:
    freeze = pd.read_csv(OUTPUT_ROOT / "freeze_2024" / "frozen_specifications_2024.csv")
    frames = _load_frames(processed_directory, include_2025=True)
    manifest = _build_test_2025_manifest(freeze, frames)
    output = OUTPUT_ROOT / "test_2025"
    _write(manifest, output / "manifest.csv", manifest.columns)
    jobs, forecasts, diagnostics = _load_checkpoint(output)
    terminal = _terminal_keys(jobs)
    pending = manifest.loc[
        ~manifest.apply(lambda row: _job_key(row) in terminal, axis=1)
    ].copy()
    if not pending.empty:
        if workers is None:
            selection = pd.read_csv(OUTPUT_ROOT / "benchmark" / "worker_selection.csv")
            workers = int(selection.iloc[0]["selected_workers"])

        def persist(result: dict[str, object]) -> None:
            nonlocal jobs, forecasts, diagnostics
            jobs, forecasts, diagnostics = _persist_result(
                jobs, forecasts, diagnostics, result, output
            )

        _run_parallel(
            pending,
            workers,
            processed_directory,
            include_2025=True,
            on_result=persist,
        )
    summary = _summary_from_jobs(jobs, forecasts, stage="test_2025")
    _write(summary, output / "summary.csv", summary.columns)
    _write(summary, output / "native_summary.csv", summary.columns)
    _write(summary, output / "common_summary.csv", summary.columns)
    smard_benchmark = pd.read_csv(
        PROJECT_ROOT / "results" / "official_benchmarks" / "smard_2025_benchmark.csv"
    )
    comparison = build_final_comparison(
        summary, forecasts, smard_benchmark=smard_benchmark
    )
    _write(
        comparison,
        OUTPUT_ROOT / "comparison" / "enhanced_sarima_final_comparison_2025.csv",
        comparison.columns,
    )
    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(jobs["status"].astype(str).eq("completed").sum()),
        "failed_jobs": int(jobs["status"].astype(str).eq("failed").sum()),
        "forecast_rows": len(forecasts),
        "diagnostic_rows": len(diagnostics),
        "comparison_rows": len(comparison),
    }


def _comparison_row_from_frame(
    frame: pd.DataFrame,
    country: str,
    model: str,
    aliases: Iterable[str] = (),
) -> dict[str, object]:
    allowed = {model, *aliases}
    matching = frame.loc[
        frame["country"].astype(str).eq(country)
        & frame["model"].astype(str).isin(allowed)
    ]
    if matching.empty:
        raise FileNotFoundError(f"comparison row missing: {country} {model}")
    row = matching.iloc[0].to_dict()
    for metric in ("mae", "rmse", "mape"):
        row.setdefault(f"native_{metric}", row.get(metric))
        row.setdefault(f"common_{metric}", row.get(metric))
    observations = row.get("evaluated_observations")
    expected = row.get("expected_observations", observations)
    coverage = row.get("coverage", 0.0)
    row.setdefault("native_observations", observations)
    row.setdefault("common_observations", observations)
    row.setdefault("native_coverage", coverage)
    row.setdefault("common_coverage", coverage)
    row.setdefault("common_timestamp_count", row.get("common_observations", observations))
    row.setdefault("expected_observations", expected)
    row.setdefault("valid_dates", 0)
    row.setdefault("invalid_dates", 0)
    row.setdefault("invalid_date_reasons", "")
    row.setdefault("evaluation_role", "external_comparator")
    row.setdefault("comparison_role", "external_comparator")
    row.setdefault("selection_role", "not_used_for_selection")
    return row


def build_final_comparison(
    summary: pd.DataFrame,
    forecasts: pd.DataFrame,
    comparator_metrics: pd.DataFrame | None = None,
    smard_benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    existing_path = PROJECT_ROOT / "results" / "enhanced_arima" / "comparison" / "enhanced_arima_final_comparison_2025.csv"
    existing = (
        comparator_metrics.copy()
        if comparator_metrics is not None
        else pd.read_csv(existing_path)
    )
    rows: list[dict[str, object]] = []
    model_order = (
        "Original SARIMA",
        "Enhanced SARIMA",
        "Original ARIMA",
        "Enhanced ARIMA",
        "Weekly seasonal naive",
        "Holt-Winters",
        "SMARD",
    )
    source_models = {
        "Original SARIMA": "original SARIMA",
        "Original ARIMA": "original ARIMA (2,1,2)",
        "Enhanced ARIMA": "enhanced ARIMA",
        "Weekly seasonal naive": "weekly seasonal naive",
        "Holt-Winters": "Holt-Winters",
        "SMARD": "SMARD",
    }
    enhanced_sarima = {
        str(row["country"]): row
        for row in summary.to_dict("records")
    }
    for country in base.COUNTRY_CONFIG:
        original_sarima = _comparison_row_from_frame(
            existing,
            country,
            "Original SARIMA",
            aliases=(source_models["Original SARIMA"],),
        )
        original_mae = float(original_sarima["mae"])
        for model in model_order:
            if model == "Enhanced SARIMA":
                source = enhanced_sarima[country]
                row = {
                    "country": country,
                    "model": model,
                    "model_family": "enhanced_sarima",
                    "specification_id": source["specification_id"],
                    "mae": source["mae"],
                    "rmse": source["rmse"],
                    "mape": source["mape"],
                    "evaluated_observations": source["evaluated_observations"],
                    "expected_observations": source["expected_observations"],
                    "coverage": source["coverage"],
                    "native_mae": source["mae"],
                    "native_rmse": source["rmse"],
                    "native_mape": source["mape"],
                    "native_observations": source["evaluated_observations"],
                    "native_coverage": source["coverage"],
                    "common_mae": source["mae"],
                    "common_rmse": source["rmse"],
                    "common_mape": source["mape"],
                    "common_observations": source["evaluated_observations"],
                    "common_coverage": source["coverage"],
                    "common_timestamp_count": source["evaluated_observations"],
                    "valid_dates": source.get("valid_dates", 0),
                    "invalid_dates": source.get("invalid_dates", 0),
                    "invalid_date_reasons": "",
                    "evaluation_role": "descriptive_post_hoc_robustness",
                    "comparison_role": "descriptive_post_hoc_robustness",
                    "selection_role": "not_used_for_selection",
                    "source_kind": "enhanced_sarima_2025",
                }
            elif model == "SMARD" and smard_benchmark is not None:
                benchmark_rows = smard_benchmark.loc[
                    smard_benchmark["country"].astype(str).eq(country)
                ]
                if benchmark_rows.empty:
                    raise FileNotFoundError(f"SMARD benchmark row missing: {country}")
                source = benchmark_rows.iloc[0].to_dict()
                observations = source.get("n_observations", source.get("evaluated_observations"))
                coverage = source.get("coverage", 0.0)
                row = {
                    "country": country,
                    "model": model,
                    "model_family": "smard",
                    "specification_id": "official_smard_benchmark",
                    "mae": source.get("mae"),
                    "rmse": source.get("rmse"),
                    "mape": source.get("mape"),
                    "evaluated_observations": observations,
                    "expected_observations": observations,
                    "coverage": coverage,
                    "native_mae": source.get("mae"),
                    "native_rmse": source.get("rmse"),
                    "native_mape": source.get("mape"),
                    "native_observations": observations,
                    "native_coverage": coverage,
                    "common_mae": source.get("mae"),
                    "common_rmse": source.get("rmse"),
                    "common_mape": source.get("mape"),
                    "common_observations": observations,
                    "common_coverage": coverage,
                    "common_timestamp_count": observations,
                    "valid_dates": 0,
                    "invalid_dates": 0,
                    "invalid_date_reasons": "",
                    "evaluation_role": "external_comparator",
                    "comparison_role": "external_comparator",
                    "selection_role": "not_used_for_selection",
                    "source_kind": "official_smard_benchmark_2025",
                }
            else:
                row = _comparison_row_from_frame(
                    existing,
                    country,
                    model,
                    aliases=(source_models[model],),
                )
                row["model"] = model
            row["analysis_label"] = ANALYSIS_LABEL
            row["mae_change_vs_original_sarima"] = (
                float(row["mae"]) - original_mae
                if pd.notna(row.get("mae"))
                else float("nan")
            )
            row["mae_improvement_vs_original_sarima"] = (
                original_mae - float(row["mae"])
                if pd.notna(row.get("mae"))
                else float("nan")
            )
            row["mae_improvement_pct_vs_original_sarima"] = (
                (original_mae - float(row["mae"])) / original_mae * 100.0
                if pd.notna(row.get("mae")) and original_mae
                else float("nan")
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    result["_country_order"] = result["country"].map(
        {country: index for index, country in enumerate(base.COUNTRY_CONFIG)}
    )
    result["_model_order"] = result["model"].map(
        {model: index for index, model in enumerate(model_order)}
    )
    return result.sort_values(["_country_order", "_model_order"], kind="stable").drop(
        columns=["_country_order", "_model_order"]
    ).reset_index(drop=True)


def _run_feasibility_child(
    country: str,
    candidate: dict[str, object],
    processed_directory: str,
    connection: object,
) -> None:
    started = perf_counter()
    peak_rss = 0.0
    try:
        frame = _load_all_country_data(country, processed_directory)
        target_date = FEASIBILITY_DATE
        context = base.build_information_context(frame, country, target_date)
        available = base.information_set(frame, context.cutoff_utc)
        target = base.extract_target_day(frame, target_date)
        order = _order_from_row(candidate)
        fit = fit_sarima(
            available["actual_load_mwh"].to_numpy(dtype=float),
            order,
            fit_kwargs={"method": "lbfgs", "maxiter": 50, "disp": False},
            retry_maxiter=None,
            require_standard_errors=False,
        )
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        result = {
            "country": country,
            "specification_id": candidate["specification_id"],
            "seasonal_period": candidate["seasonal_period"],
            "target_date": target_date,
            "status": "completed" if fit.eligible and fit.fitted_result is not None else "not_converged",
            "convergence_status": fit.convergence_status,
            "converged": fit.converged,
            "runtime_seconds": perf_counter() - started,
            "optimizer_iterations": fit.optimizer_iterations,
            "state_dimension": getattr(getattr(fit.fitted_result, "model", None), "k_states", None),
            "peak_rss_mb": peak_rss,
            "target_day_mae": float("nan"),
            "error_message": fit.error_message or "",
            "analysis_label": ANALYSIS_LABEL,
        }
        if fit.eligible and fit.fitted_result is not None:
            path = base.build_sarima_forecast_path(
                frame,
                country,
                target_date,
                str(candidate["specification_id"]),
                fit.fitted_result,
                context=context,
                order=order,
                trend=fit.trend,
            )
            metrics = base.evaluate_target_day(path, expected_observations=len(target))
            result["target_day_mae"] = float(metrics["mae"])
        connection.send(result)
    except Exception as error:
        connection.send(
            {
                "country": country,
                "specification_id": candidate["specification_id"],
                "seasonal_period": candidate["seasonal_period"],
                "target_date": FEASIBILITY_DATE,
                "status": "failed",
                "convergence_status": "failed",
                "converged": False,
                "runtime_seconds": perf_counter() - started,
                "optimizer_iterations": None,
                "state_dimension": None,
                "peak_rss_mb": peak_rss,
                "target_day_mae": float("nan"),
                "error_message": f"{type(error).__name__}: {error}",
                "analysis_label": ANALYSIS_LABEL,
            }
        )
    finally:
        connection.close()


def run_s168_feasibility(
    processed_directory: str | Path = PROCESSED_ROOT,
) -> dict[str, object]:
    candidates = candidate_definitions().loc[lambda frame: frame["branch"].eq("s168_feasibility")]
    context = mp.get_context("spawn")
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        for country in base.COUNTRY_CONFIG:
            parent, child = context.Pipe(False)
            process = context.Process(
                target=_run_feasibility_child,
                args=(country, candidate, str(processed_directory), child),
            )
            started = perf_counter()
            process.start()
            child.close()
            process.join(900)
            if process.is_alive():
                process.terminate()
                process.join(30)
                rows.append(
                    {
                        "country": country,
                        "specification_id": candidate["specification_id"],
                        "seasonal_period": candidate["seasonal_period"],
                        "target_date": FEASIBILITY_DATE,
                        "status": "timed_out",
                        "convergence_status": "timeout",
                        "converged": False,
                        "runtime_seconds": perf_counter() - started,
                        "optimizer_iterations": None,
                        "state_dimension": None,
                        "peak_rss_mb": float("nan"),
                        "target_day_mae": float("nan"),
                        "error_message": "15-minute feasibility limit exceeded",
                        "analysis_label": ANALYSIS_LABEL,
                    }
                )
            elif parent.poll():
                rows.append(parent.recv())
            else:
                rows.append(
                    {
                        "country": country,
                        "specification_id": candidate["specification_id"],
                        "seasonal_period": candidate["seasonal_period"],
                        "target_date": FEASIBILITY_DATE,
                        "status": "failed",
                        "convergence_status": "failed",
                        "converged": False,
                        "runtime_seconds": perf_counter() - started,
                        "optimizer_iterations": None,
                        "state_dimension": None,
                        "peak_rss_mb": float("nan"),
                        "target_day_mae": float("nan"),
                        "error_message": "feasibility child returned no result",
                        "analysis_label": ANALYSIS_LABEL,
                    }
                )
            parent.close()
    result = pd.DataFrame(rows)
    _write(result, OUTPUT_ROOT / "feasibility" / "s168_feasibility.csv", result.columns)
    return {
        "feasibility_rows": len(result),
        "completed": int(result["status"].eq("completed").sum()),
        "timed_out": int(result["status"].eq("timed_out").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated enhanced SARIMA robustness analysis")
    parser.add_argument(
        "--stage",
        choices=("audit", "feasibility", "screening", "benchmark", "full-validation", "freeze", "test-2025", "comparison", "all"),
        default="all",
    )
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    candidates = candidate_definitions()
    _write(candidates, OUTPUT_ROOT / "specifications" / "candidate_definitions.csv", candidates.columns)
    results: dict[str, object] = {}
    if args.stage in {"audit", "all"}:
        results["audit"] = audit_authoritative_reuse(candidates).to_dict("records")
    if args.stage in {"feasibility", "all"}:
        results["feasibility"] = run_s168_feasibility(args.processed_directory)
    if args.stage in {"screening", "all"}:
        results["screening"] = run_screening(args.processed_directory, args.workers or 1)
    if args.stage in {"benchmark", "all"}:
        results["benchmark"] = run_worker_benchmark(args.processed_directory)
    if args.stage in {"full-validation", "all"}:
        results["full_validation"] = run_full_validation(args.processed_directory, args.workers)
    if args.stage in {"freeze", "all"}:
        results["freeze"] = freeze_2024()
    if args.stage in {"test-2025", "all"}:
        results["test_2025"] = run_test_2025(args.processed_directory, args.workers)
    if args.stage in {"comparison", "all"}:
        summary = pd.read_csv(OUTPUT_ROOT / "test_2025" / "summary.csv")
        forecasts = pd.concat(
            [
                pd.read_csv(OUTPUT_ROOT / "test_2025" / "forecasts.csv"),
            ],
            ignore_index=True,
        )
        smard_benchmark = pd.read_csv(
            PROJECT_ROOT / "results" / "official_benchmarks" / "smard_2025_benchmark.csv"
        )
        comparison = build_final_comparison(
            summary, forecasts, smard_benchmark=smard_benchmark
        )
        _write(
            comparison,
            OUTPUT_ROOT / "comparison" / "enhanced_sarima_final_comparison_2025.csv",
            comparison.columns,
        )
        results["comparison"] = {"rows": len(comparison)}
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
