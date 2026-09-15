from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import (  # noqa: E402
    COUNTRY_CONFIG,
    PROCESSED,
    _as_prepared,
    calculate_metrics,
    country_forecast_origin,
    extract_target_day,
    load_country_data,
)
from enhanced_arima_models import (  # noqa: E402
    EnhancedSpecification,
    derive_weekly_invalid_target_dates,
    enhanced_specifications,
)
import enhanced_arima_screening_2024 as screening  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "results" / "enhanced_arima"
OUTPUT_DIRECTORY = OUTPUT_ROOT / "test_2025"
FREEZE_PATH = OUTPUT_ROOT / "freeze_2024" / "frozen_specifications_2024.csv"
COMPARISON_PATH = OUTPUT_ROOT / "comparison" / "enhanced_arima_final_comparison_2025.csv"
VALIDATION_YEAR = 2025
EXPECTED_DAYS = 365
EXPECTED_OBSERVATIONS = 8760
MODEL_FAMILY = "enhanced_arima"
DESCRIPTIVE_LABEL = "descriptive_post_hoc_robustness"
SELECTION_LABEL = "not_used_for_selection"

JOB_KEY = screening.SCREENING_JOB_KEY
FORECAST_KEY = screening.SCREENING_FORECAST_KEY
DIAGNOSTIC_KEY = screening.SCREENING_DIAGNOSTIC_KEY
JOB_COLUMNS = screening.JOB_COLUMNS
FORECAST_COLUMNS = screening.FORECAST_COLUMNS
DIAGNOSTIC_COLUMNS = screening.DIAGNOSTICS_COLUMNS
MAPPING_ISSUE_COLUMNS = screening.MAPPING_ISSUE_COLUMNS

MANIFEST_COLUMNS = [
    "country",
    "target_date",
    "forecast_origin_local",
    "forecast_origin_utc",
    "information_cutoff_utc",
    "expected_observations",
    "branch",
    "specification_id",
    "specification_order",
    "p",
    "d",
    "q",
    "trend",
    "transform",
    "weekly_invalid",
    "weekly_invalid_reason",
]

SUMMARY_COLUMNS = [
    "country",
    "branch",
    "specification_id",
    "specification_order",
    "p",
    "d",
    "q",
    "trend",
    "transform",
    "expected_jobs",
    "completed_jobs",
    "failed_jobs",
    "expected_dates",
    "valid_dates",
    "invalid_dates",
    "unresolved_dates",
    "invalid_reasons",
    "expected_observations",
    "native_observations",
    "native_coverage",
    "native_mae",
    "native_rmse",
    "native_mape",
    "common_observations",
    "common_coverage",
    "common_mae",
    "common_rmse",
    "common_mape",
    "mae",
    "rmse",
    "mape",
    "evaluated_observations",
    "coverage",
    "common_timestamp_count",
    "evaluation_role",
    "selection_role",
]

COMPARISON_COLUMNS = [
    "country",
    "model",
    "model_family",
    "specification_id",
    "mae",
    "rmse",
    "mape",
    "evaluated_observations",
    "expected_observations",
    "coverage",
    "native_mae",
    "native_rmse",
    "native_mape",
    "native_observations",
    "native_coverage",
    "common_mae",
    "common_rmse",
    "common_mape",
    "common_observations",
    "common_coverage",
    "common_timestamp_count",
    "valid_dates",
    "invalid_dates",
    "invalid_date_reasons",
    "evaluation_role",
    "comparison_role",
    "selection_role",
    "source_kind",
]

COMPARATOR_FORECAST_PATHS = {
    "original_arima": PROJECT_ROOT
    / "results"
    / "arima"
    / "test_2025"
    / "arima_test_2025_forecasts.csv",
    "original_sarima": PROJECT_ROOT
    / "results"
    / "arima_sarima"
    / "test_2025"
    / "sarima_test_2025_forecasts.csv",
    "weekly_seasonal_naive": PROJECT_ROOT
    / "results"
    / "baselines"
    / "forecasts"
    / "baseline_forecasts_2025.csv",
    "holt_winters": PROJECT_ROOT
    / "results"
    / "exponential_smoothing"
    / "holt_winters"
    / "test_2025"
    / "holt_winters_test_2025_forecasts.csv",
}
SMARD_BENCHMARK_PATH = (
    PROJECT_ROOT / "results" / "official_benchmarks" / "smard_2025_benchmark.csv"
)

_SPECIFICATIONS = {specification.specification_id: specification for specification in enhanced_specifications()}
_SPECIFICATION_ORDER = {
    (specification.branch, specification.specification_id): order
    for branch in ("ordinary", "weekly_differenced")
    for order, specification in enumerate(
        (item for item in enhanced_specifications() if item.branch == branch), 1
    )
}
_COUNTRY_ORDER = {country: index for index, country in enumerate(COUNTRY_CONFIG)}


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _blank(value: object) -> bool:
    return value is None or str(value).strip() in {"", "nan", "None", "NaT"}


def _date_text(value: object) -> str:
    target = pd.Timestamp(value).date()
    return target.isoformat()


def _canonical_timestamp(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _job_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["country"]),
        _date_text(row["target_date"]),
        str(row["branch"]),
        str(row["specification_id"]),
    )


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _specification_from_row(row: Mapping[str, object]) -> EnhancedSpecification:
    try:
        specification_id = str(row["specification_id"])
        specification = _SPECIFICATIONS[specification_id]
    except (KeyError, TypeError):
        raise ValueError("freeze contains an unknown enhanced specification") from None
    if (
        str(row.get("branch")) != specification.branch
        or tuple(int(row[column]) for column in ("p", "d", "q")) != specification.order
        or str(row.get("trend")) != specification.trend
        or str(row.get("transform")) != specification.transform
    ):
        raise ValueError("freeze contains invalid enhanced specification settings")
    return specification


def _normalise_freeze(freeze: pd.DataFrame) -> pd.DataFrame:
    required = {
        "country",
        "branch",
        "specification_id",
        "p",
        "d",
        "q",
        "trend",
        "transform",
        "selection_status",
        "adequate",
    }
    missing = sorted(required.difference(freeze.columns))
    if missing:
        raise ValueError(f"freeze is missing required columns: {missing}")
    result = freeze.copy()
    result["country"] = result["country"].astype(str)
    if set(result["country"]) != set(COUNTRY_CONFIG) or len(result) != len(COUNTRY_CONFIG):
        raise ValueError("freeze must contain exactly one row per country")
    if result.duplicated("country").any():
        raise ValueError("freeze must contain exactly one row per country")
    if not result["selection_status"].astype(str).str.lower().eq("frozen").all():
        raise ValueError("freeze contains a specification that is not frozen")
    if not result["adequate"].map(_bool_value).all():
        raise ValueError("freeze contains a specification that is not adequate")
    for row in result.to_dict("records"):
        _specification_from_row(row)
    result["specification_order"] = [
        _SPECIFICATION_ORDER[(str(row["branch"]), str(row["specification_id"]))]
        for row in result.to_dict("records")
    ]
    result["adequate"] = True
    return result.sort_values(
        "country",
        key=lambda values: values.map(_COUNTRY_ORDER),
        kind="mergesort",
    ).reset_index(drop=True)


def verify_freeze(freeze_path: str | Path) -> pd.DataFrame:
    path = Path(freeze_path)
    if not path.exists():
        raise FileNotFoundError(f"required 2024 freeze is missing: {path}")
    try:
        freeze = pd.read_csv(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read 2024 freeze: {path}") from error
    return _normalise_freeze(freeze)


def _expected_2025_dates(frame: pd.DataFrame) -> list[str]:
    prepared = _as_prepared(frame)
    observed = sorted(
        set(
            pd.to_datetime(prepared["local_date"], errors="raise")
            .dt.date.map(date.isoformat)
            .loc[lambda values: values.str.startswith(f"{VALIDATION_YEAR}-")]
        )
    )
    expected = [
        (date(VALIDATION_YEAR, 1, 1) + timedelta(days=offset)).isoformat()
        for offset in range(EXPECTED_DAYS)
    ]
    if observed != expected:
        raise ValueError(
            "2025 enhanced ARIMA test requires every local date from "
            "2025-01-01 through 2025-12-31"
        )
    return expected


def _invalid_by_date(frame: pd.DataFrame) -> dict[str, str]:
    invalid = derive_weekly_invalid_target_dates(_as_prepared(frame), VALIDATION_YEAR)
    if invalid.empty:
        return {}
    grouped: dict[str, list[str]] = {}
    for row in invalid.to_dict("records"):
        target = _date_text(row["target_date"])
        reason = str(row.get("reason") or "weekly source mapping issue")
        if reason in {"", "nan", "None"}:
            reason = "weekly source mapping issue"
        grouped.setdefault(target, []).append(reason)
    return {
        target: "; ".join(dict.fromkeys(reasons))
        for target, reasons in grouped.items()
    }


def build_2025_manifest(
    freeze: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build one origin-bounded job for every 2025 local date and country."""
    verified = _normalise_freeze(freeze)
    missing_frames = sorted(set(COUNTRY_CONFIG).difference(frames))
    if missing_frames:
        raise ValueError(f"country frames are missing: {missing_frames}")
    dates_by_country = {
        country: _expected_2025_dates(frames[country]) for country in COUNTRY_CONFIG
    }
    if dates_by_country["Germany"] != dates_by_country["Austria"]:
        raise ValueError("2025 target dates differ between Germany and Austria")

    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        frame = _as_prepared(frames[country])
        invalid_by_date = _invalid_by_date(frame)
        selected = verified.loc[verified["country"].eq(country)].iloc[0].to_dict()
        specification = _specification_from_row(selected)
        for target_date in dates_by_country[country]:
            expected_observations = len(extract_target_day(frame, target_date))
            origin = country_forecast_origin(target_date, country)
            reason = invalid_by_date.get(target_date, "")
            rows.append(
                {
                    "country": country,
                    "target_date": target_date,
                    "forecast_origin_local": origin.tz_convert(
                        COUNTRY_CONFIG[country]["timezone"]
                    ).isoformat(),
                    "forecast_origin_utc": origin.isoformat(),
                    "information_cutoff_utc": origin.isoformat(),
                    "expected_observations": expected_observations,
                    "branch": specification.branch,
                    "specification_id": specification.specification_id,
                    "specification_order": _SPECIFICATION_ORDER[
                        (specification.branch, specification.specification_id)
                    ],
                    "p": specification.order[0],
                    "d": specification.order[1],
                    "q": specification.order[2],
                    "trend": specification.trend,
                    "transform": specification.transform,
                    "weekly_invalid": bool(
                        specification.branch == "weekly_differenced" and reason
                    ),
                    "weekly_invalid_reason": (
                        reason if specification.branch == "weekly_differenced" else ""
                    ),
                }
            )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(manifest) != 2 * EXPECTED_DAYS or manifest.duplicated(list(JOB_KEY)).any():
        raise ValueError("2025 enhanced ARIMA manifest is not the exact 730-job set")
    return manifest


def _load_checkpoint(path: Path, columns: list[str]) -> pd.DataFrame:
    loaded = screening._load_csv(path, columns)
    return loaded.reindex(columns=columns)


def _reconcile_checkpoints(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = {_job_key(row) for row in manifest.to_dict("records")}
    manifest_by_key = {_job_key(row): row for row in manifest.to_dict("records")}

    def valid_job(row: pd.Series) -> bool:
        try:
            key = _job_key(row)
            return key in expected and screening._manifest_job_matches(
                row.to_dict(), manifest_by_key[key]
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    def belongs(row: pd.Series, columns: Iterable[str]) -> bool:
        try:
            values = tuple(
                screening._canonical_key_value(column, row[column])
                for column in columns
            )
            return values in expected
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    jobs = jobs.loc[jobs.apply(valid_job, axis=1)].copy() if not jobs.empty else jobs
    forecasts = (
        forecasts.loc[forecasts.apply(lambda row: belongs(row, JOB_KEY), axis=1)].copy()
        if not forecasts.empty
        else forecasts
    )
    expected_invalid_keys = {
        _job_key(row)
        for row in manifest.to_dict("records")
        if str(row["branch"]) == "weekly_differenced"
        and _bool_value(row["weekly_invalid"])
    }
    if not forecasts.empty and expected_invalid_keys:
        forecasts = forecasts.loc[
            ~forecasts.apply(
                lambda row: tuple(
                    screening._canonical_key_value(column, row[column])
                    for column in JOB_KEY
                )
                in expected_invalid_keys,
                axis=1,
            )
        ].copy()
    diagnostics = (
        diagnostics.loc[diagnostics.apply(lambda row: belongs(row, JOB_KEY), axis=1)].copy()
        if not diagnostics.empty
        else diagnostics
    )
    return (
        screening.upsert_frame(pd.DataFrame(columns=JOB_COLUMNS), jobs, JOB_KEY, sort=False),
        screening.upsert_frame(
            pd.DataFrame(columns=FORECAST_COLUMNS), forecasts, FORECAST_KEY, sort=False
        ),
        screening.upsert_frame(
            pd.DataFrame(columns=DIAGNOSTIC_COLUMNS),
            diagnostics,
            DIAGNOSTIC_KEY,
            sort=False,
        ),
    )


def _drop_job_rows(
    frame: pd.DataFrame,
    key_columns: Iterable[str],
    key: tuple[str, str, str, str],
) -> pd.DataFrame:
    return screening._drop_job_rows(frame, key_columns, key)


def _persist_result(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    result: Mapping[str, object],
    branch_manifest: pd.DataFrame,
    branch_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    job = dict(result["job"])
    key = _job_key(job)
    branch_keys = {_job_key(row) for row in branch_manifest.to_dict("records")}
    if key not in branch_keys:
        return jobs, forecasts, diagnostics
    incoming_job = pd.DataFrame([job]).reindex(columns=JOB_COLUMNS)
    incoming_forecasts = pd.DataFrame(result.get("forecasts", [])).reindex(
        columns=FORECAST_COLUMNS
    )
    incoming_diagnostics = pd.DataFrame(result.get("diagnostics", [])).reindex(
        columns=DIAGNOSTIC_COLUMNS
    )
    jobs = screening.upsert_frame(jobs, incoming_job, JOB_KEY, sort=False)
    forecasts = _drop_job_rows(forecasts, FORECAST_KEY, key)
    if not incoming_forecasts.empty:
        forecasts = screening.upsert_frame(
            forecasts, incoming_forecasts, FORECAST_KEY, sort=False
        )
    diagnostics = _drop_job_rows(diagnostics, DIAGNOSTIC_KEY, key)
    if not incoming_diagnostics.empty:
        diagnostics = screening.upsert_frame(
            diagnostics, incoming_diagnostics, DIAGNOSTIC_KEY, sort=False
        )
    _atomic_write_csv(forecasts, branch_output / "forecasts.csv")
    _atomic_write_csv(diagnostics, branch_output / "diagnostics.csv")
    _atomic_write_csv(jobs, branch_output / "jobs.csv")
    return jobs, forecasts, diagnostics


def _execute_job_2025(
    job: Mapping[str, object],
    frame: pd.DataFrame,
) -> dict[str, object]:
    return screening._execute_job(job, frame)


def _mapping_issues(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        frame = _as_prepared(frames[country])
        invalid = derive_weekly_invalid_target_dates(frame, VALIDATION_YEAR)
        if invalid.empty:
            continue
        counts = invalid.groupby(invalid["target_date"].astype(str)).size().to_dict()
        expected = {
            str(target): len(extract_target_day(frame, str(target)))
            for target in invalid["target_date"].astype(str).unique()
        }
        for issue in invalid.to_dict("records"):
            target = str(issue["target_date"])
            source_key = issue.get("source_key")
            records.append(
                {
                    "country": country,
                    "branch": "weekly_differenced",
                    "target_date": target,
                    "target_timestamp_utc": issue.get("target_timestamp_utc"),
                    "affected_timestamp_utc": issue.get("target_timestamp_utc"),
                    "source_key": source_key,
                    "source_local_key": source_key,
                    "source_timestamp_utc": issue.get("source_timestamp_utc"),
                    "reason": issue.get("reason"),
                    "source_row_count": issue.get("source_row_count"),
                    "coverage": float(max(0, expected[target] - counts[target]) / expected[target]),
                }
            )
    return pd.DataFrame(records, columns=MAPPING_ISSUE_COLUMNS)


def _fit_metadata_is_complete(job: Mapping[str, object]) -> bool:
    required = (
        "status",
        "fit_status",
        "convergence_status",
        "converged",
        "parameters_finite",
        "standard_errors_finite",
        "stationarity_ok",
        "invertibility_ok",
        "log_likelihood",
        "aic",
        "aicc",
        "bic",
    )
    if any(column not in job or _blank(job[column]) for column in required):
        return False
    if str(job["status"]) != "completed" or str(job["fit_status"]) != "success":
        return False
    if str(job["convergence_status"]) != "converged":
        return False
    if not all(_bool_value(job[column]) for column in required[3:8]):
        return False
    if not _blank(job.get("error_message")):
        return False
    try:
        return bool(np.isfinite([float(job[column]) for column in required[8:]]).all())
    except (TypeError, ValueError):
        return False


def _expected_invalid_jobs(manifest: pd.DataFrame) -> dict[tuple[str, str, str, str], str]:
    return {
        _job_key(row): str(row["weekly_invalid_reason"])
        for row in manifest.to_dict("records")
        if str(row["branch"]) == "weekly_differenced"
        and _bool_value(row["weekly_invalid"])
        and str(row["weekly_invalid_reason"]).strip()
    }


def _valid_target_rows(
    forecasts: pd.DataFrame,
    key: tuple[str, str, str, str],
) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()
    mask = np.ones(len(forecasts), dtype=bool)
    for column, expected in zip(JOB_KEY, key):
        mask &= forecasts[column].map(
            lambda value, name=column: screening._canonical_key_value(name, value)
        ).eq(expected)
    result = forecasts.loc[mask].copy()
    if result.empty:
        return result
    result = result.loc[
        result["is_target_day"].map(_bool_value)
        & result["evaluated"].map(_bool_value)
        & result["status"].astype(str).eq("completed")
    ].copy()
    if result.empty:
        return result
    actual = pd.to_numeric(result["actual_load_mwh"], errors="coerce")
    forecast = pd.to_numeric(result["forecast_mwh"], errors="coerce")
    timestamps = pd.to_datetime(
        result["timestamp_utc"], format="mixed", utc=True, errors="coerce"
    )
    result = result.loc[
        timestamps.notna() & np.isfinite(actual) & np.isfinite(forecast)
    ].copy()
    result["timestamp_utc"] = timestamps.loc[result.index]
    return result


def _metric_values(rows: pd.DataFrame, expected: int) -> dict[str, float | int]:
    if rows.empty:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "evaluated_observations": 0,
            "coverage": 0.0 if expected else float("nan"),
        }
    metrics = calculate_metrics(
        rows["actual_load_mwh"], rows["forecast_mwh"], expected_observations=expected
    )
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mape": float(metrics["mape"]),
        "evaluated_observations": int(metrics["evaluated_observations"]),
        "coverage": float(metrics["coverage"]),
    }


def _build_summary(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    manifest: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    expected_invalid = _expected_invalid_jobs(manifest)
    candidate_rows: list[dict[str, object]] = []
    valid_rows_by_country: dict[str, list[pd.DataFrame]] = {country: [] for country in COUNTRY_CONFIG}
    valid_dates_by_key: dict[tuple[str, str, str, str], set[str]] = {}
    for candidate_key, expected in manifest.groupby(
        ["country", "branch", "specification_id"], sort=False
    ):
        country, branch, specification_id = (str(value) for value in candidate_key)
        key_jobs = jobs.loc[
            jobs["country"].astype(str).eq(country)
            & jobs["branch"].astype(str).eq(branch)
            & jobs["specification_id"].astype(str).eq(specification_id)
        ].copy()
        valid_jobs: list[dict[str, object]] = []
        invalid_dates: set[str] = set()
        reasons: list[str] = []
        for job in key_jobs.to_dict("records"):
            key = _job_key(job)
            if key in expected_invalid and (
                str(job.get("status")) == "weekly_lag_invalid"
                and str(job.get("error_message", "")).strip() == expected_invalid[key]
            ):
                invalid_dates.add(key[1])
                reasons.append(expected_invalid[key])
                continue
            if _fit_metadata_is_complete(job) and screening._forecast_is_complete(
                job,
                forecasts,
                frames[country],
            ):
                valid_jobs.append(job)
        valid_dates = {_date_text(job["target_date"]) for job in valid_jobs}
        valid_forecasts = pd.concat(
            [_valid_target_rows(forecasts, _job_key(job)) for job in valid_jobs],
            ignore_index=True,
        ) if valid_jobs else pd.DataFrame(columns=FORECAST_COLUMNS)
        expected_observations = int(expected["expected_observations"].sum())
        native = _metric_values(valid_forecasts, expected_observations)
        valid_rows_by_country[country].append(valid_forecasts)
        valid_dates_by_key[(country, branch, specification_id, "dates")] = valid_dates
        specification = _SPECIFICATIONS[specification_id]
        candidate_rows.append(
            {
                "country": country,
                "branch": branch,
                "specification_id": specification_id,
                "specification_order": _SPECIFICATION_ORDER[(branch, specification_id)],
                "p": specification.order[0],
                "d": specification.order[1],
                "q": specification.order[2],
                "trend": specification.trend,
                "transform": specification.transform,
                "expected_jobs": len(expected),
                "completed_jobs": int(key_jobs["status"].astype(str).eq("completed").sum())
                if not key_jobs.empty
                else 0,
                "failed_jobs": int(key_jobs["status"].astype(str).eq("failed").sum())
                if not key_jobs.empty
                else len(expected),
                "expected_dates": int(expected["target_date"].nunique()),
                "valid_dates": len(valid_dates),
                "invalid_dates": len(invalid_dates),
                "unresolved_dates": max(
                    0,
                    int(expected["target_date"].nunique())
                    - len(valid_dates)
                    - len(invalid_dates),
                ),
                "invalid_reasons": "; ".join(dict.fromkeys(reasons)),
                "expected_observations": expected_observations,
                "native_observations": int(native["evaluated_observations"]),
                "native_coverage": float(native["coverage"]),
                "native_mae": float(native["mae"]),
                "native_rmse": float(native["rmse"]),
                "native_mape": float(native["mape"]),
                "mae": float(native["mae"]),
                "rmse": float(native["rmse"]),
                "mape": float(native["mape"]),
                "evaluated_observations": int(native["evaluated_observations"]),
                "coverage": float(len(valid_dates) / expected["target_date"].nunique()),
                "evaluation_role": DESCRIPTIVE_LABEL,
                "selection_role": SELECTION_LABEL,
            }
        )

    common_by_country: dict[str, set[pd.Timestamp]] = {}
    for country in COUNTRY_CONFIG:
        sets = [
            set(frame["timestamp_utc"])
            for frame in valid_rows_by_country[country]
            if not frame.empty
        ]
        common_by_country[country] = set.intersection(*sets) if sets else set()
    for row in candidate_rows:
        country = str(row["country"])
        common_timestamps = common_by_country[country]
        key = (country, str(row["branch"]), str(row["specification_id"]))
        valid = next(
            (
                frame
                for frame in valid_rows_by_country[country]
                if not frame.empty
                and key[1] == str(frame["branch"].iloc[0])
                and key[2] == str(frame["specification_id"].iloc[0])
            ),
            pd.DataFrame(columns=FORECAST_COLUMNS),
        )
        common = valid.loc[valid["timestamp_utc"].isin(common_timestamps)].copy()
        metrics = _metric_values(common, len(common_timestamps))
        row.update(
            {
                "common_observations": int(metrics["evaluated_observations"]),
                "common_coverage": float(metrics["coverage"]),
                "common_mae": float(metrics["mae"]),
                "common_rmse": float(metrics["rmse"]),
                "common_mape": float(metrics["mape"]),
                "common_timestamp_count": len(common_timestamps),
            }
        )
    return pd.DataFrame(candidate_rows, columns=SUMMARY_COLUMNS)


def _validate_run(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    manifest: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> None:
    expected_keys = {_job_key(row) for row in manifest.to_dict("records")}
    observed_keys = {_job_key(row) for row in jobs.to_dict("records")}
    if observed_keys != expected_keys or len(jobs) != len(expected_keys):
        raise ValueError("2025 enhanced ARIMA jobs do not match the exact manifest")
    expected_invalid = _expected_invalid_jobs(manifest)
    for job in jobs.to_dict("records"):
        key = _job_key(job)
        if key in expected_invalid:
            if not (
                str(job.get("status")) == "weekly_lag_invalid"
                and str(job.get("error_message", "")).strip() == expected_invalid[key]
            ):
                raise ValueError("2025 weekly invalidity does not match the derived mapping")
            if not forecasts.empty:
                matching = forecasts.apply(
                    lambda row: tuple(
                        screening._canonical_key_value(column, row[column])
                        for column in JOB_KEY
                    )
                    == key,
                    axis=1,
                )
                if matching.any():
                    raise ValueError("derived-invalid weekly job must have no forecast rows")
            continue
        if not _fit_metadata_is_complete(job) or not screening._forecast_is_complete(
            job, forecasts, frames[key[0]]
        ):
            raise ValueError(f"2025 enhanced ARIMA job is incomplete: {key}")


def _write_run_outputs(
    output: Path,
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    summary = _build_summary(jobs, forecasts, manifest, frames)
    _atomic_write_csv(manifest, output / "manifest.csv")
    _atomic_write_csv(_mapping_issues(frames), output / "weekly_differenced" / "mapping_issues.csv")
    _atomic_write_csv(summary, output / "summary.csv")
    _atomic_write_csv(
        summary.loc[:, [column for column in summary.columns if column.startswith("native_") or column in {
            "country", "branch", "specification_id", "valid_dates", "invalid_dates", "invalid_reasons", "coverage",
        }]],
        output / "native_summary.csv",
    )
    _atomic_write_csv(
        summary.loc[:, [column for column in summary.columns if column.startswith("common_") or column in {
            "country", "branch", "specification_id", "valid_dates", "invalid_dates",
        }]],
        output / "common_summary.csv",
    )
    common_rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        country_rows = forecasts.loc[
            forecasts["country"].astype(str).eq(country)
            & forecasts["is_target_day"].map(_bool_value)
            & forecasts["evaluated"].map(_bool_value)
            & forecasts["status"].astype(str).eq("completed")
        ]
        if country_rows.empty:
            continue
        common_rows.extend(
            {"country": country, "timestamp_utc": timestamp}
            for timestamp in sorted(set(pd.to_datetime(country_rows["timestamp_utc"], utc=True)))
        )
    _atomic_write_csv(
        pd.DataFrame(common_rows, columns=["country", "timestamp_utc"]),
        output / "common_evaluation_timestamps.csv",
    )
    for branch in ("ordinary", "weekly_differenced"):
        branch_summary = summary.loc[summary["branch"].eq(branch)].copy()
        _atomic_write_csv(branch_summary, output / branch / "summary.csv")
    return summary


def _read_comparator(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required read-only comparator forecast is missing: {path}")
    return pd.read_csv(path)


def _load_comparator_forecasts() -> dict[str, pd.DataFrame]:
    return {name: _read_comparator(path) for name, path in COMPARATOR_FORECAST_PATHS.items()}


def _load_smard_benchmark() -> pd.DataFrame:
    return _read_comparator(SMARD_BENCHMARK_PATH)


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _normalise_comparison_frame(frame: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["country", "timestamp_utc", "actual_load_mwh", "forecast_mwh"])
    source = frame.copy()
    if country is not None and "country" in source:
        source = source.loc[source["country"].astype(str).eq(country)].copy()
    country_column = "country" if "country" in source else None
    timestamp_column = _first_column(
        source, ("timestamp_utc", "interval_start_utc", "timestamp", "interval_start")
    )
    actual_column = _first_column(
        source, ("actual_load_mwh", "actual_grid_load_mwh", "actual_mwh", "actual")
    )
    forecast_column = _first_column(
        source,
        (
            "forecast_mwh",
            "forecasted_load_mwh",
            "forecasted_grid_load_mwh",
            "forecast_mwh",
            "forecast",
        ),
    )
    if timestamp_column is None or actual_column is None or forecast_column is None:
        return pd.DataFrame(columns=["country", "timestamp_utc", "actual_load_mwh", "forecast_mwh"])
    if "evaluated" in source:
        source = source.loc[source["evaluated"].map(_bool_value)].copy()
    if "is_target_day" in source:
        source = source.loc[source["is_target_day"].map(_bool_value)].copy()
    if "local_date" in source:
        source = source.loc[
            source["local_date"].astype(str).str[:4].eq(str(VALIDATION_YEAR))
        ].copy()
    else:
        timestamps = pd.to_datetime(
            source[timestamp_column], format="mixed", utc=True, errors="coerce"
        )
        source = source.loc[timestamps.dt.year.eq(VALIDATION_YEAR)].copy()
    result = pd.DataFrame(
        {
            "country": source[country_column].astype(str) if country_column else country,
            "timestamp_utc": pd.to_datetime(
                source[timestamp_column], format="mixed", utc=True, errors="coerce"
            ),
            "actual_load_mwh": pd.to_numeric(source[actual_column], errors="coerce"),
            "forecast_mwh": pd.to_numeric(source[forecast_column], errors="coerce"),
        }
    )
    result = result.loc[
        result["timestamp_utc"].notna()
        & np.isfinite(result["actual_load_mwh"])
        & np.isfinite(result["forecast_mwh"])
    ].copy()
    return result.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _comparator_frame(
    comparator_forecasts: Mapping[str, pd.DataFrame],
    name: str,
) -> pd.DataFrame:
    aliases = {
        "original_arima": ("original_arima", "arima", "ARIMA"),
        "original_sarima": ("original_sarima", "sarima", "SARIMA"),
        "weekly_seasonal_naive": (
            "weekly_seasonal_naive",
            "Weekly seasonal naive",
            "baselines",
        ),
        "holt_winters": ("holt_winters", "Holt-Winters"),
    }
    for alias in aliases[name]:
        if alias in comparator_forecasts:
            frame = comparator_forecasts[alias]
            if name == "weekly_seasonal_naive":
                if "model_family" in frame:
                    allowed = frame["model_family"].astype(str).isin(
                        {"Weekly seasonal naive", "weekly_seasonal_naive"}
                    )
                    if allowed.any():
                        frame = frame.loc[allowed]
                elif "specification_id" in frame:
                    allowed = frame["specification_id"].astype(str).eq(
                        "Weekly seasonal naive"
                    )
                    if allowed.any():
                        frame = frame.loc[allowed]
            return frame
    return pd.DataFrame()


def _finite_or(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _comparison_metric_rows(frame: pd.DataFrame, expected: int, common: set[pd.Timestamp]) -> dict[str, object]:
    native = _metric_values(frame, expected)
    common_frame = frame.loc[frame["timestamp_utc"].isin(common)].copy()
    common_metrics = _metric_values(common_frame, len(common))
    return {
        "mae": native["mae"],
        "rmse": native["rmse"],
        "mape": native["mape"],
        "evaluated_observations": native["evaluated_observations"],
        "coverage": native["coverage"],
        "native_mae": native["mae"],
        "native_rmse": native["rmse"],
        "native_mape": native["mape"],
        "native_observations": native["evaluated_observations"],
        "native_coverage": native["coverage"],
        "common_mae": common_metrics["mae"],
        "common_rmse": common_metrics["rmse"],
        "common_mape": common_metrics["mape"],
        "common_observations": common_metrics["evaluated_observations"],
        "common_coverage": common_metrics["coverage"],
        "common_timestamp_count": len(common),
    }


def _smard_metric_rows(
    benchmark: pd.DataFrame, country: str, expected: int
) -> dict[str, object] | None:
    if benchmark.empty or "country" not in benchmark:
        return None
    matching = benchmark.loc[benchmark["country"].astype(str).eq(country)]
    if matching.empty:
        return None
    row = matching.iloc[0]
    observations = int(_finite_or(row.get("n_observations"), expected))
    coverage = _finite_or(
        row.get("coverage"),
        observations / expected if expected else float("nan"),
    )
    metrics = {
        name: _finite_or(row.get(name), float("nan"))
        for name in ("mae", "rmse", "mape")
    }
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "evaluated_observations": observations,
        "coverage": coverage,
        "native_mae": metrics["mae"],
        "native_rmse": metrics["rmse"],
        "native_mape": metrics["mape"],
        "native_observations": observations,
        "native_coverage": coverage,
        "common_mae": metrics["mae"],
        "common_rmse": metrics["rmse"],
        "common_mape": metrics["mape"],
        "common_observations": observations,
        "common_coverage": coverage,
        "common_timestamp_count": observations,
    }


def build_final_comparison(
    enhanced_summary: pd.DataFrame,
    enhanced_forecasts: pd.DataFrame,
    comparator_forecasts: dict[str, pd.DataFrame],
    smard_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Build the descriptive six-model comparison separately for each country."""
    model_names = (
        "original ARIMA (2,1,2)",
        "enhanced ARIMA",
        "original SARIMA",
        "weekly seasonal naive",
        "Holt-Winters",
        "SMARD",
    )
    model_families = {
        "original ARIMA (2,1,2)": "arima",
        "enhanced ARIMA": "enhanced_arima",
        "original SARIMA": "sarima",
        "weekly seasonal naive": "baselines",
        "Holt-Winters": "holt_winters",
        "SMARD": "smard",
    }
    source_kinds = {
        "original ARIMA (2,1,2)": "original_arima_2025",
        "enhanced ARIMA": "enhanced_arima_2025",
        "original SARIMA": "original_sarima_2025",
        "weekly seasonal naive": "weekly_seasonal_naive_2025",
        "Holt-Winters": "holt_winters_2025",
        "SMARD": "official_smard_benchmark_2025",
    }
    comparator_names = {
        "original ARIMA (2,1,2)": "original_arima",
        "original SARIMA": "original_sarima",
        "weekly seasonal naive": "weekly_seasonal_naive",
        "Holt-Winters": "holt_winters",
    }
    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        summary_rows = enhanced_summary.loc[
            enhanced_summary["country"].astype(str).eq(country)
        ] if not enhanced_summary.empty and "country" in enhanced_summary else pd.DataFrame()
        enhanced_row = summary_rows.iloc[0] if not summary_rows.empty else pd.Series(dtype=object)
        frame_by_model: dict[str, pd.DataFrame] = {
            "enhanced ARIMA": _normalise_comparison_frame(enhanced_forecasts, country),
            "SMARD": _normalise_comparison_frame(smard_benchmark, country),
        }
        for label, comparator_name in comparator_names.items():
            frame_by_model[label] = _normalise_comparison_frame(
                _comparator_frame(comparator_forecasts, comparator_name), country
            )
        expected = _finite_or(
            enhanced_row.get("expected_observations", np.nan),
            max((len(frame) for frame in frame_by_model.values()), default=0),
        )
        expected_count = int(expected)
        valid_sets = [
            set(frame["timestamp_utc"])
            for frame in frame_by_model.values()
            if not frame.empty
        ]
        common = set.intersection(*valid_sets) if valid_sets else set()
        for model in model_names:
            frame = frame_by_model.get(model, pd.DataFrame())
            metrics = _comparison_metric_rows(frame, expected_count, common)
            if model == "SMARD" and frame.empty:
                metrics = _smard_metric_rows(smard_benchmark, country, expected_count) or metrics
            specification_id = {
                "original ARIMA (2,1,2)": "arima_p2_d1_q2",
                "enhanced ARIMA": str(enhanced_row.get("specification_id", "")),
                "original SARIMA": (
                    str(frame.get("specification_id", pd.Series(dtype=object)).iloc[0])
                    if "specification_id" in frame and not frame.empty
                    else "original_sarima"
                ),
                "weekly seasonal naive": "Weekly seasonal naive",
                "Holt-Winters": "hw_mul_s168_damped",
                "SMARD": "official_smard_benchmark",
            }[model]
            invalid_dates = int(_finite_or(enhanced_row.get("invalid_dates", 0), 0)) if model == "enhanced ARIMA" else 0
            invalid_reasons = str(
                enhanced_row.get("invalid_reasons", "")
                if model == "enhanced ARIMA"
                else ""
            )
            if model == "enhanced ARIMA" and frame.empty:
                for metric in (
                    "native_mae", "native_rmse", "native_mape", "mae", "rmse", "mape"
                ):
                    if metric in enhanced_row:
                        metrics[metric] = enhanced_row[metric]
                metrics["native_coverage"] = enhanced_row.get("native_coverage", metrics["native_coverage"])
                metrics["coverage"] = enhanced_row.get("native_coverage", metrics["coverage"])
            rows.append(
                {
                    "country": country,
                    "model": model,
                    "model_family": model_families[model],
                    "specification_id": specification_id,
                    **metrics,
                    "expected_observations": expected_count,
                    "valid_dates": int(
                        _finite_or(enhanced_row.get("valid_dates", 0), 0)
                        if model == "enhanced ARIMA"
                        else 0
                    ),
                    "invalid_dates": invalid_dates,
                    "invalid_date_reasons": invalid_reasons,
                    "evaluation_role": DESCRIPTIVE_LABEL,
                    "comparison_role": (
                        DESCRIPTIVE_LABEL if model == "enhanced ARIMA" else "external_comparator"
                    ),
                    "selection_role": SELECTION_LABEL,
                    "source_kind": source_kinds[model],
                }
            )
    result = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    result["_country_order"] = result["country"].map(_COUNTRY_ORDER)
    result["_model_order"] = result["model"].map({name: index for index, name in enumerate(model_names)})
    return result.sort_values(["_country_order", "_model_order"], kind="mergesort").drop(
        columns=["_country_order", "_model_order"]
    ).reset_index(drop=True)


def run_descriptive_2025_test(
    output_root: str | Path,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    output_root = Path(output_root)
    freeze = verify_freeze(output_root / "freeze_2024" / "frozen_specifications_2024.csv")
    frames = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    manifest = build_2025_manifest(freeze, frames)
    output = output_root / "test_2025"
    output.mkdir(parents=True, exist_ok=True)
    branch_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for branch in ("ordinary", "weekly_differenced"):
        branch_manifest = manifest.loc[manifest["branch"].eq(branch)].copy()
        branch_output = output / branch
        jobs = _load_checkpoint(branch_output / "jobs.csv", JOB_COLUMNS)
        forecasts = _load_checkpoint(branch_output / "forecasts.csv", FORECAST_COLUMNS)
        diagnostics = _load_checkpoint(branch_output / "diagnostics.csv", DIAGNOSTIC_COLUMNS)
        jobs, forecasts, diagnostics = _reconcile_checkpoints(
            jobs, forecasts, diagnostics, branch_manifest
        )
        terminal = screening._terminal_job_keys(
            jobs, forecasts, branch_manifest, source_frames=frames
        )
        for job in branch_manifest.to_dict("records"):
            key = _job_key(job)
            if key in terminal:
                continue
            result = _execute_job_2025(job, frames[str(job["country"])])
            jobs, forecasts, diagnostics = _persist_result(
                jobs,
                forecasts,
                diagnostics,
                result,
                branch_manifest,
                branch_output,
            )
        jobs = screening._stable_checkpoint_frame(jobs, JOB_KEY)
        forecasts = screening._stable_checkpoint_frame(forecasts, FORECAST_KEY)
        diagnostics = screening._stable_checkpoint_frame(diagnostics, DIAGNOSTIC_KEY)
        _atomic_write_csv(jobs, branch_output / "jobs.csv")
        _atomic_write_csv(forecasts, branch_output / "forecasts.csv")
        _atomic_write_csv(diagnostics, branch_output / "diagnostics.csv")
        branch_frames[branch] = (jobs, forecasts, diagnostics)

    jobs = pd.concat([value[0] for value in branch_frames.values()], ignore_index=True)
    forecasts = pd.concat([value[1] for value in branch_frames.values()], ignore_index=True)
    diagnostics = pd.concat([value[2] for value in branch_frames.values()], ignore_index=True)
    jobs, forecasts, diagnostics = _reconcile_checkpoints(
        jobs, forecasts, diagnostics, manifest
    )
    _validate_run(jobs, forecasts, manifest, frames)
    summary = _write_run_outputs(output, jobs, forecasts, diagnostics, manifest, frames)
    comparator_forecasts = _load_comparator_forecasts()
    smard_benchmark = _load_smard_benchmark()
    comparison = build_final_comparison(
        summary, forecasts, comparator_forecasts, smard_benchmark
    )
    comparison_path = output_root / "comparison" / "enhanced_arima_final_comparison_2025.csv"
    _atomic_write_csv(comparison, comparison_path)
    expected_invalid = _expected_invalid_jobs(manifest)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(jobs["status"].astype(str).eq("completed").sum()),
        "failed_jobs": int(jobs["status"].astype(str).eq("failed").sum()),
        "terminal_weekly_invalid_jobs": int(
            jobs["status"].astype(str).eq("weekly_lag_invalid").sum()
        ),
        "expected_weekly_invalid_jobs": len(expected_invalid),
        "forecast_rows": len(forecasts),
        "diagnostic_rows": len(diagnostics),
        "summary": summary.to_dict(orient="records"),
        "comparison_rows": len(comparison),
        "output_directory": str(output),
        "comparison_path": str(comparison_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the freeze-gated descriptive enhanced ARIMA 2025 test"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    args = parser.parse_args()
    print(
        json.dumps(
            run_descriptive_2025_test(args.output_root, args.processed_directory),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
