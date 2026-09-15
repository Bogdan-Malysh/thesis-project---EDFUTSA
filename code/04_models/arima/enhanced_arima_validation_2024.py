from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
import json
import multiprocessing as mp
from pathlib import Path
import sys
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
)
from enhanced_arima_models import (  # noqa: E402
    derive_weekly_invalid_target_dates,
    enhanced_specifications,
)
import enhanced_arima_screening_2024 as screening  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2024
JOB_KEY = screening.SCREENING_JOB_KEY
FORECAST_KEY = screening.SCREENING_FORECAST_KEY
DIAGNOSTIC_KEY = screening.SCREENING_DIAGNOSTIC_KEY
JOB_COLUMNS = screening.JOB_COLUMNS
FORECAST_COLUMNS = screening.FORECAST_COLUMNS
DIAGNOSTICS_COLUMNS = screening.DIAGNOSTICS_COLUMNS
MAPPING_ISSUE_COLUMNS = screening.MAPPING_ISSUE_COLUMNS
_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _finite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _int_value(value: object, default: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number


def _json_object(value: object) -> dict[str, object]:
    if value is None:
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_list(value: object) -> list[object]:
    if value is None:
        return []
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def _bounded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    local_dates = pd.to_datetime(prepared["local_date"], errors="raise").dt.date
    start = date(VALIDATION_YEAR, 1, 1) - timedelta(days=8)
    end = date(VALIDATION_YEAR, 12, 31)
    return prepared.loc[(local_dates >= start) & (local_dates <= end)].copy()


def _candidate_order(row: Mapping[str, object], fallback: int) -> int:
    try:
        return int(row["specification_order"])
    except (KeyError, TypeError, ValueError):
        return fallback


def _normalise_shortlist(shortlist: pd.DataFrame) -> pd.DataFrame:
    required = {
        "country",
        "branch",
        "specification_id",
        "p",
        "d",
        "q",
        "trend",
        "transform",
    }
    missing = sorted(required.difference(shortlist.columns))
    if missing:
        raise ValueError(f"shortlist is missing required columns: {missing}")
    result = shortlist.copy()
    result["country"] = result["country"].astype(str)
    result["branch"] = result["branch"].astype(str)
    result["specification_id"] = result["specification_id"].astype(str)
    if not set(result["country"]) == set(COUNTRY_CONFIG):
        raise ValueError("shortlist must contain exactly Germany and Austria")
    if result.duplicated(["country", "branch", "specification_id"]).any():
        raise ValueError("shortlist contains duplicate candidate keys")
    counts = result.groupby(["country", "branch"]).size().to_dict()
    maximum_counts = {
        (country, "ordinary"): 3
        for country in COUNTRY_CONFIG
    }
    maximum_counts.update(
        {(country, "weekly_differenced"): 2 for country in COUNTRY_CONFIG}
    )
    if set(counts) != set(maximum_counts) or any(
        not 1 <= counts[key] <= maximum for key, maximum in maximum_counts.items()
    ):
        raise ValueError(
            "shortlist must contain at least one and at most three ordinary or two "
            "weekly candidates per country"
        )
    result["p"] = pd.to_numeric(result["p"], errors="raise").astype(int)
    result["d"] = pd.to_numeric(result["d"], errors="raise").astype(int)
    result["q"] = pd.to_numeric(result["q"], errors="raise").astype(int)
    order_lookup = {
        (branch, specification.specification_id): order
        for branch in ("ordinary", "weekly_differenced")
        for order, specification in enumerate(
            (
                item
                for item in enhanced_specifications()
                if item.branch == branch
            ),
            1,
        )
    }
    if "specification_order" not in result:
        result["specification_order"] = [
            order_lookup.get((row["branch"], row["specification_id"]))
            for row in result.to_dict("records")
        ]
    else:
        result["specification_order"] = pd.to_numeric(
            result["specification_order"], errors="raise"
        )
    expected_orders = pd.Series(
        [
            order_lookup.get((row["branch"], row["specification_id"]))
            for row in result.to_dict("records")
        ],
        index=result.index,
    )
    if (
        result["specification_order"].isna().any()
        or expected_orders.isna().any()
        or not result["specification_order"].eq(expected_orders).all()
    ):
        raise ValueError("shortlist contains an unknown specification order")
    specification_lookup = {
        (specification.branch, specification.specification_id): specification
        for specification in enhanced_specifications()
    }
    for row in result.to_dict("records"):
        specification = specification_lookup.get(
            (row["branch"], row["specification_id"])
        )
        if specification is None:
            raise ValueError("shortlist contains an unknown branch or specification")
        if (
            tuple(row[column] for column in ("p", "d", "q"))
            != specification.order
            or str(row["trend"]) != specification.trend
            or str(row["transform"]) != specification.transform
        ):
            raise ValueError("shortlist contains mismatched specification settings")
    result["specification_order"] = result["specification_order"].astype(int)
    return result.reset_index(drop=True)


def _weekly_invalid_lookup(frame: pd.DataFrame) -> dict[str, str]:
    invalid = derive_weekly_invalid_target_dates(frame, VALIDATION_YEAR)
    if invalid.empty:
        return {}
    grouped: dict[str, list[str]] = {}
    for row in invalid.to_dict("records"):
        target_date = _date_text(row["target_date"])
        reason = str(row.get("reason") or "weekly source mapping issue")
        if reason in {"nan", "None", ""}:
            reason = "weekly source mapping issue"
        grouped.setdefault(target_date, []).append(reason)
    return {
        target_date: "; ".join(dict.fromkeys(reasons))
        for target_date, reasons in grouped.items()
    }


def build_full_validation_manifest(
    shortlist: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build the 2024 country/date/candidate manifest without future data."""
    shortlist = _normalise_shortlist(shortlist)
    missing_frames = sorted(set(COUNTRY_CONFIG).difference(frames))
    if missing_frames:
        raise ValueError(f"country frames are missing: {missing_frames}")

    target_dates = pd.date_range(
        f"{VALIDATION_YEAR}-01-01",
        f"{VALIDATION_YEAR}-12-31",
        freq="D",
    ).date
    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        frame = _bounded_frame(frames[country])
        invalid_by_date = _weekly_invalid_lookup(frame)
        candidates = shortlist.loc[shortlist["country"].eq(country)]
        for target in target_dates:
            target_text = target.isoformat()
            expected_observations = len(extract_target_day(frame, target_text))
            origin = country_forecast_origin(target_text, country)
            origin_local = origin.tz_convert(COUNTRY_CONFIG[country]["timezone"])
            invalid_reason = invalid_by_date.get(target_text, "")
            for candidate in candidates.to_dict("records"):
                rows.append(
                    {
                        "country": country,
                        "target_date": target_text,
                        "forecast_origin_local": origin_local.isoformat(),
                        "forecast_origin_utc": origin.isoformat(),
                        "information_cutoff_utc": origin.isoformat(),
                        "expected_observations": expected_observations,
                        "branch": candidate["branch"],
                        "specification_id": candidate["specification_id"],
                        "specification_order": candidate["specification_order"],
                        "p": candidate["p"],
                        "d": candidate["d"],
                        "q": candidate["q"],
                        "trend": candidate["trend"],
                        "transform": candidate["transform"],
                        "weekly_invalid": bool(
                            candidate["branch"] == "weekly_differenced"
                            and invalid_reason
                        ),
                        "weekly_invalid_reason": (
                            invalid_reason
                            if candidate["branch"] == "weekly_differenced"
                            else ""
                        ),
                    }
                )

    manifest = pd.DataFrame(rows)
    key = list(JOB_KEY)
    if manifest.duplicated(key).any():
        raise ValueError("full validation manifest contains duplicate job keys")
    return manifest


def _load_validation_frames(
    processed_directory: str | Path = PROCESSED,
) -> dict[str, pd.DataFrame]:
    return {
        country: screening._load_setup_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _load_reusable_candidates() -> dict[tuple[str, str, str], dict[str, object]]:
    return screening._read_reusable_original_artifacts()


def _execute_validation_job(
    job: Mapping[str, object],
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Use Task 3 execution semantics for one origin-bounded job."""
    return screening._execute_job(job, frame)


def _initialize_validation_worker(processed_directory: str) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = _load_validation_frames(processed_directory)


def _worker_execute_validation(job: Mapping[str, object]) -> dict[str, object]:
    return _execute_validation_job(job, _WORKER_FRAMES[str(job["country"])])


def _job_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["country"]),
        _date_text(row["target_date"]),
        str(row["branch"]),
        str(row["specification_id"]),
    )


def _expected_invalid_jobs(
    manifest: pd.DataFrame,
) -> dict[tuple[str, str, str, str], str]:
    if manifest.empty or "weekly_invalid" not in manifest:
        return {}
    expected: dict[tuple[str, str, str, str], str] = {}
    for row in manifest.to_dict("records"):
        if str(row.get("branch")) != "weekly_differenced":
            continue
        if not _bool_value(row.get("weekly_invalid", False)):
            continue
        reason = str(row.get("weekly_invalid_reason", "")).strip()
        if reason:
            expected[_job_key(row)] = reason
    return expected


def _blank(value: object) -> bool:
    return value is None or str(value).strip() in {"", "nan", "None", "NaT"}


def _completed_job_metadata_complete(job: Mapping[str, object]) -> bool:
    required = (
        "status",
        "fit_status",
        "convergence_status",
        "converged",
        "parameters_finite",
        "standard_errors_finite",
        "stationarity_ok",
        "invertibility_ok",
        "expected_observations",
        "log_likelihood",
        "aic",
        "aicc",
        "bic",
    )
    if any(column not in job or _blank(job.get(column)) for column in required):
        return False
    if "error_message" not in job:
        return False
    if str(job.get("status")) != "completed":
        return False
    if str(job.get("fit_status")) != "success" or str(
        job.get("convergence_status")
    ) != "converged":
        return False
    if not all(
        _bool_value(job.get(column))
        for column in (
            "converged",
            "parameters_finite",
            "standard_errors_finite",
            "stationarity_ok",
            "invertibility_ok",
        )
    ):
        return False
    if not _blank(job.get("error_message")):
        return False
    if not all(
        _finite(job.get(column))
        for column in ("log_likelihood", "aic", "aicc", "bic")
    ):
        return False
    if _int_value(job.get("expected_observations")) <= 0:
        return False
    return _residual_metadata_complete(job)[0]


def _job_common_eligible(job: Mapping[str, object]) -> bool:
    return _completed_job_metadata_complete(job) and not _bool_value(
        job.get("weekly_invalid", False)
    )


def _terminal_job_keys(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    expected_invalid: Mapping[tuple[str, str, str, str], str] | None = None,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> set[tuple[str, str, str, str]]:
    expected_invalid = expected_invalid or {}
    terminal: set[tuple[str, str, str, str]] = set()
    for job in jobs.to_dict("records"):
        key = _job_key(job)
        status = str(job.get("status", ""))
        if (
            status == "weekly_lag_invalid"
            and _bool_value(job.get("weekly_invalid", False))
            and expected_invalid.get(key) == str(job.get("error_message", "")).strip()
        ):
            terminal.add(key)
        elif (
            status == "completed"
            and _completed_job_metadata_complete(job)
            and _forecast_complete_for_job(
                job,
                forecasts,
                source_frames.get(str(job["country"]))
                if source_frames is not None
                else None,
            )
        ):
            terminal.add(key)
    return terminal


def _load_checkpoint(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        loaded = screening._load_csv(path, columns)
    except (TypeError, ValueError):
        if columns != FORECAST_COLUMNS or not path.exists():
            raise
        loaded = pd.read_csv(path)
        for column in columns:
            if column not in loaded:
                loaded[column] = np.nan
        timestamps = pd.to_datetime(
            loaded["timestamp_utc"], format="mixed", utc=True, errors="coerce"
        )
        loaded = loaded.loc[timestamps.notna()].copy()
        loaded["timestamp_utc"] = timestamps.loc[loaded.index].map(
            pd.Timestamp.isoformat
        )
        loaded = screening.upsert_frame(
            pd.DataFrame(columns=columns), loaded, FORECAST_KEY
        )
    return loaded.reindex(columns=columns)


def _reconcile_checkpoint_frames(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_keys = {_job_key(row) for row in manifest.to_dict("records")}
    manifest_by_key = {
        _job_key(row): row for row in manifest.to_dict("records")
    }

    def metadata_matches_manifest(
        row: Mapping[str, object], expected: Mapping[str, object]
    ) -> bool:
        try:
            for column in expected:
                if column in JOB_KEY:
                    continue
                if column not in row:
                    return False
                if _blank(row[column]) and _blank(expected[column]):
                    continue
                if _blank(expected[column]):
                    return False
                if column == "target_date":
                    if _date_text(row[column]) != _date_text(expected[column]):
                        return False
                elif column in {
                    "p",
                    "d",
                    "q",
                    "specification_order",
                    "expected_observations",
                }:
                    if _int_value(row[column], -1) != _int_value(expected[column], -1):
                        return False
                elif column in {
                    "forecast_origin_local",
                    "forecast_origin_utc",
                    "information_cutoff_utc",
                }:
                    if _utc(row[column]) != _utc(expected[column]):
                        return False
                elif column == "weekly_invalid":
                    if _bool_value(row[column]) != _bool_value(expected[column]):
                        return False
                elif str(row[column]).strip() != str(expected[column]).strip():
                    return False
        except (TypeError, ValueError, OverflowError):
            return False
        return True

    def reconcile(
        frame: pd.DataFrame,
        columns: list[str],
        key_columns: Iterable[str],
        deduplication_key: Iterable[str],
        validate_job_metadata: bool = False,
    ) -> pd.DataFrame:
        result = frame.reindex(columns=columns).copy()
        if result.empty:
            return result.reset_index(drop=True)
        key_columns = tuple(key_columns)

        def belongs(row: Mapping[str, object]) -> bool:
            try:
                key = tuple(
                    screening._canonical_key_value(column, row[column])
                    for column in key_columns
                )
                return key in expected_keys
            except (KeyError, TypeError, ValueError):
                return False

        result = result.loc[result.apply(belongs, axis=1)].reset_index(drop=True)
        if validate_job_metadata and not result.empty:
            result = result.loc[
                result.apply(
                    lambda row: metadata_matches_manifest(
                        row, manifest_by_key[_job_key(row)]
                    ),
                    axis=1,
                )
            ].reset_index(drop=True)
        return screening.upsert_frame(
            pd.DataFrame(columns=columns),
            result,
            deduplication_key,
            sort=False,
        )

    return (
        reconcile(
            jobs, JOB_COLUMNS, JOB_KEY, JOB_KEY, validate_job_metadata=True
        ),
        reconcile(forecasts, FORECAST_COLUMNS, FORECAST_KEY[:4], FORECAST_KEY),
        reconcile(diagnostics, DIAGNOSTICS_COLUMNS, DIAGNOSTIC_KEY[:4], DIAGNOSTIC_KEY),
    )


def _drop_job_rows(
    frame: pd.DataFrame,
    job_key: tuple[str, str, str, str],
    key_columns: Iterable[str],
) -> pd.DataFrame:
    return screening._drop_job_rows(frame, key_columns, job_key)


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
    jobs = screening.upsert_frame(
        jobs,
        pd.DataFrame([job]),
        JOB_KEY,
        sort=False,
    )
    forecasts = _drop_job_rows(forecasts, key, FORECAST_KEY)
    forecast_rows = pd.DataFrame(result.get("forecasts", []))
    if not forecast_rows.empty:
        forecasts = screening.upsert_frame(
            forecasts,
            forecast_rows,
            FORECAST_KEY,
            sort=False,
        )
    diagnostics = _drop_job_rows(diagnostics, key, DIAGNOSTIC_KEY)
    diagnostic_rows = pd.DataFrame(result.get("diagnostics", []))
    if not diagnostic_rows.empty:
        diagnostics = screening.upsert_frame(
            diagnostics,
            diagnostic_rows,
            DIAGNOSTIC_KEY,
            sort=False,
        )
    screening._atomic_write_csv(forecasts, branch_output / "forecasts.csv")
    screening._atomic_write_csv(diagnostics, branch_output / "diagnostics.csv")
    screening._atomic_write_csv(jobs, branch_output / "jobs.csv")
    screening._atomic_write_csv(
        screening._summary_frame(jobs, branch_manifest, forecasts),
        branch_output / "summary.csv",
    )
    return jobs, forecasts, diagnostics


def _mapping_issue_frame(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for country, raw_frame in frames.items():
        frame = _bounded_frame(raw_frame)
        invalid = derive_weekly_invalid_target_dates(frame, VALIDATION_YEAR)
        if invalid.empty:
            continue
        expected_by_date = {
            target_date: len(extract_target_day(frame, target_date))
            for target_date in invalid["target_date"].astype(str).unique()
        }
        counts = invalid.groupby(invalid["target_date"].astype(str)).size().to_dict()
        for issue in invalid.to_dict("records"):
            target_date = _date_text(issue["target_date"])
            source_key = issue.get("source_key")
            expected = expected_by_date[target_date]
            affected = int(counts[target_date])
            records.append(
                {
                    "country": country,
                    "branch": "weekly_differenced",
                    "target_date": target_date,
                    "target_timestamp_utc": issue.get("target_timestamp_utc"),
                    "affected_timestamp_utc": issue.get("target_timestamp_utc"),
                    "source_key": source_key,
                    "source_local_key": source_key,
                    "source_timestamp_utc": issue.get("source_timestamp_utc"),
                    "reason": issue.get("reason"),
                    "source_row_count": issue.get("source_row_count"),
                    "coverage": float(max(0, expected - affected) / expected),
                }
            )
    return pd.DataFrame(records, columns=MAPPING_ISSUE_COLUMNS)


def _valid_target_forecasts(
    forecasts: pd.DataFrame,
    country: str,
    branch: str,
    specification_id: str,
    dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    if forecasts.empty:
        return forecasts.copy()
    mask = (
        forecasts["country"].astype(str).eq(country)
        & forecasts["branch"].astype(str).eq(branch)
        & forecasts["specification_id"].astype(str).eq(specification_id)
        & forecasts["is_target_day"].map(_bool_value)
        & forecasts["evaluated"].map(_bool_value)
        & forecasts["status"].astype(str).eq("completed")
    )
    result = forecasts.loc[mask].copy()
    if dates is not None:
        result = result.loc[result["target_date"].map(_date_text).isin(set(dates))]
    if result.empty:
        return result
    result["actual_load_mwh"] = pd.to_numeric(
        result["actual_load_mwh"], errors="coerce"
    )
    result["forecast_mwh"] = pd.to_numeric(result["forecast_mwh"], errors="coerce")
    result = result.loc[
        np.isfinite(result["actual_load_mwh"])
        & np.isfinite(result["forecast_mwh"])
    ].copy()
    timestamps = pd.to_datetime(
        result["timestamp_utc"], format="mixed", utc=True, errors="coerce"
    )
    result = result.loc[timestamps.notna()].copy()
    result["timestamp_utc"] = timestamps.loc[result.index]
    return result


def build_common_timestamp_set(
    forecasts: pd.DataFrame,
    candidates: pd.DataFrame,
    country: str,
    jobs: pd.DataFrame | None = None,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return the exact valid target UTC timestamp intersection for a country."""
    required = {"country", "branch", "specification_id"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidates are missing required columns: {missing}")
    country_candidates = candidates.loc[candidates["country"].astype(str).eq(country)]
    intersection: set[pd.Timestamp] | None = None
    for candidate in country_candidates.to_dict("records"):
        valid_dates: set[str] | None = None
        if jobs is not None:
            candidate_jobs = _job_groups(
                jobs,
                country,
                str(candidate["branch"]),
                str(candidate["specification_id"]),
            )
            valid_dates = {
                _date_text(job["target_date"])
                for job in candidate_jobs.to_dict("records")
                if _job_common_eligible(job)
                and _forecast_complete_for_job(
                    job,
                    forecasts,
                    source_frames.get(country) if source_frames is not None else None,
                )
            }
        valid = _valid_target_forecasts(
            forecasts,
            country,
            str(candidate["branch"]),
            str(candidate["specification_id"]),
            valid_dates,
        )
        timestamps = (
            set(valid["timestamp_utc"].drop_duplicates())
            if not valid.empty and "timestamp_utc" in valid
            else set()
        )
        intersection = timestamps if intersection is None else intersection & timestamps
    timestamps = sorted(intersection or set())
    return pd.DataFrame(
        {
            "country": [country] * len(timestamps),
            "timestamp_utc": timestamps,
        },
        columns=["country", "timestamp_utc"],
    )


def _candidate_rows(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "country",
        "branch",
        "specification_id",
        "specification_order",
        "p",
        "d",
        "q",
        "trend",
        "transform",
    ]
    source = manifest if not manifest.empty else jobs
    result = source.copy()
    for column in columns:
        if column not in result:
            result[column] = np.nan
    result = result[columns].drop_duplicates(
        ["country", "branch", "specification_id"]
    )
    return result.reset_index(drop=True)


def _job_groups(
    jobs: pd.DataFrame,
    country: str,
    branch: str,
    specification_id: str,
) -> pd.DataFrame:
    if jobs.empty:
        return jobs.copy()
    return jobs.loc[
        jobs["country"].astype(str).eq(country)
        & jobs["branch"].astype(str).eq(branch)
        & jobs["specification_id"].astype(str).eq(specification_id)
    ].copy()


def _expected_observations(group: pd.DataFrame) -> int:
    if "expected_observations" not in group:
        return 0
    values = pd.to_numeric(group["expected_observations"], errors="coerce")
    return int(values.fillna(0).sum())


def build_native_and_common_summary(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    manifest: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    expected_keys = {
        _job_key(row) for row in manifest.to_dict("records")
    }

    def retain_manifest_keys(
        frame: pd.DataFrame,
        key_columns: Iterable[str],
        deduplication_key: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        key_columns = tuple(key_columns)
        if not set(key_columns).issubset(frame.columns):
            return frame.iloc[0:0].copy()
        mask = frame.apply(
            lambda row: tuple(
                screening._canonical_key_value(column, row[column])
                for column in key_columns
            ) in expected_keys,
            axis=1,
        )
        return screening.upsert_frame(
            pd.DataFrame(columns=frame.columns),
            frame.loc[mask],
            deduplication_key or key_columns,
            sort=False,
        )

    # Checkpoint reconciliation already happens before aggregation; retain
    # rejected job rows here so their failure reasons remain reportable.
    jobs = retain_manifest_keys(jobs, JOB_KEY)
    forecasts = retain_manifest_keys(forecasts, FORECAST_KEY[:4], FORECAST_KEY)
    candidates = _candidate_rows(jobs, manifest)
    expected_invalid = _expected_invalid_jobs(manifest)
    rows: list[dict[str, object]] = []
    common_sets = {
        country: build_common_timestamp_set(
            forecasts,
            candidates,
            country,
            jobs=jobs,
            source_frames=source_frames,
        )
        for country in COUNTRY_CONFIG
    }
    for candidate in candidates.to_dict("records"):
        country = str(candidate["country"])
        branch = str(candidate["branch"])
        specification_id = str(candidate["specification_id"])
        expected = manifest.loc[
            manifest["country"].astype(str).eq(country)
            & manifest["branch"].astype(str).eq(branch)
            & manifest["specification_id"].astype(str).eq(specification_id)
        ]
        group = _job_groups(jobs, country, branch, specification_id)
        completed = group.get("status", pd.Series(dtype=object)).astype(str).eq(
            "completed"
        )
        expected_invalid_mask = group.apply(
            lambda row: _job_key(row) in expected_invalid
            and str(row.get("status")) == "weekly_lag_invalid"
            and expected_invalid[_job_key(row)]
            == str(row.get("error_message", "")).strip(),
            axis=1,
        ) if not group.empty else pd.Series(dtype=bool)
        valid_job_dates = {
            _date_text(job["target_date"])
            for job in group.to_dict("records")
            if _job_common_eligible(job)
            and _forecast_complete_for_job(
                job,
                forecasts,
                source_frames.get(country) if source_frames is not None else None,
            )
        }
        invalid_rows = group.loc[
            expected_invalid_mask
        ]
        invalid_dates = set(invalid_rows["target_date"].map(_date_text))
        expected_dates = int(expected["target_date"].nunique()) if not expected.empty else len(group)
        unresolved_dates = max(0, expected_dates - len(valid_job_dates) - len(invalid_dates))
        native = _valid_target_forecasts(
            forecasts,
            country,
            branch,
            specification_id,
            valid_job_dates,
        )
        expected_observations = _expected_observations(expected)
        native_actual = (
            native["actual_load_mwh"]
            if "actual_load_mwh" in native
            else pd.Series(dtype=float)
        )
        native_forecast = (
            native["forecast_mwh"]
            if "forecast_mwh" in native
            else pd.Series(dtype=float)
        )
        native_metrics = calculate_metrics(
            native_actual,
            native_forecast,
            expected_observations=expected_observations,
        )
        common_timestamps = common_sets[country]["timestamp_utc"]
        common = (
            native.loc[native["timestamp_utc"].isin(set(common_timestamps))]
            if "timestamp_utc" in native
            else native.copy()
        )
        common_actual = (
            common["actual_load_mwh"]
            if "actual_load_mwh" in common
            else pd.Series(dtype=float)
        )
        common_forecast = (
            common["forecast_mwh"]
            if "forecast_mwh" in common
            else pd.Series(dtype=float)
        )
        common_metrics = calculate_metrics(
            common_actual,
            common_forecast,
            expected_observations=len(common_timestamps),
        )
        invalid_reasons = [
            reason
            for job in group.to_dict("records")
            for reason in _job_invalid_reasons(
                job,
                forecasts,
                expected_invalid,
                source_frames.get(country) if source_frames is not None else None,
            )
        ]
        expected_job_keys = sorted(
            [_job_key(row) for row in expected.to_dict("records")]
        )
        native_coverage = float(native_metrics["coverage"])
        rows.append(
            {
                **candidate,
                "expected_jobs": len(expected),
                "completed_jobs": int(completed.sum()),
                "failed_jobs": int(
                    sum(
                        not _expected_weekly_invalid(job, expected_invalid)
                        and bool(
                            _job_invalid_reasons(
                                job,
                                forecasts,
                                expected_invalid,
                                source_frames.get(country)
                                if source_frames is not None
                                else None,
                            )
                        )
                        for job in group.to_dict("records")
                    )
                ),
                "expected_dates": expected_dates,
                "valid_dates": len(valid_job_dates),
                "invalid_dates": len(invalid_dates),
                "unresolved_dates": unresolved_dates,
                "invalid_reasons": "; ".join(dict.fromkeys(invalid_reasons)),
                "expected_job_keys": json.dumps(
                    [list(key) for key in expected_job_keys]
                ),
                "expected_observations": expected_observations,
                "native_observations": int(native_metrics["evaluated_observations"]),
                "native_coverage": native_coverage,
                "native_mae": float(native_metrics["mae"]),
                "native_rmse": float(native_metrics["rmse"]),
                "native_mape": float(native_metrics["mape"]),
                "common_observations": int(common_metrics["evaluated_observations"]),
                "common_coverage": float(common_metrics["coverage"]),
                "common_mae": float(common_metrics["mae"]),
                "common_rmse": float(common_metrics["rmse"]),
                "common_mape": float(common_metrics["mape"]),
                "mae": float(native_metrics["mae"]),
                "rmse": float(native_metrics["rmse"]),
                "mape": float(native_metrics["mape"]),
                "evaluated_observations": int(native_metrics["evaluated_observations"]),
                "coverage": native_coverage,
                "common_timestamp_count": len(common_timestamps),
            }
        )
    return pd.DataFrame(rows)


def _forecast_complete_for_job(
    job: Mapping[str, object],
    forecasts: pd.DataFrame,
    source_frame: pd.DataFrame | None = None,
) -> bool:
    try:
        required = {
            *JOB_KEY,
            "model_family",
            "forecast_origin_local",
            "timestamp_utc",
            "interval_end_utc",
            "timestamp_local",
            "local_date",
            "forecast_origin_utc",
            "information_cutoff_utc",
            "actual_load_mwh",
            "forecast_mwh",
            "bridge_used",
            "source_kind",
            "is_target_day",
            "evaluated",
            "status",
        }
        if forecasts.empty or not required.issubset(forecasts.columns):
            return False
        key = _job_key(job)
        mask = np.ones(len(forecasts), dtype=bool)
        for column, expected in zip(JOB_KEY, key):
            mask &= forecasts[column].map(
                lambda value, name=column: screening._canonical_key_value(name, value)
            ).eq(expected)
        matching = forecasts.loc[mask].copy()
        if matching.empty:
            return False
        timestamps = pd.to_datetime(
            matching["timestamp_utc"], format="mixed", utc=True, errors="coerce"
        )
        interval_end = pd.to_datetime(
            matching["interval_end_utc"], format="mixed", utc=True, errors="coerce"
        )
        forecast_origin = pd.to_datetime(
            matching["forecast_origin_utc"], format="mixed", utc=True, errors="coerce"
        )
        information_cutoff = pd.to_datetime(
            matching["information_cutoff_utc"], format="mixed", utc=True, errors="coerce"
        )
        if (
            timestamps.isna().any()
            or interval_end.isna().any()
            or forecast_origin.isna().any()
            or information_cutoff.isna().any()
        ):
            return False
        expected_origin = country_forecast_origin(key[1], key[0])
        expected_origin_local = expected_origin.tz_convert(
            COUNTRY_CONFIG[key[0]]["timezone"]
        ).isoformat()
        if not forecast_origin.eq(expected_origin).all() or not information_cutoff.eq(
            expected_origin
        ).all():
            return False
        if not matching["forecast_origin_local"].astype(str).eq(
            expected_origin_local
        ).all():
            return False
        if not interval_end.eq(timestamps + pd.Timedelta(hours=1)).all():
            return False
        timezone = COUNTRY_CONFIG[key[0]]["timezone"]
        expected_end = pd.Timestamp(
            f"{pd.Timestamp(key[1]).date() + timedelta(days=1)} 00:00:00",
            tz=timezone,
        ).tz_convert("UTC")
        expected_path = pd.date_range(
            expected_origin,
            expected_end,
            inclusive="left",
            freq="h",
        )
        if len(matching) != len(expected_path) or not timestamps.is_unique:
            return False
        matching = matching.assign(_timestamp_utc=timestamps).sort_values(
            "_timestamp_utc", kind="stable"
        )
        timestamps = matching["_timestamp_utc"]
        if list(timestamps) != list(expected_path):
            return False
        expected_local = expected_path.tz_convert(timezone)
        if matching["timestamp_local"].astype(str).tolist() != [
            timestamp.isoformat() for timestamp in expected_local
        ]:
            return False
        if matching["local_date"].astype(str).tolist() != [
            timestamp.date().isoformat() for timestamp in expected_local
        ]:
            return False
        expected_target = expected_local.date == pd.Timestamp(key[1]).date()
        if matching["status"].astype(str).tolist() != [
            "completed"
        ] * len(matching):
            return False
        for column, expected_values in (
            ("is_target_day", expected_target),
            ("evaluated", expected_target),
            ("bridge_used", ~expected_target),
        ):
            if matching[column].map(_blank).any() or matching[column].map(
                _bool_value
            ).tolist() != expected_values.tolist():
                return False
        for column in ("model_family", "source_kind"):
            if matching[column].map(_blank).any():
                return False
        if key[2] == "weekly_differenced":
            if not screening._weekly_source_metadata_complete(
                job, matching, expected_origin, source_frame
            ):
                return False
        target = matching.loc[
            matching["is_target_day"].map(_bool_value)
            & matching["evaluated"].map(_bool_value)
            & matching["status"].astype(str).eq("completed")
            & matching["local_date"].astype(str).str[:10].eq(key[1])
        ]
        expected_observations = _int_value(job.get("expected_observations"))
        if expected_observations <= 0 or len(target) != expected_observations:
            return False
        if set(target["_timestamp_utc"]) != set(
            screening._expected_target_index(key[0], key[1])
        ):
            return False
        actual = pd.to_numeric(matching["actual_load_mwh"], errors="coerce")
        forecast = pd.to_numeric(matching["forecast_mwh"], errors="coerce")
        return bool(np.isfinite(actual).all() and np.isfinite(forecast).all())
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _residual_metadata_complete(job: Mapping[str, object]) -> tuple[bool, int]:
    required = (
        "residual_n_effective",
        "residual_rejected",
        "residual_rejection_reason",
        "residual_acf_values",
        "residual_acf_flagged_lags",
        "residual_ljung_box_statistics",
        "residual_ljung_box_pvalues",
    )
    if any(
        column not in job
        or str(job.get(column)).strip() in {"", "nan", "None"}
        for column in required
        if column != "residual_rejection_reason"
    ):
        return False, 0
    if "residual_rejection_reason" not in job:
        return False, 0
    acf = _json_object(job.get("residual_acf_values"))
    statistics = _json_object(job.get("residual_ljung_box_statistics"))
    pvalues = _json_object(job.get("residual_ljung_box_pvalues"))
    try:
        flagged = json.loads(str(job.get("residual_acf_flagged_lags")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, 0
    if not isinstance(flagged, list):
        return False, 0
    if not acf or not {"24", "48"}.issubset(statistics) or not {"24", "48"}.issubset(pvalues):
        return False, 0
    flagged_keys = {str(lag) for lag in flagged}
    if not flagged_keys.issubset({str(lag) for lag in acf}):
        return False, 0
    rejected = _bool_value(job.get("residual_rejected"))
    rejection_reason = str(job.get("residual_rejection_reason", "")).strip()
    if rejected != bool(rejection_reason and rejection_reason not in {"nan", "None"}):
        return False, 0
    if not _finite(job.get("residual_n_effective")) or _int_value(job.get("residual_n_effective")) <= 0:
        return False, 0
    try:
        complete = all(
            np.isfinite(float(value))
            for values in (acf, statistics, pvalues)
            for value in values.values()
        )
    except (TypeError, ValueError):
        return False, 0
    if not complete:
        return False, 0
    warning_count = len(flagged)
    warning_count += sum(
        1
        for value in pvalues.values()
        if _finite(value) and float(value) < 0.01
    )
    return True, warning_count


def _expected_weekly_invalid(
    job: Mapping[str, object],
    expected_invalid: Mapping[tuple[str, str, str, str], str],
) -> bool:
    if not (
        str(job.get("status")) == "weekly_lag_invalid"
        and str(job.get("branch")) == "weekly_differenced"
        and _bool_value(job.get("weekly_invalid", False))
    ):
        return False
    expected_reason = expected_invalid.get(_job_key(job), "")
    actual_reason = str(job.get("error_message", "")).strip()
    return bool(expected_reason and actual_reason == expected_reason)


def _job_invalid_reasons(
    job: Mapping[str, object],
    forecasts: pd.DataFrame,
    expected_invalid: Mapping[tuple[str, str, str, str], str],
    source_frame: pd.DataFrame | None = None,
) -> list[str]:
    expected = _expected_weekly_invalid(job, expected_invalid)
    if expected:
        return [
            str(job.get("error_message", "")).strip()
            or expected_invalid.get(_job_key(job), "expected weekly mapping invalidity")
        ]
    if _job_common_eligible(job) and _forecast_complete_for_job(
        job, forecasts, source_frame
    ):
        return []
    reasons: list[str] = []
    error = str(job.get("error_message", "")).strip()
    if error and error not in {"nan", "None"}:
        reasons.append(error)
    status = str(job.get("status", "")).strip()
    if status != "completed":
        reasons.append(f"status={status or 'missing'}")
    if status == "completed" and not _completed_job_metadata_complete(job):
        reasons.append("completed job metadata is invalid")
    if status == "completed" and not _forecast_complete_for_job(
        job, forecasts, source_frame
    ):
        reasons.append("forecast rows are incomplete or non-finite")
    return list(dict.fromkeys(reason for reason in reasons if reason)) or [
        "validation rejected job"
    ]


def _diagnostics_complete_for_job(
    job: Mapping[str, object],
    diagnostics: pd.DataFrame | None,
    expected_invalid: Mapping[tuple[str, str, str, str], str],
) -> bool:
    if _expected_weekly_invalid(job, expected_invalid):
        return True
    if diagnostics is None or diagnostics.empty:
        return False
    required_columns = {
        "country",
        "target_date",
        "branch",
        "specification_id",
        "diagnostic",
        "lag",
        "value",
        "threshold",
        "flagged",
    }
    if not required_columns.issubset(diagnostics.columns):
        return False
    target_date = _date_text(job["target_date"])
    matching = diagnostics.loc[
        diagnostics["country"].astype(str).eq(str(job["country"]))
        & diagnostics["target_date"].map(_date_text).eq(target_date)
        & diagnostics["branch"].astype(str).eq(str(job["branch"]))
        & diagnostics["specification_id"].astype(str).eq(
            str(job["specification_id"])
        )
    ].copy()
    if matching.empty:
        return False
    try:
        observed_keys = list(
            zip(matching["diagnostic"].astype(str), matching["lag"].astype(int))
        )
    except (TypeError, ValueError):
        return False
    if len(observed_keys) != len(set(observed_keys)):
        return False
    acf = _json_object(job.get("residual_acf_values"))
    statistics = _json_object(job.get("residual_ljung_box_statistics"))
    pvalues = _json_object(job.get("residual_ljung_box_pvalues"))
    try:
        flagged_lags = {int(lag) for lag in _json_list(job.get("residual_acf_flagged_lags"))}
        expected_values = {
            ("residual_acf", int(lag)): (float(value), int(lag) in flagged_lags)
            for lag, value in acf.items()
        }
        expected_values.update(
            {
                ("ljung_box_statistic", int(lag)): (
                    float(value),
                    float(pvalues[str(lag)]) < 0.01,
                )
                for lag, value in statistics.items()
            }
        )
        expected_values.update(
            {
                ("ljung_box_pvalue", int(lag)): (
                    float(value),
                    float(value) < 0.01,
                )
                for lag, value in pvalues.items()
            }
        )
    except (TypeError, ValueError):
        return False
    if not expected_values or set(observed_keys) != set(expected_values):
        return False
    for row in matching.to_dict("records"):
        if not _finite(row.get("value")) or not _finite(row.get("threshold")):
            return False
        if str(row.get("flagged")).strip() in {"", "nan", "None"}:
            return False
        key = (str(row["diagnostic"]), int(row["lag"]))
        expected_value, expected_flag = expected_values[key]
        if not np.isclose(float(row["value"]), expected_value, rtol=1e-12, atol=1e-12):
            return False
        if _bool_value(row["flagged"]) != expected_flag:
            return False
    return True


def _diagnostic_warning_count(
    job: Mapping[str, object],
    diagnostics: pd.DataFrame,
) -> int:
    if diagnostics.empty:
        return 0
    target_date = _date_text(job["target_date"])
    matching = diagnostics.loc[
        diagnostics["country"].astype(str).eq(str(job["country"]))
        & diagnostics["target_date"].map(_date_text).eq(target_date)
        & diagnostics["branch"].astype(str).eq(str(job["branch"]))
        & diagnostics["specification_id"].astype(str).eq(
            str(job["specification_id"])
        )
    ]
    return int(matching["flagged"].map(_bool_value).sum())


def _job_adequacy(
    job: Mapping[str, object],
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame | None,
    expected_invalid: Mapping[tuple[str, str, str, str], str],
    source_frame: pd.DataFrame | None = None,
) -> tuple[bool, bool, int, list[str]]:
    status = str(job.get("status", ""))
    is_expected_invalid = _expected_weekly_invalid(job, expected_invalid)
    if is_expected_invalid:
        return True, True, 0, []
    reasons: list[str] = []
    if status != "completed":
        reasons.append("implementation or data failure")
    if str(job.get("fit_status", "")) != "success":
        reasons.append("fit is not successful")
    if str(job.get("convergence_status", "")) != "converged" or not _bool_value(
        job.get("converged", False)
    ):
        reasons.append("convergence is not eligible")
    for column in (
        "parameters_finite",
        "standard_errors_finite",
        "stationarity_ok",
        "invertibility_ok",
    ):
        if not _bool_value(job.get(column, False)):
            reasons.append(f"{column} is false")
    for column in ("log_likelihood", "aic", "aicc", "bic"):
        if not _finite(job.get(column)):
            reasons.append(f"{column} is not finite")
    if str(job.get("error_message", "")).strip() not in {"", "nan", "None"}:
        reasons.append("implementation or data failure")
    if not _forecast_complete_for_job(job, forecasts, source_frame):
        reasons.append("forecast rows are incomplete or non-finite")
    metadata_complete, warning_count = _residual_metadata_complete(job)
    diagnostics_complete = metadata_complete and _diagnostics_complete_for_job(
        job, diagnostics, expected_invalid
    )
    if diagnostics_complete and diagnostics is not None:
        warning_count = _diagnostic_warning_count(job, diagnostics)
    if not metadata_complete:
        reasons.append("residual diagnostics metadata is incomplete")
    elif not diagnostics_complete:
        reasons.append("diagnostics CSV is incomplete")
    return not reasons, diagnostics_complete, warning_count, reasons


def build_adequacy_review(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
    *,
    manifest: pd.DataFrame | None = None,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if manifest is not None:
        jobs, forecasts, diagnostics = _reconcile_checkpoint_frames(
            jobs,
            forecasts,
            diagnostics if diagnostics is not None else pd.DataFrame(),
            manifest,
        )
    candidates = _candidate_rows(jobs, manifest if manifest is not None else summary)
    expected_invalid = _expected_invalid_jobs(manifest) if manifest is not None else {}
    rows: list[dict[str, object]] = []
    for candidate in candidates.to_dict("records"):
        group = _job_groups(
            jobs,
            str(candidate["country"]),
            str(candidate["branch"]),
            str(candidate["specification_id"]),
        )
        valid_jobs = 0
        invalid_jobs = 0
        terminal_invalid_jobs = 0
        warning_count = 0
        reasons: list[str] = []
        reviewed = True
        for job in group.to_dict("records"):
            adequate, diagnostics_reviewed, warnings, job_reasons = _job_adequacy(
                job,
                forecasts,
                diagnostics,
                expected_invalid,
                source_frames.get(str(candidate["country"]))
                if source_frames is not None
                else None,
            )
            is_expected_invalid = _expected_weekly_invalid(job, expected_invalid)
            if is_expected_invalid:
                terminal_invalid_jobs += 1
            elif adequate:
                valid_jobs += 1
            else:
                invalid_jobs += 1
                reasons.extend(job_reasons)
            reviewed = reviewed and diagnostics_reviewed
            warning_count += warnings
        if group.empty:
            reasons.append("no jobs present")
            reviewed = False
        candidate_summary = summary.loc[
            summary["country"].astype(str).eq(str(candidate["country"]))
            & summary["branch"].astype(str).eq(str(candidate["branch"]))
            & summary["specification_id"].astype(str).eq(
                str(candidate["specification_id"])
            )
        ]
        expected_jobs = (
            _int_value(candidate_summary.iloc[0]["expected_jobs"], len(group))
            if not candidate_summary.empty and "expected_jobs" in candidate_summary
            else len(group)
        )
        expected_dates = (
            _int_value(candidate_summary.iloc[0]["expected_dates"], len(group))
            if not candidate_summary.empty and "expected_dates" in candidate_summary
            else len(group["target_date"].unique())
        )
        present_dates = set(group["target_date"].map(_date_text)) if not group.empty else set()
        expected_keys: set[tuple[str, str, str, str]] | None = None
        if manifest is not None:
            expected_keys = {
                _job_key(row)
                for row in manifest.to_dict("records")
                if str(row.get("country")) == str(candidate["country"])
                and str(row.get("branch")) == str(candidate["branch"])
                and str(row.get("specification_id"))
                == str(candidate["specification_id"])
            }
        elif not candidate_summary.empty and "expected_job_keys" in candidate_summary:
            try:
                expected_keys = {
                    tuple(key)
                    for key in json.loads(
                        str(candidate_summary.iloc[0]["expected_job_keys"])
                    )
                }
            except (TypeError, ValueError, json.JSONDecodeError):
                expected_keys = None
        present_keys = {_job_key(row) for row in group.to_dict("records")}
        if expected_keys is not None:
            expected_jobs = len(expected_keys)
            if present_keys != expected_keys:
                reasons.append("manifest job key set is incomplete or substituted")
                invalid_jobs += len(expected_keys.symmetric_difference(present_keys))
        elif len(group) != expected_jobs:
            reasons.append("manifest job count is incomplete")
            invalid_jobs += abs(expected_jobs - len(group))
        if len(present_dates) != expected_dates:
            reasons.append("manifest target-date coverage is incomplete")
        unique_reasons = list(dict.fromkeys(reasons))
        rows.append(
            {
                **candidate,
                "total_jobs": len(group),
                "expected_jobs": expected_jobs,
                "expected_dates": expected_dates,
                "present_dates": len(present_dates),
                "valid_jobs": valid_jobs,
                "invalid_jobs": invalid_jobs,
                "terminal_invalid_jobs": terminal_invalid_jobs,
                "warning_count": warning_count,
                "residual_warning_count": warning_count,
                "diagnostics_reviewed": bool(reviewed),
                "residual_review_status": "reviewed" if reviewed else "unreviewed",
                "convergence_adequate": bool(
                    valid_jobs + terminal_invalid_jobs == len(group)
                    and bool(valid_jobs)
                    and not unique_reasons
                ),
                "adequate": bool(
                    valid_jobs + terminal_invalid_jobs == len(group)
                    and bool(valid_jobs)
                    and reviewed
                    and not unique_reasons
                ),
                "inadequacy_reasons": "; ".join(unique_reasons),
            }
        )
    return pd.DataFrame(rows)


SELECTION_RULE = "common_mae, common_rmse, common_mape, specification_order"


FREEZE_COLUMNS = [
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
    "total_jobs",
    "valid_jobs",
    "invalid_jobs",
    "terminal_invalid_jobs",
    "warning_count",
    "residual_warning_count",
    "diagnostics_reviewed",
    "residual_review_status",
    "convergence_adequate",
    "adequate",
    "inadequacy_reasons",
    "selection_status",
    "selection_rule",
]


def freeze_specifications(
    summary: pd.DataFrame,
    adequacy: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = ["country", "branch", "specification_id"]
    if not set(merge_keys).issubset(adequacy.columns):
        adequacy = pd.DataFrame(
            columns=merge_keys + ["adequate", "diagnostics_reviewed"]
        )
    if summary.empty:
        countries = list(COUNTRY_CONFIG)
        merged = pd.DataFrame(columns=["country"])
    else:
        merged = summary.merge(
            adequacy,
            on=["country", "branch", "specification_id"],
            how="left",
            suffixes=("", "_adequacy"),
        )
        countries = list(dict.fromkeys([*COUNTRY_CONFIG, *summary["country"].astype(str)]))
    rows: list[dict[str, object]] = []
    for country in countries:
        candidates = merged.loc[
            merged.get("country", pd.Series(dtype=object)).astype(str).eq(country)
            & merged.get("adequate", pd.Series(False, index=merged.index)).map(
                _bool_value
            )
            & merged.get(
                "diagnostics_reviewed", pd.Series(False, index=merged.index)
            ).map(_bool_value)
        ].copy()
        if not candidates.empty:
            candidates["_mae"] = pd.to_numeric(candidates["common_mae"], errors="coerce")
            candidates["_rmse"] = pd.to_numeric(candidates["common_rmse"], errors="coerce")
            candidates["_mape"] = pd.to_numeric(candidates["common_mape"], errors="coerce")
            candidates = candidates.loc[
                np.isfinite(candidates["_mae"])
                & np.isfinite(candidates["_rmse"])
                & np.isfinite(candidates["_mape"])
            ].copy()
        if candidates.empty:
            rows.append(
                {
                    "country": country,
                    "selection_status": "no_eligible_candidate",
                    "selection_rule": SELECTION_RULE,
                    "adequate": False,
                }
            )
            continue
        candidates["_order"] = pd.to_numeric(
            candidates.get("specification_order", pd.Series(np.nan, index=candidates.index)),
            errors="coerce",
        ).fillna(999999)
        chosen = candidates.sort_values(
            ["_mae", "_rmse", "_mape", "_order", "specification_id"],
            kind="stable",
        ).iloc[0].to_dict()
        chosen.pop("_mae", None)
        chosen.pop("_rmse", None)
        chosen.pop("_mape", None)
        chosen.pop("_order", None)
        chosen.update(
            {
                "selection_status": "frozen",
                "selection_rule": SELECTION_RULE,
                "adequate": True,
            }
        )
        rows.append(chosen)
    result = pd.DataFrame(rows)
    for column in FREEZE_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return result.reindex(
        columns=FREEZE_COLUMNS
        + [column for column in result.columns if column not in FREEZE_COLUMNS]
    )


def _write_common_outputs(
    output: Path,
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    manifest: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = build_native_and_common_summary(
        jobs, forecasts, manifest, source_frames=source_frames
    )
    candidate_frame = _candidate_rows(jobs, manifest)
    timestamp_frames = [
        build_common_timestamp_set(
            forecasts,
            candidate_frame,
            country,
            jobs=jobs,
            source_frames=source_frames,
        )
        for country in COUNTRY_CONFIG
    ]
    timestamps = pd.concat(timestamp_frames, ignore_index=True)
    screening._atomic_write_csv(timestamps, output / "common_evaluation_timestamps.csv")
    screening._atomic_write_csv(
        summary.loc[:, [
            column
            for column in summary.columns
            if column.startswith("native_")
            or column
            in {
                "country",
                "branch",
                "specification_id",
                "valid_dates",
                "invalid_dates",
                "invalid_reasons",
                "coverage",
            }
        ]],
        output / "native_summary.csv",
    )
    screening._atomic_write_csv(
        summary.loc[:, [
            column
            for column in summary.columns
            if column.startswith("common_")
            or column in {"country", "branch", "specification_id", "valid_dates", "invalid_dates"}
        ]],
        output / "common_summary.csv",
    )
    screening._atomic_write_csv(summary, output / "summary.csv")
    return summary, timestamps


def _freeze_publishable(
    manifest: pd.DataFrame,
    jobs: pd.DataFrame,
    adequacy: pd.DataFrame,
    freeze: pd.DataFrame,
) -> bool:
    manifest_keys = {_job_key(row) for row in manifest.to_dict("records")}
    present_keys = {_job_key(row) for row in jobs.to_dict("records")}
    candidate_keys = {
        (str(row["country"]), str(row["branch"]), str(row["specification_id"]))
        for row in manifest.drop_duplicates(
            ["country", "branch", "specification_id"]
        ).to_dict("records")
    }
    adequacy_keys = {
        (str(row["country"]), str(row["branch"]), str(row["specification_id"]))
        for row in adequacy.to_dict("records")
        if {"country", "branch", "specification_id"}.issubset(adequacy.columns)
    }
    if present_keys != manifest_keys or adequacy_keys != candidate_keys:
        return False
    if len(adequacy) != len(candidate_keys) or adequacy.empty:
        return False
    adequate_countries = set(
        adequacy.loc[adequacy["adequate"].map(_bool_value), "country"].astype(str)
    )
    if adequate_countries != set(COUNTRY_CONFIG):
        return False
    if not {"country", "selection_status", "adequate"}.issubset(freeze.columns):
        return False
    countries = freeze["country"].astype(str)
    return bool(
        len(freeze) == len(COUNTRY_CONFIG)
        and set(countries) == set(COUNTRY_CONFIG)
        and countries.value_counts().eq(1).all()
        and freeze["selection_status"].astype(str).eq("frozen").all()
        and freeze["adequate"].map(_bool_value).all()
    )


def _invalidate_freeze_artifacts(output: Path) -> None:
    freeze_output = output.parent / "freeze_2024"
    for filename in (
        "adequacy_review_2024.csv",
        "frozen_specifications_2024.csv",
    ):
        (freeze_output / filename).unlink(missing_ok=True)


def run_full_validation(
    shortlist: pd.DataFrame,
    output_root: str | Path,
    workers: int,
    processed_directory: str | Path = PROCESSED,
    reuse_authoritative: bool = True,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    _invalidate_freeze_artifacts(output)
    frames = _load_validation_frames(processed_directory)
    manifest = build_full_validation_manifest(shortlist, frames)
    screening._atomic_write_csv(
        _mapping_issue_frame(frames),
        output / "weekly_differenced" / "mapping_issues.csv",
    )
    reusable = _load_reusable_candidates() if reuse_authoritative else {}
    branch_jobs: dict[str, pd.DataFrame] = {}
    branch_forecasts: dict[str, pd.DataFrame] = {}
    branch_diagnostics: dict[str, pd.DataFrame] = {}
    expected_invalid = _expected_invalid_jobs(manifest)

    for branch in ("ordinary", "weekly_differenced"):
        branch_manifest = manifest.loc[manifest["branch"].eq(branch)].copy()
        branch_output = output / branch
        branch_output.mkdir(parents=True, exist_ok=True)
        jobs = _load_checkpoint(branch_output / "jobs.csv", JOB_COLUMNS)
        forecasts = _load_checkpoint(branch_output / "forecasts.csv", FORECAST_COLUMNS)
        diagnostics = _load_checkpoint(
            branch_output / "diagnostics.csv", DIAGNOSTICS_COLUMNS
        )
        jobs, forecasts, diagnostics = _reconcile_checkpoint_frames(
            jobs, forecasts, diagnostics, branch_manifest
        )
        terminal = _terminal_job_keys(
            jobs, forecasts, expected_invalid, source_frames=frames
        )
        pending: list[dict[str, object]] = []
        for job in branch_manifest.to_dict("records"):
            key = _job_key(job)
            if key in terminal:
                continue
            if branch == "ordinary" and reuse_authoritative:
                original_id = screening._original_specification_id(job)
                reused = reusable.get((key[0], key[1], original_id))
                if reused is not None:
                    try:
                        reused_result = screening._adapt_reused_result(job, reused)
                    except ValueError:
                        reused_result = None
                    if reused_result is not None:
                        jobs, forecasts, diagnostics = _persist_result(
                            jobs,
                            forecasts,
                            diagnostics,
                            reused_result,
                            branch_manifest,
                            branch_output,
                        )
                        terminal.add(key)
                        continue
            pending.append(job)

        def persist(result: dict[str, object]) -> None:
            nonlocal jobs, forecasts, diagnostics
            jobs, forecasts, diagnostics = _persist_result(
                jobs,
                forecasts,
                diagnostics,
                result,
                branch_manifest,
                branch_output,
            )

        if workers == 1:
            for job in pending:
                persist(_execute_validation_job(job, frames[str(job["country"])]))
        elif pending:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_validation_worker,
                initargs=(str(processed_directory),),
            ) as executor:
                for result in executor.map(_worker_execute_validation, pending):
                    persist(result)
        branch_jobs[branch] = screening._stable_checkpoint_frame(jobs, JOB_KEY)
        branch_forecasts[branch] = screening._stable_checkpoint_frame(
            forecasts, FORECAST_KEY
        )
        branch_diagnostics[branch] = screening._stable_checkpoint_frame(
            diagnostics, DIAGNOSTIC_KEY
        )
        screening._atomic_write_csv(branch_jobs[branch], branch_output / "jobs.csv")
        screening._atomic_write_csv(
            branch_forecasts[branch], branch_output / "forecasts.csv"
        )
        screening._atomic_write_csv(
            branch_diagnostics[branch], branch_output / "diagnostics.csv"
        )
        screening._atomic_write_csv(
            screening._summary_frame(
                branch_jobs[branch], branch_manifest, branch_forecasts[branch]
            ),
            branch_output / "summary.csv",
        )

    jobs = pd.concat(branch_jobs.values(), ignore_index=True, sort=False)
    forecasts = pd.concat(branch_forecasts.values(), ignore_index=True, sort=False)
    diagnostics = pd.concat(branch_diagnostics.values(), ignore_index=True, sort=False)
    jobs, forecasts, diagnostics = _reconcile_checkpoint_frames(
        jobs, forecasts, diagnostics, manifest
    )
    summary, timestamps = _write_common_outputs(
        output, jobs, forecasts, manifest, source_frames=frames
    )
    adequacy = build_adequacy_review(
        jobs,
        forecasts,
        summary,
        diagnostics,
        manifest=manifest,
        source_frames=frames,
    )
    freeze = freeze_specifications(summary, adequacy)
    freeze_output = output.parent / "freeze_2024"
    if _freeze_publishable(manifest, jobs, adequacy, freeze):
        screening._atomic_write_csv(adequacy, freeze_output / "adequacy_review_2024.csv")
        screening._atomic_write_csv(
            freeze,
            freeze_output / "frozen_specifications_2024.csv",
        )
    else:
        _invalidate_freeze_artifacts(output)
    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(jobs["status"].astype(str).eq("completed").sum())
        if not jobs.empty
        else 0,
        "failed_jobs": int(jobs["status"].astype(str).eq("failed").sum())
        if not jobs.empty
        else 0,
        "terminal_weekly_invalid_jobs": int(
            jobs["status"].astype(str).eq("weekly_lag_invalid").sum()
        )
        if not jobs.empty
        else 0,
        "forecast_rows": len(forecasts),
        "diagnostic_rows": len(diagnostics),
        "common_timestamp_rows": len(timestamps),
        "summary": summary.to_dict("records"),
        "adequacy": adequacy.to_dict("records"),
        "freeze": freeze.to_dict("records"),
        "output_directory": str(output),
    }
