from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import scipy
import statsmodels


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import (  # noqa: E402
    _as_prepared,
    COUNTRY_CONFIG,
    PROCESSED,
    build_information_context,
    calculate_metrics,
    country_forecast_origin,
    evaluate_target_day,
    extract_target_day,
    information_set,
)
from arima_screening import build_candidate_orders  # noqa: E402
from arima_models import forecast_values, trend_for_d  # noqa: E402
from enhanced_arima_models import (  # noqa: E402
    EnhancedSpecification,
    build_weekly_forecast_path,
    derive_weekly_invalid_target_dates,
    enhanced_specifications,
    fit_enhanced,
    transform_weekly_series,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARIMA_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "arima"
ORIGINAL_TABLE_DIRECTORY = RESULTS_DIRECTORY / "tables"
ORIGINAL_VALIDATION_DIRECTORY = RESULTS_DIRECTORY / "validation_2024" / "full_validation"
SCREENING_YEAR = 2024

SCREENING_DATES = (
    date(2024, 1, 1),
    date(2024, 1, 8),
    date(2024, 1, 27),
    date(2024, 2, 15),
    date(2024, 3, 31),
    date(2024, 4, 1),
    date(2024, 5, 9),
    date(2024, 6, 17),
    date(2024, 7, 25),
    date(2024, 8, 18),
    date(2024, 9, 12),
    date(2024, 10, 3),
    date(2024, 10, 26),
    date(2024, 10, 27),
    date(2024, 11, 11),
    date(2024, 12, 25),
)
SCREENING_CATEGORIES = (
    "New Year holiday / cold season",
    "Monday / cold season",
    "Saturday / cold season",
    "ordinary weekday",
    "spring-DST Sunday",
    "Easter Monday holiday",
    "Ascension holiday",
    "Monday / warm season",
    "warm-season weekday",
    "warm-season Sunday",
    "ordinary weekday",
    "Germany-specific public holiday",
    "Austria-specific public holiday / Saturday",
    "autumn-DST Sunday",
    "Monday",
    "Christmas holiday / cold season",
)
CALENDAR_COLUMNS = ["target_date", "category"]
SPECIFICATION_COLUMNS = [
    "branch",
    "specification_id",
    "p",
    "d",
    "q",
    "trend",
    "transform",
]
MANIFEST_COLUMNS = [
    "country",
    "target_date",
    "category",
    "forecast_origin_local",
    "forecast_origin_utc",
    "information_cutoff_utc",
    "expected_observations",
    "branch",
    "specification_id",
    "p",
    "d",
    "q",
    "trend",
    "transform",
    "weekly_invalid",
    "weekly_invalid_reason",
]
AUDIT_COLUMNS = [
    "audit_type",
    "country",
    "branch",
    "specification_id",
    "order",
    "overlap_status",
    "evidence_completeness",
    "source_paths",
    "reuse_eligible",
    "original_candidate_grid",
    "selected_order",
    "nearby_orders_tested",
    "training_evidence",
    "validation_evidence",
    "notes",
]
RUN_MANIFEST_COLUMNS = [
    "python_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "statsmodels_version",
    "platform",
    "logical_cpu_count",
    "physical_ram_bytes",
    "available_ram_bytes",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "selected_worker",
    "run_timestamp_utc",
    "enhanced_source_files",
]
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SCREENING_JOB_KEY = (
    "country",
    "target_date",
    "branch",
    "specification_id",
)
SCREENING_FORECAST_KEY = SCREENING_JOB_KEY + ("timestamp_utc",)
SCREENING_DIAGNOSTIC_KEY = SCREENING_JOB_KEY + ("diagnostic", "lag")
JOB_COLUMNS = [
    "country",
    "target_date",
    "category",
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
    "hard_warning_messages",
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
    "verified_reuse",
    "reuse_status",
    "artifact_source",
    "source_artifact_path",
    "source_job_path",
    "source_forecasts_path",
    "reuse_verified",
]
FORECAST_COLUMNS = [
    "country",
    "model_family",
    "target_date",
    "branch",
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
    "source_timestamp_utc",
    "source_actual_mwh",
    "source_kind",
    "is_target_day",
    "evaluated",
    "status",
    "mapping_issue_count",
]
DIAGNOSTICS_COLUMNS = [
    "country",
    "target_date",
    "branch",
    "specification_id",
    "diagnostic",
    "lag",
    "value",
    "threshold",
    "flagged",
]
MAPPING_ISSUE_COLUMNS = [
    "country",
    "branch",
    "target_date",
    "target_timestamp_utc",
    "affected_timestamp_utc",
    "source_key",
    "source_local_key",
    "source_timestamp_utc",
    "reason",
    "source_row_count",
    "coverage",
]
_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


def _relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _order_text(order: tuple[int, int, int]) -> str:
    return str(tuple(int(value) for value in order))


def build_screening_calendar() -> pd.DataFrame:
    """Return the shared 2024 target-date calendar in its persisted order."""
    return pd.DataFrame(
        {
            "target_date": [target.isoformat() for target in SCREENING_DATES],
            "category": list(SCREENING_CATEGORIES),
        },
        columns=CALENDAR_COLUMNS,
    )


def _candidate_definition_frame(
    specifications: Iterable[EnhancedSpecification],
) -> pd.DataFrame:
    rows = [
        {
            "branch": specification.branch,
            "specification_id": specification.specification_id,
            "p": specification.order[0],
            "d": specification.order[1],
            "q": specification.order[2],
            "trend": specification.trend,
            "transform": specification.transform,
        }
        for specification in specifications
    ]
    return pd.DataFrame(rows, columns=SPECIFICATION_COLUMNS)


def _read_original_sources() -> dict[str, object]:
    candidate_path = ORIGINAL_TABLE_DIRECTORY / "arima_candidate_screening_training_2024.csv"
    shortlist_path = ORIGINAL_TABLE_DIRECTORY / "arima_shortlists_2024.csv"
    selected_path = ORIGINAL_TABLE_DIRECTORY / "arima_selected_2024.csv"
    source_path = ARIMA_DIRECTORY / "arima_screening.py"
    if not source_path.read_text(encoding="utf-8"):
        raise ValueError(f"existing ARIMA source is empty: {source_path}")

    validation_paths = sorted(ORIGINAL_VALIDATION_DIRECTORY.glob("*.csv"))
    sources = {
        "source": source_path,
        "candidate": candidate_path,
        "shortlist": shortlist_path,
        "selected": selected_path,
        "validation": validation_paths,
    }
    missing = [
        path
        for value in sources.values()
        for path in (value if isinstance(value, list) else [value])
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"required original ARIMA source is missing: {missing}")
    return {
        **sources,
        "candidate_frame": pd.read_csv(candidate_path),
        "shortlist_frame": pd.read_csv(shortlist_path),
        "selected_frame": pd.read_csv(selected_path),
        "validation_frames": {
            path.name: pd.read_csv(path) for path in validation_paths
        },
    }


def _parse_arima_order(specification_id: object) -> tuple[int, int, int]:
    match = re.fullmatch(r"arima_p(\d+)_d(\d+)_q(\d+)", str(specification_id))
    if match is None:
        raise ValueError(f"cannot parse ARIMA order from specification id: {specification_id}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _validation_order_set(validation_frames: dict[str, pd.DataFrame]) -> set[tuple[int, int, int]]:
    orders: set[tuple[int, int, int]] = set()
    jobs = validation_frames.get("arima_validation_2024_jobs.csv")
    if jobs is None or jobs.empty:
        return orders
    required = {"p", "d", "q"}
    if required.issubset(jobs.columns):
        orders.update(
            (int(row.p), int(row.d), int(row.q))
            for row in jobs.itertuples(index=False)
        )
    return orders


def _source_path_text(paths: Iterable[Path]) -> str:
    return json.dumps([_relative_path(path) for path in paths])


def _has_complete_evidence(rows: pd.DataFrame) -> bool:
    required = {
        "fit_status",
        "converged",
        "parameters_finite",
        "standard_errors_finite",
        "stationarity_ok",
        "invertibility_ok",
        "aic",
        "aicc",
        "bic",
    }
    if rows.empty or not required.issubset(rows.columns):
        return False
    for _, row in rows.iterrows():
        if str(row["fit_status"]) != "success":
            return False
        if not all(
            str(row[column]).strip().lower() == "true"
            for column in (
                "converged",
                "parameters_finite",
                "standard_errors_finite",
                "stationarity_ok",
                "invertibility_ok",
            )
        ):
            return False
        if not all(np.isfinite(float(row[column])) for column in ("aic", "aicc", "bic")):
            return False
    return True


def audit_existing_arima() -> pd.DataFrame:
    """Audit the existing 2024 ARIMA evidence without writing to its results."""
    sources = _read_original_sources()
    candidate = sources["candidate_frame"]
    shortlist = sources["shortlist_frame"]
    selected = sources["selected_frame"]
    validation_frames = sources["validation_frames"]
    original_grid = tuple(build_candidate_orders(1))
    original_grid_text = json.dumps([_order_text(order) for order in original_grid])

    original_orders = {
        (int(row.p), int(row.d), int(row.q))
        for row in candidate.itertuples(index=False)
        if {"p", "d", "q"}.issubset(candidate.columns)
    }
    selected_orders = {
        _parse_arima_order(specification_id)
        for specification_id in selected["selected_specification_id"].astype(str)
    }
    if not selected_orders:
        raise ValueError("existing ARIMA selection contains no selected order")
    selected_order = next(iter(selected_orders))
    validation_orders = _validation_order_set(validation_frames)
    nearby_orders = sorted(
        order
        for order in original_orders | validation_orders
        if order[1] == selected_order[1]
        and abs(order[0] - selected_order[0]) <= 1
        and abs(order[2] - selected_order[2]) <= 1
    )

    source_paths = [
        sources["source"],
        sources["candidate"],
        sources["shortlist"],
        sources["selected"],
        *sources["validation"],
    ]
    source_text = _source_path_text(source_paths)
    nearby_text = json.dumps([_order_text(order) for order in nearby_orders])
    selected_text = _order_text(selected_order)
    rows: list[dict[str, object]] = []

    rows.extend(
        {
            "audit_type": "original_candidate_grid",
            "country": "",
            "branch": "ordinary",
            "specification_id": "",
            "order": _order_text(order),
            "overlap_status": "original",
            "evidence_completeness": "complete",
            "source_paths": source_text,
            "reuse_eligible": True,
            "original_candidate_grid": original_grid_text,
            "selected_order": selected_text,
            "nearby_orders_tested": nearby_text,
            "training_evidence": "candidate screening source read",
            "validation_evidence": "validation source read",
            "notes": "order in the frozen original candidate grid",
        }
        for order in original_grid
    )

    selected_required = {"country", "selected_specification_id"}
    if not selected_required.issubset(selected.columns):
        raise ValueError(
            f"selected ARIMA source is missing required columns: "
            f"{sorted(selected_required.difference(selected.columns))}"
        )
    for country in COUNTRY_CONFIG:
        country_selected = selected.loc[selected["country"].eq(country)]
        selected_country_order = (
            _parse_arima_order(country_selected.iloc[0]["selected_specification_id"])
            if not country_selected.empty
            else selected_order
        )
        rows.append(
            {
                "audit_type": "selected_model",
                "country": country,
                "branch": "ordinary",
                "specification_id": str(
                    country_selected.iloc[0]["selected_specification_id"]
                    if not country_selected.empty
                    else ""
                ),
                "order": _order_text(selected_country_order),
                "overlap_status": "selected",
                "evidence_completeness": "complete" if not country_selected.empty else "missing",
                "source_paths": source_text,
                "reuse_eligible": not country_selected.empty,
                "original_candidate_grid": original_grid_text,
                "selected_order": _order_text(selected_country_order),
                "nearby_orders_tested": nearby_text,
                "training_evidence": "candidate screening source read",
                "validation_evidence": "validation source read",
                "notes": "selected order from original 2024 ARIMA results",
            }
        )

    ordinary_specifications = tuple(
        specification
        for specification in enhanced_specifications()
        if specification.branch == "ordinary"
    )
    validation_jobs = validation_frames.get("arima_validation_2024_jobs.csv")
    for country in COUNTRY_CONFIG:
        country_candidate = candidate.loc[candidate["country"].eq(country)]
        country_validation = (
            validation_jobs.loc[validation_jobs["country"].eq(country)]
            if validation_jobs is not None and not validation_jobs.empty
            else pd.DataFrame()
        )
        for specification in ordinary_specifications:
            order = specification.order
            matching = country_candidate.loc[
                country_candidate["p"].eq(order[0])
                & country_candidate["d"].eq(order[1])
                & country_candidate["q"].eq(order[2])
            ]
            overlap = order in original_orders and not matching.empty
            complete = overlap and _has_complete_evidence(matching)
            validation_match = (
                not country_validation.empty
                and country_validation["p"].eq(order[0])
                & country_validation["d"].eq(order[1])
                & country_validation["q"].eq(order[2])
            )
            rows.append(
                {
                    "audit_type": "ordinary_enhanced_candidate",
                    "country": country,
                    "branch": specification.branch,
                    "specification_id": specification.specification_id,
                    "order": _order_text(order),
                    "overlap_status": "overlap" if overlap else "unexplored",
                    "evidence_completeness": "complete" if complete else "missing",
                    "source_paths": source_text,
                    "reuse_eligible": bool(complete),
                    "original_candidate_grid": original_grid_text,
                    "selected_order": selected_text,
                    "nearby_orders_tested": nearby_text,
                    "training_evidence": (
                        "country candidate row complete"
                        if complete
                        else "no complete country candidate row"
                    ),
                    "validation_evidence": (
                        "validation job evidence present"
                        if bool(validation_match.any())
                        else "no validation job evidence"
                    ),
                    "notes": (
                        "exact ordinary specification can reuse original training evidence"
                        if complete
                        else "ordinary specification was not covered by the original grid"
                    ),
                }
            )

    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def _normalise_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"target_date", "category"}
    missing = sorted(required.difference(calendar.columns))
    if missing:
        raise ValueError(f"screening calendar is missing required columns: {missing}")
    normalised = calendar.copy()
    normalised["target_date"] = pd.to_datetime(
        normalised["target_date"], errors="raise"
    ).dt.date.map(date.isoformat)
    if not normalised["target_date"].map(lambda value: value[:4]).eq(str(SCREENING_YEAR)).all():
        raise ValueError("screening calendar contains a date outside 2024")
    if normalised["target_date"].duplicated().any() and "country" not in normalised.columns:
        raise ValueError("screening calendar contains duplicate target dates")
    return normalised


def _invalid_date_lookup(invalid_dates: pd.DataFrame) -> dict[tuple[str | None, str], str]:
    if invalid_dates.empty:
        return {}
    if "target_date" not in invalid_dates.columns:
        raise ValueError("invalid-date frame is missing target_date")
    has_country = "country" in invalid_dates.columns
    lookup: dict[tuple[str | None, str], list[str]] = {}
    for row in invalid_dates.itertuples(index=False):
        target_date = pd.Timestamp(getattr(row, "target_date")).date().isoformat()
        if target_date[:4] != str(SCREENING_YEAR):
            continue
        country_value = str(getattr(row, "country")) if has_country else None
        if country_value == "nan":
            country_value = None
        reason_value = getattr(row, "reason", "weekly source mapping issue")
        reason = str(reason_value)
        if reason in {"", "nan", "None"}:
            reason = "weekly source mapping issue"
        lookup.setdefault((country_value, target_date), []).append(reason)
    return {
        key: "; ".join(dict.fromkeys(reasons))
        for key, reasons in lookup.items()
    }


def build_screening_manifest(
    calendar: pd.DataFrame,
    specifications: tuple[EnhancedSpecification, ...],
    frames: dict[str, pd.DataFrame],
    invalid_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 2024 country/date/branch/specification job manifest."""
    calendar = _normalise_calendar(calendar)
    if not isinstance(specifications, tuple):
        raise TypeError("specifications must be a tuple of EnhancedSpecification values")
    if len({specification.specification_id for specification in specifications}) != len(specifications):
        raise ValueError("enhanced specifications contain duplicate identifiers")
    missing_frames = sorted(set(COUNTRY_CONFIG).difference(frames))
    if missing_frames:
        raise ValueError(f"country frames are missing: {missing_frames}")

    invalid_lookup = _invalid_date_lookup(invalid_dates)
    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        country_calendar = calendar
        if "country" in calendar.columns:
            country_calendar = calendar.loc[calendar["country"].eq(country)]
        frame = frames[country]
        expected_observations: dict[str, int] = {}
        for row in country_calendar.itertuples(index=False):
            target_date = str(row.target_date)
            expected_observations[target_date] = len(extract_target_day(frame, target_date))
            origin_utc = country_forecast_origin(target_date, country)
            origin_local = origin_utc.tz_convert(COUNTRY_CONFIG[country]["timezone"])
            country_reason = invalid_lookup.get((country, target_date))
            global_reason = invalid_lookup.get((None, target_date))
            invalid_reason = country_reason or global_reason
            for specification in specifications:
                rows.append(
                    {
                        "country": country,
                        "target_date": target_date,
                        "category": str(row.category),
                        "forecast_origin_local": origin_local.isoformat(),
                        "forecast_origin_utc": origin_utc.isoformat(),
                        "information_cutoff_utc": origin_utc.isoformat(),
                        "expected_observations": expected_observations[target_date],
                        "branch": specification.branch,
                        "specification_id": specification.specification_id,
                        "p": specification.order[0],
                        "d": specification.order[1],
                        "q": specification.order[2],
                        "trend": specification.trend,
                        "transform": specification.transform,
                        "weekly_invalid": bool(
                            specification.branch == "weekly_differenced"
                            and invalid_reason is not None
                        ),
                        "weekly_invalid_reason": (
                            invalid_reason
                            if specification.branch == "weekly_differenced"
                            else ""
                        ),
                    }
                )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    key = ["country", "target_date", "branch", "specification_id"]
    if manifest.duplicated(key).any():
        raise ValueError("screening manifest contains duplicate job keys")
    return manifest


def _sysconf_bytes(page_name: str, count_name: str) -> int | float:
    try:
        page_size = int(os.sysconf(page_name))
        page_count = int(os.sysconf(count_name))
    except (AttributeError, OSError, ValueError):
        return float("nan")
    return page_size * page_count


def _run_manifest_frame() -> pd.DataFrame:
    row: dict[str, object] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "statsmodels_version": statsmodels.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count() or float("nan"),
        "physical_ram_bytes": _sysconf_bytes("SC_PAGESIZE", "SC_PHYS_PAGES"),
        "available_ram_bytes": _sysconf_bytes("SC_PAGESIZE", "SC_AVPHYS_PAGES"),
        "selected_worker": "",
        "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "enhanced_source_files": json.dumps(
            ["enhanced_arima_models.py", "enhanced_arima_screening_2024.py"]
        ),
    }
    row.update({name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT_NAMES})
    return pd.DataFrame([row], columns=RUN_MANIFEST_COLUMNS)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _year_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "local_date" not in frame.columns:
        dates = pd.to_datetime(frame["timestamp_local"], errors="raise").dt.date
    else:
        dates = pd.to_datetime(frame["local_date"], errors="raise").dt.date
    bridge_start = date(SCREENING_YEAR, 1, 1) - timedelta(days=8)
    year_end = date(SCREENING_YEAR, 12, 31)
    selected = [bridge_start <= value <= year_end for value in dates]
    return frame.loc[selected].copy()


def _load_setup_country_data(
    country: str,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    path = Path(processed_directory) / COUNTRY_CONFIG[country]["filename"]
    first_row = pd.read_csv(path, usecols=["timestamp_utc"], nrows=1)
    if first_row.empty:
        raise ValueError(f"country data is empty: {path}")
    first_timestamp = pd.to_datetime(
        first_row["timestamp_utc"], utc=True, errors="raise"
    ).iloc[0]
    timezone = COUNTRY_CONFIG[country]["timezone"]
    bridge_start = pd.Timestamp(
        f"{SCREENING_YEAR}-01-01 00:00:00", tz=timezone
    ) - pd.Timedelta(days=8)
    year_end = pd.Timestamp(
        f"{SCREENING_YEAR}-12-31 23:00:00", tz=timezone
    )
    start_utc = bridge_start.tz_convert("UTC")
    end_utc = year_end.tz_convert("UTC")
    if first_timestamp > end_utc:
        raise ValueError(f"country data starts after the 2024 setup period: {path}")
    start_offset = max(
        0.0, (start_utc - first_timestamp).total_seconds() / 3600
    )
    end_offset = (end_utc - first_timestamp).total_seconds() / 3600
    if start_offset < 0:
        start_offset = 0
    if not start_offset.is_integer() or not end_offset.is_integer():
        raise ValueError(f"country data is not aligned to hourly UTC observations: {path}")
    rows_to_skip = int(start_offset)
    rows_to_read = int(end_offset - start_offset) + 1
    if rows_to_read <= 0:
        raise ValueError(f"country data has no bounded 2024 setup period: {path}")
    bounded = pd.read_csv(
        path,
        skiprows=range(1, rows_to_skip + 1),
        nrows=rows_to_read,
    )
    return _year_frame(_as_prepared(bounded))


def write_initial_artifacts(output_root: str | Path) -> dict[str, Path]:
    """Write only the four setup artifacts, each with an atomic CSV replacement."""
    output = Path(output_root)
    specifications = enhanced_specifications()
    calendar = build_screening_calendar()
    frames = {
        country: _load_setup_country_data(country, PROCESSED)
        for country in COUNTRY_CONFIG
    }
    invalid_frames = []
    for country, frame in frames.items():
        invalid = derive_weekly_invalid_target_dates(frame, SCREENING_YEAR)
        invalid.insert(0, "country", country)
        invalid_frames.append(invalid)
    invalid_dates = pd.concat(invalid_frames, ignore_index=True)
    screening_manifest = build_screening_manifest(
        calendar,
        specifications,
        frames,
        invalid_dates,
    )
    del screening_manifest

    audit_path = output / "audit" / "original_arima_audit.csv"
    run_manifest_path = output / "audit" / "enhanced_arima_run_manifest.csv"
    candidate_path = (
        output / "specifications" / "enhanced_arima_candidate_definitions.csv"
    )
    calendar_path = output / "specifications" / "screening_calendar_2024.csv"
    _atomic_write_csv(audit_existing_arima(), audit_path)
    _atomic_write_csv(_run_manifest_frame(), run_manifest_path)
    _atomic_write_csv(_candidate_definition_frame(specifications), candidate_path)
    _atomic_write_csv(calendar, calendar_path)
    return {
        "audit": audit_path,
        "run_manifest": run_manifest_path,
        "candidate_definitions": candidate_path,
        "calendar": calendar_path,
    }


def run_screening(
    output_root: str | Path,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")

    output = Path(output_root) / "screening_2024"
    frames = {
        country: _load_setup_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    calendar = build_screening_calendar()
    specifications = enhanced_specifications()
    invalid_frames: list[pd.DataFrame] = []
    for country, frame in frames.items():
        invalid = derive_weekly_invalid_target_dates(frame, SCREENING_YEAR).copy()
        invalid.insert(0, "country", country)
        invalid_frames.append(invalid)
    invalid_dates = pd.concat(invalid_frames, ignore_index=True)
    manifest = build_screening_manifest(
        calendar,
        specifications,
        frames,
        invalid_dates,
    )
    _write_mapping_issues(output / "weekly_differenced", invalid_dates, frames)
    reusable = _read_reusable_original_artifacts()

    summaries: list[dict[str, object]] = []
    for branch in ("ordinary", "weekly_differenced"):
        branch_manifest = manifest.loc[manifest["branch"].eq(branch)].copy()
        branch_output = output / branch
        jobs_path = branch_output / "jobs.csv"
        forecasts_path = branch_output / "forecasts.csv"
        diagnostics_path = branch_output / "diagnostics.csv"
        summary_path = branch_output / "summary.csv"
        jobs = _load_csv(jobs_path, JOB_COLUMNS)
        forecasts = _load_csv(forecasts_path, FORECAST_COLUMNS)
        diagnostics = _load_csv(diagnostics_path, DIAGNOSTICS_COLUMNS)
        terminal_keys = _terminal_job_keys(
            jobs, forecasts, branch_manifest, source_frames=frames
        )

        pending: list[dict[str, object]] = []
        for row in branch_manifest.itertuples(index=False, name=None):
            job = dict(zip(branch_manifest.columns, row))
            key = _job_key(job)
            if key in terminal_keys:
                continue
            if branch == "ordinary":
                reusable_result = reusable.get(
                    (key[0], key[1], _original_specification_id(job))
                )
                if reusable_result is not None:
                    result = _adapt_reused_result(job, reusable_result)
                    jobs, forecasts, diagnostics = _persist_screening_result(
                        jobs,
                        forecasts,
                        diagnostics,
                        result,
                        jobs_path,
                        forecasts_path,
                        diagnostics_path,
                        summary_path,
                        branch_manifest,
                    )
                    terminal_keys.add(key)
                    continue
            pending.append(job)

        def persist(result: dict[str, object]) -> None:
            nonlocal jobs, forecasts, diagnostics
            jobs, forecasts, diagnostics = _persist_screening_result(
                jobs,
                forecasts,
                diagnostics,
                result,
                jobs_path,
                forecasts_path,
                diagnostics_path,
                summary_path,
                branch_manifest,
            )

        if workers == 1:
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

        summary = _summary_frame(jobs, branch_manifest, forecasts)
        jobs = _stable_checkpoint_frame(jobs, SCREENING_JOB_KEY)
        forecasts = _stable_checkpoint_frame(forecasts, SCREENING_FORECAST_KEY)
        diagnostics = _stable_checkpoint_frame(
            diagnostics, SCREENING_DIAGNOSTIC_KEY
        )
        _atomic_write_csv(jobs, jobs_path)
        _atomic_write_csv(forecasts, forecasts_path)
        _atomic_write_csv(diagnostics, diagnostics_path)
        _atomic_write_csv(summary, summary_path)
        summaries.extend(summary.to_dict("records"))

    all_jobs = pd.concat(
        [_load_csv(output / branch / "jobs.csv", JOB_COLUMNS) for branch in ("ordinary", "weekly_differenced")],
        ignore_index=True,
    )
    expected_keys = {_job_key(row) for row in manifest.to_dict("records")}
    represented_keys = {_job_key(row) for row in all_jobs.to_dict("records")}
    shortlist_path = output / "shortlist_2024.csv"
    shortlist_input = pd.DataFrame(summaries)
    shortlist = (
        build_shortlist(shortlist_input)
        if expected_keys == represented_keys
        else pd.DataFrame()
    )
    maximum_counts = {
        (country, "ordinary"): 3 for country in COUNTRY_CONFIG
    }
    maximum_counts.update({
        (country, "weekly_differenced"): 2 for country in COUNTRY_CONFIG
    })
    actual_counts = (
        shortlist.groupby(["country", "branch"]).size().to_dict()
        if not shortlist.empty
        else {}
    )
    shortlist_ready = expected_keys == represented_keys and all(
        1 <= actual_counts.get(key, 0) <= maximum
        for key, maximum in maximum_counts.items()
    )
    if shortlist_ready:
        _atomic_write_csv(shortlist, shortlist_path)
    else:
        shortlist_path.unlink(missing_ok=True)

    return {
        "total_jobs": len(manifest),
        "completed_jobs": int(all_jobs["status"].eq("completed").sum())
        if not all_jobs.empty
        else 0,
        "failed_jobs": int(all_jobs["status"].eq("failed").sum())
        if not all_jobs.empty
        else 0,
        "terminal_weekly_invalid_jobs": int(
            all_jobs["status"].eq("weekly_lag_invalid").sum()
        )
        if not all_jobs.empty
        else 0,
        "forecast_rows": int(
            sum(
                len(_load_csv(output / branch / "forecasts.csv", FORECAST_COLUMNS))
                for branch in ("ordinary", "weekly_differenced")
            )
        ),
        "summary": summaries,
        "shortlist_rows": len(shortlist) if shortlist_ready else 0,
        "shortlist_written": shortlist_ready,
        "output_directory": str(output),
    }


def build_shortlist(screening_jobs: pd.DataFrame) -> pd.DataFrame:
    summary_input = "target_date" not in screening_jobs.columns
    required = {
        "country",
        "branch",
        "specification_id",
        "mae",
        "rmse",
        "mape",
    }
    if summary_input:
        required.update({"valid_dates", "invalid_dates", "expected_dates"})
    else:
        required.update({"target_date", "status"})
    missing = sorted(required.difference(screening_jobs.columns))
    if missing:
        raise ValueError(f"screening jobs are missing required columns: {missing}")
    if screening_jobs.empty:
        return pd.DataFrame(columns=_shortlist_columns())

    specifications = enhanced_specifications()
    specification_lookup = {
        specification.specification_id: (order, specification)
        for branch in ("ordinary", "weekly_differenced")
        for order, specification in enumerate(
            (item for item in specifications if item.branch == branch), 1
        )
    }
    rows: list[dict[str, object]] = []
    for (country, branch, specification_id), group in screening_jobs.groupby(
        ["country", "branch", "specification_id"], sort=False
    ):
        specification_entry = specification_lookup.get(str(specification_id))
        if specification_entry is None:
            continue
        specification_order, specification = specification_entry
        group = group.copy()
        if summary_input:
            source = group.iloc[0]
            expected_dates = int(source["expected_dates"])
            valid_dates = int(source["valid_dates"])
            invalid_dates = int(source["invalid_dates"])
            unresolved_dates = int(
                source.get(
                    "unresolved_dates",
                    max(0, expected_dates - valid_dates - invalid_dates),
                )
            )
            valid_rows = group
        else:
            group["target_date"] = pd.to_datetime(
                group["target_date"], errors="raise"
            ).dt.date.map(date.isoformat)
            target_dates = set(group["target_date"])
            weekly_invalid = (
                group["weekly_invalid"].map(_bool_value)
                if "weekly_invalid" in group.columns
                else pd.Series(False, index=group.index)
            )
            valid_rows = group.loc[
                group["status"].eq("completed")
                & (~weekly_invalid if branch == "weekly_differenced" else True)
                & group[["mae", "rmse", "mape"]]
                .apply(pd.to_numeric, errors="coerce")
                .notna()
                .all(axis=1)
            ]
            valid_dates_set = set(valid_rows["target_date"])
            invalid_dates_set = set(
                group.loc[
                    group["status"].eq("weekly_lag_invalid") & weekly_invalid,
                    "target_date",
                ]
            )
            valid_dates = len(valid_dates_set)
            invalid_dates = len(invalid_dates_set - valid_dates_set)
            expected_dates = len(target_dates)
            unresolved_dates = len(
                target_dates - valid_dates_set - invalid_dates_set
            )

        finite_metrics = not valid_rows.empty and all(
            np.isfinite(value)
            for value in _aggregate_job_metrics(valid_rows).values()
        )
        if branch == "ordinary":
            eligible = (
                expected_dates == len(SCREENING_DATES)
                and valid_dates == len(SCREENING_DATES)
                and invalid_dates == 0
                and unresolved_dates == 0
                and finite_metrics
            )
        else:
            eligible = (
                expected_dates == len(SCREENING_DATES)
                and valid_dates >= 15
                and valid_dates + invalid_dates == expected_dates
                and unresolved_dates == 0
                and finite_metrics
            )
        if not eligible:
            continue

        metric_values = _aggregate_job_metrics(valid_rows)
        evaluated_observations = int(
            pd.to_numeric(
                valid_rows.get("evaluated_observations", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
        expected_observations = int(
            pd.to_numeric(
                group.get("expected_observations", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
        rows.append(
            {
                "country": str(country),
                "branch": branch,
                "specification_id": str(specification_id),
                "specification_order": specification_order,
                "order": _order_text(specification.order),
                "p": specification.order[0],
                "d": specification.order[1],
                "q": specification.order[2],
                "trend": specification.trend,
                "transform": specification.transform,
                "mae": metric_values["mae"],
                "rmse": metric_values["rmse"],
                "mape": metric_values["mape"],
                "valid_dates": valid_dates,
                "invalid_dates": invalid_dates,
                "expected_dates": expected_dates,
                "evaluated_observations": evaluated_observations,
                "expected_observations": expected_observations,
                "coverage": float(valid_dates / expected_dates),
                "eligible": True,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_shortlist_columns())
    result = pd.DataFrame(rows, columns=_shortlist_columns())
    country_rank = {country: index for index, country in enumerate(COUNTRY_CONFIG)}
    branch_rank = {"ordinary": 0, "weekly_differenced": 1}
    result["_country_rank"] = result["country"].map(country_rank)
    result["_branch_rank"] = result["branch"].map(branch_rank)
    result = result.sort_values(
        [
            "_country_rank",
            "_branch_rank",
            "mae",
            "rmse",
            "mape",
            "specification_order",
        ],
        kind="stable",
        ignore_index=True,
    )
    result["shortlist_rank"] = result.groupby(
        ["country", "branch"], sort=False
    ).cumcount() + 1
    limits = result["branch"].map({"ordinary": 3, "weekly_differenced": 2})
    result = result.loc[result["shortlist_rank"].le(limits)].copy()
    return result.drop(columns=["_country_rank", "_branch_rank"])[
        _shortlist_columns() + ["shortlist_rank"]
    ]


def _shortlist_columns() -> list[str]:
    return [
        "country",
        "branch",
        "specification_id",
        "specification_order",
        "order",
        "p",
        "d",
        "q",
        "trend",
        "transform",
        "mae",
        "rmse",
        "mape",
        "valid_dates",
        "invalid_dates",
        "expected_dates",
        "evaluated_observations",
        "expected_observations",
        "coverage",
        "eligible",
    ]


def _aggregate_job_metrics(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {column: float("nan") for column in ("mae", "rmse", "mape")}
    values = {
        column: pd.to_numeric(rows[column], errors="coerce")
        for column in ("mae", "rmse", "mape")
    }
    if "evaluated_observations" in rows.columns:
        weights = pd.to_numeric(rows["evaluated_observations"], errors="coerce")
        usable = (
            weights.notna()
            & weights.gt(0)
            & values["mae"].notna()
            & values["rmse"].notna()
            & values["mape"].notna()
        )
        if usable.any():
            weights = weights.loc[usable]
            total = float(weights.sum())
            return {
                "mae": float((values["mae"].loc[usable] * weights).sum() / total),
                "rmse": float(
                    np.sqrt(
                        (values["rmse"].loc[usable].pow(2) * weights).sum()
                        / total
                    )
                ),
                "mape": float((values["mape"].loc[usable] * weights).sum() / total),
            }
    return {column: float(values[column].mean()) for column in values}


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _job_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row["country"]),
        pd.Timestamp(row["target_date"]).date().isoformat(),
        str(row["branch"]),
        str(row["specification_id"]),
    )


def _canonical_key_value(column: str, value: object) -> str:
    if column == "target_date":
        return pd.Timestamp(value).date().isoformat()
    if column == "timestamp_utc":
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()
    return str(value)


def upsert_frame(
    existing: pd.DataFrame,
    replacement: pd.DataFrame,
    key_columns: Iterable[str],
    *,
    sort: bool = True,
) -> pd.DataFrame:
    keys = list(key_columns)
    if existing.empty and replacement.empty:
        return existing.copy()
    combined = pd.concat([existing, replacement], ignore_index=True, sort=False)
    for column in keys:
        if column not in combined.columns:
            raise ValueError(f"upsert key column is missing: {column}")
        combined[column] = combined[column].map(
            lambda value, name=column: _canonical_key_value(name, value)
        )
    combined = combined.drop_duplicates(keys, keep="last")
    if sort:
        return combined.sort_values(keys, kind="stable", ignore_index=True)
    return combined.reset_index(drop=True)


def _load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    loaded = pd.read_csv(path)
    for column in columns:
        if column not in loaded.columns:
            loaded[column] = np.nan
    if "target_date" in loaded.columns:
        loaded["target_date"] = loaded["target_date"].map(
            lambda value: _canonical_key_value("target_date", value)
        )
    if "timestamp_utc" in loaded.columns:
        loaded["timestamp_utc"] = pd.to_datetime(
            loaded["timestamp_utc"], format="mixed", utc=True, errors="raise"
        ).map(pd.Timestamp.isoformat)
    if "lag" in loaded.columns:
        lag = pd.to_numeric(loaded["lag"], errors="raise")
        if not np.isfinite(lag).all() or not np.equal(lag, lag.astype(int)).all():
            raise ValueError(f"checkpoint contains a non-integral diagnostic lag: {path}")
        loaded["lag"] = lag.astype(int)
    if set(SCREENING_FORECAST_KEY).issubset(loaded.columns):
        return upsert_frame(
            pd.DataFrame(columns=columns),
            loaded,
            SCREENING_FORECAST_KEY,
        )
    if set(SCREENING_DIAGNOSTIC_KEY).issubset(loaded.columns):
        return upsert_frame(
            pd.DataFrame(columns=columns),
            loaded,
            SCREENING_DIAGNOSTIC_KEY,
        )
    if set(SCREENING_JOB_KEY).issubset(loaded.columns):
        return upsert_frame(
            pd.DataFrame(columns=columns),
            loaded,
            SCREENING_JOB_KEY,
        )
    return loaded


def _blank(value: object) -> bool:
    return value is None or str(value).strip() in {"", "nan", "None", "NaT"}


def _fit_metadata_complete(job: Mapping[str, object]) -> bool:
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
    if "error_message" not in job or not _blank(job.get("error_message")):
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
    try:
        expected_observations = float(job["expected_observations"])
        criteria = [float(job[column]) for column in ("log_likelihood", "aic", "aicc", "bic")]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(expected_observations)
        and expected_observations > 0
        and np.isfinite(criteria).all()
    )


def _weekly_source_metadata_complete(
    job: Mapping[str, object],
    matching: pd.DataFrame,
    expected_origin: pd.Timestamp,
    source_frame: pd.DataFrame | None = None,
) -> bool:
    if str(job.get("branch")) != "weekly_differenced":
        return True
    required = {
        "timestamp_utc",
        "source_timestamp_utc",
        "source_actual_mwh",
        "source_kind",
        "mapping_issue_count",
    }
    if not required.issubset(matching.columns):
        return False
    target_timestamps = pd.to_datetime(
        matching["timestamp_utc"], format="mixed", utc=True, errors="coerce"
    )
    source_timestamps = pd.to_datetime(
        matching["source_timestamp_utc"], format="mixed", utc=True, errors="coerce"
    )
    source_actuals = pd.to_numeric(matching["source_actual_mwh"], errors="coerce")
    mapping_counts = pd.to_numeric(matching["mapping_issue_count"], errors="coerce")
    if (
        target_timestamps.isna().any()
        or source_timestamps.isna().any()
        or not np.isfinite(source_actuals).all()
    ):
        return False
    if mapping_counts.isna().any() or not np.isfinite(mapping_counts).all():
        return False
    if not np.equal(mapping_counts, mapping_counts.astype(int)).all():
        return False
    timezone = COUNTRY_CONFIG[str(job["country"])]
    timezone = timezone["timezone"]
    target_local = target_timestamps.dt.tz_convert(timezone)
    source_local = source_timestamps.dt.tz_convert(timezone)
    target_occurrences = pd.DataFrame(
        {
            "timestamp_utc": target_timestamps,
            "local_date": target_local.dt.date,
            "hour": target_local.dt.hour,
        },
        index=matching.index,
    ).sort_values("timestamp_utc", kind="stable")
    target_occurrences["occurrence"] = target_occurrences.groupby(
        ["local_date", "hour"], sort=False
    ).cumcount()
    target_occurrences = target_occurrences.sort_index()
    expected_source_timestamps = []
    for timestamp, occurrence in zip(
        target_local, target_occurrences["occurrence"]
    ):
        source_date = timestamp.date() - timedelta(days=7)
        source_local_timestamp = pd.Timestamp(
            f"{source_date.isoformat()} {timestamp.hour:02d}:00:00"
        ).tz_localize(timezone, ambiguous=int(occurrence) == 0)
        expected_source_timestamps.append(source_local_timestamp.tz_convert("UTC"))
    expected_source_timestamps = pd.Series(
        expected_source_timestamps,
        index=matching.index,
    )
    if (
        not source_timestamps.is_unique
        or not source_timestamps.eq(expected_source_timestamps).all()
        or source_local.dt.date.ne(
            pd.Series(
                [timestamp.date() - timedelta(days=7) for timestamp in target_local],
                index=matching.index,
            )
        ).any()
        or source_local.dt.hour.ne(target_local.dt.hour).any()
    ):
        return False
    if source_frame is not None:
        try:
            prepared_source = _as_prepared(source_frame)
        except (KeyError, TypeError, ValueError, OverflowError):
            prepared_source = source_frame
        actual_column = (
            "actual_load_mwh"
            if "actual_load_mwh" in prepared_source.columns
            else "actual_grid_load_mwh"
        )
        if prepared_source.empty or not {
            "timestamp_utc",
            actual_column,
        }.issubset(prepared_source.columns):
            return False
        prepared_timestamps = pd.to_datetime(
            prepared_source["timestamp_utc"],
            format="mixed",
            utc=True,
            errors="coerce",
        )
        prepared_actuals = pd.to_numeric(
            prepared_source[actual_column], errors="coerce"
        )
        if (
            prepared_timestamps.isna().any()
            or prepared_actuals.isna().any()
            or not np.isfinite(prepared_actuals).all()
            or not prepared_timestamps.is_unique
        ):
            return False
        source_lookup = pd.Series(
            prepared_actuals.to_numpy(dtype=float), index=prepared_timestamps
        )
        expected_actuals = source_timestamps.map(source_lookup)
        if (
            expected_actuals.isna().any()
            or not np.isclose(
                source_actuals.to_numpy(dtype=float),
                expected_actuals.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            ).all()
        ):
            return False
    return bool(
        mapping_counts.eq(0).all()
        and matching["source_kind"].astype(str).str.strip().eq("observed").all()
        and source_timestamps.add(pd.Timedelta(hours=1)).le(expected_origin).all()
    )


def _manifest_job_matches(
    row: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    try:
        for column, expected_value in expected.items():
            if column in SCREENING_JOB_KEY:
                continue
            if column not in row:
                return False
            actual_value = row[column]
            if _blank(actual_value) and _blank(expected_value):
                continue
            if _blank(expected_value) or _blank(actual_value):
                return False
            if column in {"p", "d", "q", "expected_observations"}:
                actual_number = float(actual_value)
                expected_number = float(expected_value)
                if (
                    not np.isfinite(actual_number)
                    or not np.isfinite(expected_number)
                    or not actual_number.is_integer()
                    or not expected_number.is_integer()
                    or int(actual_number) != int(expected_number)
                ):
                    return False
            elif column in {
                "forecast_origin_local",
                "forecast_origin_utc",
                "information_cutoff_utc",
            }:
                actual_timestamp = pd.Timestamp(actual_value)
                expected_timestamp = pd.Timestamp(expected_value)
                if actual_timestamp.tzinfo is None:
                    actual_timestamp = actual_timestamp.tz_localize("UTC")
                else:
                    actual_timestamp = actual_timestamp.tz_convert("UTC")
                if expected_timestamp.tzinfo is None:
                    expected_timestamp = expected_timestamp.tz_localize("UTC")
                else:
                    expected_timestamp = expected_timestamp.tz_convert("UTC")
                if actual_timestamp != expected_timestamp:
                    return False
            elif column == "weekly_invalid":
                if _bool_value(actual_value) != _bool_value(expected_value):
                    return False
            elif str(actual_value).strip() != str(expected_value).strip():
                return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _terminal_job_keys(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    manifest: pd.DataFrame | None = None,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> set[tuple[str, str, str, str]]:
    if jobs.empty:
        return set()
    manifest_by_key = (
        {
            _job_key(row): row
            for row in manifest.to_dict("records")
        }
        if manifest is not None
        else {}
    )
    terminal: set[tuple[str, str, str, str]] = set()
    for row in jobs.to_dict("records"):
        key = _job_key(row)
        expected = manifest_by_key.get(key)
        if manifest is not None and expected is None:
            continue
        if expected is not None and not _manifest_job_matches(row, expected):
            continue
        if (
            str(row.get("status")) == "weekly_lag_invalid"
            and expected is not None
            and str(expected.get("branch")) == "weekly_differenced"
            and _bool_value(expected.get("weekly_invalid", False))
            and _bool_value(row.get("weekly_invalid", False))
            and str(expected.get("weekly_invalid_reason", "")).strip()
            == str(row.get("weekly_invalid_reason", "")).strip()
            and str(row.get("weekly_invalid_reason", "")).strip()
            == str(row.get("error_message", "")).strip()
        ):
            terminal.add(key)
        elif (
            str(row.get("status")) == "completed"
            and _fit_metadata_complete(row)
            and _forecast_is_complete(
                row,
                forecasts,
                source_frames.get(str(row["country"]))
                if source_frames is not None
                else None,
            )
        ):
            terminal.add(key)
    return terminal


def _forecast_is_complete(
    job: Mapping[str, object],
    forecasts: pd.DataFrame,
    source_frame: pd.DataFrame | None = None,
) -> bool:
    try:
        required = {
            *SCREENING_FORECAST_KEY[:4],
            "model_family",
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
            "status",
            "mapping_issue_count",
        }
        if forecasts.empty or not required.issubset(forecasts.columns):
            return False
        key = _job_key(job)
        mask = np.ones(len(forecasts), dtype=bool)
        for column, expected in zip(SCREENING_JOB_KEY, key):
            mask &= forecasts[column].map(
                lambda value, name=column: _canonical_key_value(name, value)
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
        origins = pd.to_datetime(
            matching["forecast_origin_utc"], format="mixed", utc=True, errors="coerce"
        )
        cutoffs = pd.to_datetime(
            matching["information_cutoff_utc"], format="mixed", utc=True, errors="coerce"
        )
        if timestamps.isna().any() or interval_end.isna().any() or origins.isna().any() or cutoffs.isna().any():
            return False
        expected_origin = country_forecast_origin(key[1], key[0])
        expected_origin_local = expected_origin.tz_convert(
            COUNTRY_CONFIG[key[0]]["timezone"]
        ).isoformat()
        if (
            not origins.eq(expected_origin).all()
            or not cutoffs.eq(expected_origin).all()
            or not matching["forecast_origin_local"].astype(str).eq(
                expected_origin_local
            ).all()
            or not interval_end.eq(timestamps + pd.Timedelta(hours=1)).all()
        ):
            return False
        timezone = COUNTRY_CONFIG[key[0]]["timezone"]
        expected_path = _expected_forecast_path_index(key[0], key[1])
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
        if not matching["status"].astype(str).eq("completed").all():
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
        if matching[["model_family", "source_kind"]].apply(
            lambda column: column.map(_blank).any()
        ).any():
            return False
        if not _weekly_source_metadata_complete(
            job, matching, expected_origin, source_frame
        ):
            return False
        mapping_counts = pd.to_numeric(
            matching["mapping_issue_count"], errors="coerce"
        )
        if (
            mapping_counts.isna().any()
            or not np.isfinite(mapping_counts).all()
            or not np.equal(mapping_counts, mapping_counts.astype(int)).all()
            or not mapping_counts.eq(0).all()
        ):
            return False
        target = matching.loc[
            matching["is_target_day"].map(_bool_value)
            & matching["evaluated"].map(_bool_value)
            & matching["local_date"].astype(str).str[:10].eq(key[1])
        ]
        expected_observations = int(float(job["expected_observations"]))
        if expected_observations <= 0 or len(target) != expected_observations:
            return False
        if set(target["_timestamp_utc"]) != set(
            _expected_target_index(key[0], key[1])
        ):
            return False
        actual = pd.to_numeric(matching["actual_load_mwh"], errors="coerce")
        forecast = pd.to_numeric(matching["forecast_mwh"], errors="coerce")
        return bool(np.isfinite(actual).all() and np.isfinite(forecast).all())
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _fit_metadata(fit: object) -> dict[str, object]:
    residual = getattr(fit, "residual_diagnostics", None)
    return {
        "fit_status": getattr(fit, "fit_status", "failed"),
        "convergence_status": getattr(fit, "convergence_status", "failed"),
        "converged": getattr(fit, "converged", False),
        "parameters_finite": getattr(fit, "parameters_finite", False),
        "standard_errors_finite": getattr(fit, "standard_errors_finite", False),
        "stationarity_ok": getattr(fit, "stationarity_ok", False),
        "invertibility_ok": getattr(fit, "invertibility_ok", False),
        "nobs": getattr(fit, "nobs", 0),
        "n_params": getattr(fit, "n_params", 0),
        "log_likelihood": getattr(fit, "log_likelihood", float("nan")),
        "aic": getattr(fit, "aic", float("nan")),
        "aicc": getattr(fit, "aicc", float("nan")),
        "bic": getattr(fit, "bic", float("nan")),
        "ar_root_minimum": getattr(fit, "ar_root_minimum", float("nan")),
        "ma_root_minimum": getattr(fit, "ma_root_minimum", float("nan")),
        "warning_messages": json.dumps(
            list(getattr(fit, "warning_messages", ())), default=str
        ),
        "hard_warning_messages": json.dumps(
            list(getattr(fit, "hard_warning_messages", ())), default=str
        ),
        "optimizer_retry_count": getattr(fit, "optimizer_retry_count", 0),
        "residual_n_effective": residual.n_effective if residual else 0,
        "residual_rejected": residual.rejected if residual else True,
        "residual_rejection_reason": (
            residual.rejection_reason
            if residual
            else "residual diagnostics unavailable"
        ),
        "residual_acf_values": json.dumps(
            residual.acf_values if residual else {}, default=str
        ),
        "residual_acf_flagged_lags": json.dumps(
            list(residual.flagged_acf_lags) if residual else []
        ),
        "residual_ljung_box_statistics": json.dumps(
            residual.ljung_box_statistics if residual else {}, default=str
        ),
        "residual_ljung_box_pvalues": json.dumps(
            residual.ljung_box_pvalues if residual else {}, default=str
        ),
    }


def _diagnostic_rows(
    country: str,
    target_date: str,
    branch: str,
    specification_id: str,
    fit: object,
) -> list[dict[str, object]]:
    residual = getattr(fit, "residual_diagnostics", None)
    if residual is None:
        return []
    rows: list[dict[str, object]] = []
    for lag, value in residual.acf_values.items():
        rows.append(
            {
                "country": country,
                "target_date": target_date,
                "branch": branch,
                "specification_id": specification_id,
                "diagnostic": "residual_acf",
                "lag": lag,
                "value": value,
                "threshold": residual.acf_threshold,
                "flagged": lag in residual.flagged_acf_lags,
            }
        )
    for lag, value in residual.ljung_box_statistics.items():
        rows.append(
            {
                "country": country,
                "target_date": target_date,
                "branch": branch,
                "specification_id": specification_id,
                "diagnostic": "ljung_box_statistic",
                "lag": lag,
                "value": value,
                "threshold": 0.01,
                "flagged": residual.ljung_box_pvalues[lag] < 0.01,
            }
        )
        rows.append(
            {
                "country": country,
                "target_date": target_date,
                "branch": branch,
                "specification_id": specification_id,
                "diagnostic": "ljung_box_pvalue",
                "lag": lag,
                "value": residual.ljung_box_pvalues[lag],
                "threshold": 0.01,
                "flagged": residual.ljung_box_pvalues[lag] < 0.01,
            }
        )
    return rows


def _specification_from_job(job: Mapping[str, object]) -> EnhancedSpecification:
    return EnhancedSpecification(
        branch=str(job["branch"]),
        specification_id=str(job["specification_id"]),
        order=(int(job["p"]), int(job["d"]), int(job["q"])),
        trend=str(job["trend"]),
        transform=str(job["transform"]),
    )


def _empty_metric_values() -> dict[str, object]:
    return {
        "mae": float("nan"),
        "rmse": float("nan"),
        "mape": float("nan"),
        "evaluated_observations": 0,
        "coverage": 0.0,
    }


def _ordinary_forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: str,
    specification_id: str,
    fitted_result: object,
) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    context = build_information_context(prepared, country, target_date)
    target = extract_target_day(prepared, target_date)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(context.cutoff_utc)
        & prepared["local_date"].le(target_date)
    ].copy()
    if path.empty or not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError(f"no complete ordinary forecast path exists for {country} {target_date}")
    path["country"] = country
    path["model_family"] = "enhanced_arima"
    path["target_date"] = target_date
    path["branch"] = "ordinary"
    path["specification_id"] = specification_id
    path["forecast_origin_local"] = context.origin_local
    path["forecast_origin_utc"] = context.cutoff_utc.isoformat()
    path["information_cutoff_utc"] = context.cutoff_utc.isoformat()
    path["forecast_mwh"] = forecast_values(fitted_result, len(path))
    path["bridge_used"] = ~path["local_date"].eq(target_date)
    path["source_timestamp_utc"] = pd.NaT
    path["source_actual_mwh"] = np.nan
    path["source_kind"] = "arima_forecast"
    path["is_target_day"] = path["local_date"].eq(target_date)
    path["evaluated"] = path["is_target_day"]
    path["status"] = "completed"
    path["mapping_issue_count"] = 0
    return path.reindex(columns=FORECAST_COLUMNS)


def _execute_job(job: Mapping[str, object], frame: pd.DataFrame) -> dict[str, object]:
    job_row = dict(job)
    job_row.update(
        {
            "verified_reuse": False,
            "reuse_status": "",
            "artifact_source": "",
            "source_artifact_path": "",
            "source_job_path": "",
            "source_forecasts_path": "",
            "reuse_verified": False,
        }
    )
    specification = _specification_from_job(job)
    if _bool_value(job.get("weekly_invalid", False)):
        job_row.update(_fit_metadata(_not_run_fit()))
        job_row.update(_empty_metric_values())
        job_row.update(
            {
                "status": "weekly_lag_invalid",
                "error_message": str(
                    job.get("weekly_invalid_reason") or "weekly source mapping issue"
                ),
            }
        )
        return {"job": job_row, "forecasts": [], "diagnostics": []}

    try:
        context = build_information_context(frame, str(job["country"]), str(job["target_date"]))
        if specification.branch == "ordinary":
            available = information_set(frame, context.cutoff_utc)
            fit_values: object = available["actual_load_mwh"].to_numpy(dtype=float)
        else:
            fit_values = transform_weekly_series(
                frame, context.cutoff_utc
            ).values
        fit = fit_enhanced(specification, fit_values)
        job_row.update(_fit_metadata(fit))
        job_row["error_message"] = getattr(fit, "error_message", None)
        diagnostics = _diagnostic_rows(
            str(job["country"]),
            str(job["target_date"]),
            specification.branch,
            specification.specification_id,
            fit,
        )
        if not getattr(fit, "eligible", False) or getattr(fit, "fitted_result", None) is None:
            job_row.update(_empty_metric_values())
            job_row["status"] = "failed"
            return {"job": job_row, "forecasts": [], "diagnostics": diagnostics}

        if specification.branch == "ordinary":
            path = _ordinary_forecast_path(
                frame,
                str(job["country"]),
                str(job["target_date"]),
                specification.specification_id,
                fit.fitted_result,
            )
        else:
            path = build_weekly_forecast_path(
                frame,
                str(job["country"]),
                str(job["target_date"]),
                specification,
                fit.fitted_result,
            )
            path["branch"] = specification.branch
            if path["status"].ne("completed").any():
                job_row.update(_empty_metric_values())
                job_row.update(
                    {
                        "status": "failed",
                        "error_message": "unexpected weekly source mapping issue",
                    }
                )
                return {"job": job_row, "forecasts": [], "diagnostics": diagnostics}

        metrics = evaluate_target_day(
            path,
            expected_observations=int(job["expected_observations"]),
        )
        job_row.update(metrics)
        job_row.update({"status": "completed", "error_message": None})
        return {
            "job": job_row,
            "forecasts": path.to_dict("records"),
            "diagnostics": diagnostics,
        }
    except Exception as error:
        job_row.update(_empty_metric_values())
        job_row.update(
            {
                "status": "failed",
                "error_message": f"{type(error).__name__}: {error}",
            }
        )
        return {"job": job_row, "forecasts": [], "diagnostics": []}


class _NotRunFit:
    fit_status = "not_run"
    convergence_status = "not_run"
    converged = False
    parameters_finite = False
    standard_errors_finite = False
    stationarity_ok = False
    invertibility_ok = False
    nobs = 0
    n_params = 0
    log_likelihood = float("nan")
    aic = float("nan")
    aicc = float("nan")
    bic = float("nan")
    ar_root_minimum = float("nan")
    ma_root_minimum = float("nan")
    warning_messages: tuple[str, ...] = ()
    hard_warning_messages: tuple[str, ...] = ()
    optimizer_retry_count = 0
    residual_diagnostics = None


def _not_run_fit() -> _NotRunFit:
    return _NotRunFit()


def _write_mapping_issues(
    branch_output: Path,
    invalid_dates: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> None:
    records: list[dict[str, object]] = []
    if not invalid_dates.empty:
        for country in COUNTRY_CONFIG:
            country_issues = invalid_dates.loc[invalid_dates["country"].eq(country)].copy()
            if country_issues.empty:
                continue
            expected_by_date = {
                target_date: len(extract_target_day(frames[country], target_date))
                for target_date in country_issues["target_date"].astype(str).unique()
            }
            issue_counts = country_issues.groupby("target_date").size().to_dict()
            for issue in country_issues.to_dict("records"):
                source_key = issue.get("source_key")
                target_date = str(issue["target_date"])
                expected = expected_by_date[target_date]
                affected = int(issue_counts[target_date])
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
    _atomic_write_csv(
        pd.DataFrame(records, columns=MAPPING_ISSUE_COLUMNS),
        branch_output / "mapping_issues.csv",
    )


def _summary_frame(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (country, branch, specification_id), expected in manifest.groupby(
        ["country", "branch", "specification_id"], sort=False
    ):
        group = jobs.loc[
            jobs["country"].eq(country)
            & jobs["branch"].eq(branch)
            & jobs["specification_id"].eq(specification_id)
        ].copy()
        weekly_invalid = (
            group["weekly_invalid"].map(_bool_value)
            if "weekly_invalid" in group.columns
            else pd.Series(False, index=group.index)
        )
        valid = group.loc[
            group.get("status", pd.Series(dtype=object)).eq("completed")
            & (~weekly_invalid if branch == "weekly_differenced" else True)
            & group[["mae", "rmse", "mape"]]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .all(axis=1)
        ] if not group.empty else group
        invalid = group.loc[
            group.get("status", pd.Series(dtype=object)).eq("weekly_lag_invalid")
            & weekly_invalid
        ]
        evaluated_observations = int(
            pd.to_numeric(
                valid.get("evaluated_observations", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        )
        expected_observations = int(expected["expected_observations"].sum())
        metrics: dict[str, float | int]
        if (
            forecasts is not None
            and not forecasts.empty
            and {"branch", "status"}.issubset(forecasts.columns)
        ):
            evaluated = forecasts.loc[
                forecasts["country"].eq(country)
                & forecasts["branch"].eq(branch)
                & forecasts["specification_id"].eq(specification_id)
                & forecasts["target_date"].astype(str).str[:10].isin(
                    valid["target_date"] if not valid.empty else []
                )
                & forecasts["is_target_day"].map(_bool_value)
                & forecasts["evaluated"].map(_bool_value)
                & forecasts["status"].eq("completed")
            ]
        else:
            evaluated = pd.DataFrame()
        if not evaluated.empty:
            metrics = calculate_metrics(
                evaluated["actual_load_mwh"],
                evaluated["forecast_mwh"],
                expected_observations=expected_observations,
            )
        else:
            aggregate = _aggregate_job_metrics(valid)
            metrics = {
                "mae": aggregate["mae"],
                "rmse": aggregate["rmse"],
                "mape": aggregate["mape"],
                "evaluated_observations": evaluated_observations,
                "coverage": float(
                    evaluated_observations / expected_observations
                )
                if expected_observations
                else 0.0,
            }
        expected_dates = int(expected["target_date"].nunique())
        valid_dates = int(valid["target_date"].nunique()) if not valid.empty else 0
        invalid_dates = int(invalid["target_date"].nunique()) if not invalid.empty else 0
        unresolved_dates = max(0, expected_dates - valid_dates - invalid_dates)
        rows.append(
            {
                "country": country,
                "branch": branch,
                "specification_id": specification_id,
                "p": int(expected["p"].iloc[0]),
                "d": int(expected["d"].iloc[0]),
                "q": int(expected["q"].iloc[0]),
                "trend": str(expected["trend"].iloc[0]),
                "transform": str(expected["transform"].iloc[0]),
                "expected_jobs": len(expected),
                "completed_jobs": int(group["status"].eq("completed").sum())
                if not group.empty
                else 0,
                "failed_jobs": int(group["status"].eq("failed").sum())
                if not group.empty
                else len(expected),
                "expected_dates": expected_dates,
                "valid_dates": valid_dates,
                "invalid_dates": invalid_dates,
                "unresolved_dates": unresolved_dates,
                "expected_observations": expected_observations,
                "evaluated_observations": int(metrics["evaluated_observations"]),
                "coverage": float(valid_dates / expected_dates)
                if expected_dates
                else 0.0,
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
                "verified_reuse_jobs": int(
                    group.get("verified_reuse", pd.Series(dtype=bool))
                    .map(_bool_value)
                    .sum()
                )
                if not group.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _persist_screening_result(
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    result: dict[str, object],
    jobs_path: Path,
    forecasts_path: Path,
    diagnostics_path: Path,
    summary_path: Path,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    job = dict(result["job"])
    key = _job_key(job)
    job_frame = pd.DataFrame([job])
    forecast_frame = pd.DataFrame(result.get("forecasts", []))
    if forecast_frame.empty:
        forecast_frame = pd.DataFrame(columns=FORECAST_COLUMNS)
    diagnostic_frame = pd.DataFrame(result.get("diagnostics", []))
    if diagnostic_frame.empty:
        diagnostic_frame = pd.DataFrame(columns=DIAGNOSTICS_COLUMNS)
    jobs = upsert_frame(jobs, job_frame, SCREENING_JOB_KEY, sort=False)
    forecasts = _drop_job_rows(forecasts, SCREENING_FORECAST_KEY, key)
    if not forecast_frame.empty:
        forecasts = upsert_frame(
            forecasts,
            forecast_frame,
            SCREENING_FORECAST_KEY,
            sort=False,
        )
    diagnostics = _drop_job_rows(diagnostics, SCREENING_DIAGNOSTIC_KEY, key)
    if not diagnostic_frame.empty:
        diagnostics = upsert_frame(
            diagnostics,
            diagnostic_frame,
            SCREENING_DIAGNOSTIC_KEY,
            sort=False,
        )
    summary = _summary_frame(jobs, manifest, forecasts)
    _atomic_write_csv(forecasts, forecasts_path,)
    _atomic_write_csv(diagnostics, diagnostics_path)
    _atomic_write_csv(jobs, jobs_path)
    _atomic_write_csv(summary, summary_path)
    return jobs, forecasts, diagnostics


def _stable_checkpoint_frame(
    frame: pd.DataFrame,
    key_columns: Iterable[str],
) -> pd.DataFrame:
    return upsert_frame(
        pd.DataFrame(columns=frame.columns),
        frame,
        key_columns,
    )


def _drop_job_rows(
    frame: pd.DataFrame,
    key_columns: Iterable[str],
    job_key: tuple[str, str, str, str],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    columns = list(key_columns)
    mask = np.ones(len(frame), dtype=bool)
    for column, expected in zip(columns[:4], job_key):
        mask &= frame[column].map(
            lambda value, name=column: _canonical_key_value(name, value)
        ).eq(expected)
    return frame.loc[~mask].copy().reset_index(drop=True)


def _original_specification_id(job: Mapping[str, object]) -> str:
    return f"arima_p{int(job['p'])}_d{int(job['d'])}_q{int(job['q'])}"


def _expected_target_index(country: str, target_date: str) -> pd.DatetimeIndex:
    timezone = COUNTRY_CONFIG[country]["timezone"]
    start = pd.Timestamp(target_date).date()
    end = start + timedelta(days=1)
    local_index = pd.date_range(
        f"{start.isoformat()} 00:00:00",
        f"{end.isoformat()} 00:00:00",
        freq="h",
        inclusive="left",
        tz=timezone,
    )
    return local_index.tz_convert("UTC")


def _expected_target_observations(country: str, target_date: str) -> int:
    return len(_expected_target_index(country, target_date))


def _expected_forecast_path_index(
    country: str, target_date: str
) -> pd.DatetimeIndex:
    timezone = COUNTRY_CONFIG[country]["timezone"]
    origin = country_forecast_origin(target_date, country)
    end_date = pd.Timestamp(target_date).date() + timedelta(days=1)
    end = pd.Timestamp(
        f"{end_date.isoformat()} 00:00:00", tz=timezone
    ).tz_convert("UTC")
    return pd.date_range(origin, end, inclusive="left", freq="h")


def _source_job_is_complete(row: Mapping[str, object]) -> bool:
    required = (
        "trend",
        "fit_status",
        "convergence_status",
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
        "residual_acf_values",
        "residual_acf_flagged_lags",
        "residual_ljung_box_statistics",
        "residual_ljung_box_pvalues",
    )
    if any(
        row.get(column) is None
        or pd.isna(row.get(column))
        or str(row.get(column)).strip() in {"", "nan", "None"}
        for column in required
    ):
        return False
    if str(row.get("fit_status")) != "success" or str(
        row.get("convergence_status")
    ) != "converged":
        return False
    if str(row.get("error_message", "")).strip() not in {"", "nan", "None"}:
        return False
    if not all(
        _bool_value(row.get(column))
        for column in (
            "converged",
            "parameters_finite",
            "standard_errors_finite",
            "stationarity_ok",
            "invertibility_ok",
        )
    ):
        return False
    try:
        numeric = [
            float(row.get(column))
            for column in (
                "nobs",
                "n_params",
                "log_likelihood",
                "aic",
                "aicc",
                "bic",
                "residual_n_effective",
                "mae",
                "rmse",
                "mape",
            )
        ]
    except (TypeError, ValueError):
        return False
    if not np.isfinite(numeric).all() or numeric[0] <= 0 or numeric[6] <= 0:
        return False
    acf_values = _json_mapping(row.get("residual_acf_values"))
    statistics = _json_mapping(row.get("residual_ljung_box_statistics"))
    pvalues = _json_mapping(row.get("residual_ljung_box_pvalues"))
    if not acf_values or not statistics or not pvalues:
        return False
    if not {"24", "48"}.issubset(statistics) or not {"24", "48"}.issubset(pvalues):
        return False
    try:
        flagged_lags = json.loads(str(row.get("residual_acf_flagged_lags")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(flagged_lags, list):
        return False
    for values in (acf_values, statistics, pvalues):
        try:
            if not all(np.isfinite(float(value)) for value in values.values()):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _read_reusable_original_artifacts() -> dict[tuple[str, str, str], dict[str, object]]:
    try:
        audit = audit_existing_arima()
        jobs_path = ORIGINAL_VALIDATION_DIRECTORY / "arima_validation_2024_jobs.csv"
        forecasts_path = ORIGINAL_VALIDATION_DIRECTORY / "arima_validation_2024_forecasts.csv"
        if not jobs_path.exists() or not forecasts_path.exists():
            return {}
        original_jobs = pd.read_csv(jobs_path)
        original_forecasts = pd.read_csv(forecasts_path)
        original_jobs["target_date"] = original_jobs["target_date"].map(
            lambda value: _canonical_key_value("target_date", value)
        )
        original_forecasts["target_date"] = original_forecasts["target_date"].map(
            lambda value: _canonical_key_value("target_date", value)
        )
        original_forecasts["timestamp_utc"] = pd.to_datetime(
            original_forecasts["timestamp_utc"],
            format="mixed",
            utc=True,
            errors="raise",
        ).map(pd.Timestamp.isoformat)
    except (FileNotFoundError, ValueError, OSError, KeyError):
        return {}

    eligible_audit = audit.loc[
        audit["audit_type"].eq("ordinary_enhanced_candidate")
        & audit["reuse_eligible"].map(_bool_value)
    ]
    eligible = {
        (str(row.country), str(row.specification_id))
        for row in eligible_audit.itertuples(index=False)
    }
    duplicate_job_keys = {
        key
        for key, count in original_jobs.groupby(
            ["country", "target_date", "specification_id"], dropna=False
        ).size().items()
        if count != 1
    }
    reusable: dict[tuple[str, str, str], dict[str, object]] = {}
    provenance = {
        "artifact_source": "original_arima_validation_2024",
        "source_artifact_path": json.dumps(
            [str(jobs_path), str(forecasts_path)]
        ),
        "source_job_path": str(jobs_path),
        "source_forecasts_path": str(forecasts_path),
    }
    for row in original_jobs.to_dict("records"):
        country = str(row.get("country"))
        target_date = str(row.get("target_date"))
        original_id = str(row.get("specification_id"))
        if (country, target_date, original_id) in duplicate_job_keys:
            continue
        if (country, _enhanced_id_from_original(original_id)) not in eligible:
            continue
        if str(row.get("status")) != "completed" or not _source_job_is_complete(row):
            continue
        try:
            source_order = (
                int(row["p"]),
                int(row["d"]),
                int(row["q"]),
            )
            if source_order != _parse_arima_order(original_id):
                continue
            if not _source_fit_settings_match(
                row,
                {
                    "p": source_order[0],
                    "d": source_order[1],
                    "q": source_order[2],
                    "trend": trend_for_d(source_order[1]),
                },
            ):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        matching_forecasts = original_forecasts.loc[
            original_forecasts["country"].eq(country)
            & original_forecasts["target_date"].astype(str).str[:10].eq(target_date)
            & original_forecasts["specification_id"].eq(original_id)
        ].copy()
        if not {
            "forecast_origin_utc",
            "information_cutoff_utc",
            "is_target_day",
            "evaluated",
            "timestamp_utc",
            "local_date",
            "actual_load_mwh",
            "forecast_mwh",
        }.issubset(matching_forecasts.columns):
            continue
        if not _source_forecasts_complete(
            country, target_date, original_id, matching_forecasts
        ):
            continue
        expected_origin = country_forecast_origin(target_date, country)
        try:
            source_origins = pd.to_datetime(
                matching_forecasts["forecast_origin_utc"], utc=True, errors="raise"
            )
            source_cutoffs = pd.to_datetime(
                matching_forecasts["information_cutoff_utc"], utc=True, errors="raise"
            )
        except (TypeError, ValueError):
            continue
        if not source_origins.eq(expected_origin).all() or not source_cutoffs.eq(expected_origin).all():
            continue
        target_forecasts = matching_forecasts.loc[
            matching_forecasts["is_target_day"].map(_bool_value)
            & matching_forecasts["local_date"].astype(str).str[:10].eq(target_date)
        ]
        expected_observations = _expected_target_observations(country, target_date)
        if len(target_forecasts) != expected_observations:
            continue
        if not target_forecasts["timestamp_utc"].is_unique:
            continue
        try:
            target_timestamps = pd.to_datetime(
                target_forecasts["timestamp_utc"],
                format="mixed",
                utc=True,
                errors="raise",
            )
        except (TypeError, ValueError):
            continue
        if set(target_timestamps) != set(_expected_target_index(country, target_date)):
            continue
        if not target_forecasts["evaluated"].map(_bool_value).all():
            continue
        if not np.isfinite(
            pd.to_numeric(target_forecasts["actual_load_mwh"], errors="coerce")
        ).all():
            continue
        if not np.isfinite(
            pd.to_numeric(target_forecasts["forecast_mwh"], errors="coerce")
        ).all():
            continue
        if not np.isfinite(
            pd.to_numeric(matching_forecasts["forecast_mwh"], errors="coerce")
        ).all():
            continue
        if not _source_metrics_match(
            row, matching_forecasts, country, target_date, original_id
        ):
            continue
        reusable[(country, target_date, original_id)] = {
            "job": row,
            "forecasts": matching_forecasts,
            "provenance": provenance.copy(),
        }
    return reusable


def _enhanced_id_from_original(original_id: str) -> str:
    match = re.fullmatch(r"arima_p(\d+)_d(\d+)_q(\d+)", original_id)
    if match is None:
        return ""
    p, d, q = match.groups()
    return f"enhanced_ordinary_arima_p{p}_d{d}_q{q}"


def _source_forecasts_complete(
    country: str,
    target_date: str,
    specification_id: str,
    forecasts: pd.DataFrame,
) -> bool:
    required = {
        "model_family",
        "country",
        "target_date",
        "specification_id",
        "forecast_origin_local",
        "forecast_origin_utc",
        "information_cutoff_utc",
        "is_target_day",
        "evaluated",
        "timestamp_utc",
        "interval_end_utc",
        "timestamp_local",
        "local_date",
        "actual_load_mwh",
        "forecast_mwh",
        "bridge_used",
        "source_kind",
    }
    if forecasts.empty or not required.issubset(forecasts.columns):
        return False
    matching = forecasts.loc[
        forecasts["country"].astype(str).eq(country)
        & forecasts["target_date"].astype(str).str[:10].eq(target_date)
        & forecasts["specification_id"].astype(str).eq(specification_id)
    ].copy()
    if matching.empty or len(matching) != len(forecasts):
        return False
    for column in (
        "model_family",
        "forecast_origin_local",
        "forecast_origin_utc",
        "information_cutoff_utc",
        "timestamp_local",
        "local_date",
        "source_kind",
    ):
        if matching[column].map(
            lambda value: value is None or str(value).strip() in {"", "nan", "None"}
        ).any():
            return False
    if "status" in matching and not matching["status"].astype(str).eq("completed").all():
        return False
    timestamps = pd.to_datetime(
        matching["timestamp_utc"], format="mixed", utc=True, errors="coerce"
    )
    interval_end = pd.to_datetime(
        matching["interval_end_utc"], format="mixed", utc=True, errors="coerce"
    )
    origins = pd.to_datetime(
        matching["forecast_origin_utc"], format="mixed", utc=True, errors="coerce"
    )
    cutoffs = pd.to_datetime(
        matching["information_cutoff_utc"], format="mixed", utc=True, errors="coerce"
    )
    if timestamps.isna().any() or interval_end.isna().any() or origins.isna().any() or cutoffs.isna().any():
        return False
    expected_origin = country_forecast_origin(target_date, country)
    expected_origin_local = expected_origin.tz_convert(COUNTRY_CONFIG[country]["timezone"]).isoformat()
    if (
        not origins.eq(expected_origin).all()
        or not cutoffs.eq(expected_origin).all()
        or not matching["forecast_origin_local"].astype(str).eq(expected_origin_local).all()
    ):
        return False
    if not interval_end.eq(timestamps + pd.Timedelta(hours=1)).all():
        return False
    timezone = COUNTRY_CONFIG[country]["timezone"]
    expected_path = _expected_forecast_path_index(country, target_date)
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
    if matching["local_date"].astype(str).str[:10].tolist() != [
        timestamp.date().isoformat() for timestamp in expected_local
    ]:
        return False
    expected_target = expected_local.date == pd.Timestamp(target_date).date()
    for column, expected_values in (
        ("is_target_day", expected_target),
        ("evaluated", expected_target),
        ("bridge_used", ~expected_target),
    ):
        if matching[column].map(
            lambda value: value is None
            or str(value).strip() in {"", "nan", "None"}
        ).any():
            return False
        if matching[column].map(_bool_value).tolist() != expected_values.tolist():
            return False
    target = matching.loc[
        matching["is_target_day"].map(_bool_value)
        & matching["evaluated"].map(_bool_value)
        & matching["local_date"].astype(str).str[:10].eq(target_date)
    ].copy()
    expected = _expected_target_observations(country, target_date)
    if len(target) != expected or not target["timestamp_utc"].is_unique:
        return False
    if set(target["_timestamp_utc"]) != set(
        _expected_target_index(country, target_date)
    ):
        return False
    actual = pd.to_numeric(matching["actual_load_mwh"], errors="coerce")
    forecast = pd.to_numeric(matching["forecast_mwh"], errors="coerce")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        return False
    for column in ("is_target_day", "evaluated", "bridge_used"):
        if matching[column].map(
            lambda value: value is None or str(value).strip() in {"", "nan", "None"}
        ).any():
            return False
    return True


def _source_fit_settings_match(
    source_job: Mapping[str, object], target_job: Mapping[str, object]
) -> bool:
    try:
        source_order = tuple(
            int(source_job[column]) for column in ("p", "d", "q")
        )
        target_order = tuple(
            int(target_job[column]) for column in ("p", "d", "q")
        )
        if source_order != target_order:
            return False
        if _parse_arima_order(str(source_job["specification_id"])) != source_order:
            return False
        expected_trend = trend_for_d(target_order[1])
    except (KeyError, TypeError, ValueError):
        return False
    source_trend = str(source_job.get("trend", "")).strip()
    target_trend = str(target_job.get("trend", "")).strip()
    return bool(
        source_trend
        and target_trend
        and source_trend == target_trend == expected_trend
    )


def _source_metrics_match(
    source_job: Mapping[str, object],
    forecasts: pd.DataFrame,
    country: str,
    target_date: str,
    specification_id: str,
) -> bool:
    target = forecasts.loc[
        forecasts["is_target_day"].map(_bool_value)
        & forecasts["evaluated"].map(_bool_value)
        & forecasts["local_date"].astype(str).str[:10].eq(target_date)
    ]
    try:
        expected = int(float(source_job.get("expected_observations")))
    except (TypeError, ValueError):
        expected = 0
    if expected <= 0 or len(target) != expected:
        return False
    metrics = calculate_metrics(
        pd.to_numeric(target["actual_load_mwh"], errors="coerce"),
        pd.to_numeric(target["forecast_mwh"], errors="coerce"),
        expected_observations=expected,
    )
    try:
        if int(float(source_job.get("evaluated_observations"))) != int(
            metrics["evaluated_observations"]
        ):
            return False
        if not np.isclose(float(source_job["coverage"]), float(metrics["coverage"])):
            return False
        return all(
            np.isclose(float(source_job[column]), float(metrics[column]))
            for column in ("mae", "rmse", "mape")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _adapt_reused_result(
    job: Mapping[str, object],
    reusable: Mapping[str, object],
) -> dict[str, object]:
    source_job = dict(reusable["job"])
    if str(source_job.get("status")) != "completed":
        raise ValueError("authoritative reused job is not completed")
    try:
        requested_source_key = (
            str(job["country"]),
            _canonical_key_value("target_date", job["target_date"]),
            _original_specification_id(job),
        )
        source_key = (
            str(source_job["country"]),
            _canonical_key_value("target_date", source_job["target_date"]),
            str(source_job["specification_id"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("authoritative reused source and requested job keys do not match")
    if (
        source_key != requested_source_key
        or str(job.get("branch")) != "ordinary"
        or str(job.get("specification_id"))
        != _enhanced_id_from_original(requested_source_key[2])
        or (
            "branch" in source_job
            and str(source_job.get("branch")) != "ordinary"
        )
    ):
        raise ValueError("authoritative reused source and requested job keys do not match")
    if not _source_fit_settings_match(source_job, job):
        raise ValueError("authoritative reused fit settings do not match")
    original_forecasts = pd.DataFrame(reusable["forecasts"]).copy()
    original_specification_id = str(source_job.get("specification_id", ""))
    if not _source_forecasts_complete(
        str(source_job.get("country", job["country"])),
        str(source_job.get("target_date", job["target_date"])),
        original_specification_id,
        original_forecasts,
    ):
        raise ValueError("authoritative reused forecast is incomplete")
    if not _source_job_is_complete(source_job):
        raise ValueError("authoritative reused fit metadata is incomplete")
    if not _source_metrics_match(
        source_job,
        original_forecasts,
        str(source_job.get("country", job["country"])),
        str(source_job.get("target_date", job["target_date"])),
        original_specification_id,
    ):
        raise ValueError("authoritative reused metrics do not match forecasts")
    job_row = dict(job)
    for column in JOB_COLUMNS:
        if column in source_job and column not in {
            "country",
            "target_date",
            "branch",
            "specification_id",
        }:
            job_row[column] = source_job[column]
    job_row.update(
        {
            "status": "completed",
            "error_message": None,
            "verified_reuse": True,
            "reuse_status": "verified_reuse",
            "artifact_source": str(
                reusable.get("provenance", {}).get(
                    "artifact_source", "original_arima_validation_2024"
                )
            ),
            "source_artifact_path": str(
                reusable.get("provenance", {}).get("source_artifact_path", "")
            ),
            "source_job_path": str(
                reusable.get("provenance", {}).get("source_job_path", "")
            ),
            "source_forecasts_path": str(
                reusable.get("provenance", {}).get("source_forecasts_path", "")
            ),
            "reuse_verified": True,
        }
    )
    forecasts = original_forecasts.copy()
    forecasts["country"] = str(job["country"])
    forecasts["target_date"] = _canonical_key_value(
        "target_date", job["target_date"]
    )
    forecasts["model_family"] = "enhanced_arima"
    forecasts["branch"] = "ordinary"
    forecasts["specification_id"] = str(job["specification_id"])
    forecasts["source_timestamp_utc"] = pd.NaT
    forecasts["source_actual_mwh"] = np.nan
    forecasts["source_kind"] = "arima_forecast"
    forecasts["status"] = "completed"
    forecasts["mapping_issue_count"] = 0
    forecasts = forecasts.reindex(columns=FORECAST_COLUMNS)
    diagnostics = _diagnostics_from_source_job(job, source_job)
    return {
        "job": job_row,
        "forecasts": forecasts.to_dict("records"),
        "diagnostics": diagnostics,
    }


def _json_mapping(value: object) -> dict[object, object]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _diagnostics_from_source_job(
    job: Mapping[str, object],
    source_job: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    acf_values = _json_mapping(source_job.get("residual_acf_values"))
    flagged_value = source_job.get("residual_acf_flagged_lags", "[]")
    try:
        decoded_flagged = json.loads(str(flagged_value))
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded_flagged = []
    flagged = set(decoded_flagged) if isinstance(decoded_flagged, list) else set()
    try:
        threshold = float(source_job.get("residual_acf_threshold", float("nan")))
    except (TypeError, ValueError):
        threshold = float("nan")
    if not np.isfinite(threshold):
        try:
            threshold = max(
                2.576 / np.sqrt(float(source_job["residual_n_effective"])),
                0.05,
            )
        except (KeyError, TypeError, ValueError, FloatingPointError):
            threshold = float("nan")
    for lag, value in acf_values.items():
        rows.append(
            {
                "country": job["country"],
                "target_date": job["target_date"],
                "branch": job["branch"],
                "specification_id": job["specification_id"],
                "diagnostic": "residual_acf",
                "lag": int(lag),
                "value": float(value),
                "threshold": threshold,
                "flagged": int(lag) in flagged,
            }
        )
    for diagnostic, value_key in (
        ("ljung_box_statistic", "residual_ljung_box_statistics"),
        ("ljung_box_pvalue", "residual_ljung_box_pvalues"),
    ):
        values = _json_mapping(source_job.get(value_key))
        for lag, value in values.items():
            rows.append(
                {
                    "country": job["country"],
                    "target_date": job["target_date"],
                    "branch": job["branch"],
                    "specification_id": job["specification_id"],
                    "diagnostic": diagnostic,
                    "lag": int(lag),
                    "value": float(value),
                    "threshold": 0.01,
                    "flagged": (
                        float(value) < 0.01
                        if diagnostic == "ljung_box_pvalue"
                        else False
                    ),
                }
            )
    return rows


def _initialize_worker(processed_directory: str) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = {
        country: _load_setup_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _worker_execute(job: Mapping[str, object]) -> dict[str, object]:
    return _execute_job(job, _WORKER_FRAMES[str(job["country"])])
