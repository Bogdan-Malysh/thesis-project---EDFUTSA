from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Mapping

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import COUNTRY_CONFIG, PROCESSED, country_forecast_origin  # noqa: E402
from enhanced_arima_models import (  # noqa: E402
    derive_weekly_invalid_target_dates,
    enhanced_specifications,
)
import enhanced_arima_screening_2024 as screening  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DATES = (
    "2024-02-15",
    "2024-03-31",
    "2024-07-25",
    "2024-11-03",
)
WORKER_COUNTS = tuple(range(1, 9))
EQUIVALENCE_RTOL = 1e-10
EQUIVALENCE_ATOL = 1e-12
WORKLOAD_COLUMNS = [
    "country",
    "target_date",
    "preferred_date",
    "date_selection_reason",
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
]
SUMMARY_COLUMNS = [
    "requested_workers",
    "active_workers",
    "supported",
    "status",
    "elapsed_seconds",
    "jobs_per_minute",
    "failures",
    "peak_rss_bytes",
    "peak_swap_bytes",
    "resource_status",
    "max_absolute_forecast_difference",
    "max_relative_forecast_difference",
    "equivalence",
    "safe",
    "reason",
]
_WORKER_FRAMES: dict[str, pd.DataFrame] = {}


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


def _date_text(value: object) -> str:
    target = pd.Timestamp(value).date()
    if target.year != 2024:
        raise ValueError("benchmark dates must be in 2024")
    return target.isoformat()


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


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


def _forecast_key(row: Mapping[str, object]) -> tuple[str, str, str, str, str]:
    return (*_job_key(row), _canonical_timestamp(row["timestamp_utc"]))


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
    result = shortlist.copy().reset_index(drop=True)
    result["country"] = result["country"].astype(str)
    result["branch"] = result["branch"].astype(str)
    result["specification_id"] = result["specification_id"].astype(str)
    if set(result["country"]) != set(COUNTRY_CONFIG):
        raise ValueError("shortlist must contain exactly Germany and Austria")
    if result.duplicated(["country", "branch", "specification_id"]).any():
        raise ValueError("shortlist contains duplicate candidate keys")
    counts = result.groupby(["country", "branch"]).size().to_dict()
    maximum_counts = {
        (country, "ordinary"): 3 for country in COUNTRY_CONFIG
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

    specification_lookup = {
        specification.specification_id: specification
        for specification in enhanced_specifications()
    }
    for row in result.to_dict("records"):
        specification = specification_lookup.get(row["specification_id"])
        if specification is None or specification.branch != row["branch"]:
            raise ValueError("shortlist contains an unknown branch or specification")
        if tuple(int(row[column]) for column in ("p", "d", "q")) != specification.order:
            raise ValueError("shortlist specification order does not match its identifier")
        if str(row["trend"]) != specification.trend or str(row["transform"]) != specification.transform:
            raise ValueError("shortlist specification settings do not match its identifier")
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
    result["specification_order"] = pd.to_numeric(
        result["specification_order"], errors="raise"
    ).astype(int)
    return result


def _selected_shortlist(shortlist: pd.DataFrame) -> pd.DataFrame:
    result = _normalise_shortlist(shortlist)
    result["_input_order"] = np.arange(len(result))
    if "shortlist_rank" in result:
        result["_rank"] = pd.to_numeric(result["shortlist_rank"], errors="coerce")
    else:
        result["_rank"] = result["specification_order"]
    result = result.sort_values(
        ["country", "branch", "_rank", "specification_order", "_input_order"],
        kind="stable",
    )
    result = result.groupby(["country", "branch"], sort=False).head(2).copy()
    return result.drop(columns=["_input_order", "_rank"])


def _invalid_date_set(invalid_dates: pd.DataFrame) -> set[str]:
    if invalid_dates.empty:
        return set()
    if "target_date" not in invalid_dates.columns:
        raise ValueError("invalid-date frame is missing target_date")
    return {_date_text(value) for value in invalid_dates["target_date"]}


def _selected_dates(invalid_dates: pd.DataFrame) -> list[tuple[str, str, str]]:
    invalid = _invalid_date_set(invalid_dates)
    chosen: list[tuple[str, str, str]] = []
    used: set[str] = set()
    candidates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    for preferred in BENCHMARK_DATES:
        if preferred not in invalid and preferred not in used:
            selected = preferred
            reason = "preferred date is derived-valid"
        else:
            selected = next(
                candidate.date().isoformat()
                for candidate in candidates
                if candidate.date().isoformat() not in invalid
                and candidate.date().isoformat() not in used
            )
            reason = "preferred date replaced because it is derived-invalid"
        chosen.append((preferred, selected, reason))
        used.add(selected)
    return chosen


def build_benchmark_workload(
    shortlist: pd.DataFrame,
    invalid_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Build the fixed, valid workload from the screening shortlist."""
    candidates = _selected_shortlist(shortlist)
    selected_dates = _selected_dates(invalid_dates)
    rows: list[dict[str, object]] = []
    for country in COUNTRY_CONFIG:
        country_candidates = candidates.loc[candidates["country"].eq(country)]
        for candidate in country_candidates.to_dict("records"):
            for preferred, target_date, reason in selected_dates:
                origin = country_forecast_origin(target_date, country)
                origin_local = origin.tz_convert(COUNTRY_CONFIG[country]["timezone"])
                rows.append(
                    {
                        "country": country,
                        "target_date": target_date,
                        "preferred_date": preferred,
                        "date_selection_reason": reason,
                        "category": "benchmark",
                        "forecast_origin_local": origin_local.isoformat(),
                        "forecast_origin_utc": origin.isoformat(),
                        "information_cutoff_utc": origin.isoformat(),
                        "expected_observations": len(
                            screening._expected_target_index(country, target_date)
                        ),
                        "branch": candidate["branch"],
                        "specification_id": candidate["specification_id"],
                        "specification_order": candidate["specification_order"],
                        "p": candidate["p"],
                        "d": candidate["d"],
                        "q": candidate["q"],
                        "trend": candidate["trend"],
                        "transform": candidate["transform"],
                        "weekly_invalid": False,
                        "weekly_invalid_reason": "",
                    }
                )
    workload = pd.DataFrame(rows, columns=WORKLOAD_COLUMNS)
    key = ["country", "target_date", "branch", "specification_id"]
    expected_jobs = len(candidates) * len(selected_dates)
    if len(workload) != expected_jobs or workload.duplicated(key).any():
        raise ValueError("benchmark workload is not the required unique candidate/date set")
    if workload["target_date"].nunique() != 4:
        raise ValueError("benchmark workload must contain four target dates")
    if workload["weekly_invalid"].map(_bool_value).any():
        raise ValueError("benchmark workload contains a derived-invalid weekly date")
    return workload


def _load_benchmark_frames(
    processed_directory: str | Path = PROCESSED,
) -> dict[str, pd.DataFrame]:
    return {
        country: screening._load_setup_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }


def _derive_invalid_dates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for country, frame in frames.items():
        invalid = derive_weekly_invalid_target_dates(frame, 2024).copy()
        if invalid.empty:
            continue
        invalid.insert(0, "country", country)
        records.append(invalid)
    if not records:
        return pd.DataFrame(columns=["country", "target_date", "reason"])
    return pd.concat(records, ignore_index=True)


def _read_proc_status(pid: int) -> tuple[int, int | None, int | None] | None:
    try:
        values: dict[str, int] = {}
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition(":")
            if name in {"PPid", "VmRSS", "VmSwap"}:
                parsed = int(value.strip().split()[0])
                values[name] = parsed if name == "PPid" else parsed * 1024
        if "PPid" not in values:
            return None
        return values["PPid"], values.get("VmRSS"), values.get("VmSwap")
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        return None


def _process_command(pid: int) -> str:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", errors="replace")
        )
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _process_tree_memory(root_pid: int) -> tuple[int | None, int | None]:
    rss, swap, _ = _process_tree_metrics(root_pid)
    return rss, swap


def _process_tree_metrics(
    root_pid: int,
) -> tuple[int | None, int | None, int]:
    try:
        pids = [int(value) for value in os.listdir("/proc") if value.isdigit()]
    except (FileNotFoundError, PermissionError, OSError):
        return None, None, 0
    statuses = {
        pid: _read_proc_status(pid)
        for pid in pids
    }
    statuses = {pid: value for pid, value in statuses.items() if value is not None}
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, value in statuses.items():
            if pid not in tree and value[0] in tree:
                tree.add(pid)
                changed = True
    rss = [statuses[pid][1] for pid in tree if pid in statuses and statuses[pid][1] is not None]
    swap = [statuses[pid][2] for pid in tree if pid in statuses and statuses[pid][2] is not None]
    active_workers = sum(
        "multiprocessing.spawn" in _process_command(pid)
        and "spawn_main" in _process_command(pid)
        for pid in tree
        if pid != root_pid
    )
    return (
        sum(rss) if rss else None,
        sum(swap) if swap else None,
        active_workers,
    )


class _ResourceMonitor:
    def __init__(self, root_pid: int):
        self.root_pid = root_pid
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss: int | None = None
        self.peak_swap: int | None = None
        self.peak_active_workers = 0

    def _sample(self) -> None:
        rss, swap, active_workers = _process_tree_metrics(self.root_pid)
        if rss is not None:
            self.peak_rss = max(self.peak_rss or 0, rss)
        if swap is not None:
            self.peak_swap = max(self.peak_swap or 0, swap)
        self.peak_active_workers = max(self.peak_active_workers, active_workers)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(0.05)

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[int | None, int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        return self.peak_rss, self.peak_swap


def _spawn_probe(_: int) -> bool:
    return True


def _worker_count_support(workers: int) -> tuple[bool, str]:
    if workers < 1:
        return False, "worker count must be positive"
    logical_cpus = os.cpu_count() or 0
    if workers > logical_cpus:
        return False, "logical CPU capacity"
    try:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            list(executor.map(_spawn_probe, range(workers)))
    except (OSError, RuntimeError, ValueError) as error:
        return False, f"spawn startup failed: {type(error).__name__}: {error}"
    return True, ""


def _execute_benchmark_job(
    job: Mapping[str, object],
    frame: pd.DataFrame,
) -> dict[str, object]:
    return screening._execute_job(job, frame)


def _initialize_benchmark_worker(processed_directory: str) -> None:
    global _WORKER_FRAMES
    _WORKER_FRAMES = _load_benchmark_frames(processed_directory)


def _worker_execute(job: Mapping[str, object]) -> dict[str, object]:
    return _execute_benchmark_job(job, _WORKER_FRAMES[str(job["country"])])


def _run_worker_count(
    workload: pd.DataFrame,
    workers: int,
    source_frames: dict[str, pd.DataFrame],
    processed_directory: str | Path,
) -> dict[str, object]:
    monitor = _ResourceMonitor(os.getpid())
    monitor.start()
    try:
        jobs = workload.to_dict("records")
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_benchmark_worker,
            initargs=(str(processed_directory),),
        ) as executor:
            results = list(executor.map(_worker_execute, jobs))
        result_jobs = pd.DataFrame([result["job"] for result in results])
        result_forecasts = pd.DataFrame(
            [row for result in results for row in result.get("forecasts", [])]
        )
        result = {
            "jobs": result_jobs,
            "forecasts": result_forecasts,
            "diagnostics": pd.DataFrame(
                [row for result in results for row in result.get("diagnostics", [])]
            ),
        }
    except Exception as error:
        result = {
            "jobs": pd.DataFrame(),
            "forecasts": pd.DataFrame(),
            "diagnostics": pd.DataFrame(),
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        peak_rss, peak_swap = monitor.stop()
    result["peak_rss_bytes"] = peak_rss
    result["peak_swap_bytes"] = peak_swap
    result["active_workers"] = monitor.peak_active_workers
    result["resource_status"] = (
        "available" if peak_rss is not None and peak_swap is not None else "unavailable"
    )
    return result


def _result_keys(
    frame: pd.DataFrame,
    key_function,
) -> tuple[list[object], bool]:
    if frame.empty:
        return [], False
    try:
        values = [key_function(row) for row in frame.to_dict("records")]
    except (KeyError, TypeError, ValueError, OverflowError):
        return [], False
    return values, len(values) == len(set(values))


def _safe_job_key(row: Mapping[str, object]) -> tuple[str, str, str, str] | None:
    try:
        return _job_key(row)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _validate_run(
    workload: pd.DataFrame,
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    expected_keys = {_job_key(row) for row in workload.to_dict("records")}
    job_keys, unique_jobs = _result_keys(jobs, _job_key)
    forecast_keys, unique_forecasts = _result_keys(forecasts, _forecast_key)
    reasons: list[str] = []
    expected_forecast_keys: set[tuple[str, str, str, str, str]] = set()
    try:
        for row in workload.to_dict("records"):
            key = _job_key(row)
            expected_forecast_keys.update(
                (*key, _canonical_timestamp(timestamp))
                for timestamp in screening._expected_forecast_path_index(
                    key[0], key[1]
                )
            )
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons.append("workload forecast path is malformed")
    if not unique_jobs or set(job_keys) != expected_keys:
        reasons.append("duplicate or incomplete job keys")
    if "status" not in jobs.columns:
        reasons.append("job output schema is incomplete")
    if not unique_forecasts:
        reasons.append("duplicate or malformed forecast keys")
    if set(forecast_keys) != expected_forecast_keys:
        reasons.append("forecast key set or path is incomplete or foreign")
    required_forecast_columns = {
        "forecast_mwh",
        "actual_load_mwh",
        "is_target_day",
        "evaluated",
        "status",
    }
    if not required_forecast_columns.issubset(forecasts.columns):
        reasons.append("forecast output schema is incomplete")
    if (
        "status" in forecasts
        and forecasts["status"].astype(str).eq("weekly_lag_invalid").any()
    ):
        reasons.append("unexpected weekly mapping invalidity")
    if not forecasts.empty:
        for column in ("forecast_mwh", "actual_load_mwh"):
            if column in forecasts:
                values = pd.to_numeric(forecasts[column], errors="coerce")
                if not np.isfinite(values).all():
                    reasons.append("non-finite forecast output")
                    break
    failures = 0
    if not jobs.empty and "status" in jobs:
        failures = int((~jobs["status"].astype(str).eq("completed")).sum())
        if failures:
            reasons.append("unexpected job failures")
    if not jobs.empty:
        for job in jobs.to_dict("records"):
            try:
                key = _job_key(job)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if str(job.get("status")) == "weekly_lag_invalid":
                reasons.append("unexpected weekly mapping invalidity")
            matching = forecasts.loc[
                forecasts.apply(
                    lambda row, expected=key: _safe_job_key(row) == expected,
                    axis=1,
                )
            ] if not forecasts.empty else forecasts
            target = matching.loc[
                matching.get("is_target_day", pd.Series(False, index=matching.index)).map(
                    _bool_value
                )
                & matching.get("evaluated", pd.Series(False, index=matching.index)).map(
                    _bool_value
                )
                & matching.get("status", pd.Series(dtype=object)).astype(str).eq(
                    "completed"
                )
            ] if not matching.empty else matching
            expected_observations = int(
                workload.loc[
                    workload.apply(lambda row, expected=key: _job_key(row) == expected, axis=1),
                    "expected_observations",
                ].iloc[0]
            ) if key in expected_keys else 0
            if str(job.get("status")) == "completed" and len(target) != expected_observations:
                reasons.append("incorrect target length")
            if str(job.get("status")) == "completed" and not screening._blank(
                job.get("error_message")
            ):
                reasons.append("completed job has a non-empty error")
            if str(job.get("status")) == "completed" and not target.empty:
                timestamps = pd.to_datetime(
                    target["timestamp_utc"], format="mixed", utc=True, errors="coerce"
                )
                expected_timestamps = screening._expected_target_index(
                    key[0], key[1]
                )
                if (
                    timestamps.isna().any()
                    or not timestamps.is_unique
                    or set(timestamps) != set(expected_timestamps)
                ):
                    reasons.append("incorrect target timestamps")
            if key[2] == "weekly_differenced" and not matching.empty:
                if not {
                    "source_timestamp_utc",
                    "source_actual_mwh",
                    "source_kind",
                    "mapping_issue_count",
                }.issubset(matching.columns):
                    reasons.append("weekly source metadata is incomplete")
                try:
                    weekly_metadata_valid = screening._weekly_source_metadata_complete(
                        job,
                        matching,
                        pd.Timestamp(job["forecast_origin_utc"]),
                        source_frames.get(key[0]) if source_frames is not None else None,
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    weekly_metadata_valid = False
                if not weekly_metadata_valid:
                    reasons.append("weekly source metadata is invalid")
    return {
        "valid": not reasons,
        "failures": failures,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _forecast_values(forecasts: pd.DataFrame) -> dict[tuple[str, str, str, str, str], float]:
    values: dict[tuple[str, str, str, str, str], float] = {}
    for row in forecasts.to_dict("records"):
        values[_forecast_key(row)] = float(row["forecast_mwh"])
    return values


def _compare_forecasts(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[float, float, bool, list[str]]:
    reasons: list[str] = []
    try:
        baseline_values = _forecast_values(baseline)
        current_values = _forecast_values(current)
    except (KeyError, TypeError, ValueError, OverflowError):
        return float("nan"), float("nan"), False, ["forecast comparison keys are malformed"]
    if set(baseline_values) != set(current_values):
        reasons.append("forecast key sets differ")
        return float("nan"), float("nan"), False, reasons
    if not baseline_values:
        return float("nan"), float("nan"), False, ["forecast output is empty"]
    base = np.asarray([baseline_values[key] for key in sorted(baseline_values)], dtype=float)
    actual = np.asarray([current_values[key] for key in sorted(baseline_values)], dtype=float)
    if not np.isfinite(base).all() or not np.isfinite(actual).all():
        return float("nan"), float("nan"), False, ["non-finite forecast output"]
    difference = np.abs(actual - base)
    denominator = np.abs(base)
    relative = np.divide(
        difference,
        denominator,
        out=np.where(difference == 0, 0.0, np.inf),
        where=denominator != 0,
    )
    maximum_absolute = float(np.max(difference))
    maximum_relative = float(np.max(relative))
    equivalent = bool(
        np.allclose(
            actual,
            base,
            rtol=EQUIVALENCE_RTOL,
            atol=EQUIVALENCE_ATOL,
            equal_nan=False,
        )
    )
    if not equivalent:
        reasons.append("forecast equivalence failed")
    return maximum_absolute, maximum_relative, equivalent, reasons


def _summary_row(
    requested_workers: int,
    result: Mapping[str, object] | None,
    workload: pd.DataFrame,
    baseline: pd.DataFrame | None,
    elapsed_seconds: float,
    supported: bool,
    support_reason: str,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    if not supported:
        return {
            "requested_workers": requested_workers,
            "active_workers": 0,
            "supported": False,
            "status": "unsupported",
            "elapsed_seconds": np.nan,
            "jobs_per_minute": np.nan,
            "failures": np.nan,
            "peak_rss_bytes": np.nan,
            "peak_swap_bytes": np.nan,
            "resource_status": "unavailable",
            "max_absolute_forecast_difference": np.nan,
            "max_relative_forecast_difference": np.nan,
            "equivalence": False,
            "safe": False,
            "reason": support_reason,
        }
    result = result or {}
    jobs = result.get("jobs", pd.DataFrame())
    forecasts = result.get("forecasts", pd.DataFrame())
    if not isinstance(jobs, pd.DataFrame):
        jobs = pd.DataFrame(jobs)
    if not isinstance(forecasts, pd.DataFrame):
        forecasts = pd.DataFrame(forecasts)
    validation = _validate_run(
        workload, jobs, forecasts, source_frames=source_frames
    )
    if baseline is None:
        equivalent = bool(validation["valid"])
        max_absolute = 0.0 if equivalent else np.nan
        max_relative = 0.0 if equivalent else np.nan
        comparison_reasons: list[str] = []
    else:
        max_absolute, max_relative, equivalent, comparison_reasons = _compare_forecasts(
            baseline, forecasts
        )
    reasons = list(validation["reasons"]) + comparison_reasons
    error = str(result.get("error", "")).strip()
    if error:
        reasons.append(error)
    resource_status = str(
        result.get(
            "resource_status",
            "available"
            if result.get("peak_rss_bytes") is not None
            and result.get("peak_swap_bytes") is not None
            else "unavailable",
        )
    )
    resource_exhausted = _resource_exhausted(result)
    if resource_exhausted:
        reasons.append("resource exhaustion")
    if not resource_status == "available":
        reasons.append("resource telemetry unavailable")
    safe = bool(validation["valid"] and equivalent and not error and not resource_exhausted)
    if not safe and not reasons:
        reasons.append("worker run is unsafe")
    return {
        "requested_workers": requested_workers,
        "active_workers": _active_worker_count(result.get("active_workers", 0)),
        "supported": True,
        "status": "supported",
        "elapsed_seconds": elapsed_seconds,
        "jobs_per_minute": float(len(workload) / elapsed_seconds * 60)
        if elapsed_seconds > 0
        else np.nan,
        "failures": int(validation["failures"]),
        "peak_rss_bytes": result.get("peak_rss_bytes", np.nan),
        "peak_swap_bytes": result.get("peak_swap_bytes", np.nan),
        "resource_status": resource_status,
        "max_absolute_forecast_difference": max_absolute,
        "max_relative_forecast_difference": max_relative,
        "equivalence": equivalent,
        "safe": safe,
        "reason": "; ".join(dict.fromkeys(str(reason) for reason in reasons if reason))
        or "safe",
    }


def _active_worker_count(value: object) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return int(number) if np.isfinite(number) and number >= 0 else 0


def _resource_exhausted(result: Mapping[str, object]) -> bool:
    if _bool_value(result.get("resource_exhausted", False)):
        return True
    try:
        peak_swap = float(result.get("peak_swap_bytes", 0))
    except (TypeError, ValueError, OverflowError):
        peak_swap = 0.0
    return bool(np.isfinite(peak_swap) and peak_swap > 0)


def select_fastest_safe_worker(summary: pd.DataFrame) -> int:
    required = {"requested_workers", "supported", "safe", "elapsed_seconds"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"worker summary is missing required columns: {missing}")
    supported = summary.loc[
        summary["supported"].map(_bool_value)
        & summary["safe"].map(_bool_value)
    ].copy()
    if supported.empty:
        raise ValueError("no safe supported worker count")
    elapsed = pd.to_numeric(supported["elapsed_seconds"], errors="coerce")
    supported = supported.loc[np.isfinite(elapsed)].copy()
    if supported.empty:
        raise ValueError("no safe supported worker count has finite elapsed time")
    supported["_elapsed"] = pd.to_numeric(
        supported["elapsed_seconds"], errors="raise"
    )
    return int(
        supported.sort_values(
            ["_elapsed", "requested_workers"], kind="stable"
        ).iloc[0]["requested_workers"]
    )


def _update_run_manifest(output_root: Path, selected_worker: int) -> None:
    path = output_root / "audit" / "enhanced_arima_run_manifest.csv"
    if path.exists():
        manifest = pd.read_csv(path)
    else:
        manifest = screening._run_manifest_frame()
    if manifest.empty:
        manifest = screening._run_manifest_frame()
    if "selected_worker" not in manifest:
        manifest["selected_worker"] = ""
    manifest["selected_worker"] = manifest["selected_worker"].astype(object)
    manifest.loc[manifest.index[0], "selected_worker"] = selected_worker
    _atomic_write_csv(manifest, path)


def run_worker_benchmark(
    shortlist: pd.DataFrame,
    output_root: str | Path,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    """Run the fixed valid workload for supported worker counts 1 through 8."""
    output = Path(output_root)
    benchmark_output = output / "benchmark"
    frames = _load_benchmark_frames(processed_directory)
    invalid_dates = _derive_invalid_dates(frames)
    workload = build_benchmark_workload(shortlist, invalid_dates)
    workload_path = benchmark_output / "workload.csv"
    _atomic_write_csv(workload, workload_path)
    persisted_workload = pd.read_csv(workload_path)

    rows: list[dict[str, object]] = []
    baseline_forecasts: pd.DataFrame | None = None
    for requested_workers in WORKER_COUNTS:
        supported, support_reason = _worker_count_support(requested_workers)
        worker_output = benchmark_output / f"workers_{requested_workers}"
        worker_output.mkdir(parents=True, exist_ok=True)
        if not supported:
            row = _summary_row(
                requested_workers,
                None,
                persisted_workload,
                baseline_forecasts,
                0.0,
                False,
                support_reason,
            )
            _atomic_write_csv(pd.DataFrame([row], columns=SUMMARY_COLUMNS), worker_output / "result.csv")
            rows.append(row)
            continue

        started = time.perf_counter()
        try:
            result = _run_worker_count(
                persisted_workload,
                requested_workers,
                frames,
                processed_directory,
            )
        except Exception as error:
            result = {
                "jobs": pd.DataFrame(),
                "forecasts": pd.DataFrame(),
                "error": f"{type(error).__name__}: {error}",
                "resource_status": "unavailable",
            }
        elapsed = time.perf_counter() - started
        jobs = result.get("jobs", pd.DataFrame())
        forecasts = result.get("forecasts", pd.DataFrame())
        if not isinstance(jobs, pd.DataFrame):
            jobs = pd.DataFrame(jobs)
        if not isinstance(forecasts, pd.DataFrame):
            forecasts = pd.DataFrame(forecasts)
        _atomic_write_csv(jobs, worker_output / "jobs.csv")
        _atomic_write_csv(forecasts, worker_output / "forecasts.csv")
        if requested_workers == 1:
            baseline_forecasts = forecasts.copy()
        row = _summary_row(
            requested_workers,
            result,
            persisted_workload,
            None
            if requested_workers == 1
            else baseline_forecasts
            if baseline_forecasts is not None
            else pd.DataFrame(),
            elapsed,
            True,
            "",
            source_frames=frames,
        )
        _atomic_write_csv(pd.DataFrame([row], columns=SUMMARY_COLUMNS), worker_output / "result.csv")
        rows.append(row)

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    _atomic_write_csv(summary, benchmark_output / "worker_benchmark_summary.csv")
    try:
        selected_worker = select_fastest_safe_worker(summary)
    except ValueError:
        selection = pd.DataFrame(
            [{"selection_status": "no_safe_supported_worker", "selected_worker": np.nan}]
        )
    else:
        selection = pd.DataFrame(
            [{"selection_status": "selected", "selected_worker": selected_worker}]
        )
        _update_run_manifest(output, selected_worker)
    _atomic_write_csv(selection, benchmark_output / "worker_selection.csv")
    return summary


if __name__ == "__main__":
    raise SystemExit("Use run_worker_benchmark from Python; production execution is explicit.")
