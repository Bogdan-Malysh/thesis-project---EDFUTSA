from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import PROCESSED
from sarimax_models import (
    SARIMAX_BENCHMARK_FIT_KWARGS,
    build_exog,
    fit_sarimax,
    order_for_country,
)
from sarimax_state_update import build_initial_training_set
from sarimax_validation_2024 import load_validation_country_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_full_fit_benchmark.json"
)
RUNTIME_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_fixed_state_runtime_segments.jsonl"
)


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_bytes() -> int:
    if hasattr(ctypes, "windll"):
        process = ctypes.windll.kernel32.GetCurrentProcess()
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
        api.restype = wintypes.BOOL
        if not api(process, ctypes.byref(counters), counters.cb):
            return -1
        return int(counters.WorkingSetSize)

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return -1
    return -1


def benchmark_row(
    fit: object,
    *,
    observations: int,
    fit_seconds: float,
    rss_start_bytes: int,
    rss_finish_bytes: int,
    fit_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    base_fit = fit.sarima_fit
    result = fit.fitted_result
    options = fit_kwargs or SARIMAX_BENCHMARK_FIT_KWARGS
    return {
        "observations": observations,
        "parameters": base_fit.n_params,
        "state_dimension": result.model.ssm.k_states,
        "optimizer_method": options["method"],
        "function_evaluations": base_fit.optimizer_function_calls,
        "iterations": base_fit.optimizer_iterations,
        "converged": base_fit.converged,
        "termination_message": base_fit.optimizer_message,
        "optimizer_status": base_fit.optimizer_status,
        "optimizer_warnflag": base_fit.optimizer_warnflag,
        "parameters_finite": base_fit.parameters_finite,
        "stationarity_ok": base_fit.stationarity_ok,
        "invertibility_ok": base_fit.invertibility_ok,
        "gradient_norm": base_fit.optimizer_gradient_norm,
        "maxiter_limit_reached": (
            base_fit.optimizer_iterations is not None
            and base_fit.optimizer_iterations >= int(options["maxiter"])
            and not base_fit.converged
        ),
        "maxfun_limit_reached": (
            base_fit.optimizer_function_calls is not None
            and base_fit.optimizer_function_calls >= int(options["maxfun"])
            and not base_fit.converged
        ),
        "log_likelihood": base_fit.log_likelihood,
        "fit_seconds": fit_seconds,
        "rss_start_bytes": rss_start_bytes,
        "rss_finish_bytes": rss_finish_bytes,
        "covariance_skipped": (
            getattr(result, "cov_type", None) == "none"
            and not base_fit.standard_errors_finite
        ),
    }


def append_runtime(row: dict[str, object]) -> None:
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    frame = load_validation_country_data("Germany", PROCESSED)
    training, _ = build_initial_training_set(frame, "Germany")
    exog = build_exog(training["timestamp_utc"], "Germany", "sarimax_calendar")
    exog.index = training["timestamp_utc"]
    values = pd.Series(
        training["actual_load_mwh"].to_numpy(dtype=float),
        index=training["timestamp_utc"],
    )

    segment_id = f"full-fit-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
    segment_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rss_start = working_set_bytes()
    append_runtime(
        {
            "segment_id": segment_id,
            "event": "start",
            "segment_start_utc": segment_start,
            "workers": 1,
            "memory_working_set_start_bytes": rss_start,
        }
    )
    started = perf_counter()
    try:
        fit = fit_sarimax(
            values,
            exog,
            order_for_country("Germany"),
            fit_kwargs=SARIMAX_BENCHMARK_FIT_KWARGS,
        )
    finally:
        fit_seconds = perf_counter() - started
        rss_finish = working_set_bytes()
        append_runtime(
            {
                "segment_id": segment_id,
                "event": "finish",
                "segment_finish_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "segment_wall_clock_seconds": fit_seconds,
                "workers": 1,
                "memory_working_set_finish_bytes": rss_finish,
            }
        )
    row = benchmark_row(
        fit,
        observations=len(training),
        fit_seconds=fit_seconds,
        rss_start_bytes=rss_start,
        rss_finish_bytes=rss_finish,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    if not fit.eligible:
        raise SystemExit(f"SARIMAX benchmark fit is not eligible: {fit.error_message}")


if __name__ == "__main__":
    main()
