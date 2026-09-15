from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
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
from benchmark_sarimax_fit_2024 import (
    OUTPUT_PATH as PREVIOUS_OUTPUT_PATH,
    RUNTIME_PATH,
    append_runtime,
    benchmark_row,
    working_set_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_full_fit_continuation_benchmark.json"
)
CONTINUATION_FIT_KWARGS = {
    **SARIMAX_BENCHMARK_FIT_KWARGS,
    "maxiter": 150,
    "maxfun": 10000,
}


def load_warm_start(path: Path) -> tuple[np.ndarray | None, str]:
    if not path.exists():
        return None, "50-iteration benchmark output is absent"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("parameter_vector", "start_params", "params"):
        candidate = payload.get(key)
        if not isinstance(candidate, list):
            continue
        try:
            values = np.asarray(candidate, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(values) and np.isfinite(values).all():
            return values, f"loaded from {path.name}:{key}"
    return None, "50-iteration benchmark output did not persist a parameter vector"


def main() -> None:
    warm_start, warm_start_reason = load_warm_start(PREVIOUS_OUTPUT_PATH)
    previous = json.loads(PREVIOUS_OUTPUT_PATH.read_text(encoding="utf-8"))
    fit_kwargs = dict(CONTINUATION_FIT_KWARGS)
    if warm_start is not None:
        fit_kwargs["start_params"] = warm_start

    frame = load_validation_country_data("Germany", PROCESSED)
    training, _ = build_initial_training_set(frame, "Germany")
    exog = build_exog(training["timestamp_utc"], "Germany", "sarimax_calendar")
    exog.index = training["timestamp_utc"]
    values = pd.Series(
        training["actual_load_mwh"].to_numpy(dtype=float),
        index=training["timestamp_utc"],
    )

    segment_id = (
        "full-fit-continuation-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
    )
    rss_start = working_set_bytes()
    append_runtime(
        {
            "segment_id": segment_id,
            "event": "start",
            "segment_start_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "workers": 1,
            "memory_working_set_start_bytes": rss_start,
            "warm_start_used": warm_start is not None,
        }
    )
    started = perf_counter()
    try:
        fit = fit_sarimax(
            values,
            exog,
            order_for_country("Germany"),
            fit_kwargs=fit_kwargs,
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
                "warm_start_used": warm_start is not None,
            }
        )

    row = benchmark_row(
        fit,
        observations=len(training),
        fit_seconds=fit_seconds,
        rss_start_bytes=rss_start,
        rss_finish_bytes=rss_finish,
        fit_kwargs=CONTINUATION_FIT_KWARGS,
    )
    base_fit = fit.sarima_fit
    previous_log_likelihood = previous.get("log_likelihood")
    row.update(
        {
            "warm_start_used": warm_start is not None,
            "warm_start_reason": warm_start_reason,
            "previous_iterations": previous.get("iterations"),
            "previous_function_evaluations": previous.get("function_evaluations"),
            "additional_iterations": (
                base_fit.optimizer_iterations - int(previous["iterations"])
                if warm_start is not None and base_fit.optimizer_iterations is not None
                else None
            ),
            "additional_function_evaluations": (
                base_fit.optimizer_function_calls - int(previous["function_evaluations"])
                if warm_start is not None and base_fit.optimizer_function_calls is not None
                else None
            ),
            "previous_log_likelihood": previous_log_likelihood,
            "log_likelihood_change_vs_50_iteration": (
                base_fit.log_likelihood - float(previous_log_likelihood)
                if previous_log_likelihood is not None
                else None
            ),
        }
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    if not fit.eligible:
        raise SystemExit(f"SARIMAX continuation benchmark is not eligible: {fit.error_message}")


if __name__ == "__main__":
    main()
