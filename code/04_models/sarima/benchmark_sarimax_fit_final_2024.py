from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
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
    OUTPUT_PATH as BENCHMARK_OUTPUT_PATH,
    RUNTIME_PATH,
    append_runtime,
    benchmark_row,
    working_set_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREVIOUS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_full_fit_continuation_benchmark.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_full_fit_final_benchmark.json"
)
PROGRESS_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_full_fit_progress.json"
)
FINAL_PARAMETER_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_full_fit_final_parameter_vector.json"
)
FINAL_FIT_KWARGS = {
    **SARIMAX_BENCHMARK_FIT_KWARGS,
    "maxiter": 500,
    "maxfun": 30000,
}


def load_previous_start(path: Path) -> tuple[np.ndarray | None, str]:
    if not path.exists():
        return None, "150-iteration benchmark output is absent"
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
    return None, "150-iteration benchmark output did not persist a parameter vector"


def _root_validity(
    param_names: list[str],
    params: np.ndarray,
    prefix: str,
    sign: float,
) -> bool:
    coefficients: dict[int, float] = {}
    pattern = re.compile(rf"^{re.escape(prefix)}\.L(\d+)$")
    for name, value in zip(param_names, params):
        match = pattern.fullmatch(name)
        if match is not None:
            coefficients[int(match.group(1))] = float(value)
    if not coefficients:
        return True
    degree = max(coefficients)
    polynomial = np.ones(degree + 1, dtype=float)
    for lag, value in coefficients.items():
        polynomial[lag] = sign * value
    try:
        roots = np.roots(polynomial[::-1])
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return False
    return bool(len(roots) and np.isfinite(roots).all() and (np.abs(roots) > 1.01).all())


def valid_parameter_vector(model: object, params: np.ndarray) -> bool:
    names = [str(name) for name in model.param_names]
    if len(params) != len(names) or not np.isfinite(params).all():
        return False
    return (
        _root_validity(names, params, "ar", -1.0)
        and _root_validity(names, params, "ma", 1.0)
    )


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.iteration = 0

    def __call__(self, model: object, unconstrained_params: np.ndarray) -> None:
        self.iteration += 1
        if self.iteration % 10:
            return
        try:
            transformed = np.asarray(
                model.transform_params(unconstrained_params),
                dtype=float,
            ).reshape(-1)
            if not valid_parameter_vector(model, transformed):
                return
            payload = {
                "source": "optimizer_callback",
                "optimizer_iteration": self.iteration,
                "saved_utc": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "parameter_vector": transformed.tolist(),
            }
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
        except Exception:
            return


def persist_final_parameters(fit: object) -> None:
    result = fit.fitted_result
    params = np.asarray(result.params, dtype=float).reshape(-1)
    payload = {
        "source": "final_result",
        "saved_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "parameter_vector": params.tolist(),
        "parameters_finite": bool(np.isfinite(params).all()),
    }
    FINAL_PARAMETER_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_PARAMETER_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    start_params, warm_start_reason = load_previous_start(PREVIOUS_OUTPUT_PATH)
    fit_kwargs = dict(FINAL_FIT_KWARGS)
    if start_params is not None:
        fit_kwargs["start_params"] = start_params

    frame = load_validation_country_data("Germany", PROCESSED)
    training, _ = build_initial_training_set(frame, "Germany")
    exog = build_exog(training["timestamp_utc"], "Germany", "sarimax_calendar")
    exog.index = training["timestamp_utc"]
    values = pd.Series(
        training["actual_load_mwh"].to_numpy(dtype=float),
        index=training["timestamp_utc"],
    )

    progress = ProgressWriter(PROGRESS_PATH)
    segment_id = (
        "full-fit-final-"
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
            "warm_start_used": start_params is not None,
        }
    )
    started = perf_counter()
    try:
        fit = fit_sarimax(
            values,
            exog,
            order_for_country("Germany"),
            fit_kwargs=fit_kwargs,
            progress_callback=progress,
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
                "warm_start_used": start_params is not None,
            }
        )

    persist_final_parameters(fit)
    row = benchmark_row(
        fit,
        observations=len(training),
        fit_seconds=fit_seconds,
        rss_start_bytes=rss_start,
        rss_finish_bytes=rss_finish,
        fit_kwargs=FINAL_FIT_KWARGS,
    )
    row.update(
        {
            "warm_start_used": start_params is not None,
            "warm_start_reason": warm_start_reason,
            "final_parameter_vector_path": str(FINAL_PARAMETER_PATH),
            "progress_parameter_vector_path": str(PROGRESS_PATH),
        }
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2))
    if not fit.eligible:
        raise SystemExit(f"Final SARIMAX benchmark is not eligible: {fit.error_message}")


if __name__ == "__main__":
    main()
