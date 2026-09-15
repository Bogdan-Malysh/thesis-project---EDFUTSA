from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import sarimax_state_validation_2024 as state_validation
import sarimax_validation_2024 as sarimax_validation
from benchmark_sarimax_fit_2024 import working_set_bytes
from benchmark_sarimax_fit_final_2024 import valid_parameter_vector
from common.forecasting_framework import COUNTRY_CONFIG, PROCESSED, extract_target_day
from sarimax_models import (
    SARIMAX_SPECIFICATION_IDS,
    build_exog,
    order_for_country,
    specification_columns,
)
from sarimax_state_update import build_initial_training_set


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "fixed_state_mini_four_workers"
)
FINAL_PARAMETER_PATH = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "sarimax_full_fit_final_parameter_vector.json"
)
RUNTIME_FILENAME = "sarimax_fixed_state_mini_worker_runtime.jsonl"
CHAIN_FILENAME = "sarimax_fixed_state_mini_chain_summary.json"
RUN_FILENAME = "sarimax_fixed_state_mini_run.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: object) -> str:
    return json.dumps(value, default=str, allow_nan=True, sort_keys=True)


def _model_parameter_spec(
    frame: pd.DataFrame,
    country: str,
    specification_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    training, _ = build_initial_training_set(frame, country)
    exog = build_exog(training["timestamp_utc"], country, specification_id)
    model = SARIMAX(
        training["actual_load_mwh"].to_numpy(dtype=float),
        exog=exog,
        order=order_for_country(country).nonseasonal_order,
        seasonal_order=order_for_country(country).seasonal_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return tuple(str(name) for name in model.param_names), tuple(str(column) for column in exog)


def _load_final_parameter_vector(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    payload = json.loads(FINAL_PARAMETER_PATH.read_text(encoding="utf-8"))
    vector = np.asarray(payload.get("parameter_vector"), dtype=float).reshape(-1)
    names, columns = _model_parameter_spec(frame, "Germany", "sarimax_calendar")
    if len(vector) != len(names) or not np.isfinite(vector).all():
        raise ValueError("persisted Germany calendar vector does not match the exact model size")
    if not payload.get("parameters_finite", False):
        raise ValueError("persisted Germany calendar vector is not marked finite")
    training, _ = build_initial_training_set(frame, "Germany")
    exog = build_exog(training["timestamp_utc"], "Germany", "sarimax_calendar")
    model = SARIMAX(
        training["actual_load_mwh"].to_numpy(dtype=float),
        exog=exog,
        order=order_for_country("Germany").nonseasonal_order,
        seasonal_order=order_for_country("Germany").seasonal_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    if tuple(str(name) for name in model.param_names) != names:
        raise ValueError("Germany calendar parameter ordering is not reproducible")
    if not valid_parameter_vector(model, vector):
        raise ValueError("persisted Germany calendar vector fails finite/root validity checks")
    return vector, names, columns


def _crisis_start_params(
    frame: pd.DataFrame,
    calendar_vector: np.ndarray,
    calendar_names: tuple[str, ...],
) -> tuple[list[float], tuple[str, ...]]:
    crisis_names, _ = _model_parameter_spec(frame, "Germany", "sarimax_calendar_crisis")
    calendar_by_name = dict(zip(calendar_names, calendar_vector, strict=True))
    crisis_only = set(crisis_names).difference(calendar_names)
    if crisis_only != {"is_covid_period", "is_post_invasion"}:
        raise ValueError(f"unexpected Germany crisis parameter mapping: {sorted(crisis_only)}")
    start = [float(calendar_by_name.get(name, 0.0)) for name in crisis_names]
    if not np.isfinite(start).all():
        raise ValueError("Germany crisis informed start contains non-finite values")
    return start, crisis_names


def _chain_worker(task: dict[str, object]) -> dict[str, object]:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    started_utc = _utc_now()
    rss_start = working_set_bytes()
    started = perf_counter()
    country = str(task["country"])
    specification_id = str(task["specification_id"])
    processed_directory = str(task["processed_directory"])
    frame = sarimax_validation.load_validation_country_data(country, processed_directory)
    metadata: dict[str, object] = {}
    results = list(
        state_validation.iter_chain(
            country,
            specification_id,
            [str(value) for value in task["target_dates"]],
            frame,
            set(),
            fit_kwargs=dict(task["fit_kwargs"]),
            fixed_params=task.get("fixed_params"),
            chain_metadata=metadata,
            strict_initial_fit=True,
        )
    )
    if len(results) != 3:
        raise RuntimeError(f"{country} {specification_id} did not produce three mini jobs")
    failed = [result["job"] for result in results if result["job"].get("status") != "completed"]
    if failed:
        raise RuntimeError(
            f"{country} {specification_id} produced failed mini jobs: "
            f"{[row.get('error_message') for row in failed]}"
        )
    fit = metadata["fit"]
    if fit.fitted_result is None or not fit.eligible:
        raise RuntimeError(f"{country} {specification_id} initial fit is not eligible")
    params = np.asarray(fit.fitted_result.params, dtype=float).reshape(-1)
    names = tuple(str(name) for name in getattr(fit.fitted_result, "param_names", ()))
    rss_finish = working_set_bytes()
    elapsed = perf_counter() - started
    return {
        "country": country,
        "specification_id": specification_id,
        "initial_fit_metadata": state_validation._fit_metadata(fit),
        "initial_parameter_vector": params.tolist(),
        "initial_parameter_names": list(names),
        "initial_training_observations": int(metadata["training_observations"]),
        "initial_cutoff": str(metadata["initial_cutoff"]),
        "initial_runtime_seconds": float(metadata["initial_runtime_seconds"]),
        "rolling_state_update_seconds": float(
            sum(float(result["job"].get("state_update_runtime_seconds") or 0.0) for result in results)
        ),
        "rolling_forecast_seconds": float(
            sum(float(result["job"].get("forecast_runtime_seconds") or 0.0) for result in results)
        ),
        "worker_wall_clock_seconds": float(elapsed),
        "worker_started_utc": started_utc,
        "worker_finished_utc": _utc_now(),
        "memory_working_set_start_bytes": int(rss_start),
        "memory_working_set_finish_bytes": int(rss_finish),
        "results": results,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=True) + "\n", encoding="utf-8")


def _persist_result_frames(
    output: Path,
    manifest: pd.DataFrame,
    expected_timestamps: dict[tuple[str, str, str], set[str]],
    chain_results: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs_path = output / state_validation.JOB_FILENAME
    forecasts_path = output / state_validation.FORECAST_FILENAME
    diagnostics_path = output / state_validation.DIAGNOSTIC_FILENAME
    summary_path = output / state_validation.SUMMARY_FILENAME
    jobs = sarimax_validation._load_csv(jobs_path, state_validation.JOB_COLUMNS)
    forecasts = sarimax_validation._load_csv(forecasts_path, state_validation.FORECAST_COLUMNS)
    diagnostics = sarimax_validation._load_csv(diagnostics_path, state_validation.DIAGNOSTIC_COLUMNS)
    for chain in sorted(chain_results, key=lambda value: (str(value["country"]), str(value["specification_id"]))):
        for result in chain["results"]:
            jobs = sarimax_validation.upsert_frame(
                jobs,
                pd.DataFrame([result["job"]]),
                state_validation.JOB_KEY_COLUMNS,
            )
            forecasts = sarimax_validation._without_job_rows(forecasts, result["job"])
            forecasts = sarimax_validation.upsert_frame(
                forecasts,
                pd.DataFrame(result["forecasts"]),
                state_validation.FORECAST_KEY_COLUMNS,
            )
            diagnostics = sarimax_validation.upsert_frame(
                diagnostics,
                pd.DataFrame([result["diagnostics"]]),
                sarimax_validation.DIAGNOSTIC_KEY_COLUMNS,
            )
    for frame, path, columns in (
        (forecasts, forecasts_path, state_validation.FORECAST_COLUMNS),
        (diagnostics, diagnostics_path, state_validation.DIAGNOSTIC_COLUMNS),
        (jobs, jobs_path, state_validation.JOB_COLUMNS),
    ):
        sarimax_validation._atomic_write(frame, path, columns)
    summary = state_validation._summary_frame(
        jobs,
        manifest,
        forecasts,
        diagnostics,
        expected_timestamps,
    )
    state_validation._write_summary(summary, summary_path)
    return jobs, forecasts, diagnostics, summary


def _validate_mini_outputs(
    manifest: pd.DataFrame,
    jobs: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    expected_timestamps: dict[tuple[str, str, str], set[str]],
) -> dict[str, object]:
    expected_keys = {
        tuple(str(row[column]) for column in state_validation.JOB_KEY_COLUMNS)
        for row in manifest.to_dict("records")
    }
    job_keys = {
        tuple(str(row[column]) for column in state_validation.JOB_KEY_COLUMNS)
        for row in jobs.to_dict("records")
    }
    if job_keys != expected_keys or len(jobs) != 12:
        raise AssertionError("mini output does not contain exactly the 12 approved job keys")
    if jobs[list(state_validation.JOB_KEY_COLUMNS)].duplicated().any():
        raise AssertionError("mini jobs contain duplicate keys")
    if not jobs["status"].astype(str).eq("completed").all():
        raise AssertionError("mini output contains failed jobs")
    if not pd.to_numeric(jobs["parameter_vector_unchanged"], errors="coerce").fillna(0).astype(bool).all():
        raise AssertionError("fixed-state parameter invariance was not recorded for every job")
    if jobs["initial_training_end_local"].astype(str).ne("2023-12-31").any():
        raise AssertionError("mini jobs do not use the required initial training cutoff")
    forecast_keys = forecasts[list(state_validation.FORECAST_KEY_COLUMNS)].astype(str)
    if forecast_keys.duplicated().any():
        raise AssertionError("mini forecasts contain duplicate keys")
    forecast_timestamp_sets = sarimax_validation._forecast_timestamps_by_job(forecasts)
    if set(forecast_timestamp_sets) != expected_keys:
        raise AssertionError("mini forecasts are missing job keys")
    for key in expected_keys:
        if forecast_timestamp_sets[key] != expected_timestamps[key]:
            raise AssertionError(f"forecast timestamps do not match the expected path for {key}")
    numeric_forecasts = pd.to_numeric(forecasts["forecast_mwh"], errors="coerce")
    if not np.isfinite(numeric_forecasts.to_numpy(dtype=float)).all():
        raise AssertionError("mini forecasts contain non-finite values")
    if forecasts["local_date"].astype(str).str.startswith("2025").any():
        raise AssertionError("mini forecast paths contain 2025 observations")
    if (pd.to_datetime(forecasts["timestamp_utc"], utc=True) < pd.to_datetime(forecasts["information_cutoff_utc"], utc=True)).any():
        raise AssertionError("mini forecast paths precede their information cutoffs")
    expected_by_key = {
        tuple(str(row[column]) for column in state_validation.JOB_KEY_COLUMNS): int(row["expected_observations"])
        for row in manifest.to_dict("records")
    }
    for row in jobs.to_dict("records"):
        key = tuple(str(row[column]) for column in state_validation.JOB_KEY_COLUMNS)
        if int(row["target_intervals"]) != expected_by_key[key]:
            raise AssertionError(f"target interval count mismatch for {key}")
        if int(row["evaluated_observations"]) != expected_by_key[key] or float(row["coverage"]) != 1.0:
            raise AssertionError(f"target coverage mismatch for {key}")
    if len(diagnostics) != 12 or diagnostics[list(sarimax_validation.DIAGNOSTIC_KEY_COLUMNS)].duplicated().any():
        raise AssertionError("mini diagnostics are incomplete or duplicated")
    expected_lengths = sorted(int(value) for value in manifest["expected_observations"].unique())
    return {
        "total_jobs": int(len(jobs)),
        "completed_jobs": int(jobs["status"].astype(str).eq("completed").sum()),
        "forecast_rows": int(len(forecasts)),
        "diagnostic_rows": int(len(diagnostics)),
        "target_day_lengths": expected_lengths,
        "parameter_vector_unchanged_all": True,
        "coverage_all": True,
        "timestamps_exact": True,
        "no_2025_path_rows": True,
    }


def run_mini(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to mix new mini results with existing files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        country: sarimax_validation.load_validation_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    dates = [value.isoformat() for value in sarimax_validation.MINI_DATES]
    dates_by_country = {country: dates for country in COUNTRY_CONFIG}
    manifest = sarimax_validation.build_manifest(dates_by_country, full_year=False)
    manifest["expected_observations"] = [
        len(extract_target_day(frames[country], target_date))
        for country, target_date in zip(manifest["country"], manifest["target_date"])
    ]
    expected_timestamps = {
        tuple(str(row[column]) for column in state_validation.JOB_KEY_COLUMNS): sarimax_validation.expected_forecast_timestamps(
            frames[str(row["country"])], str(row["country"]), str(row["target_date"])
        )
        for row in manifest.to_dict("records")
    }
    sarimax_validation._atomic_write(
        pd.DataFrame(
            [
                {
                    "specification_order": index,
                    "specification_id": specification_id,
                    "exog_columns": _json_value(list(specification_columns(specification_id))),
                }
                for index, specification_id in enumerate(SARIMAX_SPECIFICATION_IDS, 1)
            ]
        ),
        output / state_validation.SPECIFICATION_FILENAME,
        ["specification_order", "specification_id", "exog_columns"],
    )

    germany_vector, germany_names, germany_columns = _load_final_parameter_vector(frames["Germany"])
    germany_crisis_start, germany_crisis_names = _crisis_start_params(
        frames["Germany"], germany_vector, germany_names
    )
    if germany_columns != _model_parameter_spec(frames["Germany"], "Germany", "sarimax_calendar")[1]:
        raise ValueError("Germany calendar exogenous column ordering changed")
    tasks = [
        {
            "country": "Germany",
            "specification_id": "sarimax_calendar",
            "target_dates": dates,
            "processed_directory": str(processed_directory),
            "fit_kwargs": dict(state_validation.STATE_INITIAL_FIT_KWARGS),
            "fixed_params": germany_vector.tolist(),
            "parameter_source": "reused_final_germany_calendar_vector",
        },
        {
            "country": "Germany",
            "specification_id": "sarimax_calendar_crisis",
            "target_dates": dates,
            "processed_directory": str(processed_directory),
            "fit_kwargs": {
                **state_validation.STATE_INITIAL_FIT_KWARGS,
                "start_params": germany_crisis_start,
            },
            "fixed_params": None,
            "parameter_source": "joint_crisis_optimization_with_mapped_calendar_start",
        },
        {
            "country": "Austria",
            "specification_id": "sarimax_calendar",
            "target_dates": dates,
            "processed_directory": str(processed_directory),
            "fit_kwargs": dict(state_validation.STATE_INITIAL_FIT_KWARGS),
            "fixed_params": None,
            "parameter_source": "independent_austria_optimization",
        },
        {
            "country": "Austria",
            "specification_id": "sarimax_calendar_crisis",
            "target_dates": dates,
            "processed_directory": str(processed_directory),
            "fit_kwargs": dict(state_validation.STATE_INITIAL_FIT_KWARGS),
            "fixed_params": None,
            "parameter_source": "independent_austria_optimization",
        },
    ]
    started_utc = _utc_now()
    started = perf_counter()
    chain_results: list[dict[str, object]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        futures = [executor.submit(_chain_worker, task) for task in tasks]
        try:
            for future in as_completed(futures):
                chain_results.append(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            raise
    elapsed = perf_counter() - started
    chain_results.sort(key=lambda value: (str(value["country"]), str(value["specification_id"])))

    parameter_directory = output / "initial_parameters"
    for chain, task in zip(chain_results, sorted(tasks, key=lambda value: (str(value["country"]), str(value["specification_id"]))), strict=True):
        _write_json(
            parameter_directory / f"{chain['country'].lower()}_{chain['specification_id']}.json",
            {
                "country": chain["country"],
                "specification_id": chain["specification_id"],
                "parameter_source": task["parameter_source"],
                "parameter_names": chain["initial_parameter_names"],
                "parameter_vector": chain["initial_parameter_vector"],
                "fit_kwargs": task["fit_kwargs"],
                "fit_metadata": chain["initial_fit_metadata"],
                "training_observations": chain["initial_training_observations"],
                "initial_cutoff": chain["initial_cutoff"],
            },
        )
    _write_json(
        output / CHAIN_FILENAME,
        [
            {key: value for key, value in chain.items() if key != "results"}
            for chain in chain_results
        ],
    )
    with (output / RUNTIME_FILENAME).open("w", encoding="utf-8") as handle:
        for chain in chain_results:
            handle.write(json.dumps({key: value for key, value in chain.items() if key != "results"}, default=str) + "\n")

    jobs, forecasts, diagnostics, summary = _persist_result_frames(
        output,
        manifest,
        expected_timestamps,
        chain_results,
    )
    checks = _validate_mini_outputs(
        manifest,
        jobs,
        forecasts,
        diagnostics,
        expected_timestamps,
    )
    report = {
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "workers": 4,
        "thread_environment": {
            variable: os.environ[variable]
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "initial_fit_kwargs": state_validation.STATE_INITIAL_FIT_KWARGS,
        "germany_calendar_parameter_names": list(germany_names),
        "germany_crisis_parameter_names": list(germany_crisis_names),
        "summary": summary.to_dict("records"),
        "checks": checks,
        "output_directory": str(output),
    }
    _write_json(output / RUN_FILENAME, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-worker fixed-state SARIMAX mini")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    args = parser.parse_args()
    print(json.dumps(run_mini(args.output_directory, args.processed_directory), indent=2, default=str))


if __name__ == "__main__":
    main()
