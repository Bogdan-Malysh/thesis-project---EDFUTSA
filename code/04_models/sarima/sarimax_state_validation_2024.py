from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from datetime import date, datetime
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import sarimax_validation_2024 as sarimax_validation
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    PROCESSED,
    _as_prepared,
    build_information_context,
    evaluate_target_day,
    extract_target_day,
    information_set,
)
from sarima_models import forecast_values
from sarimax_models import (
    SARIMAX_BENCHMARK_FIT_KWARGS,
    build_exog,
    fit_sarimax,
    order_for_country,
    specification_columns,
)
from sarimax_state_update import (
    FixedParameterState,
    build_initial_training_set,
    observations_between_cutoffs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_FAMILY = "SARIMAX"
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "arima_sarimax"
    / "validation_2024"
    / "fixed_state_validation"
)
FULL_OUTPUT_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY
JOB_KEY_COLUMNS = sarimax_validation.JOB_KEY_COLUMNS
FORECAST_KEY_COLUMNS = sarimax_validation.FORECAST_KEY_COLUMNS
EXTRA_COLUMNS = (
    "initial_training_end_local",
    "initial_estimation_runtime_seconds",
    "state_update_observations",
    "state_update_runtime_seconds",
    "forecast_runtime_seconds",
    "state_update_cutoff_utc",
    "parameter_vector_unchanged",
)
JOB_COLUMNS = sarimax_validation.JOB_COLUMNS + list(EXTRA_COLUMNS)
FORECAST_COLUMNS = sarimax_validation.FORECAST_COLUMNS
DIAGNOSTIC_COLUMNS = sarimax_validation.DIAGNOSTIC_COLUMNS + list(EXTRA_COLUMNS)
JOB_FILENAME = "sarimax_fixed_state_validation_2024_jobs.csv"
FORECAST_FILENAME = "sarimax_fixed_state_validation_2024_forecasts.csv"
DIAGNOSTIC_FILENAME = "sarimax_fixed_state_validation_2024_diagnostics.csv"
SUMMARY_FILENAME = "sarimax_fixed_state_validation_2024_summary.csv"
SPECIFICATION_FILENAME = "sarimax_fixed_state_specifications_2024.csv"
STATE_INITIAL_FIT_KWARGS = {
    **SARIMAX_BENCHMARK_FIT_KWARGS,
    "maxiter": 500,
    "maxfun": 30000,
    # statsmodels.extend() requires the filtered state discarded by low_memory=True.
    "low_memory": False,
}


def _json_value(value: object) -> str:
    return json.dumps(value, default=str, allow_nan=True, sort_keys=True)


def _coerce_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _parameter_metadata(result: object) -> tuple[str, str]:
    if result is None:
        return _json_value({}), _json_value({})
    params = np.asarray(getattr(result, "params", []), dtype=float).reshape(-1)
    bse = np.asarray(getattr(result, "bse", []), dtype=float).reshape(-1)
    names = [str(name) for name in getattr(result, "param_names", [])]
    if len(names) != len(params):
        names = [f"parameter_{index}" for index in range(len(params))]
    return (
        _json_value({name: float(value) for name, value in zip(names, params)}),
        _json_value({name: float(value) for name, value in zip(names[: len(bse)], bse)}),
    )


def _fit_metadata(fit: object) -> dict[str, object]:
    base = sarimax_validation._fit_metadata(fit)
    estimates, errors = _parameter_metadata(fit.fitted_result)
    base.update(
        {
            "exog_columns": _json_value(list(fit.exog_columns)),
            "exog_rank": fit.exog_rank,
            "exog_condition_number": fit.exog_condition_number,
            "transformed_exog_rank": fit.transformed_exog_rank,
            "parameter_estimates": estimates,
            "standard_errors": errors,
        }
    )
    return base


def _diagnostic_row(job: Mapping[str, object]) -> dict[str, object]:
    row = sarimax_validation._diagnostic_row(job)
    for column in EXTRA_COLUMNS:
        row[column] = job.get(column)
    return row


def _forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: str,
    specification_id: str,
    state_result: object,
    context: object,
    order: object,
) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    target = extract_target_day(prepared, target_date)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(context.cutoff_utc)
        & prepared["local_date"].le(target_date)
    ].copy()
    if path.empty or not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError("fixed-state SARIMAX path does not contain the complete target day")
    sarimax_validation._validate_hourly_forecast_window(
        path, target, context.cutoff_utc, target_date
    )
    exog = build_exog(path["timestamp_utc"], country, specification_id)
    if tuple(exog.columns) != specification_columns(specification_id):
        raise ValueError("fixed-state forecast exog columns are misaligned")
    exog.index = path["timestamp_utc"]
    forecasts = forecast_values(state_result, len(path), exog=exog)
    path["country"] = country
    path["model_family"] = MODEL_FAMILY
    path["target_date"] = target_date
    path["specification_id"] = specification_id
    path["p"] = order.p
    path["d"] = order.d
    path["q"] = order.q
    path["P"] = order.P
    path["D"] = order.D
    path["Q"] = order.Q
    path["seasonal_period"] = order.seasonal_period
    path["trend"] = "n"
    path["forecast_origin_local"] = context.origin_local
    path["forecast_origin_utc"] = context.cutoff_utc.isoformat()
    path["information_cutoff_utc"] = context.cutoff_utc.isoformat()
    path["forecast_mwh"] = forecasts
    path["bridge_used"] = ~path["local_date"].eq(target_date)
    path["source_kind"] = "sarimax_fixed_state_forecast"
    path["is_target_day"] = path["local_date"].eq(target_date)
    path["evaluated"] = path["is_target_day"]
    return path[FORECAST_COLUMNS]


def _failed_result(row: dict[str, object], error: Exception | str) -> dict[str, object]:
    row.update(
        {
            "status": "failed",
            "error_message": str(error),
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "target_intervals": 0,
            "path_intervals": 0,
            "bridge_intervals": 0,
            "missing_predictions": 0,
            "evaluated_observations": 0,
            "coverage": 0.0,
        }
    )
    return {"job": row, "forecasts": [], "diagnostics": _diagnostic_row(row)}


def iter_chain(
    country: str,
    specification_id: str,
    target_dates: Iterable[str],
    frame: pd.DataFrame,
    completed_keys: set[tuple[str, str, str]],
    *,
    fit_kwargs: Mapping[str, object] | None = None,
    fixed_params: object | None = None,
    chain_metadata: dict[str, object] | None = None,
    strict_initial_fit: bool = False,
):
    training, initial_cutoff = build_initial_training_set(frame, country)
    training_exog = build_exog(training["timestamp_utc"], country, specification_id)
    training_exog.index = training["timestamp_utc"]
    order = order_for_country(country)
    fit_started = perf_counter()
    fit = fit_sarimax(
        pd.Series(
            training["actual_load_mwh"].to_numpy(dtype=float),
            index=training["timestamp_utc"],
        ),
        training_exog,
        order,
        fit_kwargs=dict(fit_kwargs or STATE_INITIAL_FIT_KWARGS),
        fixed_params=fixed_params,
    )
    initial_runtime = perf_counter() - fit_started
    if chain_metadata is not None:
        chain_metadata.update(
            {
                "fit": fit,
                "training_observations": len(training),
                "initial_cutoff": initial_cutoff,
                "initial_runtime_seconds": initial_runtime,
            }
        )
    if strict_initial_fit and not fit.eligible:
        raise RuntimeError(f"initial {country} {specification_id} fit failed: {fit.error_message}")
    fit_metadata = _fit_metadata(fit)
    state = (
        FixedParameterState(
            result=fit.fitted_result,
            parameter_vector=np.asarray(fit.fitted_result.params, dtype=float).copy(),
        )
        if fit.eligible and fit.fitted_result is not None
        else None
    )
    previous_cutoff = initial_cutoff
    for target_date in sorted(str(value) for value in target_dates):
        key = (country, target_date, specification_id)
        row: dict[str, object] = {
            "country": country,
            "target_date": target_date,
            "specification_id": specification_id,
            "specification_order": sarimax_validation.SARIMAX_SPECIFICATION_IDS.index(
                specification_id
            )
            + 1,
            "initial_training_end_local": "2023-12-31",
            "initial_estimation_runtime_seconds": initial_runtime,
            **fit_metadata,
        }
        context = build_information_context(frame, country, target_date)
        row["forecast_origin_local"] = context.origin_local
        row["forecast_origin_utc"] = context.cutoff_utc.isoformat()
        row["information_cutoff_utc"] = context.cutoff_utc.isoformat()
        row["training_observations"] = len(training)
        row["expected_observations"] = len(extract_target_day(frame, target_date))
        if state is None:
            yield _failed_result(row, fit.error_message or "initial fixed-state SARIMAX fit failed")
            previous_cutoff = context.cutoff_utc
            continue
        update_rows = observations_between_cutoffs(frame, previous_cutoff, context.cutoff_utc)
        update_started = perf_counter()
        unchanged = True
        try:
            if not update_rows.empty:
                update_exog = build_exog(update_rows["timestamp_utc"], country, specification_id)
                update_exog.index = update_rows["timestamp_utc"]
                update_endog = pd.Series(
                    update_rows["actual_load_mwh"].to_numpy(dtype=float),
                    index=update_rows["timestamp_utc"],
                )
                state.update(update_endog, update_exog)
            row.update(
                {
                    "state_update_observations": len(update_rows),
                    "state_update_runtime_seconds": perf_counter() - update_started,
                    "state_update_cutoff_utc": context.cutoff_utc.isoformat(),
                    "parameter_vector_unchanged": unchanged,
                }
            )
            forecast_started = perf_counter()
            path = _forecast_path(
                frame,
                country,
                target_date,
                specification_id,
                state.result,
                context,
                order,
            )
            forecast_runtime = perf_counter() - forecast_started
            metrics = evaluate_target_day(
                path,
                expected_observations=int(row["expected_observations"]),
            )
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
                    "forecast_runtime_seconds": forecast_runtime,
                }
            )
            result = {
                "job": row,
                "forecasts": path.to_dict("records"),
                "diagnostics": _diagnostic_row(row),
            }
        except Exception as error:
            result = _failed_result(row, error)
        previous_cutoff = context.cutoff_utc
        if key not in completed_keys:
            yield result


def _summary_frame(
    jobs: pd.DataFrame,
    manifest: pd.DataFrame,
    forecasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    expected_timestamps: Mapping[tuple[str, str, str], set[str]],
) -> pd.DataFrame:
    summary = sarimax_validation._summary_frame(
        jobs,
        manifest,
        forecasts,
        diagnostics=diagnostics,
        expected_forecast_timestamps=expected_timestamps,
    )
    if "model_family" in summary.columns:
        summary["model_family"] = MODEL_FAMILY
    else:
        summary.insert(1, "model_family", MODEL_FAMILY)
    return summary


def _write_summary(summary: pd.DataFrame, path: Path) -> None:
    sarimax_validation._atomic_write(
        summary,
        path,
        [
            "country",
            "model_family",
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


def run_validation(
    target_dates: Iterable[date | str | pd.Timestamp]
    | Mapping[str, Iterable[date | str | pd.Timestamp]]
    | None = None,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    processed_directory: str | Path = PROCESSED,
    *,
    full_year: bool = False,
) -> dict[str, object]:
    sarimax_validation._validate_selected_sarima_orders()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frames = {
        country: sarimax_validation.load_validation_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    if full_year:
        dates_by_country = {
            country: sorted(
                value
                for value in frames[country]["local_date"].astype(str).unique()
                if value in sarimax_validation.FULL_YEAR_DATE_STRINGS
            )
            for country in COUNTRY_CONFIG
        }
    else:
        dates = sarimax_validation.MINI_DATES if target_dates is None else target_dates
        if isinstance(dates, Mapping):
            dates_by_country = {
                country: [_coerce_date(value).isoformat() for value in dates.get(country, ())]
                for country in COUNTRY_CONFIG
            }
        else:
            values = [_coerce_date(value).isoformat() for value in dates]
            dates_by_country = {country: values for country in COUNTRY_CONFIG}
    manifest = sarimax_validation.build_manifest(dates_by_country, full_year=full_year)
    manifest["expected_observations"] = [
        len(extract_target_day(frames[country], target_date))
        for country, target_date in zip(manifest["country"], manifest["target_date"])
    ]
    expected_timestamps = {
        tuple(str(row[column]) for column in JOB_KEY_COLUMNS): sarimax_validation.expected_forecast_timestamps(
            frames[str(row["country"])], str(row["country"]), str(row["target_date"])
        )
        for row in manifest.to_dict("records")
    }
    jobs_path = output / JOB_FILENAME
    forecasts_path = output / FORECAST_FILENAME
    diagnostics_path = output / DIAGNOSTIC_FILENAME
    summary_path = output / SUMMARY_FILENAME
    specifications_path = output / SPECIFICATION_FILENAME
    sarimax_validation._atomic_write(
        pd.DataFrame(
            [
                {
                    "specification_order": index,
                    "specification_id": specification_id,
                    "exog_columns": _json_value(list(specification_columns(specification_id))),
                }
                for index, specification_id in enumerate(
                    sarimax_validation.SARIMAX_SPECIFICATION_IDS, 1
                )
            ]
        ),
        specifications_path,
        ["specification_order", "specification_id", "exog_columns"],
    )
    jobs = sarimax_validation._load_csv(jobs_path, JOB_COLUMNS)
    forecasts = sarimax_validation._load_csv(forecasts_path, FORECAST_COLUMNS)
    diagnostics = sarimax_validation._load_csv(diagnostics_path, DIAGNOSTIC_COLUMNS)
    completed = sarimax_validation.load_completed_job_keys(
        jobs, forecasts, diagnostics, expected_timestamps
    )
    _write_summary(
        _summary_frame(jobs, manifest, forecasts, diagnostics, expected_timestamps),
        summary_path,
    )

    def persist(result: Mapping[str, object]) -> None:
        nonlocal jobs, forecasts, diagnostics
        jobs = sarimax_validation.upsert_frame(
            jobs, pd.DataFrame([result["job"]]), JOB_KEY_COLUMNS
        )
        forecasts = sarimax_validation._without_job_rows(forecasts, result["job"])
        forecasts = sarimax_validation.upsert_frame(
            forecasts, pd.DataFrame(result.get("forecasts", [])), FORECAST_KEY_COLUMNS
        )
        diagnostics = sarimax_validation.upsert_frame(
            diagnostics,
            pd.DataFrame([result["diagnostics"]]),
            sarimax_validation.DIAGNOSTIC_KEY_COLUMNS,
        )
        sarimax_validation._atomic_write(forecasts, forecasts_path, FORECAST_COLUMNS)
        sarimax_validation._atomic_write(diagnostics, diagnostics_path, DIAGNOSTIC_COLUMNS)
        sarimax_validation._atomic_write(jobs, jobs_path, JOB_COLUMNS)
        _write_summary(
            _summary_frame(jobs, manifest, forecasts, diagnostics, expected_timestamps),
            summary_path,
        )

    for country in COUNTRY_CONFIG:
        for specification_id in sarimax_validation.SARIMAX_SPECIFICATION_IDS:
            chain_dates = dates_by_country[country]
            for result in iter_chain(
                country,
                specification_id,
                chain_dates,
                frames[country],
                completed,
            ):
                persist(result)
    summary = _summary_frame(jobs, manifest, forecasts, diagnostics, expected_timestamps)
    _write_summary(summary, summary_path)
    completed = sarimax_validation.load_completed_job_keys(
        jobs, forecasts, diagnostics, expected_timestamps
    )
    manifest_keys = {
        tuple(str(row[column]) for column in JOB_KEY_COLUMNS)
        for row in manifest.to_dict("records")
    }
    return {
        "total_jobs": len(manifest),
        "completed_jobs": len(completed.intersection(manifest_keys)),
        "failed_jobs": int(jobs.loc[jobs["status"].astype(str).eq("failed")].shape[0]),
        "pending_jobs": len(manifest) - len(completed.intersection(manifest_keys)),
        "forecast_rows": len(forecasts),
        "summary": summary.to_dict("records"),
        "output_directory": str(output),
    }


def run_mini_validation(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return run_validation(output_directory=output_directory, processed_directory=processed_directory)


def run_full_validation(
    output_directory: str | Path = FULL_OUTPUT_DIRECTORY,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return run_validation(
        output_directory=output_directory,
        processed_directory=processed_directory,
        full_year=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-parameter SARIMAX state validation")
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--mini", action="store_true")
    stage.add_argument("--full-year", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    args = parser.parse_args()
    result = (
        run_full_validation(args.output_directory, args.processed_directory)
        if args.full_year
        else run_mini_validation(args.output_directory, args.processed_directory)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
