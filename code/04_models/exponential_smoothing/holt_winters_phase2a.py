from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import COUNTRY_CONFIG, PROCESSED, evaluate_target_day, load_country_data
from holt_winters_models import (
    HoltWintersForecastError,
    OptimizerConfiguration,
    HoltWintersSpecification,
    ShortlistingError,
    build_specification_grid,
    forecast_holt_winters,
    screen_country,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "exponential_smoothing" / "holt_winters" / "tables"
FORECAST_DIRECTORY = PROJECT_ROOT / "results" / "exponential_smoothing" / "holt_winters" / "forecasts"
SCREENING_OUTPUT = TABLE_DIRECTORY / "holt_winters_screening_2024.csv"
SHORTLIST_OUTPUT = TABLE_DIRECTORY / "holt_winters_shortlists_2024.csv"
ATTEMPTS_OUTPUT = TABLE_DIRECTORY / "holt_winters_optimizer_attempts_2024.csv"
VALIDATION_YEAR = 2024
SMOKE_CASES = (
    ("Germany", date(2024, 7, 15)),
    ("Austria", date(2024, 7, 15)),
    ("Germany", date(2024, 3, 31)),
    ("Germany", date(2024, 10, 27)),
)


def _write_screening_outputs(
    screening: pd.DataFrame,
    shortlists: pd.DataFrame,
) -> None:
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if shortlists.empty and not len(shortlists.columns) and len(screening.columns):
        shortlists = screening.iloc[0:0].copy()
    screening.to_csv(TABLE_DIRECTORY / "holt_winters_screening_2024.csv", index=False)
    shortlists.to_csv(TABLE_DIRECTORY / "holt_winters_shortlists_2024.csv", index=False)
    _optimizer_attempts_frame(screening).to_csv(
        TABLE_DIRECTORY / "holt_winters_optimizer_attempts_2024.csv", index=False
    )


def _optimizer_attempts_frame(screening: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "country",
        "specification_id",
        "seasonal_form",
        "seasonal_periods",
        "damped_trend",
        "forecast_origin_local",
        "forecast_origin_utc",
        "attempt_number",
        "optimizer",
        "minimize_kwargs",
        "convergence_status",
        "message",
        "status",
        "success",
        "nit",
        "nfev",
        "njev",
        "warning_messages",
        "parameters_finite",
        "aic",
        "aic_finite",
        "aicc",
        "aicc_finite",
        "bic",
        "bic_finite",
        "fitting_time_seconds",
    ]
    rows = []
    if "optimizer_attempts_json" not in screening.columns:
        return pd.DataFrame(columns=columns)
    for candidate in screening.itertuples(index=False):
        for attempt in json.loads(candidate.optimizer_attempts_json):
            rows.append(
                {
                    "country": candidate.country,
                    "specification_id": candidate.specification_id,
                    "seasonal_form": candidate.seasonal_form,
                    "seasonal_periods": candidate.seasonal_periods,
                    "damped_trend": candidate.damped_trend,
                    "forecast_origin_local": getattr(
                        candidate, "forecast_origin_local", None
                    ),
                    "forecast_origin_utc": getattr(
                        candidate, "forecast_origin_utc", None
                    ),
                    "attempt_number": attempt["attempt_number"],
                    "optimizer": attempt["optimizer"],
                    "minimize_kwargs": json.dumps(
                        attempt["minimize_kwargs"], sort_keys=True
                    ),
                    "convergence_status": attempt.get("convergence_status"),
                    "message": attempt["message"],
                    "status": attempt["status"],
                    "success": attempt["success"],
                    "nit": attempt["nit"],
                    "nfev": attempt["nfev"],
                    "njev": attempt["njev"],
                    "warning_messages": json.dumps(
                        attempt["warning_messages"]
                    ),
                    "parameters_finite": attempt["parameters_finite"],
                    "aic": attempt["aic"],
                    "aic_finite": attempt["aic_finite"],
                    "aicc": attempt["aicc"],
                    "aicc_finite": attempt["aicc_finite"],
                    "bic": attempt["bic"],
                    "bic_finite": attempt["bic_finite"],
                    "fitting_time_seconds": attempt["fitting_time_seconds"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def run_screening(
    processed_directory: str | Path = PROCESSED,
    validation_year: int = VALIDATION_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    screening_frames = []
    shortlist_frames = []
    for country in COUNTRY_CONFIG:
        frame = load_country_data(country, processed_directory)
        try:
            screening, shortlist = screen_country(frame, country, validation_year)
        except ShortlistingError as error:
            failed_screening = getattr(error, "screening", None)
            if failed_screening is not None:
                screening_frames.append(failed_screening)
            if screening_frames:
                _write_screening_outputs(
                    pd.concat(screening_frames, ignore_index=True),
                    pd.concat(shortlist_frames, ignore_index=True)
                    if shortlist_frames
                    else pd.DataFrame(),
                )
            raise
        screening_frames.append(screening)
        shortlist_frames.append(shortlist)

    all_screening = pd.concat(screening_frames, ignore_index=True)
    all_shortlists = pd.concat(shortlist_frames, ignore_index=True)
    _write_screening_outputs(all_screening, all_shortlists)
    return all_screening, all_shortlists


def _specification_from_row(row: pd.Series) -> HoltWintersSpecification:
    return HoltWintersSpecification(
        specification_id=str(row["specification_id"]),
        seasonal_form=str(row["seasonal_form"]),
        seasonal_periods=int(row["seasonal_periods"]),
        damped_trend=_parse_boolean(row["damped_trend"], "damped_trend", int(row.name) + 2),
    )


def _parse_boolean(value: object, field: str, row_number: int) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"shortlist row {row_number} has invalid {field}: {value!r}")


def _parse_integer(value: object, field: str, row_number: int) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or float(numeric) != int(numeric):
        raise ValueError(f"shortlist row {row_number} has invalid {field}: {value!r}")
    return int(numeric)


def _optimizer_attempts_from_row(row: pd.Series, row_number: int) -> list[dict[str, object]]:
    raw = row["optimizer_attempts_json"]
    try:
        attempts = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"shortlist row {row_number} has invalid optimizer_attempts_json"
        ) from error
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(
            f"shortlist row {row_number} must contain a non-empty optimizer attempt list"
        )
    if not all(isinstance(attempt, dict) for attempt in attempts):
        raise ValueError(
            f"shortlist row {row_number} contains an invalid optimizer attempt"
        )
    return attempts


def _optimizer_configuration_from_row(row: pd.Series) -> OptimizerConfiguration:
    attempts = _optimizer_attempts_from_row(row, int(row.name) + 2)
    final = attempts[-1]
    return OptimizerConfiguration(
        optimizer=str(row["optimizer_used"]),
        minimize_kwargs=dict(final["minimize_kwargs"]),
    )


def load_frozen_shortlists(path: str | Path = SHORTLIST_OUTPUT) -> pd.DataFrame:
    shortlists = pd.read_csv(path)
    expected = {
        "country",
        "specification_id",
        "seasonal_form",
        "seasonal_periods",
        "damped_trend",
        "eligible",
        "selection_value",
        "optimizer_used",
        "optimizer_attempt_count",
        "optimizer_attempts_json",
    }
    missing = sorted(expected.difference(shortlists.columns))
    if missing:
        raise ValueError(f"shortlist file is missing required columns: {missing}")
    expected_countries = set(COUNTRY_CONFIG)
    actual_countries = set(shortlists["country"].dropna().astype(str))
    if actual_countries != expected_countries:
        raise ValueError(
            f"shortlist countries must be exactly {sorted(expected_countries)}"
        )

    approved = {
        specification.specification_id: specification
        for specification in build_specification_grid()
    }
    allowed_optimizers = {"L-BFGS-B", "least_squares"}
    normalized = shortlists.copy()
    for row_number, (index, row) in enumerate(normalized.iterrows(), 1):
        country = row["country"]
        specification_id = row["specification_id"]
        if pd.isna(country) or pd.isna(specification_id):
            raise ValueError(f"shortlist row {row_number} has missing identity fields")
        country = str(country)
        specification_id = str(specification_id)
        specification = approved.get(specification_id)
        if specification is None:
            raise ValueError(
                f"shortlist row {row_number} has unknown specification ID: {specification_id}"
            )
        seasonal_form = row["seasonal_form"]
        if seasonal_form != specification.seasonal_form:
            raise ValueError(f"shortlist row {row_number} has inconsistent seasonal form")
        seasonal_periods = _parse_integer(
            row["seasonal_periods"], "seasonal_periods", row_number
        )
        if seasonal_periods != specification.seasonal_periods:
            raise ValueError(f"shortlist row {row_number} has inconsistent seasonal period")
        damped_trend = _parse_boolean(row["damped_trend"], "damped_trend", row_number)
        if damped_trend != specification.damped_trend:
            raise ValueError(f"shortlist row {row_number} has inconsistent damping")
        eligible = _parse_boolean(row["eligible"], "eligible", row_number)
        if not eligible:
            raise ValueError(f"shortlist row {row_number} is not eligible")
        selection_value = pd.to_numeric(row["selection_value"], errors="coerce")
        if pd.isna(selection_value) or not np.isfinite(float(selection_value)):
            raise ValueError(f"shortlist row {row_number} has non-finite selection value")
        optimizer_used = row["optimizer_used"]
        if pd.isna(optimizer_used) or str(optimizer_used) not in allowed_optimizers:
            raise ValueError(f"shortlist row {row_number} has invalid optimizer_used")
        optimizer_attempt_count = _parse_integer(
            row["optimizer_attempt_count"], "optimizer_attempt_count", row_number
        )
        if optimizer_attempt_count < 1:
            raise ValueError(f"shortlist row {row_number} has no optimizer attempts")
        attempts = _optimizer_attempts_from_row(row, row_number)
        if len(attempts) != optimizer_attempt_count:
            raise ValueError(
                f"shortlist row {row_number} optimizer attempt count does not match metadata"
            )
        attempt_keys = []
        for attempt in attempts:
            if not isinstance(attempt.get("optimizer"), str) or not isinstance(
                attempt.get("minimize_kwargs"), dict
            ):
                raise ValueError(f"shortlist row {row_number} has malformed optimizer metadata")
            attempt_keys.append(
                (
                    attempt["optimizer"],
                    json.dumps(attempt["minimize_kwargs"], sort_keys=True),
                )
            )
        if len(attempt_keys) != len(set(attempt_keys)):
            raise ValueError(f"shortlist row {row_number} repeats an optimizer configuration")
        final = attempts[-1]
        if final.get("optimizer") != str(optimizer_used):
            raise ValueError(
                f"shortlist row {row_number} final optimizer does not match optimizer_used"
            )
        if final.get("success") is False or final.get("parameters_finite") is not True:
            raise ValueError(f"shortlist row {row_number} final optimizer was not successful")
        final_aic = pd.to_numeric(final.get("aic"), errors="coerce")
        final_aicc = pd.to_numeric(final.get("aicc"), errors="coerce")
        if not (
            np.isfinite(float(final_aic)) if not pd.isna(final_aic) else False
        ) and not (
            np.isfinite(float(final_aicc)) if not pd.isna(final_aicc) else False
        ):
            raise ValueError(f"shortlist row {row_number} has no finite final criterion")

        normalized.at[index, "country"] = country
        normalized.at[index, "specification_id"] = specification_id
        normalized.at[index, "seasonal_periods"] = seasonal_periods
        normalized.at[index, "damped_trend"] = damped_trend
        normalized.at[index, "eligible"] = eligible
        normalized.at[index, "selection_value"] = float(selection_value)
        normalized.at[index, "optimizer_used"] = str(optimizer_used)
        normalized.at[index, "optimizer_attempt_count"] = optimizer_attempt_count

    approved_groups = {
        (specification.seasonal_form, specification.seasonal_periods)
        for specification in approved.values()
    }
    group_sizes = normalized.groupby(
        ["country", "seasonal_form", "seasonal_periods"], dropna=False
    ).size()
    if (group_sizes > 1).any():
        raise ValueError("shortlist contains duplicate seasonal-form/period groups")
    for country in expected_countries:
        country_groups = set(
            zip(
                normalized.loc[normalized["country"].eq(country), "seasonal_form"],
                normalized.loc[normalized["country"].eq(country), "seasonal_periods"],
            )
        )
        if len(country_groups) != 4 or country_groups != approved_groups:
            raise ValueError(
                f"shortlist for {country} must contain exactly four approved seasonal-form/period groups"
            )
    return normalized


def run_smoke_tests(
    shortlists: pd.DataFrame | None = None,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    frozen = load_frozen_shortlists() if shortlists is None else shortlists
    frames = {
        country: load_country_data(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    rows = []
    for country, target_date in SMOKE_CASES:
        country_shortlists = frozen.loc[frozen["country"].eq(country)]
        for _, shortlist_row in country_shortlists.iterrows():
            specification = _specification_from_row(shortlist_row)
            base = {
                "country": country,
                "target_date": target_date.isoformat(),
                "specification_id": specification.specification_id,
                "target_intervals": None,
                "path_intervals": None,
                "bridge_intervals": None,
                "missing_predictions": None,
                "failure": False,
                "error_message": None,
                "convergence_status": None,
                "fitting_time_seconds": None,
                "optimizer_used": None,
                "optimizer_attempt_count": None,
            }
            try:
                optimizer_configuration = _optimizer_configuration_from_row(shortlist_row)
                result = forecast_holt_winters(
                    frames[country],
                    country,
                    target_date,
                    specification,
                    optimizer_configuration=optimizer_configuration,
                )
                forecast = result.forecast
                target_metrics = evaluate_target_day(forecast)
                base.update(
                    {
                        "target_intervals": int(target_metrics["evaluated_observations"]),
                        "path_intervals": len(forecast),
                        "bridge_intervals": int((~forecast["is_target_day"]).sum()),
                        "missing_predictions": int(forecast["forecast_mwh"].isna().sum()),
                        "convergence_status": result.fit_record.convergence_status,
                        "fitting_time_seconds": result.fit_record.fitting_time_seconds,
                        "optimizer_used": result.fit_record.optimizer_used,
                        "optimizer_attempt_count": len(result.fit_record.optimizer_attempts),
                    }
                )
                if base["missing_predictions"] or not forecast["forecast_mwh"].map(pd.notna).all():
                    base["failure"] = True
                    base["error_message"] = "smoke forecast contains missing predictions"
            except HoltWintersForecastError as error:
                base.update(
                    {
                        "failure": True,
                        "error_message": str(error),
                        "convergence_status": error.record.convergence_status,
                        "fitting_time_seconds": error.record.fitting_time_seconds,
                        "optimizer_used": error.record.optimizer_used,
                        "optimizer_attempt_count": len(error.record.optimizer_attempts),
                    }
                )
            rows.append(base)
    return pd.DataFrame(rows)


def estimate_full_validation_runtime(
    smoke_results: pd.DataFrame,
    target_origins_per_country: int = 366,
) -> dict[str, object]:
    if smoke_results["failure"].fillna(True).astype(bool).any():
        raise RuntimeError("cannot estimate runtime: a smoke fit failed")
    fitting_times = pd.to_numeric(smoke_results["fitting_time_seconds"], errors="coerce")
    if not np.isfinite(fitting_times).all():
        raise RuntimeError("cannot estimate runtime: a smoke fit has no finite fitting time")
    successful = smoke_results.loc[~smoke_results["failure"]].copy()
    by_country = {}
    total_seconds = 0.0
    total_fits = 0
    for country in COUNTRY_CONFIG:
        country_results = successful.loc[successful["country"].eq(country)]
        by_spec = country_results.groupby("specification_id")[
            "fitting_time_seconds"
        ].mean()
        country_seconds = float(by_spec.sum() * target_origins_per_country)
        country_fits = int(len(by_spec) * target_origins_per_country)
        by_country[country] = {
            "measured_smoke_fits": int(len(country_results)),
            "selected_specifications": int(len(by_spec)),
            "projected_fits": country_fits,
            "estimated_seconds": country_seconds,
            "estimated_minutes": country_seconds / 60.0,
        }
        total_seconds += country_seconds
        total_fits += country_fits
    return {
        "basis": "mean successful smoke fit time by country/specification scaled to 366 target origins",
        "projected_fits": total_fits,
        "estimated_seconds": total_seconds,
        "estimated_minutes": total_seconds / 60.0,
        "countries": by_country,
    }


def run_phase2a(
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    screening, _ = run_screening(processed_directory)
    frozen = load_frozen_shortlists()
    smoke = run_smoke_tests(frozen, processed_directory)
    runtime = estimate_full_validation_runtime(smoke)
    return {
        "screening_rows": len(screening),
        "shortlist_rows": len(frozen),
        "smoke_rows": len(smoke),
        "smoke_failures": int(smoke["failure"].sum()),
        "screening_fitting_times": screening[
            ["country", "specification_id", "fitting_time_seconds"]
        ].to_dict(orient="records"),
        "smoke_fitting_times": smoke[
            ["country", "target_date", "specification_id", "fitting_time_seconds"]
        ].to_dict(orient="records"),
        "smoke_results": smoke.to_dict(orient="records"),
        "runtime_estimate": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2A Holt-Winters screening and smoke tests")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run smoke tests from the existing frozen shortlist without re-screening",
    )
    args = parser.parse_args()
    try:
        if args.smoke_only:
            smoke = run_smoke_tests()
            result = {
                "smoke_rows": len(smoke),
                "smoke_failures": int(smoke["failure"].sum()),
                "smoke_results": smoke.to_dict(orient="records"),
                "runtime_estimate": estimate_full_validation_runtime(smoke),
            }
        else:
            result = run_phase2a()
    except ShortlistingError as error:
        result = {
            "status": "stopped_before_smoke",
            "error": str(error),
        }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
