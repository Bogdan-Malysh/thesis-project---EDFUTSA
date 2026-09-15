from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
import json
from time import perf_counter
from typing import Any
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import (
    COUNTRY_CONFIG,
    _as_prepared,
    country_forecast_origin,
    extract_target_day,
    information_set,
)


DEFAULT_OPTIMIZER = "L-BFGS-B"
RETRY_OPTIMIZER = "L-BFGS-B"
RETRY_MINIMIZE_KWARGS = {"options": {"maxfun": 30000, "maxiter": 2000}}
ALTERNATIVE_OPTIMIZER = "least_squares"
ALTERNATIVE_MINIMIZE_KWARGS = {"max_nfev": 2000}


@dataclass(frozen=True)
class HoltWintersSpecification:
    specification_id: str
    seasonal_form: str
    seasonal_periods: int
    damped_trend: bool


@dataclass(frozen=True)
class OptimizerConfiguration:
    optimizer: str
    minimize_kwargs: dict[str, object]


@dataclass(frozen=True)
class OptimizerAttempt:
    optimizer: str
    minimize_kwargs: dict[str, object]
    convergence_status: str
    message: str | None
    status: int | None
    success: bool | None
    nit: int | None
    nfev: int | None
    njev: int | None
    warning_messages: tuple[str, ...]
    parameters_finite: bool
    aic: float
    aic_finite: bool
    aicc: float
    aicc_finite: bool
    bic: float
    bic_finite: bool
    fitting_time_seconds: float

    @property
    def criteria_finite(self) -> bool:
        return self.aic_finite and self.aicc_finite and self.bic_finite


@dataclass(frozen=True)
class FitRecord:
    specification: HoltWintersSpecification
    specification_order: int
    fitted_result: Any | None
    aic: float
    aicc: float
    bic: float
    convergence_status: str
    fit_status: str
    fitting_time_seconds: float
    selection_criterion: str | None
    selection_value: float | None
    training_observations: int
    eligible: bool
    error_message: str | None
    optimizer_used: str | None
    optimizer_attempts: tuple[OptimizerAttempt, ...]


@dataclass(frozen=True)
class HoltWintersForecast:
    forecast: pd.DataFrame
    fit_record: FitRecord


class ShortlistingError(RuntimeError):
    def __init__(self, failed_groups: list[tuple[tuple[str, int], pd.DataFrame]]):
        self.failed_groups = failed_groups
        details = []
        for (seasonal_form, seasonal_periods), rows in failed_groups:
            candidate_details = "; ".join(
                f"{row.specification_id}: {row.error_message or row.fit_status}"
                for row in rows.itertuples(index=False)
            )
            details.append(
                f"{seasonal_form}/{seasonal_periods}: {candidate_details}"
            )
        super().__init__(
            "no eligible Holt-Winters candidate in group(s): " + " | ".join(details)
        )


class HoltWintersForecastError(RuntimeError):
    def __init__(self, record: FitRecord):
        self.record = record
        message = record.error_message or "Holt-Winters fit was not eligible for forecasting"
        super().__init__(f"{record.specification.specification_id}: {message}")


def build_specification_grid() -> tuple[HoltWintersSpecification, ...]:
    specifications = []
    for seasonal_form in ("add", "mul"):
        for seasonal_periods in (24, 168):
            for damped_trend in (False, True):
                damping = "damped" if damped_trend else "undamped"
                specifications.append(
                    HoltWintersSpecification(
                        specification_id=(
                            f"hw_{seasonal_form}_s{seasonal_periods}_{damping}"
                        ),
                        seasonal_form=seasonal_form,
                        seasonal_periods=seasonal_periods,
                        damped_trend=damped_trend,
                    )
                )
    return tuple(specifications)


def _metric(fitted_result: Any, name: str) -> float:
    try:
        value = float(getattr(fitted_result, name))
    except (AttributeError, TypeError, ValueError):
        return float("nan")
    return value


def _retvals_value(retvals: Any, name: str) -> Any | None:
    if retvals is None:
        return None
    if isinstance(retvals, Mapping):
        return retvals.get(name)
    return getattr(retvals, name, None)


def _convergence_status(fitted_result: Any) -> tuple[str, bool | None]:
    retvals = getattr(fitted_result, "mle_retvals", None)
    success = _explicit_bool(_retvals_value(retvals, "success"))
    converged = _explicit_bool(_retvals_value(retvals, "converged"))
    if success is False or converged is False:
        return "failed", False
    if success is True or converged is True:
        return "converged", True
    return "unknown", None


def _explicit_bool(value: Any | None) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _selection_values(
    aic: float, aicc: float
) -> tuple[str | None, float | None, bool]:
    if np.isfinite(aicc):
        return "AICc", float(aicc), True
    if np.isfinite(aic):
        return "AIC", float(aic), True
    return None, None, False


def _optimizer_attempt_value(retvals: Any, name: str) -> Any | None:
    return _retvals_value(retvals, name)


def _parameters_finite(fitted_result: Any) -> bool:
    try:
        parameters = fitted_result.params_formatted["param"].to_numpy(dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(parameters).all())


def _run_optimizer_attempt(
    values: np.ndarray,
    specification: HoltWintersSpecification,
    optimizer: str,
    minimize_kwargs: dict[str, object] | None,
) -> tuple[Any | None, OptimizerAttempt]:
    started = perf_counter()
    caught_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            model = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=specification.damped_trend,
                seasonal=specification.seasonal_form,
                seasonal_periods=specification.seasonal_periods,
                initialization_method="estimated",
                use_boxcox=False,
            )
            fit_kwargs: dict[str, object] = {
                "optimized": True,
                "remove_bias": False,
                "method": optimizer,
            }
            if minimize_kwargs is not None:
                fit_kwargs["minimize_kwargs"] = minimize_kwargs
            fitted_result = model.fit(**fit_kwargs)
            caught_warnings = [
                f"{type(item.message).__name__}: {item.message}"
                for item in recorded_warnings
            ]
        retvals = getattr(fitted_result, "mle_retvals", None)
        convergence_status, effective_success = _convergence_status(fitted_result)
        message = _optimizer_attempt_value(retvals, "message")
        status = _optimizer_attempt_value(retvals, "status")
        aic = _metric(fitted_result, "aic")
        aicc = _metric(fitted_result, "aicc")
        bic = _metric(fitted_result, "bic")
        attempt = OptimizerAttempt(
            optimizer=optimizer,
            minimize_kwargs=dict(minimize_kwargs or {}),
            convergence_status=convergence_status,
            message=str(message) if message is not None else None,
            status=int(status) if status is not None else None,
            success=effective_success,
            nit=_as_int(_optimizer_attempt_value(retvals, "nit")),
            nfev=_as_int(_optimizer_attempt_value(retvals, "nfev")),
            njev=_as_int(_optimizer_attempt_value(retvals, "njev")),
            warning_messages=tuple(caught_warnings),
            parameters_finite=_parameters_finite(fitted_result),
            aic=aic,
            aic_finite=bool(np.isfinite(aic)),
            aicc=aicc,
            aicc_finite=bool(np.isfinite(aicc)),
            bic=bic,
            bic_finite=bool(np.isfinite(bic)),
            fitting_time_seconds=perf_counter() - started,
        )
        return fitted_result, attempt
    except Exception as error:
        caught_warnings.extend(
            f"{type(item.message).__name__}: {item.message}"
            for item in locals().get("recorded_warnings", [])
        )
        attempt = OptimizerAttempt(
            optimizer=optimizer,
            minimize_kwargs=dict(minimize_kwargs or {}),
            convergence_status="failed",
            message=f"{type(error).__name__}: {error}",
            status=None,
            success=None,
            nit=None,
            nfev=None,
            njev=None,
            warning_messages=tuple(caught_warnings),
            parameters_finite=False,
            aic=float("nan"),
            aic_finite=False,
            aicc=float("nan"),
            aicc_finite=False,
            bic=float("nan"),
            bic_finite=False,
            fitting_time_seconds=perf_counter() - started,
        )
        return None, attempt


def _as_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iteration_limit_reached(attempt: OptimizerAttempt) -> bool:
    message = (attempt.message or "").upper()
    return any(
        marker in message
        for marker in (
            "EXCEEDS LIMIT",
            "MAXIMUM NUMBER OF ITERATIONS",
            "ITERATION LIMIT",
            "MAXFUN",
            "MAX_NFEV",
        )
    )


def _configuration_key(
    optimizer: str, minimize_kwargs: Mapping[str, object] | None
) -> tuple[str, str]:
    return (
        optimizer,
        json.dumps(minimize_kwargs or {}, sort_keys=True, separators=(",", ":")),
    )


def _fixed_fallback_configurations() -> tuple[OptimizerConfiguration, ...]:
    return (
        OptimizerConfiguration(DEFAULT_OPTIMIZER, {}),
        OptimizerConfiguration(RETRY_OPTIMIZER, RETRY_MINIMIZE_KWARGS),
        OptimizerConfiguration(ALTERNATIVE_OPTIMIZER, ALTERNATIVE_MINIMIZE_KWARGS),
    )


def _record_from_attempts(
    specification: HoltWintersSpecification,
    specification_order: int,
    attempts: list[OptimizerAttempt],
    fitted_result: Any | None,
) -> FitRecord:
    final = attempts[-1]
    criterion, value, criterion_ok = _selection_values(final.aic, final.aicc)
    optimizer_ok = fitted_result is not None and final.success is not False
    eligible = optimizer_ok and final.parameters_finite and criterion_ok
    if not optimizer_ok:
        error_message = (
            "optimizer explicitly reported failure"
            if final.success is False
            else final.message or "optimizer fit failed"
        )
    elif not final.parameters_finite:
        error_message = "estimated parameters are not finite"
    elif not criterion_ok:
        error_message = "no finite AICc or AIC selection criterion"
    else:
        error_message = None
    return FitRecord(
        specification=specification,
        specification_order=specification_order,
        fitted_result=fitted_result,
        aic=final.aic,
        aicc=final.aicc,
        bic=final.bic,
        convergence_status=final.convergence_status,
        fit_status="success" if optimizer_ok else "failed",
        fitting_time_seconds=sum(attempt.fitting_time_seconds for attempt in attempts),
        selection_criterion=criterion,
        selection_value=value,
        training_observations=0,
        eligible=eligible,
        error_message=error_message,
        optimizer_used=final.optimizer if eligible else None,
        optimizer_attempts=tuple(attempts),
    )


def fit_specification(
    training_values: Iterable[float],
    specification: HoltWintersSpecification,
    specification_order: int | None = None,
    optimizer_configuration: OptimizerConfiguration | None = None,
) -> FitRecord:
    order = (
        specification_order
        if specification_order is not None
        else build_specification_grid().index(specification)
    )
    values = np.asarray(list(training_values), dtype=float)
    started = perf_counter()
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        return FitRecord(
            specification=specification,
            specification_order=order,
            fitted_result=None,
            aic=float("nan"),
            aicc=float("nan"),
            bic=float("nan"),
            convergence_status="failed",
            fit_status="failed",
            fitting_time_seconds=perf_counter() - started,
            selection_criterion=None,
            selection_value=None,
            training_observations=int(len(values)),
            eligible=False,
            error_message="training values must be a finite non-empty series",
            optimizer_used=None,
            optimizer_attempts=(),
        )
    if specification.seasonal_form == "mul" and (values <= 0).any():
        return FitRecord(
            specification=specification,
            specification_order=order,
            fitted_result=None,
            aic=float("nan"),
            aicc=float("nan"),
            bic=float("nan"),
            convergence_status="failed",
            fit_status="failed",
            fitting_time_seconds=perf_counter() - started,
            selection_criterion=None,
            selection_value=None,
            training_observations=int(len(values)),
            eligible=False,
            error_message=(
                "multiplicative seasonal Holt-Winters requires strictly positive "
                "training values"
            ),
            optimizer_used=None,
            optimizer_attempts=(),
        )

    attempts: list[OptimizerAttempt] = []
    fitted_result: Any | None = None
    if optimizer_configuration is None:
        configurations: list[OptimizerConfiguration] = [
            OptimizerConfiguration(DEFAULT_OPTIMIZER, {}),
        ]
        configuration_kwargs: list[dict[str, object] | None] = [None]
    else:
        configurations = [optimizer_configuration]
        seen = {
            _configuration_key(
                optimizer_configuration.optimizer,
                optimizer_configuration.minimize_kwargs,
            )
        }
        for fallback in _fixed_fallback_configurations():
            key = _configuration_key(fallback.optimizer, fallback.minimize_kwargs)
            if key not in seen:
                configurations.append(fallback)
                seen.add(key)
        configuration_kwargs = [configuration.minimize_kwargs for configuration in configurations]

    for attempt_number, configuration in enumerate(configurations):
        minimize_kwargs = (
            configuration_kwargs[attempt_number]
            if optimizer_configuration is None
            else configuration.minimize_kwargs
        )
        fitted_result, attempt = _run_optimizer_attempt(
            values,
            specification,
            configuration.optimizer,
            minimize_kwargs if minimize_kwargs else None,
        )
        attempts.append(attempt)
        if (
            fitted_result is not None
            and attempt.success is not False
            and attempt.parameters_finite
            and _selection_values(attempt.aic, attempt.aicc)[2]
        ):
            record = _record_from_attempts(
                specification, order, attempts, fitted_result
            )
            return FitRecord(
                **{
                    **record.__dict__,
                    "training_observations": int(len(values)),
                }
            )
        if optimizer_configuration is not None:
            continue
        if attempt_number == 0 and _iteration_limit_reached(attempt):
            configurations.append(
                OptimizerConfiguration(RETRY_OPTIMIZER, RETRY_MINIMIZE_KWARGS)
            )
            configuration_kwargs.append(RETRY_MINIMIZE_KWARGS)
        elif attempt_number == 1 and _iteration_limit_reached(attempts[0]):
            configurations.append(
                OptimizerConfiguration(ALTERNATIVE_OPTIMIZER, ALTERNATIVE_MINIMIZE_KWARGS)
            )
            configuration_kwargs.append(ALTERNATIVE_MINIMIZE_KWARGS)
        else:
            break

    record = _record_from_attempts(specification, order, attempts, fitted_result)
    return FitRecord(
        **{
            **record.__dict__,
            "training_observations": int(len(values)),
        }
    )


def fit_record_to_row(
    record: FitRecord,
    country: str,
    forecast_origin_utc: pd.Timestamp,
) -> dict[str, object]:
    specification = record.specification
    origin_local = forecast_origin_utc.tz_convert(
        COUNTRY_CONFIG[country]["timezone"]
    )
    return {
        "country": country,
        "specification_id": specification.specification_id,
        "seasonal_form": specification.seasonal_form,
        "seasonal_periods": specification.seasonal_periods,
        "trend": "add",
        "damped_trend": specification.damped_trend,
        "specification_order": record.specification_order,
        "forecast_origin_local": origin_local.isoformat(),
        "forecast_origin_utc": forecast_origin_utc.isoformat(),
        "training_observations": record.training_observations,
        "aic": record.aic,
        "aicc": record.aicc,
        "bic": record.bic,
        "convergence_status": record.convergence_status,
        "fit_status": record.fit_status,
        "fitting_time_seconds": record.fitting_time_seconds,
        "selection_criterion": record.selection_criterion,
        "selection_value": record.selection_value,
        "eligible": record.eligible,
        "error_message": record.error_message,
        "optimizer_used": record.optimizer_used,
        "optimizer_attempt_count": len(record.optimizer_attempts),
        "optimizer_attempts_json": json.dumps(
            [optimizer_attempt_to_dict(attempt, number) for number, attempt in enumerate(record.optimizer_attempts, 1)],
            sort_keys=True,
        ),
    }


def optimizer_attempt_to_dict(
    attempt: OptimizerAttempt,
    attempt_number: int,
) -> dict[str, object]:
    return {
        "attempt_number": attempt_number,
        "optimizer": attempt.optimizer,
        "minimize_kwargs": attempt.minimize_kwargs,
        "convergence_status": attempt.convergence_status,
        "message": attempt.message,
        "status": attempt.status,
        "success": attempt.success,
        "nit": attempt.nit,
        "nfev": attempt.nfev,
        "njev": attempt.njev,
        "warning_messages": list(attempt.warning_messages),
        "parameters_finite": attempt.parameters_finite,
        "aic": attempt.aic,
        "aic_finite": attempt.aic_finite,
        "aicc": attempt.aicc,
        "aicc_finite": attempt.aicc_finite,
        "bic": attempt.bic,
        "bic_finite": attempt.bic_finite,
        "fitting_time_seconds": attempt.fitting_time_seconds,
    }


def optimizer_attempts_to_rows(
    record: FitRecord,
    country: str,
    forecast_origin_utc: pd.Timestamp,
) -> list[dict[str, object]]:
    specification = record.specification
    origin_local = forecast_origin_utc.tz_convert(
        COUNTRY_CONFIG[country]["timezone"]
    )
    rows = []
    for attempt_number, attempt in enumerate(record.optimizer_attempts, 1):
        row = optimizer_attempt_to_dict(attempt, attempt_number)
        row.update(
            {
                "country": country,
                "specification_id": specification.specification_id,
                "seasonal_form": specification.seasonal_form,
                "seasonal_periods": specification.seasonal_periods,
                "damped_trend": specification.damped_trend,
                "forecast_origin_local": origin_local.isoformat(),
                "forecast_origin_utc": forecast_origin_utc.isoformat(),
                "warning_messages": json.dumps(attempt.warning_messages),
                "minimize_kwargs": json.dumps(attempt.minimize_kwargs, sort_keys=True),
            }
        )
        rows.append(row)
    return rows


def select_shortlists(screening: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seasonal_form",
        "seasonal_periods",
        "selection_value",
        "eligible",
        "specification_order",
    }
    missing = sorted(required.difference(screening.columns))
    if missing:
        raise ValueError(f"screening frame is missing required columns: {missing}")

    failed_groups: list[tuple[tuple[str, int], pd.DataFrame]] = []
    selected_rows: list[pd.Series] = []
    for group_key, group in screening.groupby(
        ["seasonal_form", "seasonal_periods"], sort=False, dropna=False
    ):
        selection_values = pd.to_numeric(group["selection_value"], errors="coerce")
        eligible_flags = group["eligible"].map(
            lambda value: _explicit_bool(value) is True
        )
        eligible_mask = eligible_flags & np.isfinite(selection_values)
        if not eligible_mask.any():
            failed_groups.append((group_key, group.copy()))
            continue
        ordered_indices = pd.DataFrame(
            {
                "selection_value_numeric": selection_values.loc[eligible_mask],
                "specification_order": group.loc[eligible_mask, "specification_order"],
            }
        ).sort_values(
            ["selection_value_numeric", "specification_order"],
            kind="mergesort",
        )
        selected_rows.append(group.loc[ordered_indices.index[0]])
    if failed_groups:
        raise ShortlistingError(failed_groups)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def first_validation_date(frame: pd.DataFrame, year: int = 2024) -> date:
    prepared = _as_prepared(frame)
    values = sorted(
        value
        for value in prepared["local_date"].unique()
        if value.startswith(f"{year:04d}-")
    )
    if not values:
        raise ValueError(f"no validation dates found for {year}")
    return date.fromisoformat(values[0])


def screen_country(
    frame: pd.DataFrame,
    country: str,
    validation_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = _as_prepared(frame)
    target_date = first_validation_date(prepared, validation_year)
    origin = country_forecast_origin(target_date, country)
    training = information_set(prepared, origin)["actual_load_mwh"]
    rows = []
    for order, specification in enumerate(build_specification_grid()):
        record = fit_specification(training, specification, order)
        rows.append(fit_record_to_row(record, country, origin))
    screening = pd.DataFrame(rows)
    try:
        shortlist = select_shortlists(screening)
    except ShortlistingError as error:
        error.screening = screening
        raise
    return screening, shortlist


def _continuous_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    prepared = _as_prepared(frame)
    local_date = pd.Timestamp(target_date).date().isoformat()
    origin = country_forecast_origin(local_date, country)
    target = extract_target_day(prepared, local_date)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(origin)
        & prepared["local_date"].le(local_date)
    ].copy()
    if path.empty or not path["local_date"].eq(local_date).any():
        raise ValueError(f"no Holt-Winters forecast path exists for {country} {local_date}")
    if not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError(f"Holt-Winters path does not contain all target timestamps for {local_date}")
    return path, origin


def forecast_holt_winters(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
    specification: HoltWintersSpecification,
    optimizer_configuration: OptimizerConfiguration | None = None,
) -> HoltWintersForecast:
    prepared = _as_prepared(frame)
    path, origin = _continuous_path(prepared, country, target_date)
    training = information_set(prepared, origin)["actual_load_mwh"]
    order = build_specification_grid().index(specification)
    record = fit_specification(
        training,
        specification,
        order,
        optimizer_configuration=optimizer_configuration,
    )
    if (
        not record.eligible
        or record.fitted_result is None
        or record.convergence_status == "failed"
    ):
        raise HoltWintersForecastError(record)
    try:
        predictions = np.asarray(record.fitted_result.forecast(len(path)), dtype=float)
    except Exception as error:
        failed = FitRecord(
            **{
                **record.__dict__,
                "fit_status": "failed",
                "eligible": False,
                "error_message": f"forecast failed: {type(error).__name__}: {error}",
            }
        )
        raise HoltWintersForecastError(failed) from error
    if len(predictions) != len(path) or not np.isfinite(predictions).all():
        failed = FitRecord(
            **{
                **record.__dict__,
                "fit_status": "failed",
                "eligible": False,
                "error_message": "forecast returned missing or non-finite predictions",
            }
        )
        raise HoltWintersForecastError(failed)

    local_date = pd.Timestamp(target_date).date().isoformat()
    result = path[
        [
            "timestamp_utc",
            "interval_end_utc",
            "timestamp_local",
            "local_date",
            "actual_load_mwh",
        ]
    ].copy()
    result["country"] = country
    result["model_family"] = "Holt-Winters"
    result["specification_id"] = specification.specification_id
    result["target_date"] = local_date
    result["forecast_origin_local"] = origin.tz_convert(
        COUNTRY_CONFIG[country]["timezone"]
    ).isoformat()
    result["forecast_origin_utc"] = origin.isoformat()
    result["information_cutoff_utc"] = origin.isoformat()
    result["forecast_mwh"] = predictions
    result["is_target_day"] = result["local_date"].eq(local_date)
    result["evaluated"] = result["is_target_day"]
    result["fitting_time_seconds"] = record.fitting_time_seconds
    result["convergence_status"] = record.convergence_status
    result["optimizer_used"] = record.optimizer_used
    result["optimizer_attempt_count"] = len(record.optimizer_attempts)
    return HoltWintersForecast(
        result[
            [
                "country",
                "model_family",
                "specification_id",
                "target_date",
                "forecast_origin_local",
                "forecast_origin_utc",
                "information_cutoff_utc",
                "timestamp_utc",
                "interval_end_utc",
                "timestamp_local",
                "local_date",
                "actual_load_mwh",
                "forecast_mwh",
                "fitting_time_seconds",
                "convergence_status",
                "optimizer_used",
                "optimizer_attempt_count",
                "is_target_day",
                "evaluated",
            ]
        ],
        record,
    )
