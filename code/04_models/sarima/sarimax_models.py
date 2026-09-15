from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
REGRESSION_ROOT = MODEL_ROOT / "regression_arima_errors"
if str(REGRESSION_ROOT) not in sys.path:
    sys.path.insert(0, str(REGRESSION_ROOT))

from regression_arima_errors_models import (
    build_exog as build_calendar_exog,
    specification_columns as regression_specification_columns,
)
from sarima_models import SarimaFitRecord, SarimaOrder, fit_sarima


SARIMAX_SPECIFICATION_IDS = (
    "sarimax_calendar",
    "sarimax_calendar_crisis",
)
SARIMA_INTERVENTION_SPECIFICATION_IDS = {
    "Germany": "sarima_p2_d0_q0_P0_D1_Q0_s24_crisis",
    "Austria": "sarima_p1_d0_q1_P1_D1_Q0_s24_crisis",
}
SARIMAX_FORECAST_FIT_KWARGS = {
    "method": "lbfgs",
    "maxiter": 50,
    "maxfun": 1000,
    "disp": 0,
    "cov_type": "none",
    "low_memory": True,
}
SARIMAX_BENCHMARK_FIT_KWARGS = {
    **SARIMAX_FORECAST_FIT_KWARGS,
    "maxfun": 3000,
}
REGRESSION_SPECIFICATION_BY_SARIMAX = {
    "sarimax_calendar": "reg_arima_h_d_m_hol",
    "sarimax_calendar_crisis": "reg_arima_h_d_m_hol_crisis",
    "sarima_p2_d0_q0_P0_D1_Q0_s24_crisis": "arima_p2_d1_q2_crisis",
    "sarima_p1_d0_q1_P1_D1_Q0_s24_crisis": "arima_p2_d1_q2_crisis",
}
COUNTRY_ORDERS = {
    "Germany": SarimaOrder(2, 0, 0, 0, 1, 0),
    "Austria": SarimaOrder(1, 0, 1, 1, 1, 0),
}


def specification_columns(specification_id: str) -> tuple[str, ...]:
    try:
        regression_id = REGRESSION_SPECIFICATION_BY_SARIMAX[specification_id]
    except KeyError as error:
        raise ValueError(f"unsupported SARIMAX specification: {specification_id}") from error
    return regression_specification_columns(regression_id)


def build_exog(
    timestamps_utc: pd.Series | pd.DatetimeIndex,
    country: str,
    specification_id: str,
) -> pd.DataFrame:
    try:
        regression_id = REGRESSION_SPECIFICATION_BY_SARIMAX[specification_id]
    except KeyError as error:
        raise ValueError(f"unsupported SARIMAX specification: {specification_id}") from error
    exog = build_calendar_exog(timestamps_utc, country, regression_id)
    expected_columns = specification_columns(specification_id)
    if tuple(exog.columns) != expected_columns:
        raise ValueError("SARIMAX exogenous columns do not match the frozen specification")
    return exog


def order_for_country(country: str) -> SarimaOrder:
    try:
        return COUNTRY_ORDERS[country]
    except KeyError as error:
        raise ValueError(f"unsupported SARIMAX country: {country}") from error


@dataclass(frozen=True)
class SarimaxFitRecord:
    sarima_fit: SarimaFitRecord
    exog_columns: tuple[str, ...]
    exog_rank: int
    exog_condition_number: float
    transformed_exog_rank: int
    exog_error_message: str | None = None

    @property
    def fitted_result(self) -> Any | None:
        return self.sarima_fit.fitted_result

    @property
    def eligible(self) -> bool:
        return self.exog_error_message is None and self.sarima_fit.eligible

    @property
    def error_message(self) -> str | None:
        return self.exog_error_message or self.sarima_fit.error_message


def fit_sarimax(
    values: object,
    exog: pd.DataFrame,
    order: SarimaOrder,
    *,
    forecast_only: bool = True,
    fit_kwargs: dict[str, object] | None = None,
    progress_callback: Callable[[Any, np.ndarray], None] | None = None,
    fixed_params: object | None = None,
) -> SarimaxFitRecord:
    columns = tuple(str(column) for column in exog.columns)
    matrix = exog.to_numpy(dtype=float, copy=False)
    exog_rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    condition_number = float(np.linalg.cond(matrix)) if matrix.size else float("nan")
    transformed = np.diff(matrix, n=1, axis=0) if len(matrix) > 1 else matrix
    transformed_rank = int(np.linalg.matrix_rank(transformed)) if transformed.size else 0
    exog_error: str | None = None
    if len(set(columns)) != len(columns):
        exog_error = "exogenous design contains duplicate columns"
    elif matrix.ndim != 2 or matrix.shape[1] != len(columns):
        exog_error = "exogenous design has invalid shape"
    elif not np.isfinite(matrix).all():
        exog_error = "exogenous design contains non-finite values"
    elif exog_rank != len(columns):
        exog_error = "rank-deficient actual exogenous design"

    selected_fit_kwargs = (
        dict(fit_kwargs)
        if fit_kwargs is not None
        else dict(SARIMAX_FORECAST_FIT_KWARGS)
        if forecast_only
        else None
    )
    fit_options = {
        "fit_kwargs": selected_fit_kwargs,
        "require_standard_errors": not forecast_only,
        "retry_maxiter": None if forecast_only else 1000,
        "validate_exog": False,
    }
    if progress_callback is not None:
        fit_options["progress_callback"] = progress_callback
    if fixed_params is not None:
        fit_options["fixed_params"] = fixed_params
    fit = fit_sarima(
        values if exog_error is None else [],
        order,
        exog=None if exog_error else exog,
        **fit_options,
    )
    return SarimaxFitRecord(
        sarima_fit=fit,
        exog_columns=columns,
        exog_rank=exog_rank,
        exog_condition_number=condition_number,
        transformed_exog_rank=transformed_rank,
        exog_error_message=exog_error,
    )
