from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
PREPROCESSING_ROOT = Path(__file__).resolve().parents[2] / "02_preprocessing"
if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_ROOT))

from calendar_features import build_calendar_features
from crisis_features import build_crisis_features


ERROR_ORDER = (2, 1, 2)
TREND = "n"
ROOT_THRESHOLD = 1.01
RESIDUAL_LAGS = (24, 48)
RESIDUAL_ALPHA = 0.01
RESIDUAL_ACF_Z = 2.576

SPECIFICATION_IDS = (
    "reg_arima_h",
    "reg_arima_h_d",
    "reg_arima_h_d_m",
    "reg_arima_h_d_m_hol",
)
INTERVENTION_SPECIFICATION_ID = "reg_arima_h_d_m_hol_crisis"
CRISIS_SPECIFICATION_ID = "arima_p2_d1_q2_crisis"
SPECIFICATION_GROUPS = {
    "reg_arima_h": ("hour",),
    "reg_arima_h_d": ("hour", "weekday"),
    "reg_arima_h_d_m": ("hour", "weekday", "month"),
    "reg_arima_h_d_m_hol": ("hour", "weekday", "month", "holiday"),
    INTERVENTION_SPECIFICATION_ID: (
        "hour",
        "weekday",
        "month",
        "holiday",
        "intervention",
    ),
    CRISIS_SPECIFICATION_ID: ("crisis",),
}


def specification_columns(specification_id: str) -> tuple[str, ...]:
    if specification_id not in SPECIFICATION_GROUPS:
        raise ValueError(f"unsupported regression specification: {specification_id}")
    columns: list[str] = []
    groups = SPECIFICATION_GROUPS[specification_id]
    if "hour" in groups:
        columns.extend(f"hour_{hour:02d}" for hour in range(1, 24))
    if "weekday" in groups:
        columns.extend(f"weekday_{day}" for day in range(1, 7))
    if "month" in groups:
        columns.extend(f"month_{month:02d}" for month in range(2, 13))
    if "holiday" in groups:
        columns.append("public_holiday")
    if "intervention" in groups or "crisis" in groups:
        columns.extend(("is_covid_period", "is_post_invasion"))
    return tuple(columns)


def build_exog(
    timestamps_utc: pd.Series | pd.DatetimeIndex,
    country: str,
    specification_id: str,
) -> pd.DataFrame:
    columns = specification_columns(specification_id)
    calendar = build_calendar_features(timestamps_utc, country)
    exog = pd.DataFrame(index=pd.RangeIndex(len(calendar)))
    if "hour" in SPECIFICATION_GROUPS[specification_id]:
        for hour in range(1, 24):
            exog[f"hour_{hour:02d}"] = (
                calendar["hour"].to_numpy() == hour
            ).astype(float)
    if "weekday" in SPECIFICATION_GROUPS[specification_id]:
        for day in range(1, 7):
            exog[f"weekday_{day}"] = (
                calendar["day_of_week"].to_numpy() == day
            ).astype(float)
    if "month" in SPECIFICATION_GROUPS[specification_id]:
        for month in range(2, 13):
            exog[f"month_{month:02d}"] = (
                calendar["month"].to_numpy() == month
            ).astype(float)
    if "holiday" in SPECIFICATION_GROUPS[specification_id]:
        exog["public_holiday"] = calendar["is_public_holiday"].to_numpy().astype(float)
    if "intervention" in SPECIFICATION_GROUPS[specification_id] or "crisis" in SPECIFICATION_GROUPS[specification_id]:
        crisis_source = calendar[["timestamp_utc", "timestamp_local"]].copy()
        for column in crisis_source.columns:
            crisis_source[column] = crisis_source[column].map(
                lambda value: pd.Timestamp(value).isoformat()
            )
        crisis = build_crisis_features(crisis_source)
        exog["is_covid_period"] = crisis["is_covid_period"].to_numpy(dtype=float)
        exog["is_post_invasion"] = crisis["is_post_invasion"].to_numpy(dtype=float)
    exog = exog.reindex(columns=columns)
    if len(exog) != len(calendar) or not exog.index.equals(pd.RangeIndex(len(calendar))):
        raise ValueError("exogenous rows are not aligned with calendar timestamps")
    if not np.isfinite(exog.to_numpy(dtype=float)).all():
        raise ValueError("exogenous design contains non-finite values")
    return exog


@dataclass(frozen=True)
class ResidualDiagnostics:
    n_effective: int
    acf_values: dict[int, float]
    acf_threshold: float
    flagged_acf_lags: tuple[int, ...]
    ljung_box_statistics: dict[int, float]
    ljung_box_pvalues: dict[int, float]
    warning_messages: tuple[str, ...]


@dataclass(frozen=True)
class RegressionArimaFitRecord:
    specification_id: str
    order: tuple[int, int, int]
    trend: str
    fitted_result: Any | None
    fit_status: str
    convergence_status: str
    optimizer_attempts: tuple[dict[str, object], ...]
    optimizer_retry_count: int
    warning_messages: tuple[str, ...]
    coefficient_warnings: tuple[str, ...]
    error_message: str | None
    parameters_finite: bool
    standard_errors_finite: bool
    converged: bool
    nobs: int
    n_params: int
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    parameter_estimates: dict[str, float]
    standard_errors: dict[str, float]
    exog_columns: tuple[str, ...]
    exog_rank: int
    exog_condition_number: float
    transformed_exog_rank: int | None
    ar_root_minimum: float
    ma_root_minimum: float
    stationarity_ok: bool
    invertibility_ok: bool
    residual_diagnostics: ResidualDiagnostics | None

    @property
    def eligible(self) -> bool:
        return bool(
            self.error_message is None
            and self.fitted_result is not None
            and self.converged
            and self.parameters_finite
            and np.isfinite(self.log_likelihood)
            and self.stationarity_ok
            and self.invertibility_ok
            and self.exog_rank == len(self.exog_columns)
        )


def _finite_float(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _retvals(result: Any) -> Mapping[str, Any]:
    value = getattr(result, "mle_retvals", None)
    return value if isinstance(value, Mapping) else {}


def _bool_value(value: object) -> bool:
    return bool(value is True or isinstance(value, np.bool_) and bool(value))


def _serializable_value(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _attempt_record(
    method_kwargs: dict[str, int],
    result: Any | None,
    warnings_seen: tuple[str, ...],
    error_message: str | None,
) -> dict[str, object]:
    retvals = _retvals(result)
    return {
        "method_kwargs": dict(method_kwargs),
        "converged": _bool_value(retvals.get("converged")),
        "success": _bool_value(retvals.get("success")),
        "warnflag": _serializable_value(retvals.get("warnflag"))
        if retvals.get("warnflag") is not None
        else None,
        "warning_messages": list(warnings_seen),
        "error_message": error_message,
    }


def fit_with_retry(
    fit_once: Callable[
        [dict[str, int] | None], tuple[Any | None, tuple[str, ...], str | None]
    ],
) -> tuple[Any | None, tuple[dict[str, object], ...], int]:
    attempts: list[dict[str, object]] = []
    result, first_warnings, first_error = fit_once(None)
    attempts.append(_attempt_record({}, result, first_warnings, first_error))
    if first_error is not None or result is None:
        return None, tuple(attempts), 0
    if _bool_value(_retvals(result).get("converged")):
        return result, tuple(attempts), 0

    result, retry_warnings, retry_error = fit_once({"maxiter": 1000})
    attempts.append(
        _attempt_record({"maxiter": 1000}, result, retry_warnings, retry_error)
    )
    return result, tuple(attempts), 1


def _fit_once(
    values: np.ndarray,
    exog: pd.DataFrame,
    method_kwargs: dict[str, int] | None,
) -> tuple[Any | None, tuple[str, ...], str | None]:
    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            model = ARIMA(
                values,
                exog=exog,
                order=ERROR_ORDER,
                trend=TREND,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            fitted = model.fit(
                **({"method_kwargs": method_kwargs} if method_kwargs else {})
            )
        warning_messages.extend(
            f"{item.category.__name__}: {item.message}" for item in recorded
        )
        return fitted, tuple(warning_messages), None
    except Exception as error:
        return None, tuple(warning_messages), f"{type(error).__name__}: {error}"


def _minimum_root(roots: object) -> float:
    if roots is None:
        return float("inf")
    values = np.asarray(roots, dtype=complex).reshape(-1)
    if not len(values):
        return float("inf")
    moduli = np.abs(values)
    return float(np.min(moduli)) if np.isfinite(moduli).all() else float("nan")


def _stable_roots(roots: object) -> bool:
    if roots is None:
        return True
    values = np.asarray(roots, dtype=complex).reshape(-1)
    return bool(len(values) == 0 or (np.isfinite(np.abs(values)).all() and (np.abs(values) > ROOT_THRESHOLD).all()))


def _aicc(aic: float, nobs: int, n_params: int) -> float:
    denominator = nobs - n_params - 1
    if not np.isfinite(aic) or denominator <= 0:
        return float("nan")
    return float(aic + (2 * n_params * (n_params + 1)) / denominator)


def _residual_diagnostics(result: Any) -> ResidualDiagnostics:
    warning_messages: list[str] = []
    try:
        residuals = np.asarray(result.resid, dtype=float).reshape(-1)
        residuals = residuals[max(24, sum(ERROR_ORDER)) :]
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) <= 48:
            return ResidualDiagnostics(len(residuals), {}, float("nan"), (), {}, {}, ())
        acf_values_array = acf(residuals, nlags=48, fft=True, adjusted=False)
        threshold = max(RESIDUAL_ACF_Z / math.sqrt(len(residuals)), 0.05)
        acf_values = {lag: float(acf_values_array[lag]) for lag in range(1, 49)}
        flagged = tuple(
            lag for lag, value in acf_values.items() if abs(value) > threshold
        )
        ljung_box = acorr_ljungbox(
            residuals,
            lags=list(RESIDUAL_LAGS),
            model_df=ERROR_ORDER[0] + ERROR_ORDER[2],
            return_df=True,
        )
        statistics = {lag: float(ljung_box.loc[lag, "lb_stat"]) for lag in RESIDUAL_LAGS}
        pvalues = {lag: float(ljung_box.loc[lag, "lb_pvalue"]) for lag in RESIDUAL_LAGS}
        return ResidualDiagnostics(
            len(residuals),
            acf_values,
            threshold,
            flagged,
            statistics,
            pvalues,
            tuple(warning_messages),
        )
    except Exception as error:
        warning_messages.append(f"residual diagnostics failed: {type(error).__name__}: {error}")
        return ResidualDiagnostics(0, {}, float("nan"), (), {}, {}, tuple(warning_messages))


def _failed_record(
    specification_id: str,
    columns: tuple[str, ...],
    exog_rank: int,
    exog_condition_number: float,
    transformed_exog_rank: int | None,
    error_message: str,
    warnings_seen: tuple[str, ...] = (),
    attempts: tuple[dict[str, object], ...] = (),
    retry_count: int = 0,
) -> RegressionArimaFitRecord:
    return RegressionArimaFitRecord(
        specification_id,
        ERROR_ORDER,
        TREND,
        None,
        "failed",
        "failed",
        attempts,
        retry_count,
        warnings_seen,
        tuple(item for item in warnings_seen if "covariance" in item.lower() or "hessian" in item.lower()),
        error_message,
        False,
        False,
        False,
        0,
        0,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        {},
        {},
        columns,
        exog_rank,
        exog_condition_number,
        transformed_exog_rank,
        float("nan"),
        float("nan"),
        False,
        False,
        None,
    )


def fit_regression_arima(
    values: object,
    exog: pd.DataFrame,
    specification_id: str,
) -> RegressionArimaFitRecord:
    columns = specification_columns(specification_id)
    if tuple(exog.columns) != columns:
        return _failed_record(
            specification_id,
            columns,
            0,
            float("nan"),
            None,
            "exogenous columns do not match the frozen specification",
        )
    numeric_values = np.asarray(values, dtype=float).reshape(-1)
    matrix = exog.to_numpy(dtype=float)
    exog_rank = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    exog_condition_number = float(np.linalg.cond(matrix)) if matrix.size else float("nan")
    transformed = np.diff(matrix, n=1, axis=0) if len(matrix) > 1 else matrix
    transformed_rank = int(np.linalg.matrix_rank(transformed)) if transformed.size else 0
    if len(numeric_values) != len(exog):
        return _failed_record(
            specification_id,
            columns,
            exog_rank,
            exog_condition_number,
            transformed_rank,
            "endog and exog row counts do not match",
        )
    if not np.isfinite(numeric_values).all() or not np.isfinite(matrix).all():
        return _failed_record(
            specification_id,
            columns,
            exog_rank,
            exog_condition_number,
            transformed_rank,
            "endog or exog contains non-finite values",
        )
    if exog_rank != len(columns):
        return _failed_record(
            specification_id,
            columns,
            exog_rank,
            exog_condition_number,
            transformed_rank,
            "rank-deficient actual exogenous design",
        )
    result, attempts, retry_count = fit_with_retry(
        lambda method_kwargs: _fit_once(numeric_values, exog, method_kwargs)
    )
    warnings_seen = tuple(
        warning
        for attempt in attempts
        for warning in attempt.get("warning_messages", [])
    )
    coefficient_warnings = tuple(
        warning
        for warning in warnings_seen
        if "covariance" in warning.lower() or "hessian" in warning.lower()
    )
    if result is None:
        error = str(attempts[-1].get("error_message") or "fit returned no result")
        return _failed_record(
            specification_id,
            columns,
            exog_rank,
            exog_condition_number,
            transformed_rank,
            error,
            warnings_seen,
            attempts,
            retry_count,
        )

    retvals = _retvals(result)
    converged = _bool_value(retvals.get("converged"))
    params = np.asarray(getattr(result, "params", []), dtype=float).reshape(-1)
    bse = np.asarray(getattr(result, "bse", []), dtype=float).reshape(-1)
    names = [str(name) for name in getattr(result, "param_names", [])]
    if len(names) != len(params):
        names = [f"parameter_{index}" for index in range(len(params))]
    parameter_estimates = {name: float(value) for name, value in zip(names, params)}
    standard_errors = {
        name: float(value) for name, value in zip(names[: len(bse)], bse)
    }
    parameters_finite = bool(len(params) > 0 and np.isfinite(params).all())
    standard_errors_finite = bool(len(bse) > 0 and np.isfinite(bse).all())
    nobs = int(getattr(result, "nobs", len(numeric_values)))
    n_params = int(len(params))
    log_likelihood = float(getattr(result, "llf", float("nan")))
    aic = float(getattr(result, "aic", float("nan")))
    bic = float(getattr(result, "bic", float("nan")))
    native_aicc = getattr(result, "aicc", None)
    aicc = (
        float(native_aicc)
        if native_aicc is not None and _finite_float(native_aicc)
        else _aicc(aic, nobs, n_params)
    )
    stationarity_ok = _stable_roots(getattr(result, "arroots", None))
    invertibility_ok = _stable_roots(getattr(result, "maroots", None))
    reasons: list[str] = []
    if not converged:
        reasons.append("optimizer did not converge after approved retry")
    if not parameters_finite:
        reasons.append("non-finite regression coefficient estimates")
    if not _finite_float(log_likelihood):
        reasons.append("non-finite likelihood")
    if not stationarity_ok:
        reasons.append("AR roots fail stationarity threshold")
    if not invertibility_ok:
        reasons.append("MA roots fail invertibility threshold")
    if not all(_finite_float(value) for value in (aic, aicc, bic)):
        reasons.append("non-finite information criterion")
    return RegressionArimaFitRecord(
        specification_id,
        ERROR_ORDER,
        TREND,
        result,
        "success" if not reasons else "failed",
        "converged" if converged else "failed",
        attempts,
        retry_count,
        warnings_seen,
        coefficient_warnings,
        "; ".join(reasons) if reasons else None,
        parameters_finite,
        standard_errors_finite,
        converged,
        nobs,
        n_params,
        log_likelihood,
        aic,
        aicc,
        bic,
        parameter_estimates,
        standard_errors,
        columns,
        exog_rank,
        exog_condition_number,
        transformed_rank,
        _minimum_root(getattr(result, "arroots", None)),
        _minimum_root(getattr(result, "maroots", None)),
        stationarity_ok,
        invertibility_ok,
        _residual_diagnostics(result),
    )


def forecast_values(fitted_result: Any, exog: pd.DataFrame) -> np.ndarray:
    if exog.empty:
        raise ValueError("forecast exog is empty")
    forecast = np.asarray(
        fitted_result.get_forecast(steps=len(exog), exog=exog).predicted_mean,
        dtype=float,
    )
    if forecast.shape != (len(exog),) or not np.isfinite(forecast).all():
        raise ValueError("regression ARIMA forecast output is unusable")
    return forecast
