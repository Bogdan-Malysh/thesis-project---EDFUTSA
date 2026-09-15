from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping
import warnings

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf


ROOT_THRESHOLD = 1.01
RESIDUAL_ALPHA = 0.01
RESIDUAL_ACF_Z = 2.576
RESIDUAL_LAGS = (24, 48)


@dataclass(frozen=True)
class ResidualDiagnostics:
    n_effective: int
    acf_values: dict[int, float]
    acf_threshold: float
    flagged_acf_lags: tuple[int, ...]
    ljung_box_statistics: dict[int, float]
    ljung_box_pvalues: dict[int, float]
    rejected: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class ArimaFitRecord:
    order: tuple[int, int, int]
    trend: str
    fitted_result: Any | None
    fit_status: str
    convergence_status: str
    warning_messages: tuple[str, ...]
    hard_warning_messages: tuple[str, ...]
    optimizer_retry_count: int
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
    ar_root_minimum: float
    ma_root_minimum: float
    stationarity_ok: bool
    invertibility_ok: bool
    residual_diagnostics: ResidualDiagnostics | None

    @property
    def eligible(self) -> bool:
        return self.error_message is None and self.converged and self.stationarity_ok and self.invertibility_ok


def trend_for_d(d: int) -> str:
    if d == 0:
        return "c"
    if d in (1, 2):
        return "n"
    raise ValueError(f"unsupported ARIMA differencing order: {d}")


def order_id(order: tuple[int, int, int]) -> str:
    p, d, q = order
    return f"arima_p{p}_d{d}_q{q}"


def root_is_stable(roots: object, threshold: float = ROOT_THRESHOLD) -> bool:
    if roots is None:
        return True
    values = np.asarray(roots, dtype=complex).reshape(-1)
    if not len(values):
        return True
    moduli = np.abs(values)
    return bool(np.isfinite(moduli).all() and (moduli > threshold).all())


def residual_acf_threshold(n_effective: int) -> float:
    if n_effective <= 0:
        raise ValueError("n_effective must be positive")
    return max(RESIDUAL_ACF_Z / math.sqrt(n_effective), 0.05)


def residual_rejection(
    ljung_box_pvalues: dict[int, float],
    acf_values: dict[int, float],
    n_effective: int,
) -> tuple[bool, float, list[int]]:
    threshold = residual_acf_threshold(n_effective)
    flagged = sorted(
        lag
        for lag, value in acf_values.items()
        if np.isfinite(value) and abs(float(value)) > threshold
    )
    pvalues = [ljung_box_pvalues.get(lag, float("nan")) for lag in RESIDUAL_LAGS]
    if not all(np.isfinite(value) for value in pvalues):
        return True, threshold, flagged
    rejected = all(float(value) < RESIDUAL_ALPHA for value in pvalues) and len(flagged) >= 2
    return rejected, threshold, flagged


def classify_warning(category_name: str, message: str) -> tuple[bool, str]:
    normalized = f"{category_name}: {message}".strip()
    lower = normalized.lower()
    covariance_markers = (
        "hessian",
        "covariance",
        "covar",
        "singular covariance",
        "non-positive-definite hessian",
    )
    if any(marker in lower for marker in covariance_markers):
        return False, normalized
    hard_categories = {"ConvergenceWarning", "RuntimeWarning", "FloatingPointError"}
    hard_markers = (
        "converg",
        "overflow",
        "underflow",
        "invalid value",
        "nan",
        "infinite",
    )
    is_hard = category_name in hard_categories or any(
        marker in lower for marker in hard_markers
    )
    return is_hard, normalized


def _finite_float(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _minimum_root(roots: object) -> float:
    if roots is None:
        return float("inf")
    values = np.asarray(roots, dtype=complex).reshape(-1)
    if not len(values):
        return float("inf")
    moduli = np.abs(values)
    return float(np.min(moduli)) if np.isfinite(moduli).all() else float("nan")


def _retvals_mapping(result: Any) -> Mapping[str, Any]:
    retvals = getattr(result, "mle_retvals", None)
    return retvals if isinstance(retvals, Mapping) else {}


def _empty_residual_diagnostics() -> ResidualDiagnostics:
    return ResidualDiagnostics(
        n_effective=0,
        acf_values={},
        acf_threshold=float("nan"),
        flagged_acf_lags=(),
        ljung_box_statistics={},
        ljung_box_pvalues={},
        rejected=True,
        rejection_reason="residual diagnostics unavailable",
    )


def _residual_diagnostics(
    result: Any,
    order: tuple[int, int, int],
) -> ResidualDiagnostics:
    p, d, q = order
    residuals = np.asarray(getattr(result, "resid"), dtype=float).reshape(-1)
    burn_in = max(24, p + q + d)
    residuals = residuals[burn_in:]
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) <= 48:
        return _empty_residual_diagnostics()

    acf_values_array = acf(residuals, nlags=48, fft=True, adjusted=False)
    acf_values = {
        lag: float(acf_values_array[lag]) for lag in range(1, 49)
    }
    threshold = residual_acf_threshold(len(residuals))
    flagged = sorted(
        lag for lag, value in acf_values.items() if abs(value) > threshold
    )
    ljung_box = acorr_ljungbox(
        residuals,
        lags=list(RESIDUAL_LAGS),
        model_df=p + q,
        return_df=True,
    )
    statistics = {
        int(lag): float(ljung_box.loc[lag, "lb_stat"])
        for lag in RESIDUAL_LAGS
    }
    pvalues = {
        int(lag): float(ljung_box.loc[lag, "lb_pvalue"])
        for lag in RESIDUAL_LAGS
    }
    rejected, _, _ = residual_rejection(pvalues, acf_values, len(residuals))
    reason = (
        "joint Ljung-Box and residual ACF rule"
        if rejected
        else None
    )
    return ResidualDiagnostics(
        n_effective=len(residuals),
        acf_values=acf_values,
        acf_threshold=threshold,
        flagged_acf_lags=tuple(flagged),
        ljung_box_statistics=statistics,
        ljung_box_pvalues=pvalues,
        rejected=rejected,
        rejection_reason=reason,
    )


def _aicc(aic: float, nobs: int, n_params: int) -> float:
    denominator = nobs - n_params - 1
    if not np.isfinite(aic) or denominator <= 0:
        return float("nan")
    return float(aic + (2 * n_params * (n_params + 1)) / denominator)


def _failed_record(
    order: tuple[int, int, int],
    trend: str,
    message: str,
    warnings_seen: tuple[str, ...] = (),
    hard_warnings: tuple[str, ...] = (),
    optimizer_retry_count: int = 0,
) -> ArimaFitRecord:
    return ArimaFitRecord(
        order=order,
        trend=trend,
        fitted_result=None,
        fit_status="failed",
        convergence_status="failed",
        warning_messages=warnings_seen,
        hard_warning_messages=hard_warnings,
        optimizer_retry_count=optimizer_retry_count,
        error_message=message,
        parameters_finite=False,
        standard_errors_finite=False,
        converged=False,
        nobs=0,
        n_params=0,
        log_likelihood=float("nan"),
        aic=float("nan"),
        aicc=float("nan"),
        bic=float("nan"),
        ar_root_minimum=float("nan"),
        ma_root_minimum=float("nan"),
        stationarity_ok=False,
        invertibility_ok=False,
        residual_diagnostics=_empty_residual_diagnostics(),
    )


def _fit_once(
    numeric_values: np.ndarray,
    order: tuple[int, int, int],
    trend: str,
    method_kwargs: dict[str, int] | None = None,
) -> tuple[Any | None, tuple[str, ...], tuple[str, ...], str | None]:
    warning_messages: list[str] = []
    hard_warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            model = ARIMA(
                numeric_values,
                order=order,
                trend=trend,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            fitted = model.fit(**({"method_kwargs": method_kwargs} if method_kwargs else {}))
        for caught in recorded:
            is_hard, formatted = classify_warning(
                caught.category.__name__, str(caught.message)
            )
            warning_messages.append(formatted)
            if is_hard:
                hard_warning_messages.append(formatted)
        return (
            fitted,
            tuple(warning_messages),
            tuple(hard_warning_messages),
            None,
        )
    except Exception as error:
        return (
            None,
            tuple(warning_messages),
            tuple(hard_warning_messages),
            f"{type(error).__name__}: {error}",
        )


def fit_arima(values: object, order: tuple[int, int, int]) -> ArimaFitRecord:
    numeric_values = np.asarray(values, dtype=float).reshape(-1)
    trend = trend_for_d(order[1])
    if len(numeric_values) <= 50:
        return _failed_record(order, trend, "insufficient observations")
    if not np.isfinite(numeric_values).all():
        return _failed_record(order, trend, "training values are not finite")

    fitted, first_warnings, first_hard_warnings, error_message = _fit_once(
        numeric_values, order, trend
    )
    if error_message is not None or fitted is None:
        return _failed_record(
            order,
            trend,
            error_message or "ARIMA fit returned no result",
            first_warnings,
            first_hard_warnings,
        )

    warning_messages = list(first_warnings)
    retry_count = 0
    retvals = _retvals_mapping(fitted)
    converged = retvals.get("converged") is True
    if not converged:
        retry_count = 1
        retried, retry_warnings, retry_hard_warnings, retry_error = _fit_once(
            numeric_values, order, trend, {"maxiter": 1000}
        )
        warning_messages.extend(retry_warnings)
        if retry_error is not None or retried is None:
            return _failed_record(
                order,
                trend,
                f"optimizer retry failed: {retry_error or 'ARIMA fit returned no result'}",
                tuple(warning_messages),
                tuple(first_hard_warnings + retry_hard_warnings),
                retry_count,
            )
        fitted = retried
        retvals = _retvals_mapping(fitted)
        converged = retvals.get("converged") is True
        hard_warning_messages = retry_hard_warnings
    else:
        hard_warning_messages = list(first_hard_warnings)

    success = retvals.get("success")
    warnflag = retvals.get("warnflag")
    params = np.asarray(getattr(fitted, "params", []), dtype=float)
    bse = np.asarray(getattr(fitted, "bse", []), dtype=float)
    parameters_finite = bool(len(params) and np.isfinite(params).all())
    standard_errors_finite = bool(len(bse) and np.isfinite(bse).all())
    nobs = int(getattr(fitted, "nobs", len(numeric_values)))
    n_params = int(len(params))
    log_likelihood = float(getattr(fitted, "llf", float("nan")))
    aic = float(getattr(fitted, "aic", float("nan")))
    bic = float(getattr(fitted, "bic", float("nan")))
    aicc = _aicc(aic, nobs, n_params)
    ar_roots = getattr(fitted, "arroots", None)
    ma_roots = getattr(fitted, "maroots", None)
    stationarity_ok = root_is_stable(ar_roots)
    invertibility_ok = root_is_stable(ma_roots)
    residual_diagnostics = _residual_diagnostics(fitted, order)

    reasons: list[str] = []
    if not converged:
        reasons.append("optimizer did not explicitly converge")
    if success is not None and success is not True:
        reasons.append("optimizer reported failure")
    if warnflag is not None and warnflag != 0:
        reasons.append(f"optimizer warnflag={warnflag}")
    if hard_warning_messages:
        reasons.append("invalid or failed-fit warning")
    if not parameters_finite:
        reasons.append("non-finite parameter estimates")
    if not standard_errors_finite:
        reasons.append("non-finite standard errors")
    if not all(
        _finite_float(value)
        for value in (log_likelihood, aic, aicc, bic)
    ):
        reasons.append("non-finite likelihood or information criterion")
    if not stationarity_ok:
        reasons.append("AR roots fail stationarity threshold")
    if not invertibility_ok:
        reasons.append("MA roots fail invertibility threshold")
    return ArimaFitRecord(
        order=order,
        trend=trend,
        fitted_result=fitted,
        fit_status="success",
        convergence_status="converged" if converged else "failed",
        warning_messages=tuple(warning_messages),
        hard_warning_messages=tuple(hard_warning_messages),
        optimizer_retry_count=retry_count,
        error_message="; ".join(reasons) if reasons else None,
        parameters_finite=parameters_finite,
        standard_errors_finite=standard_errors_finite,
        converged=converged,
        nobs=nobs,
        n_params=n_params,
        log_likelihood=log_likelihood,
        aic=aic,
        aicc=aicc,
        bic=bic,
        ar_root_minimum=_minimum_root(ar_roots),
        ma_root_minimum=_minimum_root(ma_roots),
        stationarity_ok=stationarity_ok,
        invertibility_ok=invertibility_ok,
        residual_diagnostics=residual_diagnostics,
    )


def forecast_values(fitted_result: Any, steps: int) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive")
    forecast = np.asarray(fitted_result.get_forecast(steps=steps).predicted_mean, dtype=float)
    if forecast.shape != (steps,) or not np.isfinite(forecast).all():
        raise ValueError("ARIMA forecast output is unusable")
    return forecast
