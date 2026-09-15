from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import re
from typing import Any
import warnings

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf


SEASONAL_PERIOD = 24
ROOT_THRESHOLD = 1.01
RESIDUAL_ALPHA = 0.01
RESIDUAL_ACF_Z = 2.576
RESIDUAL_ACF_LAGS = tuple(range(1, 49))
RESIDUAL_LAGS = (24, 48)

_CORE_ORDERS = (
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (1, 1, 0, 0),
    (2, 0, 0, 0),
    (0, 2, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)
_INTERACTION_ORDERS = (
    (1, 0, 1, 0),
    (0, 1, 0, 1),
    (1, 1, 1, 0),
    (1, 1, 0, 1),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
)


@dataclass(frozen=True)
class SarimaOrder:
    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    seasonal_period: int = SEASONAL_PERIOD

    def __post_init__(self) -> None:
        bounds = {
            "p": (0, 2),
            "d": (0, 2),
            "q": (0, 2),
            "P": (0, 1),
            "D": (0, 1),
            "Q": (0, 1),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
                raise ValueError(f"unsupported SARIMA {name}: {value}")
        if self.seasonal_period != SEASONAL_PERIOD:
            raise ValueError(f"seasonal_period must be {SEASONAL_PERIOD}")

    @property
    def nonseasonal_order(self) -> tuple[int, int, int]:
        return self.p, self.d, self.q

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return self.P, self.D, self.Q, self.seasonal_period

    @property
    def specification_id(self) -> str:
        return specification_id(self)


@dataclass(frozen=True)
class ResidualDiagnostics:
    burn_in: int
    state_space_loglikelihood_burn: int
    conservative_burn_in: int
    n_effective: int
    acf_values: dict[int, float]
    acf_threshold: float
    flagged_acf_lags: tuple[int, ...]
    ljung_box_statistics: dict[int, float]
    ljung_box_pvalues: dict[int, float]
    training_adequacy_rejected: bool
    training_adequacy_reason: str | None
    warnings: tuple[str, ...] = ()

    @classmethod
    def empty(cls, n_effective: int = 0, burn_in: int = 0) -> ResidualDiagnostics:
        return cls(
            burn_in=burn_in,
            state_space_loglikelihood_burn=0,
            conservative_burn_in=burn_in,
            n_effective=n_effective,
            acf_values={},
            acf_threshold=float("nan"),
            flagged_acf_lags=(),
            ljung_box_statistics={},
            ljung_box_pvalues={},
            training_adequacy_rejected=True,
            training_adequacy_reason="residual diagnostics unavailable",
        )


@dataclass(frozen=True)
class RootDiagnostics:
    combined_ar_roots: tuple[complex, ...] | None
    combined_ma_roots: tuple[complex, ...] | None
    reconstructed_nonseasonal_ar_roots: tuple[complex, ...] | None
    reconstructed_nonseasonal_ma_roots: tuple[complex, ...] | None
    reconstructed_seasonal_ar_roots: tuple[complex, ...] | None
    reconstructed_seasonal_ma_roots: tuple[complex, ...] | None
    combined_ar_method: str
    combined_ma_method: str
    reconstructed_component_method: str
    reconstructed_components: tuple[str, ...]

    @classmethod
    def empty(cls) -> RootDiagnostics:
        return cls(
            combined_ar_roots=None,
            combined_ma_roots=None,
            reconstructed_nonseasonal_ar_roots=None,
            reconstructed_nonseasonal_ma_roots=None,
            reconstructed_seasonal_ar_roots=None,
            reconstructed_seasonal_ma_roots=None,
            combined_ar_method="result.arroots (statsmodels combined AR roots)",
            combined_ma_method="result.maroots (statsmodels combined MA roots)",
            reconstructed_component_method="numpy.roots on fitted component lag polynomials",
            reconstructed_components=(),
        )


@dataclass(frozen=True)
class SarimaFitRecord:
    order: SarimaOrder
    trend: str
    fitted_result: Any | None
    fit_status: str
    convergence_status: str
    optimizer_success: bool | None
    optimizer_status: Any | None
    optimizer_warnflag: Any | None
    warning_messages: tuple[str, ...]
    hard_warning_messages: tuple[str, ...]
    optimizer_retry_count: int
    error_message: str | None
    parameters_finite: bool
    standard_errors_finite: bool
    converged: bool
    forecast_valid: bool
    forecast_error_message: str | None
    nobs: int
    effective_nobs: int
    n_params: int
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    root_diagnostics: RootDiagnostics
    ar_root_minimum: float
    ma_root_minimum: float
    stationarity_ok: bool
    invertibility_ok: bool
    residual_diagnostics: ResidualDiagnostics
    training_adequacy_rejected: bool
    optimizer_iterations: int | None = None
    optimizer_function_calls: int | None = None
    optimizer_message: str | None = None
    optimizer_gradient_norm: float | None = None

    @property
    def specification_id(self) -> str:
        return self.order.specification_id

    @property
    def eligible(self) -> bool:
        return self.error_message is None


def trend_for_d_D(d: int, D: int) -> str:
    if d not in (0, 1, 2):
        raise ValueError(f"unsupported SARIMA d: {d}")
    if D not in (0, 1):
        raise ValueError(f"unsupported SARIMA D: {D}")
    return "c" if d == 0 and D == 0 else "n"


def specification_id(order: SarimaOrder) -> str:
    return (
        f"sarima_p{order.p}_d{order.d}_q{order.q}_"
        f"P{order.P}_D{order.D}_Q{order.Q}_s{order.seasonal_period}"
    )


def candidate_pool(d: int, D: int) -> tuple[SarimaOrder, ...]:
    trend_for_d_D(d, D)
    bounded_orders = _CORE_ORDERS + _INTERACTION_ORDERS
    return tuple(
        SarimaOrder(p, d, q, P, D, Q)
        for p, q, P, Q in bounded_orders
    )


def root_is_stable(roots: object, threshold: float = ROOT_THRESHOLD) -> bool:
    if roots is None:
        return True
    try:
        values = np.asarray(roots, dtype=complex).reshape(-1)
    except (TypeError, ValueError):
        return False
    if not len(values):
        return True
    moduli = np.abs(values)
    valid_moduli = np.isfinite(moduli) | np.isinf(moduli)
    return bool(valid_moduli.all() and (moduli > threshold).all())


def _required_roots_are_stable(roots: object, term_count: int) -> bool:
    if term_count == 0:
        return True
    if roots is None:
        return False
    try:
        values = np.asarray(roots, dtype=complex).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(len(values) and root_is_stable(values))


def residual_acf_threshold(n_effective: int) -> float:
    if n_effective <= 0:
        raise ValueError("n_effective must be positive")
    return max(RESIDUAL_ACF_Z / math.sqrt(n_effective), 0.05)


def residual_rejection(
    ljung_box_pvalues: Mapping[int, float],
    acf_values: Mapping[int, float],
    n_effective: int,
) -> tuple[bool, float, list[int]]:
    threshold = residual_acf_threshold(n_effective)
    flagged = sorted(
        lag
        for lag, value in acf_values.items()
        if np.isfinite(value) and abs(float(value)) > threshold
    )
    pvalues = [ljung_box_pvalues.get(lag, float("nan")) for lag in RESIDUAL_LAGS]
    rejected = (
        all(np.isfinite(value) for value in pvalues)
        and all(float(value) < RESIDUAL_ALPHA for value in pvalues)
        and len(flagged) >= 2
    )
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
    try:
        values = np.asarray(roots, dtype=complex).reshape(-1)
    except (TypeError, ValueError):
        return float("nan")
    if not len(values):
        return float("inf")
    moduli = np.abs(values)
    return float(np.min(moduli)) if np.isfinite(moduli).all() else float("nan")


def _as_complex_tuple(roots: object) -> tuple[complex, ...] | None:
    if roots is None:
        return None
    try:
        return tuple(complex(value) for value in np.asarray(roots, dtype=complex).reshape(-1))
    except (TypeError, ValueError):
        return None


def _retvals_mapping(result: Any) -> Mapping[str, Any]:
    retvals = getattr(result, "mle_retvals", None)
    return retvals if isinstance(retvals, Mapping) else {}


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retvals_count(retvals: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in retvals:
            return _optional_int(retvals[key])
    return None


def _retvals_message(retvals: Mapping[str, Any]) -> str | None:
    for key in ("message", "task"):
        value = retvals.get(key)
        if value is not None:
            return str(value)
    return None


def _retvals_gradient_norm(retvals: Mapping[str, Any]) -> float | None:
    for key in ("grad", "jac", "gopt"):
        value = retvals.get(key)
        if value is None:
            continue
        try:
            gradient = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(gradient) and np.isfinite(gradient).all():
            return float(np.linalg.norm(gradient))
    return None


def _is_explicit_true(value: object) -> bool:
    return value is True or isinstance(value, np.bool_) and bool(value)


def _status_failed(value: object) -> bool:
    if value is None:
        return False
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return str(value).strip().lower() not in {"0", "success", "ok"}


def _parameter_mapping(result: Any) -> dict[str, float]:
    params = getattr(result, "params", None)
    names = getattr(result, "param_names", None)
    if names is None:
        model = getattr(result, "model", None)
        names = getattr(model, "param_names", None)
    if names is None and hasattr(params, "index"):
        names = list(params.index)
    if names is None or params is None:
        return {}
    try:
        values = np.asarray(params, dtype=float).reshape(-1)
        names = list(names)
    except (TypeError, ValueError):
        return {}
    if len(values) != len(names):
        return {}
    return {str(name): float(value) for name, value in zip(names, values)}


def _reconstruct_component_roots(
    result: Any,
    prefix: str,
    degree: int,
    sign: float,
    seasonal_period: int = 1,
) -> tuple[complex, ...] | None:
    if degree == 0:
        return ()
    values = _parameter_mapping(result)
    pattern = re.compile(rf"^{re.escape(prefix)}\.L(\d+)$")
    coefficients: dict[int, float] = {}
    for name, value in values.items():
        match = pattern.fullmatch(name)
        if match is not None:
            raw_lag = int(match.group(1))
            if raw_lag % seasonal_period:
                continue
            lag = raw_lag // seasonal_period
            if 1 <= lag <= degree:
                coefficients[lag] = value
    if set(coefficients) != set(range(1, degree + 1)):
        return None
    if not all(np.isfinite(value) for value in coefficients.values()):
        return None
    polynomial = np.ones(degree + 1, dtype=float)
    for lag, value in coefficients.items():
        polynomial[lag] = sign * value
    # ``polynomial`` is ordered by increasing lag; numpy.roots expects descending powers.
    try:
        return _as_complex_tuple(np.roots(polynomial[::-1]))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _root_diagnostics(result: Any, order: SarimaOrder) -> RootDiagnostics:
    component_specs = (
        (
            "nonseasonal_ar",
            "ar",
            order.p,
            -1.0,
        ),
        (
            "nonseasonal_ma",
            "ma",
            order.q,
            1.0,
        ),
        (
            "seasonal_ar",
            "ar.S",
            order.P,
            -1.0,
        ),
        (
            "seasonal_ma",
            "ma.S",
            order.Q,
            1.0,
        ),
    )
    reconstructed: dict[str, tuple[complex, ...] | None] = {}
    calculated: list[str] = []
    for label, prefix, degree, sign in component_specs:
        roots = _reconstruct_component_roots(
            result,
            prefix,
            degree,
            sign,
            order.seasonal_period if prefix.endswith(".S") else 1,
        )
        reconstructed[label] = roots
        if roots is not None and degree:
            calculated.append(label)
    return RootDiagnostics(
        combined_ar_roots=_as_complex_tuple(getattr(result, "arroots", None)),
        combined_ma_roots=_as_complex_tuple(getattr(result, "maroots", None)),
        reconstructed_nonseasonal_ar_roots=reconstructed["nonseasonal_ar"],
        reconstructed_nonseasonal_ma_roots=reconstructed["nonseasonal_ma"],
        reconstructed_seasonal_ar_roots=reconstructed["seasonal_ar"],
        reconstructed_seasonal_ma_roots=reconstructed["seasonal_ma"],
        combined_ar_method="result.arroots (statsmodels combined AR roots)",
        combined_ma_method="result.maroots (statsmodels combined MA roots)",
        reconstructed_component_method=(
            "numpy.roots on fitted parameter-name lag polynomials; "
            "AR coefficients use [1, -phi], MA coefficients use [1, theta]"
        ),
        reconstructed_components=tuple(calculated),
    )


def _state_space_burn(result: Any) -> int:
    model = getattr(result, "model", None)
    for source in (result, model):
        value = getattr(source, "loglikelihood_burn", None)
        if value is None:
            continue
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric >= 0:
            return numeric
    return 0


def _empty_residual_diagnostics(
    burn_in: int,
    state_space_burn: int,
    conservative_burn_in: int,
    n_effective: int,
    reason: str,
    diagnostic_warnings: tuple[str, ...] = (),
) -> ResidualDiagnostics:
    return ResidualDiagnostics(
        burn_in=burn_in,
        state_space_loglikelihood_burn=state_space_burn,
        conservative_burn_in=conservative_burn_in,
        n_effective=n_effective,
        acf_values={},
        acf_threshold=float("nan"),
        flagged_acf_lags=(),
        ljung_box_statistics={},
        ljung_box_pvalues={},
        training_adequacy_rejected=True,
        training_adequacy_reason=reason,
        warnings=diagnostic_warnings,
    )


def _residual_diagnostics(result: Any, order: SarimaOrder) -> ResidualDiagnostics:
    conservative_burn_in = max(
        48,
        order.p
        + order.q
        + order.d
        + SEASONAL_PERIOD * (order.P + order.Q + order.D),
    )
    state_space_burn = _state_space_burn(result)
    burn_in = max(state_space_burn, conservative_burn_in)
    try:
        residuals = np.asarray(getattr(result, "resid"), dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError):
        return _empty_residual_diagnostics(
            burn_in,
            state_space_burn,
            conservative_burn_in,
            0,
            "residuals unavailable",
        )
    residuals = residuals[burn_in:]
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) <= max(RESIDUAL_ACF_LAGS):
        return _empty_residual_diagnostics(
            burn_in,
            state_space_burn,
            conservative_burn_in,
            len(residuals),
            "insufficient residuals after burn-in",
        )

    diagnostic_warnings: tuple[str, ...] = ()
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            acf_values_array = acf(
                residuals,
                nlags=max(RESIDUAL_ACF_LAGS),
                fft=True,
                adjusted=False,
            )
            acf_values = {
                lag: float(acf_values_array[lag]) for lag in RESIDUAL_ACF_LAGS
            }
            ljung_box = acorr_ljungbox(
                residuals,
                lags=list(RESIDUAL_LAGS),
                model_df=order.p + order.q + order.P + order.Q,
                return_df=True,
            )
        diagnostic_warnings = tuple(
            f"{caught.category.__name__}: {caught.message}"
            for caught in recorded
        )
    except Exception as error:
        return _empty_residual_diagnostics(
            burn_in,
            state_space_burn,
            conservative_burn_in,
            len(residuals),
            f"residual diagnostics failed: {type(error).__name__}: {error}",
            diagnostic_warnings,
        )

    statistics = {
        int(lag): float(ljung_box.loc[lag, "lb_stat"])
        for lag in RESIDUAL_LAGS
    }
    pvalues = {
        int(lag): float(ljung_box.loc[lag, "lb_pvalue"])
        for lag in RESIDUAL_LAGS
    }
    if not (
        np.isfinite(list(acf_values.values())).all()
        and np.isfinite(list(statistics.values())).all()
        and np.isfinite(list(pvalues.values())).all()
    ):
        return _empty_residual_diagnostics(
            burn_in,
            state_space_burn,
            conservative_burn_in,
            len(residuals),
            "non-finite residual diagnostics",
            diagnostic_warnings,
        )
    rejected, threshold, flagged = residual_rejection(
        pvalues,
        acf_values,
        len(residuals),
    )
    return ResidualDiagnostics(
        burn_in=burn_in,
        state_space_loglikelihood_burn=state_space_burn,
        conservative_burn_in=conservative_burn_in,
        n_effective=len(residuals),
        acf_values=acf_values,
        acf_threshold=threshold,
        flagged_acf_lags=tuple(flagged),
        ljung_box_statistics=statistics,
        ljung_box_pvalues=pvalues,
        training_adequacy_rejected=rejected,
        training_adequacy_reason=(
            "joint Ljung-Box and residual ACF rule" if rejected else None
        ),
        warnings=diagnostic_warnings,
    )


def _aicc(aic: float, nobs: int, n_params: int) -> float:
    denominator = nobs - n_params - 1
    if not np.isfinite(aic) or denominator <= 0:
        return float("nan")
    return float(aic + (2 * n_params * (n_params + 1)) / denominator)


def _effective_nobs(result: Any, fallback: int) -> int:
    for source in (result, getattr(result, "model", None)):
        if source is None:
            continue
        value = getattr(source, "nobs_effective", None)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric > 0:
            return int(numeric)
    return fallback


def _failed_record(
    order: SarimaOrder,
    trend: str,
    message: str,
    warnings_seen: tuple[str, ...] = (),
    hard_warnings: tuple[str, ...] = (),
    optimizer_retry_count: int = 0,
    fitted_result: Any | None = None,
) -> SarimaFitRecord:
    return SarimaFitRecord(
        order=order,
        trend=trend,
        fitted_result=fitted_result,
        fit_status="failed",
        convergence_status="failed",
        optimizer_success=False,
        optimizer_status=None,
        optimizer_warnflag=None,
        warning_messages=warnings_seen,
        hard_warning_messages=hard_warnings,
        optimizer_retry_count=optimizer_retry_count,
        error_message=message,
        parameters_finite=False,
        standard_errors_finite=False,
        converged=False,
        forecast_valid=False,
        forecast_error_message=None,
        nobs=0,
        effective_nobs=0,
        n_params=0,
        log_likelihood=float("nan"),
        aic=float("nan"),
        aicc=float("nan"),
        bic=float("nan"),
        root_diagnostics=RootDiagnostics.empty(),
        ar_root_minimum=float("nan"),
        ma_root_minimum=float("nan"),
        stationarity_ok=False,
        invertibility_ok=False,
        residual_diagnostics=ResidualDiagnostics.empty(),
        training_adequacy_rejected=True,
    )


def _fit_once(
    numeric_values: np.ndarray,
    order: SarimaOrder,
    trend: str,
    method_kwargs: Mapping[str, object] | None = None,
    exog: object | None = None,
    progress_callback: Callable[[Any, np.ndarray], None] | None = None,
    fixed_params: object | None = None,
) -> tuple[Any | None, tuple[str, ...], tuple[str, ...], str | None]:
    recorded: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model_kwargs: dict[str, object] = {
                "order": order.nonseasonal_order,
                "seasonal_order": order.seasonal_order,
                "trend": trend,
                "enforce_stationarity": True,
                "enforce_invertibility": True,
            }
            if exog is not None:
                model_kwargs["exog"] = exog
            model = SARIMAX(numeric_values, **model_kwargs)
            if fixed_params is not None:
                fitted = model.filter(
                    np.asarray(fixed_params, dtype=float).reshape(-1),
                    transformed=True,
                    includes_fixed=True,
                    cov_type="none",
                )
            else:
                fit_options = dict(method_kwargs or {})
                if progress_callback is not None:
                    fit_options["callback"] = lambda params: progress_callback(
                        model, np.asarray(params, dtype=float).copy()
                    )
                fitted = model.fit(**fit_options)
            recorded = caught
    except Exception as error:
        recorded = locals().get("caught", recorded)
        warning_messages: list[str] = []
        hard_warning_messages: list[str] = []
        for caught_warning in recorded:
            is_hard, formatted = classify_warning(
                caught_warning.category.__name__, str(caught_warning.message)
            )
            warning_messages.append(formatted)
            if is_hard:
                hard_warning_messages.append(formatted)
        return (
            None,
            tuple(warning_messages),
            tuple(hard_warning_messages),
            f"{type(error).__name__}: {error}",
        )
    warning_messages = []
    hard_warning_messages = []
    for caught_warning in recorded:
        is_hard, formatted = classify_warning(
            caught_warning.category.__name__, str(caught_warning.message)
        )
        warning_messages.append(formatted)
        if is_hard:
            hard_warning_messages.append(formatted)
    return fitted, tuple(warning_messages), tuple(hard_warning_messages), None


def _forecast_check(result: Any, exog: object | None = None) -> tuple[bool, str | None]:
    try:
        forecast_exog = None
        if exog is not None:
            forecast_exog = exog.iloc[:1] if hasattr(exog, "iloc") else np.asarray(exog)[:1]
        forecast_values(result, 1, exog=forecast_exog)
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, None


def _validate_exog(exog: object, nobs: int) -> str | None:
    try:
        matrix = np.asarray(exog, dtype=float)
    except (TypeError, ValueError) as error:
        return f"exogenous values are unusable: {error}"
    if matrix.ndim != 2 or matrix.shape[0] != nobs or matrix.shape[1] == 0:
        return "exogenous design has invalid shape"
    if not np.isfinite(matrix).all():
        return "exogenous design contains non-finite values"
    if np.unique(matrix, axis=1).shape[1] != matrix.shape[1]:
        return "exogenous design contains duplicate columns"
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        return "rank-deficient actual exogenous design"
    return None


def fit_sarima(
    values: object,
    order: SarimaOrder,
    exog: object | None = None,
    *,
    fit_kwargs: Mapping[str, object] | None = None,
    require_standard_errors: bool = True,
    retry_maxiter: int | None = 1000,
    validate_exog: bool = True,
    progress_callback: Callable[[Any, np.ndarray], None] | None = None,
    fixed_params: object | None = None,
) -> SarimaFitRecord:
    trend = trend_for_d_D(order.d, order.D)
    try:
        numeric_values = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        return _failed_record(order, trend, f"training values are unusable: {error}")
    if not len(numeric_values):
        return _failed_record(order, trend, "training values are empty")
    if not np.isfinite(numeric_values).all():
        return _failed_record(order, trend, "training values are not finite")
    if exog is not None and validate_exog:
        exog_error = _validate_exog(exog, len(numeric_values))
        if exog_error is not None:
            return _failed_record(order, trend, exog_error)

    base_fit_kwargs = dict(fit_kwargs or {})
    fitted, first_warnings, first_hard_warnings, error_message = _fit_once(
        numeric_values,
        order,
        trend,
        base_fit_kwargs,
        exog=exog,
        progress_callback=progress_callback,
        fixed_params=fixed_params,
    )
    if error_message is not None or fitted is None:
        return _failed_record(
            order,
            trend,
            error_message or "SARIMAX fit returned no result",
            first_warnings,
            first_hard_warnings,
        )

    warning_messages = list(first_warnings)
    retry_count = 0
    retvals = (
        {
            "converged": True,
            "success": True,
            "status": 0,
            "warnflag": 0,
            "iterations": 0,
            "fcalls": 0,
            "message": "fixed parameters filtered",
        }
        if fixed_params is not None
        else _retvals_mapping(fitted)
    )
    converged = _is_explicit_true(retvals.get("converged"))
    if fixed_params is None and not converged and retry_maxiter is not None:
        retry_count = 1
        retried, retry_warnings, retry_hard_warnings, retry_error = _fit_once(
                numeric_values,
                order,
                trend,
                {**base_fit_kwargs, "maxiter": retry_maxiter},
                exog=exog,
                progress_callback=progress_callback,
        )
        warning_messages.extend(retry_warnings)
        if retry_error is not None or retried is None:
            return _failed_record(
                order,
                trend,
                f"optimizer retry failed: {retry_error or 'SARIMAX fit returned no result'}",
                tuple(warning_messages),
                tuple(first_hard_warnings + retry_hard_warnings),
                retry_count,
                fitted,
            )
        fitted = retried
        retvals = _retvals_mapping(fitted)
        converged = _is_explicit_true(retvals.get("converged"))
        hard_warning_messages = tuple(retry_hard_warnings)
    else:
        hard_warning_messages = tuple(first_hard_warnings)

    params = np.asarray(getattr(fitted, "params", []), dtype=float).reshape(-1)
    bse = np.asarray(getattr(fitted, "bse", []), dtype=float).reshape(-1)
    parameters_finite = bool(len(params) and np.isfinite(params).all())
    standard_errors_finite = bool(len(bse) and np.isfinite(bse).all())
    nobs = int(getattr(fitted, "nobs", len(numeric_values)))
    effective_nobs = _effective_nobs(fitted, nobs)
    n_params = int(len(params))
    log_likelihood = float(getattr(fitted, "llf", float("nan")))
    aic = float(getattr(fitted, "aic", float("nan")))
    bic = float(getattr(fitted, "bic", float("nan")))
    aicc = _aicc(aic, effective_nobs, n_params)

    root_error: str | None = None
    if parameters_finite:
        try:
            root_diagnostics = _root_diagnostics(fitted, order)
        except Exception as error:
            root_diagnostics = RootDiagnostics.empty()
            root_error = f"root diagnostics failed: {type(error).__name__}: {error}"
    else:
        root_diagnostics = RootDiagnostics.empty()
    combined_ar_stable = _required_roots_are_stable(
        root_diagnostics.combined_ar_roots,
        order.p + order.P,
    )
    combined_ma_stable = _required_roots_are_stable(
        root_diagnostics.combined_ma_roots,
        order.q + order.Q,
    )
    reconstructed_ar_roots = (
        root_diagnostics.reconstructed_nonseasonal_ar_roots,
        root_diagnostics.reconstructed_seasonal_ar_roots,
    )
    reconstructed_ma_roots = (
        root_diagnostics.reconstructed_nonseasonal_ma_roots,
        root_diagnostics.reconstructed_seasonal_ma_roots,
    )
    stationarity_ok = combined_ar_stable and all(
        roots is None or root_is_stable(roots) for roots in reconstructed_ar_roots
    )
    invertibility_ok = combined_ma_stable and all(
        roots is None or root_is_stable(roots) for roots in reconstructed_ma_roots
    )
    forecast_valid, forecast_error_message = _forecast_check(fitted, exog=exog)
    residual_diagnostics = _residual_diagnostics(fitted, order)

    reasons: list[str] = []
    if not converged:
        reasons.append("optimizer did not explicitly converge")
    success = retvals.get("success")
    status = retvals.get("status")
    warnflag = retvals.get("warnflag")
    if success is not None and not _is_explicit_true(success):
        reasons.append("optimizer reported failure")
    if _status_failed(status):
        reasons.append(f"optimizer status={status}")
    if _status_failed(warnflag):
        reasons.append(f"optimizer warnflag={warnflag}")
    if hard_warning_messages:
        reasons.append("invalid or failed-fit warning")
    if not parameters_finite:
        reasons.append("non-finite parameter estimates")
    if root_error is not None:
        reasons.append(root_error)
    if require_standard_errors and not standard_errors_finite:
        reasons.append("non-finite standard errors")
    if not all(_finite_float(value) for value in (log_likelihood, aic, aicc, bic)):
        reasons.append("non-finite likelihood or information criterion")
    if not forecast_valid:
        reasons.append(f"unusable forecast: {forecast_error_message}")
    if not combined_ar_stable:
        reasons.append("combined AR roots fail stationarity threshold")
    if not all(
        roots is None or root_is_stable(roots) for roots in reconstructed_ar_roots
    ):
        reasons.append("reconstructed AR component roots fail stationarity threshold")
    if not combined_ma_stable:
        reasons.append("combined MA roots fail invertibility threshold")
    if not all(
        roots is None or root_is_stable(roots) for roots in reconstructed_ma_roots
    ):
        reasons.append("reconstructed MA component roots fail invertibility threshold")

    normalized_success = (
        _is_explicit_true(success) if success is not None else converged
    )
    normalized_status = status if status is not None else (0 if converged else 1)
    normalized_warnflag = warnflag if warnflag is not None else (0 if converged else 1)
    optimizer_iterations = _retvals_count(retvals, "iterations", "nit")
    optimizer_function_calls = _retvals_count(retvals, "fcalls", "funcalls", "nfev")
    optimizer_message = _retvals_message(retvals)
    optimizer_gradient_norm = _retvals_gradient_norm(retvals)

    return SarimaFitRecord(
        order=order,
        trend=trend,
        fitted_result=fitted,
        fit_status="success" if not reasons else "failed",
        convergence_status="converged" if converged else "failed",
        optimizer_success=normalized_success,
        optimizer_status=normalized_status,
        optimizer_warnflag=normalized_warnflag,
        warning_messages=tuple(warning_messages),
        hard_warning_messages=hard_warning_messages,
        optimizer_retry_count=retry_count,
        error_message="; ".join(reasons) if reasons else None,
        parameters_finite=parameters_finite,
        standard_errors_finite=standard_errors_finite,
        converged=converged,
        forecast_valid=forecast_valid,
        forecast_error_message=forecast_error_message,
        nobs=nobs,
        effective_nobs=effective_nobs,
        n_params=n_params,
        log_likelihood=log_likelihood,
        aic=aic,
        aicc=aicc,
        bic=bic,
        root_diagnostics=root_diagnostics,
        ar_root_minimum=_minimum_root(root_diagnostics.combined_ar_roots),
        ma_root_minimum=_minimum_root(root_diagnostics.combined_ma_roots),
        stationarity_ok=stationarity_ok,
        invertibility_ok=invertibility_ok,
        residual_diagnostics=residual_diagnostics,
        training_adequacy_rejected=residual_diagnostics.training_adequacy_rejected,
        optimizer_iterations=optimizer_iterations,
        optimizer_function_calls=optimizer_function_calls,
        optimizer_message=optimizer_message,
        optimizer_gradient_norm=optimizer_gradient_norm,
    )


def forecast_values(
    fitted_result: Any,
    steps: int,
    exog: object | None = None,
) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive")
    try:
        forecast_kwargs = {"steps": steps}
        if exog is not None:
            forecast_kwargs["exog"] = exog
        forecast = np.asarray(
            fitted_result.get_forecast(**forecast_kwargs).predicted_mean,
            dtype=float,
        )
    except Exception as error:
        raise ValueError(f"forecast failed: {type(error).__name__}: {error}") from error
    if forecast.shape != (steps,) or not np.isfinite(forecast).all():
        raise ValueError("forecast output is unusable")
    return forecast
