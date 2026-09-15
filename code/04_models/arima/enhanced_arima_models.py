from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from arima_models import (
    ArimaFitRecord,
    _aicc,
    _empty_residual_diagnostics,
    _minimum_root,
    _residual_diagnostics,
    _failed_record,
    classify_warning,
    fit_arima,
    root_is_stable,
    forecast_values,
    trend_for_d,
)
from common.forecasting_framework import (
    _as_prepared,
    build_information_context,
    country_forecast_origin,
    extract_target_day,
    information_set,
)


ORDINARY_ORDERS = (
    (1, 1, 1),
    (1, 1, 2),
    (1, 1, 3),
    (2, 1, 1),
    (2, 1, 2),
    (2, 1, 3),
    (3, 1, 1),
    (3, 1, 2),
    (3, 1, 3),
    (4, 1, 1),
    (4, 1, 2),
)
WEEKLY_DIFFERENCED_ORDERS = (
    (1, 0, 1),
    (2, 0, 0),
    (2, 0, 1),
    (2, 0, 2),
    (3, 0, 1),
)
ISSUE_COLUMNS = (
    "target_date",
    "target_timestamp_utc",
    "source_key",
    "source_timestamp_utc",
    "reason",
    "source_row_count",
)


@dataclass(frozen=True)
class EnhancedSpecification:
    branch: str
    specification_id: str
    order: tuple[int, int, int]
    trend: str
    transform: str


@dataclass(frozen=True)
class WeeklyTransformResult:
    values: pd.Series
    issues: pd.DataFrame
    valid_observations: int
    missing_observations: int


def _specification_id(prefix: str, order: tuple[int, int, int]) -> str:
    p, d, q = order
    return f"{prefix}_p{p}_d{d}_q{q}"


def enhanced_specifications() -> tuple[EnhancedSpecification, ...]:
    ordinary = tuple(
        EnhancedSpecification(
            branch="ordinary",
            specification_id=_specification_id("enhanced_ordinary_arima", order),
            order=order,
            trend="n",
            transform="identity",
        )
        for order in ORDINARY_ORDERS
    )
    weekly = tuple(
        EnhancedSpecification(
            branch="weekly_differenced",
            specification_id=_specification_id("weekly_differenced_arima", order),
            order=order,
            trend="c",
            transform="weekly_difference",
        )
        for order in WEEKLY_DIFFERENCED_ORDERS
    )
    return ordinary + weekly


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _local_date(row: object) -> date:
    value = str(getattr(row, "local_date"))
    return date.fromisoformat(value[:10])


def _source_key(row: object) -> tuple[str, int, int]:
    return (
        (_local_date(row) - timedelta(days=7)).isoformat(),
        int(getattr(row, "hour")),
        int(getattr(row, "local_occurrence")),
    )


def _local_key(row: object) -> tuple[str, int, int]:
    return (
        _local_date(row).isoformat(),
        int(getattr(row, "hour")),
        int(getattr(row, "local_occurrence")),
    )


def _source_lookup(frame: pd.DataFrame) -> dict[tuple[str, int, int], list[object]]:
    lookup: dict[tuple[str, int, int], list[object]] = {}
    for row in frame.itertuples(index=False):
        lookup.setdefault(_local_key(row), []).append(row)
    return lookup


def _issue_record(
    row: object,
    key: tuple[str, int, int],
    reason: str,
    matches: list[object],
) -> dict[str, object]:
    timestamp = _utc_timestamp(getattr(row, "timestamp_utc"))
    source_timestamp = (
        _utc_timestamp(getattr(matches[0], "timestamp_utc"))
        if len(matches) == 1
        else pd.NaT
    )
    return {
        "target_date": _local_date(row).isoformat(),
        "target_timestamp_utc": timestamp,
        "source_key": key,
        "source_timestamp_utc": source_timestamp,
        "reason": reason,
        "source_row_count": len(matches),
    }


def _resolve_source(
    row: object,
    lookup: dict[tuple[str, int, int], list[object]],
) -> tuple[object | None, dict[str, object] | None]:
    key = _source_key(row)
    matches = lookup.get(key, [])
    if len(matches) == 0:
        return None, _issue_record(row, key, "missing_source", matches)
    if len(matches) != 1:
        return None, _issue_record(row, key, "ambiguous_source", matches)
    source = matches[0]
    source_value = float(getattr(source, "actual_load_mwh"))
    if not np.isfinite(source_value):
        return None, _issue_record(row, key, "missing_source_value", matches)
    target_value = float(getattr(row, "actual_load_mwh"))
    if not np.isfinite(target_value):
        return None, _issue_record(row, key, "missing_target_value", matches)
    return source, None


def _issue_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=ISSUE_COLUMNS)


def _prepared_origin_frame(frame: pd.DataFrame, cutoff_utc: pd.Timestamp) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    cutoff = _utc_timestamp(cutoff_utc)
    return prepared.loc[prepared["interval_end_utc"] <= cutoff].copy()


def _hourly_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.freq is not None:
        return index
    if len(index) > 1 and (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        return pd.DatetimeIndex(index, freq="h")
    return index


def transform_weekly_series(
    frame: pd.DataFrame,
    cutoff_utc: pd.Timestamp,
) -> WeeklyTransformResult:
    prepared = _prepared_origin_frame(frame, cutoff_utc)
    index = _hourly_index(pd.DatetimeIndex(prepared["timestamp_utc"]))
    values = pd.Series(np.nan, index=index, dtype=float, name="weekly_difference")
    lookup = _source_lookup(prepared)
    issues: list[dict[str, object]] = []

    for row in prepared.itertuples(index=False):
        source, issue = _resolve_source(row, lookup)
        if issue is not None:
            issues.append(issue)
            continue
        target_value = float(getattr(row, "actual_load_mwh"))
        source_value = float(getattr(source, "actual_load_mwh"))
        values.loc[_utc_timestamp(getattr(row, "timestamp_utc"))] = (
            target_value - source_value
        )

    valid = int(values.notna().sum())
    return WeeklyTransformResult(
        values=values,
        issues=_issue_frame(issues),
        valid_observations=valid,
        missing_observations=int(len(values) - valid),
    )


def derive_weekly_invalid_target_dates(
    frame: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    lookup = _source_lookup(prepared)
    records: list[dict[str, object]] = []
    local_dates = pd.to_datetime(prepared["local_date"], errors="raise").dt.date
    for target_date in sorted(set(local_dates)):
        if target_date.year != year:
            continue
        target_rows = prepared.loc[local_dates.eq(target_date)]
        for row in target_rows.itertuples(index=False):
            _, issue = _resolve_source(row, lookup)
            if issue is not None:
                records.append(issue)
    return _issue_frame(records)


def _fit_once_missing(
    values: pd.Series,
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
                values,
                order=order,
                trend=trend,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            fitted = model.fit(
                **({"method_kwargs": method_kwargs} if method_kwargs else {})
            )
        for caught in recorded:
            is_hard, formatted = classify_warning(
                caught.category.__name__, str(caught.message)
            )
            warning_messages.append(formatted)
            if is_hard:
                hard_warning_messages.append(formatted)
        return fitted, tuple(warning_messages), tuple(hard_warning_messages), None
    except Exception as error:
        return (
            None,
            tuple(warning_messages),
            tuple(hard_warning_messages),
            f"{type(error).__name__}: {error}",
        )


def fit_missing_aware_arima(
    values: pd.Series,
    order: tuple[int, int, int],
) -> ArimaFitRecord:
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    series = values.astype(float).copy()
    trend = trend_for_d(order[1])
    numeric_values = series.to_numpy(dtype=float)
    if len(numeric_values) <= 50:
        return _failed_record(order, trend, "insufficient observations")
    if (~np.isfinite(numeric_values) & ~np.isnan(numeric_values)).any():
        return _failed_record(order, trend, "training values contain infinite values")
    if not isinstance(series.index, pd.DatetimeIndex):
        return _failed_record(order, trend, "training index is not a DatetimeIndex")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        return _failed_record(order, trend, "training index is not unique and ordered")
    series.index = _hourly_index(pd.DatetimeIndex(series.index))

    fitted, first_warnings, first_hard_warnings, error_message = _fit_once_missing(
        series, order, trend
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
    retvals = getattr(fitted, "mle_retvals", {})
    if not isinstance(retvals, Mapping):
        retvals = {}
    converged = retvals.get("converged") is True
    if not converged:
        retry_count = 1
        retried, retry_warnings, retry_hard_warnings, retry_error = _fit_once_missing(
            series, order, trend, {"maxiter": 1000}
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
        retvals = getattr(fitted, "mle_retvals", {})
        if not isinstance(retvals, Mapping):
            retvals = {}
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
    nobs = int(getattr(fitted, "nobs", len(series)))
    n_params = int(len(params))
    log_likelihood = float(getattr(fitted, "llf", float("nan")))
    aic = float(getattr(fitted, "aic", float("nan")))
    bic = float(getattr(fitted, "bic", float("nan")))
    aicc = _aicc(aic, nobs, n_params)
    ar_roots = getattr(fitted, "arroots", None)
    ma_roots = getattr(fitted, "maroots", None)
    stationarity_ok = root_is_stable(ar_roots)
    invertibility_ok = root_is_stable(ma_roots)
    try:
        residual_diagnostics = _residual_diagnostics(fitted, order)
    except Exception:
        residual_diagnostics = _empty_residual_diagnostics()

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
        np.isfinite(value)
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


def fit_enhanced(
    specification: EnhancedSpecification,
    values: object,
) -> ArimaFitRecord:
    if specification.branch == "ordinary":
        return fit_arima(values, specification.order)
    if specification.branch == "weekly_differenced":
        if not isinstance(values, pd.Series):
            raise ValueError("weekly enhanced fitting requires an indexed Series")
        return fit_missing_aware_arima(values, specification.order)
    raise ValueError(f"unsupported enhanced ARIMA branch: {specification.branch}")


def _path_frame(
    path: pd.DataFrame,
    country: str,
    target_date: str,
    specification: EnhancedSpecification,
    origin: pd.Timestamp,
    origin_local: str,
) -> pd.DataFrame:
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
    result["model_family"] = "enhanced_arima"
    result["target_date"] = target_date
    result["specification_id"] = specification.specification_id
    result["forecast_origin_local"] = origin_local
    result["forecast_origin_utc"] = origin.isoformat()
    result["information_cutoff_utc"] = origin.isoformat()
    result["is_target_day"] = result["local_date"].eq(target_date)
    result["bridge_used"] = ~result["is_target_day"]
    return result


def build_weekly_forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str,
    specification: EnhancedSpecification,
    fitted_result: object,
) -> pd.DataFrame:
    if specification.branch != "weekly_differenced":
        raise ValueError("weekly forecast path requires a weekly-differenced specification")
    prepared = _as_prepared(frame)
    target_text = pd.Timestamp(target_date).date().isoformat()
    context = build_information_context(prepared, country, target_text)
    origin = _utc_timestamp(context.cutoff_utc)
    target = extract_target_day(prepared, target_text)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(origin)
        & prepared["local_date"].le(target_text)
    ].copy()
    if path.empty or not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError(f"no complete weekly forecast path exists for {country} {target_text}")

    available = information_set(prepared, origin)
    lookup = _source_lookup(available)
    mapping_issues: list[dict[str, object]] = []
    source_rows: list[object | None] = []
    for row in path.itertuples(index=False):
        source, issue = _resolve_source(row, lookup)
        source_rows.append(source)
        if issue is not None:
            mapping_issues.append(issue)
        elif _utc_timestamp(getattr(source, "interval_end_utc")) > origin:
            raise ValueError("weekly source actual is after the forecast origin")

    result = _path_frame(
        path,
        country,
        target_text,
        specification,
        origin,
        context.origin_local,
    )
    if mapping_issues:
        result["status"] = "weekly_lag_invalid"
        result["forecast_mwh"] = np.nan
        result["source_timestamp_utc"] = pd.NaT
        result["source_actual_mwh"] = np.nan
        result["source_kind"] = "mapping_invalid"
        result["evaluated"] = False
    else:
        deltas = forecast_values(fitted_result, len(path))
        source_actuals = np.asarray(
            [float(getattr(source, "actual_load_mwh")) for source in source_rows],
            dtype=float,
        )
        if not np.isfinite(source_actuals).all():
            raise ValueError("weekly source actual values are not finite")
        result["status"] = "completed"
        result["forecast_mwh"] = source_actuals + deltas
        result["source_timestamp_utc"] = [
            _utc_timestamp(getattr(source, "timestamp_utc")) for source in source_rows
        ]
        result["source_actual_mwh"] = source_actuals
        result["source_kind"] = "observed"
        result["evaluated"] = result["is_target_day"]

    result["mapping_issue_count"] = len(mapping_issues)
    return result[
        [
            "country",
            "model_family",
            "target_date",
            "specification_id",
            "forecast_origin_local",
            "forecast_origin_utc",
            "information_cutoff_utc",
            "timestamp_utc",
            "interval_end_utc",
            "timestamp_local",
            "local_date",
            "actual_load_mwh",
            "forecast_mwh",
            "bridge_used",
            "source_timestamp_utc",
            "source_actual_mwh",
            "source_kind",
            "is_target_day",
            "evaluated",
            "status",
            "mapping_issue_count",
        ]
    ]
