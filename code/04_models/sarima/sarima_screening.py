from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, kpss, pacf


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import COUNTRY_CONFIG, PROCESSED, load_country_data
from sarima_models import SarimaOrder, candidate_pool, fit_sarima


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "arima_sarima" / "tables"
TRAINING_START = date(2020, 1, 1)
TRAINING_END = date(2023, 12, 31)
SEASONAL_PERIOD = 24
SEASONAL_LAGS = (24, 48, 72)
ORDINARY_ACF_LAGS = tuple(range(1, 25))
SIGNIFICANCE_LEVEL = 0.05
EVIDENCE_Z = 1.96
CANDIDATE_SCREENING_PATH = TABLE_DIRECTORY / "sarima_candidate_screening_training_2024.csv"
REVISED_SCREENING_PATH = (
    TABLE_DIRECTORY / "sarima_candidate_screening_training_2024_revised.csv"
)
SHORTLIST_PATH = TABLE_DIRECTORY / "sarima_shortlists_2024.csv"
RESIDUAL_DIAGNOSTICS_PATH = (
    TABLE_DIRECTORY / "sarima_residual_diagnostics_training_2024.csv"
)

_CORE_COUNT = 8
_INTERACTION_REQUIREMENTS = {
    (1, 0, 1, 0): ("ordinary_ar", "seasonal_ar"),
    (0, 1, 0, 1): ("ordinary_ma", "seasonal_ma"),
    (1, 1, 1, 0): ("ordinary_ar", "ordinary_ma", "seasonal_ar"),
    (1, 1, 0, 1): ("ordinary_ar", "ordinary_ma", "seasonal_ma"),
    (1, 0, 0, 1): ("ordinary_ar", "seasonal_ma"),
    (0, 1, 1, 0): ("ordinary_ma", "seasonal_ar"),
}


def _numeric_values(values: object) -> np.ndarray:
    try:
        numeric = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"values are unusable: {error}") from error
    if not len(numeric) or not np.isfinite(numeric).all():
        raise ValueError("values must be non-empty and finite")
    return numeric


def training_values(frame: pd.DataFrame) -> np.ndarray:
    """Return only the contiguous 2020-2023 hourly training series."""
    if "local_date" in frame:
        date_text = frame["local_date"].astype(str)
        date_source = frame["local_date"]
    elif "timestamp_local" in frame:
        date_text = frame["timestamp_local"].astype(str).str[:10]
        date_source = date_text
    else:
        raise ValueError("forecasting frame is missing local dates")

    # Filter by the training years before validating dates so future rows cannot
    # affect the training-only screening path.
    candidate_mask = date_text.str[:4].isin({"2020", "2021", "2022", "2023"})
    candidates = frame.loc[candidate_mask].copy()
    local_dates = pd.to_datetime(date_source.loc[candidate_mask], errors="coerce")
    if local_dates.isna().any():
        raise ValueError("forecasting frame contains invalid local dates")

    selected_mask = local_dates.dt.date.between(TRAINING_START, TRAINING_END)
    selected = candidates.loc[selected_mask].copy()
    if selected.empty:
        raise ValueError("training period contains no observations")
    if "timestamp_utc" not in selected:
        raise ValueError("forecasting frame is missing timestamp_utc")
    if "actual_load_mwh" not in selected:
        raise ValueError("forecasting frame is missing actual_load_mwh")

    timestamps = pd.to_datetime(selected["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("training period contains invalid UTC timestamps")
    if timestamps.duplicated().any():
        raise ValueError("training period contains duplicate UTC timestamps")
    selected = selected.assign(_timestamp_utc=timestamps).sort_values(
        "_timestamp_utc", ignore_index=True
    )
    gaps = selected["_timestamp_utc"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("training period is not hourly contiguous")

    values = pd.to_numeric(selected["actual_load_mwh"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("training period contains non-finite observations")
    return values


def _differenced(values: np.ndarray, d: int) -> np.ndarray:
    if d not in (0, 1, 2):
        raise ValueError("d must be 0, 1 or 2")
    result = values
    for _ in range(d):
        result = np.diff(result)
    return result


def _seasonal_difference(values: np.ndarray) -> np.ndarray:
    if len(values) <= SEASONAL_PERIOD:
        raise ValueError("seasonal differencing requires more than 24 observations")
    return values[SEASONAL_PERIOD:] - values[:-SEASONAL_PERIOD]


def _test_warnings(recorded: list[warnings.WarningMessage]) -> tuple[str, ...]:
    return tuple(f"{item.category.__name__}: {item.message}" for item in recorded)


def stationarity_diagnostics(values: object, D: int = 0) -> list[dict[str, object]]:
    """Run ordinary ADF/KPSS tests on one seasonal path for d=0,1,2."""
    if D not in (0, 1):
        raise ValueError("D must be 0 or 1")
    numeric_values = _numeric_values(values)
    path_values = _seasonal_difference(numeric_values) if D else numeric_values
    if len(path_values) <= 50:
        raise ValueError("stationarity diagnostics require more than 50 observations")

    records: list[dict[str, object]] = []
    for d in (0, 1, 2):
        series = _differenced(path_values, d)
        deterministic = "ct" if d == 0 else "c"
        adf_statistic = float("nan")
        adf_pvalue = float("nan")
        adf_used_lag: int | None = None
        adf_error: str | None = None
        with warnings.catch_warnings(record=True) as adf_recorded:
            warnings.simplefilter("always")
            try:
                adf_result = adfuller(
                    series,
                    regression=deterministic,
                    maxlag=24,
                    autolag="AIC",
                )
                adf_statistic = float(adf_result[0])
                adf_pvalue = float(adf_result[1])
                adf_used_lag = int(adf_result[2])
            except Exception as error:
                adf_error = f"{type(error).__name__}: {error}"

        kpss_statistic = float("nan")
        kpss_pvalue = float("nan")
        kpss_bandwidth: int | None = None
        kpss_error: str | None = None
        with warnings.catch_warnings(record=True) as kpss_recorded:
            warnings.simplefilter("always")
            try:
                kpss_result = kpss(
                    series,
                    regression=deterministic,
                    nlags="auto",
                )
                kpss_statistic = float(kpss_result[0])
                kpss_pvalue = float(kpss_result[1])
                kpss_bandwidth = int(kpss_result[2])
            except Exception as error:
                kpss_error = f"{type(error).__name__}: {error}"

        adf_stationary = bool(
            np.isfinite(adf_pvalue) and adf_pvalue < SIGNIFICANCE_LEVEL
        )
        kpss_stationary = bool(
            np.isfinite(kpss_pvalue) and kpss_pvalue >= SIGNIFICANCE_LEVEL
        )
        records.append(
            {
                "D": D,
                "d": d,
                "observations": len(series),
                "deterministic_component": deterministic,
                "adf_statistic": adf_statistic,
                "adf_pvalue": adf_pvalue,
                "adf_used_lag": adf_used_lag,
                "kpss_statistic": kpss_statistic,
                "kpss_pvalue": kpss_pvalue,
                "kpss_bandwidth": kpss_bandwidth,
                "adf_stationary": adf_stationary,
                "kpss_stationary": kpss_stationary,
                "stationary_supported": adf_stationary and kpss_stationary,
                "adf_error": adf_error,
                "kpss_error": kpss_error,
                "warnings": json.dumps(
                    list(_test_warnings(adf_recorded))
                    + list(_test_warnings(kpss_recorded))
                ),
            }
        )
    return records


def select_d(diagnostics: list[Mapping[str, object]]) -> int | None:
    supported: list[int] = []
    for row in diagnostics:
        explicit = row.get("stationary_supported")
        if explicit is None:
            try:
                explicit = (
                    float(row["adf_pvalue"]) < SIGNIFICANCE_LEVEL
                    and float(row["kpss_pvalue"]) >= SIGNIFICANCE_LEVEL
                )
            except (KeyError, TypeError, ValueError):
                explicit = False
        if bool(explicit):
            supported.append(int(row["d"]))
    return min(supported) if supported else None


def _acf_at_lags(values: np.ndarray, lags: tuple[int, ...]) -> dict[int, float]:
    try:
        values_acf = acf(values, nlags=max(lags), fft=True, adjusted=False)
    except Exception:
        return {lag: float("nan") for lag in lags}
    return {lag: float(values_acf[lag]) for lag in lags}


def seasonal_dependence_diagnostics(values: object, D: int) -> dict[str, object]:
    """Compare seasonal dependence before and after the D path transform."""
    if D not in (0, 1):
        raise ValueError("D must be 0 or 1")
    numeric_values = _numeric_values(values)
    transformed = _seasonal_difference(numeric_values) if D else numeric_values
    original_acf = _acf_at_lags(numeric_values, SEASONAL_LAGS)
    transformed_acf = _acf_at_lags(transformed, SEASONAL_LAGS)
    original_threshold = EVIDENCE_Z / np.sqrt(len(numeric_values))
    transformed_threshold = EVIDENCE_Z / np.sqrt(len(transformed))
    original_dependence = {
        lag: bool(
            np.isfinite(original_acf[lag])
            and abs(original_acf[lag]) > original_threshold
        )
        for lag in SEASONAL_LAGS
    }
    transformed_dependence = {
        lag: bool(
            np.isfinite(transformed_acf[lag])
            and abs(transformed_acf[lag]) > transformed_threshold
        )
        for lag in SEASONAL_LAGS
    }
    persistence_change = {
        lag: (
            abs(transformed_acf[lag]) - abs(original_acf[lag])
            if np.isfinite(original_acf[lag]) and np.isfinite(transformed_acf[lag])
            else float("nan")
        )
        for lag in SEASONAL_LAGS
    }
    finite_changes = [value for value in persistence_change.values() if np.isfinite(value)]
    seasonal_persistence_reduced = bool(
        finite_changes and float(np.mean(finite_changes)) < 0
    )
    seasonal_evidence_supported = (
        True
        if D == 0
        else bool(any(original_dependence.values()) and seasonal_persistence_reduced)
    )
    return {
        "D": D,
        "lags": SEASONAL_LAGS,
        "original_acf": original_acf,
        "transformed_acf": transformed_acf,
        "persistence_change": persistence_change,
        "original_threshold": float(original_threshold),
        "transformed_threshold": float(transformed_threshold),
        "original_dependence": original_dependence,
        "transformed_dependence": transformed_dependence,
        "seasonal_persistence_reduced": seasonal_persistence_reduced,
        "seasonal_evidence_supported": seasonal_evidence_supported,
    }


def assess_differencing_paths(values: object) -> dict[int, dict[str, object]]:
    numeric_values = _numeric_values(values)
    paths: dict[int, dict[str, object]] = {}
    for D in (0, 1):
        seasonal = seasonal_dependence_diagnostics(numeric_values, D)
        stationarity = stationarity_diagnostics(numeric_values, D)
        selected = select_d(stationarity)
        evidence_supported = bool(seasonal["seasonal_evidence_supported"])
        viable = selected is not None and evidence_supported
        if selected is None:
            viability_reason = "no supported ordinary differencing order"
        elif not evidence_supported:
            viability_reason = "seasonal persistence evidence does not support path"
        else:
            viability_reason = None
        paths[D] = {
            "D": D,
            "selected_d": selected,
            "seasonal_evidence_supported": evidence_supported,
            "seasonal": seasonal,
            "stationarity": stationarity,
            "viable": viable,
            "viability_reason": viability_reason,
        }
    return paths


def acf_pacf_evidence(values: object, D: int, d: int) -> dict[str, object]:
    """Calculate bounded ACF/PACF evidence for interaction candidates."""
    if D not in (0, 1) or d not in (0, 1, 2):
        raise ValueError("unsupported differencing path")
    numeric_values = _numeric_values(values)
    path_values = _seasonal_difference(numeric_values) if D else numeric_values
    transformed = _differenced(path_values, d)
    threshold = EVIDENCE_Z / np.sqrt(len(transformed))
    try:
        acf_values_array = acf(
            transformed,
            nlags=max(SEASONAL_LAGS),
            fft=True,
            adjusted=False,
        )
        pacf_values_array = pacf(
            transformed,
            nlags=max(SEASONAL_LAGS),
            method="ywm",
        )
    except Exception:
        acf_values_array = np.full(max(SEASONAL_LAGS) + 1, np.nan)
        pacf_values_array = np.full(max(SEASONAL_LAGS) + 1, np.nan)

    ordinary_acf = {
        lag: float(acf_values_array[lag]) for lag in ORDINARY_ACF_LAGS
    }
    ordinary_pacf = {
        lag: float(pacf_values_array[lag]) for lag in ORDINARY_ACF_LAGS
    }
    seasonal_acf = {
        lag: float(acf_values_array[lag]) for lag in SEASONAL_LAGS
    }
    seasonal_pacf = {
        lag: float(pacf_values_array[lag]) for lag in SEASONAL_LAGS
    }

    def has_evidence(series: Mapping[int, float]) -> bool:
        return any(np.isfinite(value) and abs(value) > threshold for value in series.values())

    return {
        "D": D,
        "d": d,
        "observations": len(transformed),
        "pointwise_threshold": float(threshold),
        "ordinary_acf": ordinary_acf,
        "ordinary_pacf": ordinary_pacf,
        "seasonal_acf": seasonal_acf,
        "seasonal_pacf": seasonal_pacf,
        "ordinary_ar": has_evidence(ordinary_pacf),
        "ordinary_ma": has_evidence(ordinary_acf),
        "seasonal_ar": has_evidence(seasonal_pacf),
        "seasonal_ma": has_evidence(seasonal_acf),
    }


def _evidence_flag(evidence: Mapping[str, object], name: str) -> bool:
    value = evidence.get(name)
    if value is None:
        value = evidence.get(f"{name}_evidence", False)
    return bool(value)


def build_candidate_orders(
    d: int,
    D: int,
    evidence: Mapping[str, object] | None = None,
) -> tuple[SarimaOrder, ...]:
    """Return all core orders plus only evidence-supported fixed interactions."""
    pool = candidate_pool(d, D)
    interaction_evidence = evidence or {}
    selected: list[SarimaOrder] = list(pool[:_CORE_COUNT])
    for order in pool[_CORE_COUNT:]:
        signature = (order.p, order.q, order.P, order.Q)
        requirements = _INTERACTION_REQUIREMENTS[signature]
        if all(_evidence_flag(interaction_evidence, name) for name in requirements):
            selected.append(order)
    return tuple(selected)


def _boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _finite_value(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _status_is_success(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"0", "success", "ok"}


def _missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalise_persisted_optimizer_fields(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    converged = _boolean_value(normalized.get("converged"))
    if _missing_value(normalized.get("optimizer_success")):
        normalized["optimizer_success"] = converged
    if _missing_value(normalized.get("optimizer_status")):
        normalized["optimizer_status"] = 0 if converged else 1
    if _missing_value(normalized.get("optimizer_warnflag")):
        normalized["optimizer_warnflag"] = 0 if converged else 1
    return normalized


def screening_row_is_model_valid(row: Mapping[str, object] | pd.Series) -> bool:
    """Check hard fit validity while deliberately ignoring residual adequacy."""
    error = row.get("error_message")
    error_text = "" if error is None or pd.isna(error) else str(error)
    residual_only = (
        not error_text
        or error_text == "joint Ljung-Box and residual ACF rule"
        or error_text.startswith("joint Ljung-Box and residual ACF rule;")
    )
    criteria_finite = True
    for criterion in ("aic", "aicc", "bic"):
        try:
            criteria_finite = criteria_finite and bool(np.isfinite(float(row[criterion])))
        except (KeyError, TypeError, ValueError):
            criteria_finite = False
    return bool(
        str(row.get("fit_status")) == "success"
        and str(row.get("convergence_status", "converged")) == "converged"
        and _boolean_value(row.get("converged"))
        and _boolean_value(row.get("parameters_finite"))
        and _boolean_value(row.get("standard_errors_finite"))
        and _boolean_value(row.get("forecast_valid"))
        and _boolean_value(row.get("stationarity_ok"))
        and _boolean_value(row.get("invertibility_ok"))
        and _status_is_success(row.get("optimizer_success"))
        and _status_is_success(row.get("optimizer_status"))
        and _status_is_success(row.get("optimizer_warnflag"))
        and _finite_value(row.get("log_likelihood"))
        and criteria_finite
        and residual_only
    )


def screening_row_is_eligible(row: Mapping[str, object] | pd.Series) -> bool:
    """Return hard numerical eligibility; residual adequacy remains diagnostic."""
    return screening_row_is_model_valid(row)


def _criterion_rankings(
    rows: list[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    rankings: dict[str, list[dict[str, object]]] = {}
    eligible = [row for row in rows if _boolean_value(row.get("eligible", False))]
    for criterion in ("aic", "aicc", "bic"):
        ranked = sorted(
            eligible,
            key=lambda row: (
                float(row[criterion]),
                int(row.get("specification_order", 0)),
                str(row.get("specification_id", "")),
            ),
        )
        rankings[criterion] = [
            {
                "specification_id": str(row["specification_id"]),
                "d": int(row["d"]),
                "D": int(row["D"]),
                "value": float(row[criterion]),
            }
            for row in ranked
        ]
    return rankings


def _residual_flag_summary(
    residual_path: Path,
    shortlist: pd.DataFrame,
) -> list[dict[str, object]]:
    if shortlist.empty or not residual_path.exists():
        return []
    residual = pd.read_csv(residual_path)
    summaries: list[dict[str, object]] = []
    for _, row in shortlist.iterrows():
        matching = residual.loc[
            residual["country"].eq(row["country"])
            & residual["specification_id"].eq(row["specification_id"])
        ]
        acf_rows = matching.loc[matching["diagnostic"].eq("residual_acf")]
        pvalue_rows = matching.loc[
            matching["diagnostic"].eq("ljung_box_pvalue")
        ]
        summaries.append(
            {
                "country": str(row["country"]),
                "specification_id": str(row["specification_id"]),
                "residual_adequacy_rejected": _boolean_value(
                    row.get("residual_adequacy_rejected", False)
                ),
                "residual_adequacy_reason": row.get("residual_adequacy_reason"),
                "acf_threshold": float(acf_rows["threshold"].iloc[0])
                if not acf_rows.empty
                else float("nan"),
                "flagged_acf_lags": [
                    int(lag)
                    for lag in acf_rows.loc[acf_rows["flagged"].map(_boolean_value), "lag"]
                ],
                "ljung_box_pvalues": {
                    str(int(lag)): float(value)
                    for lag, value in zip(
                        pvalue_rows["lag"], pvalue_rows["value"], strict=True
                    )
                },
            }
        )
    return summaries


def recompute_shortlist_from_persisted_screening(
    screening_path: str | Path = CANDIDATE_SCREENING_PATH,
    shortlist_path: str | Path = SHORTLIST_PATH,
    revised_screening_path: str | Path = REVISED_SCREENING_PATH,
    residual_path: str | Path = RESIDUAL_DIAGNOSTICS_PATH,
) -> dict[str, object]:
    """Recompute hard eligibility and IC shortlists without refitting candidates."""
    screening_path = Path(screening_path)
    shortlist_path = Path(shortlist_path)
    revised_screening_path = Path(revised_screening_path)
    residual_path = Path(residual_path)
    screening = pd.read_csv(screening_path)
    records = [_normalise_persisted_optimizer_fields(row) for row in screening.to_dict("records")]
    for row in records:
        row["model_valid"] = screening_row_is_model_valid(row)
        row["eligible"] = screening_row_is_eligible(row)
    revised = pd.DataFrame(records)
    revised_screening_path.parent.mkdir(parents=True, exist_ok=True)
    revised.to_csv(revised_screening_path, index=False)

    shortlist_frames: list[pd.DataFrame] = []
    criterion_rankings: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}
    eligible_counts: dict[str, int] = {}
    selected_paths: dict[str, list[dict[str, int]]] = {}
    for country in COUNTRY_CONFIG:
        country_rows = [row for row in records if row.get("country") == country]
        eligible_counts[country] = sum(
            _boolean_value(row.get("eligible", False)) for row in country_rows
        )
        path_groups: dict[tuple[int, int], list[Mapping[str, object]]] = {}
        for row in country_rows:
            path_groups.setdefault((int(row["d"]), int(row["D"])), []).append(row)
        criterion_rankings[country] = {
            f"d{d}_D{D}": _criterion_rankings(path_rows)
            for (d, D), path_rows in sorted(path_groups.items())
        }
        selected = narrow_shortlist(country_rows)
        shortlist = pd.DataFrame(selected)
        if not shortlist.empty:
            shortlist["shortlist_rank"] = range(1, len(shortlist) + 1)
            selected_paths[country] = [
                {"d": int(d), "D": int(D)}
                for d, D in shortlist[["d", "D"]].drop_duplicates().itertuples(
                    index=False, name=None
                )
            ]
        else:
            selected_paths[country] = []
        shortlist_frames.append(shortlist)

    shortlist = pd.concat(shortlist_frames, ignore_index=True)
    shortlist.to_csv(shortlist_path, index=False)
    return {
        "candidate_rows": int(len(revised)),
        "eligible_candidate_count": int(sum(eligible_counts.values())),
        "eligible_candidate_counts": eligible_counts,
        "shortlist_rows": int(len(shortlist)),
        "selected_paths": selected_paths,
        "criterion_rankings": criterion_rankings,
        "shortlist_residual_diagnostics": _residual_flag_summary(
            residual_path, shortlist
        ),
        "original_screening_path": str(screening_path),
        "revised_screening_path": str(revised_screening_path),
        "shortlist_path": str(shortlist_path),
        "refitted": False,
    }


def narrow_shortlist(rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
    """Narrow independently in each identical (d,D) path."""
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in rows:
        try:
            key = (int(row["d"]), int(row["D"]))
        except (KeyError, TypeError, ValueError):
            key = (0, 0)
        grouped.setdefault(key, []).append(row)

    selected: list[Mapping[str, object]] = []
    for key in sorted(grouped):
        eligible = [
            row
            for row in grouped[key]
            if _boolean_value(row.get("eligible", False))
            and all(
                np.isfinite(float(row[criterion]))
                for criterion in ("aic", "aicc", "bic")
            )
        ]
        selected_ids: set[str] = set()
        for criterion in ("aic", "aicc", "bic"):
            ranked = sorted(
                eligible,
                key=lambda row: (
                    float(row[criterion]),
                    int(row.get("specification_order", 0)),
                    str(row.get("specification_id", "")),
                ),
            )
            if ranked:
                selected_ids.add(str(ranked[0]["specification_id"]))
        selected.extend(
            row
            for row in eligible
            if str(row.get("specification_id")) in selected_ids
        )

    ordered = sorted(
        selected,
        key=lambda row: (
            int(row.get("D", 0)),
            int(row.get("d", 0)),
            int(row.get("specification_order", 0)),
            str(row.get("specification_id", "")),
        ),
    )
    return [dict(row) for row in ordered[:6]]


def _training_fields() -> dict[str, str]:
    return {
        "training_start": TRAINING_START.isoformat(),
        "training_end": TRAINING_END.isoformat(),
    }


def _fit_row(
    country: str,
    order: SarimaOrder,
    specification_order: int,
    fit: Any,
    selected_d: int,
    selected_D: int,
) -> dict[str, object]:
    residual = fit.residual_diagnostics
    row: dict[str, object] = {
        **_training_fields(),
        "country": country,
        "specification_id": order.specification_id,
        "specification_order": specification_order,
        "p": order.p,
        "d": order.d,
        "q": order.q,
        "P": order.P,
        "D": order.D,
        "Q": order.Q,
        "seasonal_period": order.seasonal_period,
        "trend": fit.trend,
        "selected_d": selected_d,
        "selected_D": selected_D,
        "fit_status": fit.fit_status,
        "convergence_status": fit.convergence_status,
        "optimizer_success": fit.optimizer_success,
        "optimizer_status": fit.optimizer_status,
        "optimizer_warnflag": fit.optimizer_warnflag,
        "optimizer_retry_count": fit.optimizer_retry_count,
        "converged": fit.converged,
        "parameters_finite": fit.parameters_finite,
        "standard_errors_finite": fit.standard_errors_finite,
        "forecast_valid": fit.forecast_valid,
        "stationarity_ok": fit.stationarity_ok,
        "invertibility_ok": fit.invertibility_ok,
        "nobs": fit.nobs,
        "effective_nobs": fit.effective_nobs,
        "n_params": fit.n_params,
        "log_likelihood": fit.log_likelihood,
        "aic": fit.aic,
        "aicc": fit.aicc,
        "bic": fit.bic,
        "ar_root_minimum": fit.ar_root_minimum,
        "ma_root_minimum": fit.ma_root_minimum,
        "residual_adequacy_rejected": residual.training_adequacy_rejected,
        "residual_adequacy_reason": residual.training_adequacy_reason,
        "warning_messages": json.dumps(list(fit.warning_messages)),
        "hard_warning_messages": json.dumps(list(fit.hard_warning_messages)),
        "error_message": fit.error_message,
    }
    row["model_valid"] = screening_row_is_model_valid(row)
    row["eligible"] = screening_row_is_eligible(row)
    return row


def _residual_rows(country: str, order: SarimaOrder, fit: Any) -> list[dict[str, object]]:
    residual = fit.residual_diagnostics
    common = {
        **_training_fields(),
        "country": country,
        "specification_id": order.specification_id,
        "p": order.p,
        "d": order.d,
        "q": order.q,
        "P": order.P,
        "D": order.D,
        "Q": order.Q,
        "seasonal_period": order.seasonal_period,
        "burn_in": residual.burn_in,
        "state_space_loglikelihood_burn": residual.state_space_loglikelihood_burn,
        "conservative_burn_in": residual.conservative_burn_in,
        "n_effective": residual.n_effective,
        "acf_threshold": residual.acf_threshold,
        "residual_adequacy_rejected": residual.training_adequacy_rejected,
        "residual_adequacy_reason": residual.training_adequacy_reason,
        "warnings": json.dumps(list(residual.warnings)),
    }
    rows: list[dict[str, object]] = []
    for lag, value in sorted(residual.acf_values.items()):
        rows.append(
            {
                **common,
                "diagnostic": "residual_acf",
                "lag": lag,
                "value": value,
                "threshold": residual.acf_threshold,
                "flagged": lag in residual.flagged_acf_lags,
            }
        )
    for lag, value in sorted(residual.ljung_box_statistics.items()):
        pvalue = residual.ljung_box_pvalues.get(lag, float("nan"))
        rows.append(
            {
                **common,
                "diagnostic": "ljung_box_statistic",
                "lag": lag,
                "value": value,
                "threshold": SIGNIFICANCE_LEVEL,
                "flagged": bool(np.isfinite(pvalue) and pvalue < SIGNIFICANCE_LEVEL),
            }
        )
        rows.append(
            {
                **common,
                "diagnostic": "ljung_box_pvalue",
                "lag": lag,
                "value": pvalue,
                "threshold": SIGNIFICANCE_LEVEL,
                "flagged": bool(np.isfinite(pvalue) and pvalue < SIGNIFICANCE_LEVEL),
            }
        )
    if not rows:
        rows.append(
            {
                **common,
                "diagnostic": "diagnostics_status",
                "lag": None,
                "value": float("nan"),
                "threshold": float("nan"),
                "flagged": True,
            }
        )
    return rows


def _seasonal_value(seasonal: Mapping[str, object], field: str, lag: int) -> object:
    values = seasonal.get(field, {})
    return values.get(lag, float("nan")) if isinstance(values, Mapping) else float("nan")


def _differencing_rows(country: str, path: Mapping[str, object]) -> list[dict[str, object]]:
    D = int(path["D"])
    seasonal = path["seasonal"]
    stationarity = path["stationarity"]
    rows = []
    for diagnostic in stationarity:
        row: dict[str, object] = {
            **_training_fields(),
            "country": country,
            "D": D,
            "d": diagnostic.get("d"),
            "selected_d": path.get("selected_d"),
            "path_viable": path.get("viable"),
            "viability_reason": path.get("viability_reason"),
            "seasonal_evidence_supported": path.get("seasonal_evidence_supported"),
            "seasonal_persistence_reduced": seasonal.get(
                "seasonal_persistence_reduced", float("nan")
            ),
            "seasonal_original_threshold": seasonal.get(
                "original_threshold", float("nan")
            ),
            "seasonal_transformed_threshold": seasonal.get(
                "transformed_threshold", float("nan")
            ),
            "observations": diagnostic.get("observations"),
            "deterministic_component": diagnostic.get("deterministic_component"),
            "adf_statistic": diagnostic.get("adf_statistic"),
            "adf_pvalue": diagnostic.get("adf_pvalue"),
            "adf_used_lag": diagnostic.get("adf_used_lag"),
            "kpss_statistic": diagnostic.get("kpss_statistic"),
            "kpss_pvalue": diagnostic.get("kpss_pvalue"),
            "kpss_bandwidth": diagnostic.get("kpss_bandwidth"),
            "adf_stationary": diagnostic.get("adf_stationary"),
            "kpss_stationary": diagnostic.get("kpss_stationary"),
            "stationary_supported": diagnostic.get("stationary_supported"),
            "adf_error": diagnostic.get("adf_error"),
            "kpss_error": diagnostic.get("kpss_error"),
            "warnings": diagnostic.get("warnings", "[]"),
        }
        for lag in SEASONAL_LAGS:
            row[f"seasonal_acf_original_lag{lag}"] = _seasonal_value(
                seasonal, "original_acf", lag
            )
            row[f"seasonal_acf_transformed_lag{lag}"] = _seasonal_value(
                seasonal, "transformed_acf", lag
            )
            row[f"seasonal_persistence_change_lag{lag}"] = _seasonal_value(
                seasonal, "persistence_change", lag
            )
            row[f"seasonal_dependence_original_lag{lag}"] = _seasonal_value(
                seasonal, "original_dependence", lag
            )
            row[f"seasonal_dependence_transformed_lag{lag}"] = _seasonal_value(
                seasonal, "transformed_dependence", lag
            )
        rows.append(row)
    return rows


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def screen_country(
    frame: pd.DataFrame, country: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = training_values(frame)
    paths = assess_differencing_paths(values)
    differencing_rows: list[dict[str, object]] = []
    screening_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []

    for D in (0, 1):
        path = paths[D]
        differencing_rows.extend(_differencing_rows(country, path))
        if not path["viable"]:
            continue
        selected_d = int(path["selected_d"])
        evidence = acf_pacf_evidence(values, D, selected_d)
        orders = build_candidate_orders(selected_d, D, evidence)
        for specification_order, order in enumerate(orders, 1):
            fit = fit_sarima(values, order)
            screening_rows.append(
                _fit_row(
                    country,
                    order,
                    specification_order,
                    fit,
                    selected_d,
                    D,
                )
            )
            residual_rows.extend(_residual_rows(country, order, fit))

    shortlist_rows = narrow_shortlist(screening_rows)
    for rank, row in enumerate(shortlist_rows, 1):
        row["shortlist_rank"] = rank

    screening_columns = list(screening_rows[0]) if screening_rows else list(_training_fields())
    differencing_columns = (
        list(differencing_rows[0]) if differencing_rows else list(_training_fields())
    )
    residual_columns = list(residual_rows[0]) if residual_rows else list(_training_fields())
    shortlist_columns = list(shortlist_rows[0]) if shortlist_rows else screening_columns + [
        "shortlist_rank"
    ]
    return (
        pd.DataFrame(differencing_rows, columns=differencing_columns),
        pd.DataFrame(screening_rows, columns=screening_columns),
        pd.DataFrame(residual_rows, columns=residual_columns),
        pd.DataFrame(shortlist_rows, columns=shortlist_columns),
    )


def run_screening(processed_directory: str | Path = PROCESSED) -> dict[str, object]:
    differencing_frames = []
    screening_frames = []
    residual_frames = []
    shortlist_frames = []
    selected_paths: dict[str, list[dict[str, object]]] = {}
    empty_shortlists: list[str] = []

    for country in COUNTRY_CONFIG:
        frame = load_country_data(country, processed_directory)
        differencing, screening, residual, shortlist = screen_country(frame, country)
        differencing_frames.append(differencing)
        screening_frames.append(screening)
        residual_frames.append(residual)
        shortlist_frames.append(shortlist)
        selected_paths[country] = (
            differencing.loc[differencing["path_viable"].eq(True), ["D", "selected_d"]]
            .drop_duplicates()
            .to_dict("records")
            if "path_viable" in differencing
            else []
        )
        if shortlist.empty:
            empty_shortlists.append(country)

    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.concat(differencing_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "sarima_differencing_diagnostics_2024.csv", index=False
    )
    pd.concat(screening_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "sarima_candidate_screening_training_2024.csv", index=False
    )
    pd.concat(residual_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "sarima_residual_diagnostics_training_2024.csv", index=False
    )
    pd.concat(shortlist_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "sarima_shortlists_2024.csv", index=False
    )
    return {
        "countries": list(COUNTRY_CONFIG),
        "training_start": TRAINING_START.isoformat(),
        "training_end": TRAINING_END.isoformat(),
        "candidate_rows": int(sum(len(frame) for frame in screening_frames)),
        "shortlist_rows": int(sum(len(frame) for frame in shortlist_frames)),
        "selected_paths": selected_paths,
        "countries_with_empty_shortlists": empty_shortlists,
        "outputs": str(TABLE_DIRECTORY),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen bounded SARIMA candidates on 2020-2023 training data"
    )
    parser.add_argument("--processed-directory", default=str(PROCESSED))
    parser.add_argument(
        "--reuse-screening",
        action="store_true",
        help="recompute eligibility and shortlists from persisted candidate fits",
    )
    args = parser.parse_args()
    if args.reuse_screening:
        summary = recompute_shortlist_from_persisted_screening()
    else:
        summary = run_screening(args.processed_directory)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
