from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, kpss, pacf

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import COUNTRY_CONFIG, PROCESSED, load_country_data
from arima_models import ArimaFitRecord, fit_arima, order_id, trend_for_d


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "arima" / "tables"
TRAINING_START = date(2020, 1, 1)
TRAINING_END = date(2023, 12, 31)
SCREENING_YEAR = 2024
SIGNIFICANCE_LEVEL = 0.05


def training_values(frame: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(frame["local_date"], errors="raise").dt.date
    selected = frame.loc[dates.between(TRAINING_START, TRAINING_END)]
    values = selected["actual_load_mwh"].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("training period contains no observations")
    if not np.isfinite(values).all():
        raise ValueError("training period contains non-finite observations")
    return values


def _differenced(values: np.ndarray, d: int) -> np.ndarray:
    result = values
    for _ in range(d):
        result = np.diff(result)
    return result


def _test_warnings(recorded: list[warnings.WarningMessage]) -> tuple[str, ...]:
    return tuple(
        f"{item.category.__name__}: {item.message}" for item in recorded
    )


def stationarity_diagnostics(values: object) -> list[dict[str, object]]:
    numeric_values = np.asarray(values, dtype=float).reshape(-1)
    if len(numeric_values) <= 50:
        raise ValueError("stationarity diagnostics require more than 50 observations")
    records: list[dict[str, object]] = []
    for d in (0, 1, 2):
        series = _differenced(numeric_values, d)
        deterministic = "ct" if d == 0 else "c"
        with warnings.catch_warnings(record=True) as adf_recorded:
            warnings.simplefilter("always")
            adf_result = adfuller(
                series,
                regression=deterministic,
                maxlag=24,
                autolag="AIC",
            )
        with warnings.catch_warnings(record=True) as kpss_recorded:
            warnings.simplefilter("always")
            kpss_result = kpss(
                series,
                regression=deterministic,
                nlags="auto",
            )
        adf_pvalue = float(adf_result[1])
        kpss_pvalue = float(kpss_result[1])
        records.append(
            {
                "d": d,
                "observations": len(series),
                "deterministic_component": deterministic,
                "adf_statistic": float(adf_result[0]),
                "adf_pvalue": adf_pvalue,
                "adf_used_lag": int(adf_result[2]),
                "kpss_statistic": float(kpss_result[0]),
                "kpss_pvalue": kpss_pvalue,
                "kpss_bandwidth": int(kpss_result[2]),
                "adf_stationary": adf_pvalue < SIGNIFICANCE_LEVEL,
                "kpss_stationary": kpss_pvalue >= SIGNIFICANCE_LEVEL,
                "stationary_supported": (
                    adf_pvalue < SIGNIFICANCE_LEVEL
                    and kpss_pvalue >= SIGNIFICANCE_LEVEL
                ),
                "warnings": json.dumps(
                    list(_test_warnings(adf_recorded))
                    + list(_test_warnings(kpss_recorded))
                ),
            }
        )
    return records


def select_d(diagnostics: list[dict[str, object]]) -> int | None:
    qualifying = [
        int(row["d"])
        for row in diagnostics
        if bool(
            row.get(
                "stationary_supported",
                float(row["adf_pvalue"]) < SIGNIFICANCE_LEVEL
                and float(row["kpss_pvalue"]) >= SIGNIFICANCE_LEVEL,
            )
        )
    ]
    return min(qualifying) if qualifying else None


def build_candidate_orders(d: int) -> tuple[tuple[int, int, int], ...]:
    if d not in (0, 1, 2):
        raise ValueError("d must be 0, 1 or 2")
    base = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (2, 0, 0),
        (0, 0, 2),
        (2, 0, 1),
        (1, 0, 2),
        (2, 0, 2),
        (3, 0, 0),
        (0, 0, 3),
    )
    return tuple((p, d, q) for p, _, q in base)


def narrow_shortlist(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    eligible = [
        row
        for row in rows
        if bool(row["eligible"])
        and all(np.isfinite(float(row[criterion])) for criterion in ("aic", "aicc", "bic"))
    ]
    if not eligible:
        return []
    selected_ids: set[str] = set()
    for criterion in ("aic", "aicc", "bic"):
        ranked = sorted(
            eligible,
            key=lambda row: (float(row[criterion]), int(row["specification_order"])),
        )
        selected_ids.update(str(row["specification_id"]) for row in ranked[:2])
    return sorted(
        [row for row in eligible if str(row["specification_id"]) in selected_ids],
        key=lambda row: int(row["specification_order"]),
    )


def _boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def screening_row_is_model_valid(row: dict[str, object] | pd.Series) -> bool:
    error = row.get("error_message")
    error_text = "" if pd.isna(error) else str(error)
    residual_error = "joint Ljung-Box and residual ACF rule" in error_text
    finite_criteria = all(
        np.isfinite(float(row[criterion])) for criterion in ("aic", "aicc", "bic")
    )
    return bool(
        str(row.get("fit_status")) == "success"
        and _boolean_value(row.get("converged"))
        and _boolean_value(row.get("parameters_finite"))
        and _boolean_value(row.get("standard_errors_finite"))
        and _boolean_value(row.get("stationarity_ok"))
        and _boolean_value(row.get("invertibility_ok"))
        and finite_criteria
        and (not error_text or residual_error)
    )


def reuse_existing_screening(
    screening_path: str | Path = TABLE_DIRECTORY / "arima_candidate_screening_training_2024.csv",
) -> dict[str, object]:
    screening = pd.read_csv(screening_path)
    screening["residual_adequacy_flag"] = screening["residual_rejected"].map(_boolean_value)
    screening["eligible"] = screening.apply(screening_row_is_model_valid, axis=1)
    residual_only = screening["error_message"].fillna("").str.contains(
        "joint Ljung-Box and residual ACF rule", regex=False
    )
    screening.loc[screening["eligible"] & residual_only, "error_message"] = None
    shortlist_frames = []
    for country in COUNTRY_CONFIG:
        country_rows = screening.loc[screening["country"].eq(country)].copy()
        selected_rows = narrow_shortlist(country_rows.to_dict("records"))
        shortlist = pd.DataFrame(selected_rows)
        if not shortlist.empty:
            shortlist["selected_d"] = int(country_rows["selected_d"].iloc[0])
            shortlist["shortlist_rank"] = range(1, len(shortlist) + 1)
        else:
            shortlist = pd.DataFrame(columns=list(screening.columns) + ["shortlist_rank"])
        shortlist_frames.append(shortlist)

    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    screening.to_csv(screening_path, index=False)
    shortlist_output = TABLE_DIRECTORY / "arima_shortlists_2024.csv"
    pd.concat(shortlist_frames, ignore_index=True).to_csv(shortlist_output, index=False)
    return {
        "screening_rows": len(screening),
        "eligible_rows": int(screening["eligible"].sum()),
        "shortlist_rows": int(sum(len(frame) for frame in shortlist_frames)),
        "selected_d": {
            country: int(screening.loc[screening["country"].eq(country), "selected_d"].iloc[0])
            for country in COUNTRY_CONFIG
        },
        "outputs": str(TABLE_DIRECTORY),
    }


def _acf_pacf_rows(values: np.ndarray, d: int, country: str) -> list[dict[str, object]]:
    differenced = _differenced(values, d)
    acf_values = acf(differenced, nlags=24, fft=True, adjusted=False)
    pacf_values = pacf(differenced, nlags=24, method="ywm")
    rows = []
    for lag in range(1, 25):
        rows.append(
            {
                "country": country,
                "d": d,
                "lag": lag,
                "acf": float(acf_values[lag]),
                "pacf": float(pacf_values[lag]),
                "pointwise_95_threshold": float(1.96 / np.sqrt(len(differenced))),
            }
        )
    return rows


def _fit_row(country: str, order: tuple[int, int, int], specification_order: int, fit: ArimaFitRecord) -> dict[str, object]:
    residual = fit.residual_diagnostics
    return {
        "country": country,
        "specification_id": order_id(order),
        "specification_order": specification_order,
        "p": order[0],
        "d": order[1],
        "q": order[2],
        "trend": fit.trend,
        "fit_status": fit.fit_status,
        "convergence_status": fit.convergence_status,
        "optimizer_retry_count": fit.optimizer_retry_count,
        "converged": fit.converged,
        "parameters_finite": fit.parameters_finite,
        "standard_errors_finite": fit.standard_errors_finite,
        "stationarity_ok": fit.stationarity_ok,
        "invertibility_ok": fit.invertibility_ok,
        "eligible": fit.eligible,
        "nobs": fit.nobs,
        "n_params": fit.n_params,
        "log_likelihood": fit.log_likelihood,
        "aic": fit.aic,
        "aicc": fit.aicc,
        "bic": fit.bic,
        "ar_root_minimum": fit.ar_root_minimum,
        "ma_root_minimum": fit.ma_root_minimum,
        "residual_rejected": residual.rejected if residual else True,
        "residual_rejection_reason": residual.rejection_reason if residual else "unavailable",
        "warning_messages": json.dumps(list(fit.warning_messages)),
        "hard_warning_messages": json.dumps(list(fit.hard_warning_messages)),
        "error_message": fit.error_message,
    }


def _residual_rows(country: str, order: tuple[int, int, int], fit: ArimaFitRecord) -> list[dict[str, object]]:
    residual = fit.residual_diagnostics
    if residual is None:
        return []
    rows = []
    for lag, value in residual.acf_values.items():
        rows.append(
            {
                "country": country,
                "specification_id": order_id(order),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "diagnostic": "residual_acf",
                "lag": lag,
                "value": value,
                "threshold": residual.acf_threshold,
                "flagged": lag in residual.flagged_acf_lags,
            }
        )
    for lag, value in residual.ljung_box_statistics.items():
        rows.append(
            {
                "country": country,
                "specification_id": order_id(order),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "diagnostic": "ljung_box_statistic",
                "lag": lag,
                "value": value,
                "threshold": 0.01,
                "flagged": residual.ljung_box_pvalues[lag] < 0.01,
            }
        )
        rows.append(
            {
                "country": country,
                "specification_id": order_id(order),
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "diagnostic": "ljung_box_pvalue",
                "lag": lag,
                "value": residual.ljung_box_pvalues[lag],
                "threshold": 0.01,
                "flagged": residual.ljung_box_pvalues[lag] < 0.01,
            }
        )
    return rows


def screen_country(frame: pd.DataFrame, country: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = training_values(frame)
    stationarity = stationarity_diagnostics(values)
    selected_d = select_d(stationarity)
    if selected_d is None:
        raise ValueError(f"no adequate differencing order for {country}")
    orders = build_candidate_orders(selected_d)
    screening_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for specification_order, order in enumerate(orders, 1):
        fit = fit_arima(values, order)
        screening_rows.append(_fit_row(country, order, specification_order, fit))
        residual_rows.extend(_residual_rows(country, order, fit))
    shortlist_rows = narrow_shortlist(screening_rows)
    stationarity_frame = pd.DataFrame(stationarity)
    stationarity_frame.insert(0, "country", country)
    screening_frame = pd.DataFrame(screening_rows)
    screening_frame["selected_d"] = selected_d
    shortlist_frame = (
        pd.DataFrame(shortlist_rows)
        if shortlist_rows
        else pd.DataFrame(columns=screening_frame.columns)
    )
    shortlist_frame["selected_d"] = selected_d
    shortlist_frame["shortlist_rank"] = range(1, len(shortlist_frame) + 1)
    return (
        stationarity_frame,
        screening_frame,
        pd.DataFrame(residual_rows),
        shortlist_frame,
    )


def run_screening(processed_directory: str | Path = PROCESSED) -> dict[str, object]:
    stationarity_frames = []
    screening_frames = []
    residual_frames = []
    shortlist_frames = []
    acf_pacf_frames = []
    for country in COUNTRY_CONFIG:
        frame = load_country_data(country, processed_directory)
        values = training_values(frame)
        stationarity, screening, residual, shortlist = screen_country(frame, country)
        stationarity_frames.append(stationarity)
        screening_frames.append(screening)
        residual_frames.append(residual)
        shortlist_frames.append(shortlist)
        selected_d = select_d(stationarity.to_dict("records"))
        if selected_d is None:
            raise ValueError(f"no adequate differencing order for {country}")
        acf_pacf_frames.append(pd.DataFrame(_acf_pacf_rows(values, selected_d, country)))

    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.concat(stationarity_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "arima_differencing_diagnostics_2024.csv", index=False
    )
    pd.concat(screening_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "arima_candidate_screening_training_2024.csv", index=False
    )
    pd.concat(residual_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "arima_residual_diagnostics_training_2024.csv", index=False
    )
    pd.concat(shortlist_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "arima_shortlists_2024.csv", index=False
    )
    pd.concat(acf_pacf_frames, ignore_index=True).to_csv(
        TABLE_DIRECTORY / "arima_acf_pacf_training_2024.csv", index=False
    )
    return {
        "countries": list(COUNTRY_CONFIG),
        "training_start": TRAINING_START.isoformat(),
        "training_end": TRAINING_END.isoformat(),
        "candidate_orders_per_country": len(build_candidate_orders(0)),
        "shortlist_rows": int(sum(len(frame) for frame in shortlist_frames)),
        "selected_d": {
            country: int(select_d(stationarity.to_dict("records")))
            for country, stationarity in zip(
                COUNTRY_CONFIG, stationarity_frames
            )
        },
        "countries_with_empty_shortlists": [
            country
            for country, frame in zip(COUNTRY_CONFIG, shortlist_frames)
            if frame.empty
        ],
        "outputs": str(TABLE_DIRECTORY),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen non-seasonal ARIMA candidates on 2020-2023 data")
    parser.add_argument("--processed-directory", default=str(PROCESSED))
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse persisted training fits and recompute eligibility/shortlists",
    )
    args = parser.parse_args()
    result = (
        reuse_existing_screening()
        if args.reuse_existing
        else run_screening(args.processed_directory)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
