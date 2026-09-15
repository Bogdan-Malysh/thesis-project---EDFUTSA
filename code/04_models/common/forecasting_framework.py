from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = PROJECT_ROOT / "data" / "processed"

COUNTRY_CONFIG = {
    "Germany": {
        "timezone": "Europe/Berlin",
        "forecast_origin_hour": 18,
        "filename": "modelling_germany_hourly.csv",
    },
    "Austria": {
        "timezone": "Europe/Vienna",
        "forecast_origin_hour": 8,
        "filename": "modelling_austria_hourly.csv",
    },
}

BASELINE_MODELS = (
    "Naive",
    "Daily seasonal naive",
    "Weekly seasonal naive",
    "Average hour-of-week profile",
)

REQUIRED_COLUMNS = {
    "timestamp_utc",
    "timestamp_local",
    "actual_grid_load_mwh",
    "hour",
    "day_of_week",
}


@dataclass(frozen=True)
class InformationContext:
    cutoff_utc: pd.Timestamp
    origin_local: str
    last_observed_value: float
    observed_utc_values: dict[pd.Timestamp, float]
    hour_of_week_profile: dict[tuple[int, int], float]


def _coerce_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"forecasting frame is missing required columns: {missing}")

    prepared = frame.copy()
    prepared["timestamp_utc"] = pd.to_datetime(
        prepared["timestamp_utc"], utc=True, errors="coerce"
    )
    if prepared["timestamp_utc"].isna().any():
        raise ValueError("forecasting frame contains invalid UTC timestamps")
    prepared["timestamp_local"] = prepared["timestamp_local"].astype(str)
    prepared["actual_load_mwh"] = pd.to_numeric(
        prepared["actual_grid_load_mwh"], errors="coerce"
    )
    if prepared["actual_load_mwh"].isna().any():
        raise ValueError("forecasting frame contains missing actual load values")
    prepared["hour"] = pd.to_numeric(prepared["hour"], errors="raise").astype(int)
    prepared["day_of_week"] = pd.to_numeric(
        prepared["day_of_week"], errors="raise"
    ).astype(int)
    prepared["interval_end_utc"] = prepared["timestamp_utc"] + pd.Timedelta(hours=1)
    prepared = prepared.sort_values("timestamp_utc", ignore_index=True)
    if prepared["timestamp_utc"].duplicated().any():
        raise ValueError("forecasting frame contains duplicate UTC timestamps")
    if not prepared["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("forecasting frame is not ordered by UTC timestamp")
    prepared["local_date"] = prepared["timestamp_local"].str[:10]
    prepared["local_occurrence"] = prepared.groupby(
        ["local_date", "hour"], sort=False
    ).cumcount()
    return prepared


def _is_prepared(frame: pd.DataFrame) -> bool:
    return {
        "actual_load_mwh",
        "interval_end_utc",
        "local_date",
        "local_occurrence",
    }.issubset(frame.columns)


def _as_prepared(frame: pd.DataFrame) -> pd.DataFrame:
    return frame if _is_prepared(frame) else _prepare_frame(frame)


def load_country_data(
    country: str,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    path = Path(processed_directory) / COUNTRY_CONFIG[country]["filename"]
    return _prepare_frame(pd.read_csv(path))


def country_forecast_origin(
    target_date: date | str | pd.Timestamp,
    country: str,
) -> pd.Timestamp:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    local_date = _coerce_date(target_date) - timedelta(days=1)
    local_time = datetime.combine(
        local_date,
        time(hour=COUNTRY_CONFIG[country]["forecast_origin_hour"]),
    )
    return pd.Timestamp(local_time).tz_localize(
        ZoneInfo(COUNTRY_CONFIG[country]["timezone"])
    ).tz_convert("UTC")


def information_set(frame: pd.DataFrame, cutoff_utc: pd.Timestamp) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    cutoff = pd.Timestamp(cutoff_utc).tz_convert("UTC")
    return prepared.loc[prepared["interval_end_utc"] <= cutoff].copy()


def extract_target_day(
    frame: pd.DataFrame,
    target_date: date | str | pd.Timestamp,
) -> pd.DataFrame:
    prepared = _as_prepared(frame)
    local_date = _coerce_date(target_date).isoformat()
    target = prepared.loc[prepared["local_date"].eq(local_date)].copy()
    if len(target) not in (23, 24, 25):
        raise ValueError(
            f"target day {local_date} has {len(target)} intervals; expected 23, 24 or 25"
        )
    if not target["timestamp_utc"].is_unique:
        raise ValueError(f"target day {local_date} contains duplicate UTC timestamps")
    return target.reset_index(drop=True)


def build_information_context(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
) -> InformationContext:
    prepared = _as_prepared(frame)
    cutoff_utc = country_forecast_origin(target_date, country)
    available = information_set(prepared, cutoff_utc)
    if available.empty:
        raise ValueError("information set is empty")
    observed_utc_values = available.set_index("timestamp_utc")[
        "actual_load_mwh"
    ].astype(float).to_dict()
    profile = available.groupby(["day_of_week", "hour"])["actual_load_mwh"].mean()
    return InformationContext(
        cutoff_utc=cutoff_utc,
        origin_local=cutoff_utc
        .tz_convert(COUNTRY_CONFIG[country]["timezone"])
        .isoformat(),
        last_observed_value=float(available["actual_load_mwh"].iloc[-1]),
        observed_utc_values=observed_utc_values,
        hour_of_week_profile={
            (int(day), int(hour)): float(value)
            for (day, hour), value in profile.items()
        },
    )


def _source_value(
    context: InformationContext,
    forecast_values: dict[pd.Timestamp, float],
    source_timestamp: pd.Timestamp,
) -> tuple[float | None, str]:
    observed = context.observed_utc_values.get(source_timestamp)
    if observed is not None:
        return observed, "observed"
    forecast = forecast_values.get(source_timestamp)
    if forecast is not None:
        return forecast, "forecast"
    return None, "fallback"


def forecast_path(
    frame: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
    model_family: str,
    context: InformationContext | None = None,
) -> pd.DataFrame:
    if model_family not in BASELINE_MODELS:
        raise ValueError(f"unsupported baseline model: {model_family}")
    prepared = _as_prepared(frame)
    local_date = _coerce_date(target_date).isoformat()
    target = extract_target_day(prepared, local_date)
    if context is None:
        context = build_information_context(prepared, country, local_date)
    cutoff_utc = context.cutoff_utc
    path = prepared.loc[
        prepared["timestamp_utc"].ge(cutoff_utc)
        & prepared["local_date"].le(local_date)
    ].copy()
    if path.empty or not path["local_date"].eq(local_date).any():
        raise ValueError(f"no forecast path exists for {country} {local_date}")
    if not target["timestamp_utc"].isin(path["timestamp_utc"]).all():
        raise ValueError(f"forecast path does not contain all target timestamps for {local_date}")

    forecast_values: dict[pd.Timestamp, float] = {}
    forecasts: list[float] = []
    bridge_used: list[bool] = []
    source_timestamps: list[pd.Timestamp] = []
    source_kinds: list[str] = []
    previous_value = context.last_observed_value
    for row in path.itertuples(index=False):
        is_target_day = row.local_date == local_date
        value: float
        source_timestamp = pd.NaT
        source_kind = "naive"
        if model_family == "Naive":
            value = previous_value
        elif model_family in ("Daily seasonal naive", "Weekly seasonal naive"):
            lag_hours = 24 if model_family == "Daily seasonal naive" else 168
            source_timestamp = row.timestamp_utc - pd.Timedelta(hours=lag_hours)
            source, source_type = _source_value(
                context,
                forecast_values,
                source_timestamp,
            )
            if source is None:
                value = previous_value
                source_kind = "fallback"
            else:
                value = source
                source_kind = source_type
        else:
            value = context.hour_of_week_profile.get(
                (int(row.day_of_week), int(row.hour)), previous_value
            )
            source_kind = "profile"

        value = float(value)
        forecasts.append(value)
        bridge_used.append(not is_target_day)
        source_timestamps.append(source_timestamp)
        source_kinds.append(source_kind)
        forecast_values[row.timestamp_utc] = value
        previous_value = value

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
    result["model_family"] = model_family
    result["target_date"] = local_date
    result["forecast_origin_local"] = context.origin_local
    result["forecast_origin_utc"] = context.cutoff_utc.isoformat()
    result["forecast_mwh"] = forecasts
    result["bridge_used"] = bridge_used
    result["source_timestamp_utc"] = source_timestamps
    result["source_kind"] = source_kinds
    result["is_target_day"] = result["local_date"].eq(local_date)
    result["evaluated"] = result["is_target_day"]
    result["information_cutoff_utc"] = context.cutoff_utc.isoformat()
    return result[
        [
            "country",
            "model_family",
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
            "bridge_used",
            "source_timestamp_utc",
            "source_kind",
            "is_target_day",
            "evaluated",
        ]
    ]


def calculate_metrics(
    actual: Iterable[float],
    forecast: Iterable[float],
    expected_observations: int | None = None,
) -> dict[str, float | int]:
    actual_values = np.asarray(list(actual), dtype=float)
    forecast_values = np.asarray(list(forecast), dtype=float)
    if actual_values.shape != forecast_values.shape:
        raise ValueError("actual and forecast arrays must have the same length")
    valid = np.isfinite(actual_values) & np.isfinite(forecast_values)
    actual_values = actual_values[valid]
    forecast_values = forecast_values[valid]
    count = int(len(actual_values))
    expected = count if expected_observations is None else int(expected_observations)
    if count == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "evaluated_observations": 0,
            "coverage": 0.0 if expected else float("nan"),
        }
    errors = forecast_values - actual_values
    nonzero_actual = actual_values != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero_actual] / actual_values[nonzero_actual])) * 100)
        if nonzero_actual.any()
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": mape,
        "evaluated_observations": count,
        "coverage": float(count / expected) if expected else float("nan"),
    }


def evaluate_target_day(
    forecast: pd.DataFrame,
    expected_observations: int | None = None,
) -> dict[str, float | int]:
    if "is_target_day" not in forecast.columns:
        raise ValueError("forecast frame is missing is_target_day")
    target = forecast.loc[forecast["is_target_day"]]
    return calculate_metrics(
        target["actual_load_mwh"],
        target["forecast_mwh"],
        expected_observations=expected_observations,
    )
