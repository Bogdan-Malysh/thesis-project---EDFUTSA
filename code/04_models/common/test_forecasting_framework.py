from datetime import date

import numpy as np
import pandas as pd
import pytest

from forecasting_framework import (
    BASELINE_MODELS,
    calculate_metrics,
    country_forecast_origin,
    evaluate_target_day,
    extract_target_day,
    forecast_path,
    information_set,
)


def _hourly_frame(start: str, end: str, timezone_name: str = "Europe/Berlin") -> pd.DataFrame:
    utc = pd.date_range(start, end, freq="h", tz="UTC")
    local = utc.tz_convert(timezone_name)
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": np.arange(len(utc), dtype=float) + 100,
            "temperature_c": 10.0,
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_public_holiday": 0,
        }
    )


def test_country_specific_forecast_origins_are_local_preceding_day_cutoffs():
    germany = country_forecast_origin(date(2024, 1, 2), "Germany")
    austria = country_forecast_origin(date(2024, 1, 2), "Austria")

    assert germany.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M") == "2024-01-01 18:00"
    assert austria.tz_convert("Europe/Vienna").strftime("%Y-%m-%d %H:%M") == "2024-01-01 08:00"


def test_information_set_excludes_all_future_observations():
    frame = _hourly_frame("2024-01-01 00:00", "2024-01-02 23:00")
    cutoff = country_forecast_origin(date(2024, 1, 2), "Germany")

    available = information_set(frame, cutoff)

    assert "interval_end_utc" in available
    assert not (available["timestamp_utc"] == cutoff).any()
    assert (available["interval_end_utc"] <= cutoff).all()
    assert available["interval_end_utc"].max() == cutoff


@pytest.mark.parametrize(
    ("country", "expected_bridge_rows", "expected_total_rows"),
    [("Germany", 6, 30), ("Austria", 16, 40)],
)
def test_forecast_path_starts_at_origin_and_has_expected_bridge_length(
    country, expected_bridge_rows, expected_total_rows
):
    frame = _hourly_frame("2024-01-01 00:00", "2024-01-02 23:00")
    target_date = date(2024, 1, 2)
    origin = country_forecast_origin(target_date, country)

    path = forecast_path(frame, country, target_date, "Naive")

    assert path.iloc[0]["timestamp_utc"] == origin
    assert len(path) == expected_total_rows
    assert int((~path["is_target_day"]).sum()) == expected_bridge_rows
    assert int(path["is_target_day"].sum()) == 24


def test_lagged_baselines_use_exact_utc_24_and_168_hour_sources():
    frame = _hourly_frame("2024-01-01 00:00", "2024-01-10 23:00")
    target_date = date(2024, 1, 8)

    daily = forecast_path(frame, "Germany", target_date, "Daily seasonal naive")
    weekly = forecast_path(frame, "Germany", target_date, "Weekly seasonal naive")

    assert (
        daily["source_timestamp_utc"]
        == daily["timestamp_utc"] - pd.Timedelta(hours=24)
    ).all()
    assert (
        weekly["source_timestamp_utc"]
        == weekly["timestamp_utc"] - pd.Timedelta(hours=168)
    ).all()
    assert (daily.loc[daily["is_target_day"], "source_kind"] == "forecast").any()


@pytest.mark.parametrize("target_date", [date(2024, 3, 31), date(2024, 10, 27)])
def test_utc_lags_remain_exact_across_dst_transitions(target_date):
    if target_date.month == 3:
        frame = _hourly_frame("2024-03-20 00:00", "2024-04-08 23:00")
    else:
        frame = _hourly_frame("2024-10-15 00:00", "2024-11-10 23:00")

    for model_family, lag_hours in (
        ("Daily seasonal naive", 24),
        ("Weekly seasonal naive", 168),
    ):
        path = forecast_path(frame, "Germany", target_date, model_family)
        assert (
            path["source_timestamp_utc"]
            == path["timestamp_utc"] - pd.Timedelta(hours=lag_hours)
        ).all()


def test_forecast_does_not_use_future_actual_loads_for_all_baselines():
    frame = _hourly_frame("2024-01-01 00:00", "2024-01-02 23:00")
    cutoff = country_forecast_origin(date(2024, 1, 2), "Germany")
    altered = frame.copy()
    altered.loc[
        pd.to_datetime(altered["timestamp_utc"], utc=True) + pd.Timedelta(hours=1)
        > cutoff,
        "actual_grid_load_mwh",
    ] = -999999.0

    for model_family in BASELINE_MODELS:
        original_path = forecast_path(
            frame, "Germany", date(2024, 1, 2), model_family
        )
        altered_path = forecast_path(
            altered, "Germany", date(2024, 1, 2), model_family
        )

        assert original_path["forecast_mwh"].tolist() == altered_path["forecast_mwh"].tolist()


def test_forecast_path_contains_bridge_intervals_before_target_day():
    frame = _hourly_frame("2024-01-01 00:00", "2024-01-02 23:00")

    path = forecast_path(frame, "Germany", date(2024, 1, 2), "Daily seasonal naive")

    assert path["timestamp_utc"].is_monotonic_increasing
    assert path["is_target_day"].any()
    assert (~path["is_target_day"]).any()
    assert path.loc[~path["is_target_day"], "bridge_used"].all()
    assert path["forecast_mwh"].notna().all()


def test_metrics_use_target_day_rows_only():
    frame = _hourly_frame("2024-01-01 00:00", "2024-01-02 23:00")
    path = forecast_path(frame, "Germany", date(2024, 1, 2), "Naive")
    path.loc[~path["is_target_day"], "forecast_mwh"] = 0.0

    metrics = evaluate_target_day(path)

    assert metrics["evaluated_observations"] == 24
    assert metrics["coverage"] == 1.0
    direct = calculate_metrics(
        path.loc[path["is_target_day"], "actual_load_mwh"],
        path.loc[path["is_target_day"], "forecast_mwh"],
    )
    assert metrics["mae"] == pytest.approx(direct["mae"])


def test_extract_target_day_returns_only_complete_local_date_rows():
    frame = _hourly_frame("2024-01-01 00:00", "2024-01-03 23:00")

    target = extract_target_day(frame, date(2024, 1, 2))

    assert len(target) == 24
    assert target["timestamp_utc"].is_unique
    assert target["local_date"].eq("2024-01-02").all()


@pytest.mark.parametrize(
    ("target_date", "expected_count"),
    [(date(2024, 3, 31), 23), (date(2024, 10, 27), 25)],
)
def test_extract_target_day_preserves_dst_day_lengths(target_date, expected_count):
    frame = _hourly_frame("2024-03-30 00:00", "2024-10-28 23:00")

    target = extract_target_day(frame, target_date)

    assert len(target) == expected_count
    assert target["timestamp_utc"].is_unique
    wall_time = target["timestamp_local"].str[:13]
    assert bool(wall_time.duplicated().any()) == (expected_count == 25)


def test_metric_calculation_reports_primary_secondary_metrics_and_coverage():
    metrics = calculate_metrics(
        actual=[1.0, 2.0, 4.0],
        forecast=[1.0, 1.0, 2.0],
        expected_observations=4,
    )

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["mape"] == pytest.approx(100 / 3)
    assert metrics["evaluated_observations"] == 3
    assert metrics["coverage"] == pytest.approx(0.75)


def test_all_phase_one_baseline_models_are_available():
    assert BASELINE_MODELS == (
        "Naive",
        "Daily seasonal naive",
        "Weekly seasonal naive",
        "Average hour-of-week profile",
    )
