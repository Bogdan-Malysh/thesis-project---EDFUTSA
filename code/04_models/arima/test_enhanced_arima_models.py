from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from arima_models import fit_arima
from enhanced_arima_models import (
    EnhancedSpecification,
    WeeklyTransformResult,
    build_weekly_forecast_path,
    derive_weekly_invalid_target_dates,
    enhanced_specifications,
    fit_enhanced,
    fit_missing_aware_arima,
    transform_weekly_series,
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
WEEKLY_ORDERS = (
    (1, 0, 1),
    (2, 0, 0),
    (2, 0, 1),
    (2, 0, 2),
    (3, 0, 1),
)


def _dst_frame(
    timezone: str = "Europe/Berlin",
    start: str = "2024-03-23",
    end: str = "2024-11-10",
) -> pd.DataFrame:
    utc_index = pd.date_range(
        start=f"{start} 00:00:00",
        end=f"{end} 23:00:00",
        freq="h",
        tz="UTC",
    )
    local = utc_index.tz_convert(timezone)
    return pd.DataFrame(
        {
            "timestamp_utc": utc_index,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": 1000.0 + np.arange(len(utc_index), dtype=float),
            "hour": local.hour,
            "day_of_week": local.dayofweek,
        }
    )


def _prepared_frame(
    timezone: str = "Europe/Berlin",
    start: str = "2024-03-23",
    end: str = "2024-11-10",
) -> pd.DataFrame:
    frame = _dst_frame(timezone, start, end)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["timestamp_local"] = frame["timestamp_local"].astype(str)
    frame["actual_load_mwh"] = frame["actual_grid_load_mwh"]
    frame["interval_end_utc"] = frame["timestamp_utc"] + pd.Timedelta(hours=1)
    frame["local_date"] = frame["timestamp_local"].str[:10]
    frame["local_occurrence"] = frame.groupby(
        ["local_date", "hour"], sort=False
    ).cumcount()
    return frame


def test_enhanced_specifications_have_exact_noncolliding_branch_contracts():
    specifications = enhanced_specifications()

    assert isinstance(specifications, tuple)
    assert [spec.order for spec in specifications if spec.branch == "ordinary"] == list(
        ORDINARY_ORDERS
    )
    assert [
        spec.order for spec in specifications if spec.branch == "weekly_differenced"
    ] == list(WEEKLY_ORDERS)
    assert len(specifications) == 16
    assert [spec.specification_id for spec in specifications] == [
        "enhanced_ordinary_arima_p1_d1_q1",
        "enhanced_ordinary_arima_p1_d1_q2",
        "enhanced_ordinary_arima_p1_d1_q3",
        "enhanced_ordinary_arima_p2_d1_q1",
        "enhanced_ordinary_arima_p2_d1_q2",
        "enhanced_ordinary_arima_p2_d1_q3",
        "enhanced_ordinary_arima_p3_d1_q1",
        "enhanced_ordinary_arima_p3_d1_q2",
        "enhanced_ordinary_arima_p3_d1_q3",
        "enhanced_ordinary_arima_p4_d1_q1",
        "enhanced_ordinary_arima_p4_d1_q2",
        "weekly_differenced_arima_p1_d0_q1",
        "weekly_differenced_arima_p2_d0_q0",
        "weekly_differenced_arima_p2_d0_q1",
        "weekly_differenced_arima_p2_d0_q2",
        "weekly_differenced_arima_p3_d0_q1",
    ]
    assert all(spec.trend == "n" for spec in specifications if spec.branch == "ordinary")
    assert all(
        spec.trend == "c"
        for spec in specifications
        if spec.branch == "weekly_differenced"
    )
    assert all(
        spec.specification_id.startswith("enhanced_ordinary_arima_")
        for spec in specifications
        if spec.branch == "ordinary"
    )
    assert all(
        spec.specification_id.startswith("weekly_differenced_arima_")
        for spec in specifications
        if spec.branch == "weekly_differenced"
    )
    assert len({spec.specification_id for spec in specifications}) == len(specifications)
    assert all(spec.transform == "identity" for spec in specifications if spec.branch == "ordinary")
    assert all(
        spec.transform == "weekly_difference"
        for spec in specifications
        if spec.branch == "weekly_differenced"
    )

    with pytest.raises(FrozenInstanceError):
        specifications[0].branch = "changed"


def test_weekly_transform_preserves_hourly_index_and_records_missing_source():
    frame = _prepared_frame(start="2024-03-24", end="2024-04-07")
    cutoff = frame["interval_end_utc"].iloc[-1]

    result = transform_weekly_series(frame, cutoff)

    assert isinstance(result, WeeklyTransformResult)
    assert result.values.index.equals(pd.DatetimeIndex(frame["timestamp_utc"]))
    assert result.values.index.freq == pd.tseries.frequencies.to_offset("h")
    assert result.values.shape == (len(frame),)
    assert result.values.isna().any()
    assert result.valid_observations + result.missing_observations == len(frame)
    assert set(result.issues.columns) >= {
        "target_timestamp_utc",
        "source_key",
        "reason",
    }
    assert result.issues["reason"].eq("missing_source").any()
    assert not result.issues["target_timestamp_utc"].eq(
        pd.Timestamp("2024-04-07 01:00:00", tz="UTC")
    ).any()


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_weekly_invalid_dates_are_derived_for_all_dst_cases(country):
    timezone = {"Germany": "Europe/Berlin", "Austria": "Europe/Vienna"}[country]
    frame = _prepared_frame(timezone)
    invalid = derive_weekly_invalid_target_dates(frame, 2024)
    invalid_dates = set(invalid["target_date"].astype(str))

    expected = {
        "2024-03-31": False,
        "2024-04-07": True,
        "2024-10-27": True,
        "2024-11-03": False,
    }
    assert {
        target: target in invalid_dates
        for target in expected
    } == expected
    assert invalid["target_date"].astype(str).str.startswith("2024-").all()
    assert country in ("Germany", "Austria")


def test_weekly_mapping_rejects_multiple_source_rows_without_utc_fallback():
    frame = _prepared_frame(start="2024-03-24", end="2024-04-07")
    source_mask = (
        frame["local_date"].eq("2024-03-31")
        & frame["hour"].eq(1)
        & frame["local_occurrence"].eq(0)
    )
    duplicate = frame.loc[source_mask].iloc[0].copy()
    duplicate["timestamp_utc"] += pd.Timedelta(minutes=30)
    duplicate["interval_end_utc"] = duplicate["timestamp_utc"] + pd.Timedelta(hours=1)
    duplicate["actual_load_mwh"] += 5.0
    frame = pd.concat([frame, pd.DataFrame([duplicate])], ignore_index=True)

    result = transform_weekly_series(
        frame,
        frame["interval_end_utc"].max(),
    )
    issue = result.issues.loc[
        result.issues["target_timestamp_utc"].eq(
            pd.Timestamp("2024-04-06 23:00:00", tz="UTC")
        )
    ]

    assert len(issue) == 1
    assert issue.iloc[0]["reason"] == "ambiguous_source"
    assert issue.iloc[0]["source_row_count"] == 2
    assert issue.iloc[0]["source_key"] == ("2024-03-31", 1, 0)
    assert pd.isna(issue.iloc[0]["source_timestamp_utc"])
    assert pd.isna(
        result.values.loc[pd.Timestamp("2024-04-06 23:00:00", tz="UTC")]
    )


def test_missing_aware_fit_retains_internal_time_position_and_differs_from_dropna():
    index = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    values = pd.Series(
        100.0 + np.sin(np.arange(96) / 4.0) + np.arange(96) / 100.0,
        index=index,
        name="weekly_difference",
    )
    values.iloc[48] = np.nan

    missing_aware = fit_missing_aware_arima(values, (1, 0, 0))
    compressed = fit_arima(values.dropna().to_numpy(), (1, 0, 0))

    assert missing_aware.error_message is None
    assert missing_aware.fitted_result is not None
    endog = missing_aware.fitted_result.model.data.orig_endog
    assert endog.shape[0] == 96
    assert pd.DatetimeIndex(endog.index).equals(index)
    assert pd.DatetimeIndex(endog.index).freq == pd.tseries.frequencies.to_offset("h")
    assert pd.isna(endog.iloc[48])
    assert compressed.fitted_result.nobs == 95
    missing_forecast = float(
        missing_aware.fitted_result.get_forecast(steps=1).predicted_mean.iloc[0]
    )
    compressed_forecast = float(
        np.asarray(
            compressed.fitted_result.get_forecast(steps=1).predicted_mean,
            dtype=float,
        ).reshape(-1)[0]
    )
    assert not np.isclose(
        missing_forecast,
        compressed_forecast,
        rtol=1e-10,
        atol=1e-12,
    )
    assert not np.isfinite(values.to_numpy()).all()
    assert fit_arima(values.to_numpy(), (1, 0, 0)).error_message == (
        "training values are not finite"
    )


def test_ordinary_enhanced_fit_delegates_to_existing_fit_arima(monkeypatch):
    specification = EnhancedSpecification(
        "ordinary",
        "enhanced_ordinary_arima_p1_d1_q1",
        (1, 1, 1),
        "n",
        "identity",
    )
    sentinel = object()

    def fake_fit(values, order):
        assert values == [1.0, 2.0]
        assert order == specification.order
        return sentinel

    import enhanced_arima_models

    monkeypatch.setattr(enhanced_arima_models, "fit_arima", fake_fit)

    assert fit_enhanced(specification, [1.0, 2.0]) is sentinel


def test_weekly_forecast_path_marks_whole_date_invalid_without_recursive_sources():
    frame = _prepared_frame(start="2024-03-24", end="2024-04-07")
    specification = next(
        spec for spec in enhanced_specifications() if spec.branch == "weekly_differenced"
    )

    class FakeFitted:
        def get_forecast(self, steps):
            return type(
                "Forecast",
                (),
                {
                    "predicted_mean": pd.Series(
                        np.arange(steps, dtype=float),
                        index=pd.date_range(
                            "2024-03-24", periods=steps, freq="h", tz="UTC"
                        ),
                    )
                },
            )()

    path = build_weekly_forecast_path(
        frame,
        "Germany",
        date(2024, 4, 7),
        specification,
        FakeFitted(),
    )

    assert len(path.loc[path["is_target_day"]]) == 24
    assert path["status"].eq("weekly_lag_invalid").all()
    assert path["forecast_mwh"].isna().all()
    assert "source_timestamp_utc" in path
    assert path["source_kind"].eq("mapping_invalid").all()


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_weekly_forecast_path_reconstructs_from_origin_available_actuals(country):
    timezone = {"Germany": "Europe/Berlin", "Austria": "Europe/Vienna"}[country]
    frame = _prepared_frame(
        timezone,
        start="2024-10-20",
        end="2024-11-03",
    )
    specification = next(
        spec for spec in enhanced_specifications() if spec.branch == "weekly_differenced"
    )

    class FakeFitted:
        def __init__(self):
            self.calls = []

        def get_forecast(self, steps):
            self.calls.append(steps)
            return type(
                "Forecast",
                (),
                {"predicted_mean": np.arange(steps, dtype=float)},
            )()

    fitted = FakeFitted()
    path = build_weekly_forecast_path(
        frame,
        country,
        date(2024, 11, 3),
        specification,
        fitted,
    )

    origin = pd.to_datetime(path["forecast_origin_utc"].iloc[0], utc=True)
    source_end_by_timestamp = frame.set_index("timestamp_utc")["interval_end_utc"]
    source_ends = path["source_timestamp_utc"].map(source_end_by_timestamp)

    assert fitted.calls == [len(path)]
    assert path["status"].eq("completed").all()
    assert len(path.loc[path["is_target_day"]]) == 24
    assert path["source_kind"].eq("observed").all()
    assert path["source_timestamp_utc"].notna().all()
    assert path["source_timestamp_utc"].isin(frame["timestamp_utc"]).all()
    assert source_ends.notna().all()
    assert (source_ends <= origin).all()
    np.testing.assert_allclose(
        path["forecast_mwh"].to_numpy(),
        path["source_actual_mwh"].to_numpy() + np.arange(len(path)),
    )
