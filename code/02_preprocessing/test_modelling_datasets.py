from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelling_datasets import (
    EXPECTED_ROWS,
    OUTPUT_COLUMNS,
    build_country_modelling_dataset,
    build_modelling_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
BINARY_COLUMNS = [
    "forecasted_grid_load_valid",
    "is_weekend",
    "is_public_holiday",
    "is_covid_period",
    "is_post_invasion",
]


def _read_inputs(country: str) -> tuple[pd.DataFrame, ...]:
    stem = country.lower()
    return (
        pd.read_csv(PROCESSED / f"smard_{stem}_hourly.csv"),
        pd.read_csv(PROCESSED / f"temperature_{stem}_hourly.csv"),
        pd.read_csv(PROCESSED / f"calendar_{stem}_hourly.csv"),
        pd.read_csv(PROCESSED / f"crisis_{stem}_hourly.csv"),
    )


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_country_modelling_dataset_has_complete_final_contract(country):
    result = build_country_modelling_dataset(country)

    assert len(result) == EXPECTED_ROWS
    assert list(result.columns) == OUTPUT_COLUMNS

    utc = pd.DatetimeIndex(pd.to_datetime(result["timestamp_utc"], utc=True))
    assert utc.is_unique
    assert utc.is_monotonic_increasing
    assert (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all()
    complete_columns = [
        column for column in result.columns if column != "forecasted_grid_load_mwh"
    ]
    assert not result[complete_columns].isna().any().any()

    complete_numeric = result[
        [
            "actual_grid_load_mwh",
            "temperature_c",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_public_holiday",
            "is_covid_period",
            "is_post_invasion",
        ]
    ]
    assert np.isfinite(complete_numeric.to_numpy()).all()
    available_forecast = result["forecasted_grid_load_mwh"].notna()
    assert np.isfinite(
        result.loc[available_forecast, "forecasted_grid_load_mwh"].to_numpy()
    ).all()
    for column in BINARY_COLUMNS:
        assert result[column].isin([0, 1]).all()

    assert list(result.columns).count("timestamp_local") == 1
    assert not any("apg" in column.lower() for column in result.columns)

    missing_forecast = result["forecasted_grid_load_mwh"].isna()
    if country == "Germany":
        assert int(missing_forecast.sum()) == 24
        expected_missing = pd.date_range(
            "2020-01-30 23:00:00+00:00", periods=24, freq="h"
        )
        assert pd.DatetimeIndex(utc[missing_forecast]).equals(expected_missing)
        assert (~result.loc[missing_forecast, "forecasted_grid_load_valid"]).all()
    else:
        assert not missing_forecast.any()

    repeated_hour = result["timestamp_local"].astype(str).str.startswith(
        "2020-10-25T02:00:00"
    )
    assert set(result.loc[repeated_hour, "timestamp_local"].str[-6:]) == {
        "+01:00",
        "+02:00",
    }


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_modelling_dataset_preserves_target_benchmark_and_calendar_local_values(country):
    target, _, calendar, _ = _read_inputs(country)
    result = build_country_modelling_dataset(country)

    expected_target = target.sort_values("interval_start_utc").reset_index(drop=True)
    expected_calendar = calendar.assign(
        _timestamp_utc=pd.to_datetime(calendar["timestamp_utc"], utc=True)
    ).sort_values("_timestamp_utc").reset_index(drop=True)

    expected_utc = pd.to_datetime(expected_target["interval_start_utc"], utc=True)
    actual_utc = pd.to_datetime(result["timestamp_utc"], utc=True)
    assert actual_utc.equals(expected_utc)
    pd.testing.assert_series_equal(
        result["actual_grid_load_mwh"],
        expected_target["actual_grid_load_mwh"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["forecasted_grid_load_mwh"],
        expected_target["forecasted_grid_load_mwh"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["forecasted_grid_load_valid"],
        expected_target["forecasted_grid_load_valid"],
        check_names=False,
    )
    assert result["timestamp_local"].tolist() == expected_calendar[
        "timestamp_local"
    ].tolist()
    assert result["timestamp_local"].tolist() != expected_target[
        "interval_start_local"
    ].tolist()


def test_duplicate_feature_key_is_rejected_before_merging():
    target, temperature, calendar, crisis = _read_inputs("Germany")
    duplicate_temperature = pd.concat(
        [temperature, temperature.iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="temperature.*duplicates"):
        build_modelling_dataset(
            target, duplicate_temperature, calendar, crisis, "Germany"
        )


def test_unmatched_feature_key_is_rejected_before_merging():
    target, temperature, calendar, crisis = _read_inputs("Germany")
    missing_calendar = calendar.iloc[:-1].copy()

    with pytest.raises(ValueError, match="calendar.*keys do not match"):
        build_modelling_dataset(target, temperature, missing_calendar, crisis, "Germany")
