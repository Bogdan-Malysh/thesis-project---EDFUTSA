from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from dateutil.easter import easter

from calendar_features import (
    OUTPUT_COLUMNS,
    build_calendar_features,
    holiday_dates,
    validate_calendar_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def official_reference_dates(country: str, year: int) -> set[date]:
    easter_sunday = easter(year)
    fixed = {
        date(year, 1, 1),
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
    }
    if country == "Germany":
        fixed.add(date(year, 10, 3))
        offsets = (-2, 1, 39, 50)
    elif country == "Austria":
        fixed.update(
            {
                date(year, 1, 6),
                date(year, 8, 15),
                date(year, 10, 26),
                date(year, 11, 1),
                date(year, 12, 8),
            }
        )
        offsets = (1, 39, 50, 60)
    else:
        raise ValueError(country)
    return fixed | {easter_sunday + timedelta(days=offset) for offset in offsets}


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_package_holidays_match_official_rules_for_project_years(country):
    for year in range(2020, 2026):
        assert holiday_dates(country, [year]) == official_reference_dates(country, year)


@pytest.mark.parametrize(
    "country, expected_count, excluded",
    [
        ("Germany", 9, {date(2020, 6, 11), date(2020, 10, 31)}),
        ("Austria", 13, {date(2020, 4, 10), date(2020, 11, 2)}),
    ],
)
def test_holiday_dates_use_only_countrywide_statutory_holidays(
    country, expected_count, excluded
):
    dates = holiday_dates(country, [2020])

    assert len(dates) == expected_count
    assert excluded.isdisjoint(dates)


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_calendar_features_preserve_distinct_autumn_offsets(country):
    timestamps = pd.date_range(
        "2020-10-25 00:00:00+00:00", periods=5, freq="h"
    )

    result = build_calendar_features(timestamps, country)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["timestamp_utc"].is_unique
    assert result["timestamp_local"].dt.tz is not None
    first_repeated = result.loc[0, "timestamp_local"]
    second_repeated = result.loc[1, "timestamp_local"]
    assert first_repeated.hour == second_repeated.hour == 2
    assert first_repeated.utcoffset() != second_repeated.utcoffset()
    assert first_repeated.isoformat() != second_repeated.isoformat()
    assert {first_repeated.strftime("%z"), second_repeated.strftime("%z")} == {
        "+0200",
        "+0100",
    }


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_calendar_features_use_the_local_date_for_holiday_status(country):
    timestamps = pd.to_datetime(
        [
            "2020-04-13 21:00:00+00:00",
            "2020-04-13 22:00:00+00:00",
            "2020-04-13 23:00:00+00:00",
            "2020-04-14 00:00:00+00:00",
        ],
        utc=True,
    )

    result = build_calendar_features(timestamps, country)

    assert result["timestamp_local"].dt.date.tolist() == [
        date(2020, 4, 13),
        date(2020, 4, 14),
        date(2020, 4, 14),
        date(2020, 4, 14),
    ]
    assert result["is_public_holiday"].tolist() == [1, 0, 0, 0]


def test_calendar_features_have_basic_integer_ranges_and_weekend_consistency():
    timestamps = pd.date_range("2020-01-03", periods=72, freq="h", tz="UTC")

    result = build_calendar_features(timestamps, "Germany")

    assert result["hour"].between(0, 23).all()
    assert result["day_of_week"].between(0, 6).all()
    assert result["month"].between(1, 12).all()
    assert result["is_weekend"].isin([0, 1]).all()
    assert result["is_public_holiday"].isin([0, 1]).all()
    assert (result["is_weekend"] == result["day_of_week"].ge(5).astype(int)).all()
    assert not result[OUTPUT_COLUMNS[2:]].isna().any().any()
    assert all(str(result[column].dtype).startswith("int") for column in OUTPUT_COLUMNS[2:])


@pytest.mark.parametrize(
    "country, source_name",
    [
        ("Germany", "smard_germany_hourly.csv"),
        ("Austria", "smard_austria_hourly.csv"),
    ],
)
def test_full_project_key_set_validates_without_changing_source(country, source_name):
    source_path = PROJECT_ROOT / "data" / "processed" / source_name
    source_bytes = source_path.read_bytes()
    source = pd.read_csv(source_path, usecols=["interval_start_utc"])
    keys = pd.to_datetime(source["interval_start_utc"], utc=True)

    result = build_calendar_features(keys, country)
    validation = validate_calendar_features(result, keys, country)

    assert source_path.read_bytes() == source_bytes
    assert validation["row_count"] == 52_608
    assert validation["timestamp_utc_unique"] is True
    assert validation["timestamp_utc_hourly_continuous"] is True
    assert validation["missing_feature_count"] == 0
    assert validation["autumn_repeated_local_rows"] == 12


def test_validation_rejects_changed_holiday_flag():
    timestamps = pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC")
    result = build_calendar_features(timestamps, "Germany")
    result.loc[0, "is_public_holiday"] = 0

    with pytest.raises(ValueError, match="holiday flags"):
        validate_calendar_features(result, timestamps, "Germany")


def test_validation_rejects_changed_weekend_flag():
    timestamps = pd.date_range("2020-01-03", periods=48, freq="h", tz="UTC")
    result = build_calendar_features(timestamps, "Germany")
    result.loc[24, "is_weekend"] = 1 - result.loc[24, "is_weekend"]

    with pytest.raises(ValueError, match="weekend flags"):
        validate_calendar_features(result, timestamps, "Germany")


def test_validation_rejects_changed_utc_key():
    timestamps = pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC")
    result = build_calendar_features(timestamps, "Germany")
    result.loc[1, "timestamp_utc"] = result.loc[1, "timestamp_utc"] + pd.Timedelta(hours=1)

    with pytest.raises(ValueError, match="UTC keys"):
        validate_calendar_features(result, timestamps, "Germany")
