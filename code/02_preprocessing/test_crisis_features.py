from pathlib import Path
import json

import pandas as pd
import pytest

from crisis_features import (
    CRISIS_SOURCES,
    OUTPUT_COLUMNS,
    build_crisis_features,
    summarize_crisis_features,
    validate_crisis_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OFFICIAL_URLS = [
    "https://www.who.int/news-room/speeches/item/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020",
    "https://www.who.int/news/item/05-05-2023-statement-on-the-fifteenth-meeting-of-the-international-health-regulations-%282005%29-emergency-committee-regarding-the-coronavirus-disease-%28covid-19%29-pandemic",
    "https://www.consilium.europa.eu/en/meetings/european-council/2022/02/24/",
    "https://www.iea.org/reports/world-energy-outlook-2022",
]
EXPECTED_RETRIEVAL = [
    {"status": "retrieved", "http_status": 200},
    {"status": "retrieved", "http_status": 200},
    {"status": "blocked", "http_status": 403},
    {"status": "retrieved", "http_status": 200},
]


def test_crisis_metadata_uses_current_urls_and_records_retrieval_limits():
    metadata_path = PROJECT_ROOT / "data" / "validation" / "crisis_feature_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert [source["url"] for source in CRISIS_SOURCES] == EXPECTED_OFFICIAL_URLS
    assert [source["url"] for source in metadata["official_sources"]] == EXPECTED_OFFICIAL_URLS
    assert [
        {
            "status": source["automated_retrieval"]["status"],
            "http_status": source["automated_retrieval"]["http_status"],
        }
        for source in metadata["official_sources"]
    ] == EXPECTED_RETRIEVAL
    assert metadata["official_sources"] == CRISIS_SOURCES

    blocked_source = metadata["official_sources"][2]
    assert blocked_source["automated_retrieval"]["content_validated"] is False
    assert "HTTP 403" in blocked_source["automated_retrieval"]["limitation"]
    assert "no source hash" in blocked_source["automated_retrieval"]["limitation"]


def source_for_utc(values: list[str], timezone_name: str) -> pd.DataFrame:
    utc = pd.to_datetime(values, utc=True)
    local = utc.tz_convert(timezone_name)
    return pd.DataFrame(
        {
            "timestamp_utc": [value.isoformat() for value in utc],
            "timestamp_local": [value.isoformat() for value in local],
        }
    )


def test_boundary_dates_use_the_local_calendar_date():
    source = source_for_utc(
        [
            "2020-03-10 22:00:00+00:00",
            "2020-03-10 23:00:00+00:00",
            "2022-02-23 22:00:00+00:00",
            "2022-02-23 23:00:00+00:00",
            "2023-05-05 21:00:00+00:00",
            "2023-05-05 22:00:00+00:00",
            "2023-05-05 23:00:00+00:00",
            "2023-05-06 00:00:00+00:00",
        ],
        "Europe/Berlin",
    )

    result = build_crisis_features(source)

    assert result["is_covid_period"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert result["is_post_invasion"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert result.loc[3, "is_covid_period"] == 1
    assert result.loc[3, "is_post_invasion"] == 1


@pytest.mark.parametrize(
    "country, timezone_name",
    [
        ("Germany", "Europe/Berlin"),
        ("Austria", "Europe/Vienna"),
    ],
)
def test_builder_preserves_timestamp_strings_and_offset_values(country, timezone_name):
    source = source_for_utc(
        [
            "2020-10-24 22:00:00+00:00",
            "2020-10-24 23:00:00+00:00",
            "2020-10-25 00:00:00+00:00",
            "2020-10-25 01:00:00+00:00",
            "2020-10-25 02:00:00+00:00",
        ],
        timezone_name,
    )

    result = build_crisis_features(source)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["timestamp_utc"].tolist() == source["timestamp_utc"].tolist()
    assert result["timestamp_local"].tolist() == source["timestamp_local"].tolist()
    assert any(value.endswith("+02:00") for value in result["timestamp_local"])
    assert any(value.endswith("+01:00") for value in result["timestamp_local"])
    assert result["is_covid_period"].isin([0, 1]).all()
    assert result["is_post_invasion"].isin([0, 1]).all()
    assert str(result["is_covid_period"].dtype).startswith("int")
    assert str(result["is_post_invasion"].dtype).startswith("int")


def test_overlap_is_one_for_both_indicators():
    source = source_for_utc(
        [
            "2022-02-24 00:00:00+00:00",
            "2023-05-05 21:00:00+00:00",
        ],
        "Europe/Berlin",
    )

    result = build_crisis_features(source)

    assert (result["is_covid_period"] == 1).all()
    assert (result["is_post_invasion"] == 1).all()


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_full_calendar_input_validates_without_source_changes(country):
    source_path = PROJECT_ROOT / "data" / "processed" / f"calendar_{country.lower()}_hourly.csv"
    before = source_path.read_bytes()
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False)
    source = source[["timestamp_utc", "timestamp_local"]].copy()

    result = build_crisis_features(source)
    validation = validate_crisis_features(result, source, country)

    assert source_path.read_bytes() == before
    assert validation["row_count"] == 52_608
    assert validation["timestamp_utc_unique"] is True
    assert validation["timestamp_utc_hourly_continuous"] is True
    assert validation["missing_value_count"] == 0
    assert validation["timestamp_local_offset_count"] == 2


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_summary_contains_counts_and_local_bounds(country):
    source_path = PROJECT_ROOT / "data" / "processed" / f"calendar_{country.lower()}_hourly.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False)
    source = source[["timestamp_utc", "timestamp_local"]].copy()
    result = build_crisis_features(source)

    summary = summarize_crisis_features(result, country)

    assert summary["country"] == country
    assert summary["total_observations"] == 52_608
    assert summary["covid_period_hours"] > 0
    assert summary["post_invasion_hours"] > 0
    assert summary["overlap_hours"] > 0
    assert summary["covid_first_local_timestamp"].startswith("2020-03-11")
    assert summary["covid_last_local_timestamp"].startswith("2023-05-05")
    assert summary["post_invasion_first_local_timestamp"].startswith("2022-02-24")


def test_validation_rejects_non_hourly_utc_keys():
    source_path = PROJECT_ROOT / "data" / "processed" / "calendar_germany_hourly.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False)
    source = source[["timestamp_utc", "timestamp_local"]].copy()
    source.loc[1, "timestamp_utc"] = "2020-01-01T00:30:00+00:00"
    result = build_crisis_features(source)

    with pytest.raises(ValueError, match="hourly"):
        validate_crisis_features(result, source, "Germany")
