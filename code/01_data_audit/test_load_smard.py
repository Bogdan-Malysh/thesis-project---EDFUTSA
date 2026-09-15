from pathlib import Path

import pandas as pd
import pytest

from load_smard import load_smard_file


ACTUAL_HEADER = (
    "Start date;End date;grid load [MWh] Calculated resolutions;"
    "Grid load incl. hydro pumped storage [MWh] Calculated resolutions;"
    "Hydro pumped storage [MWh] Calculated resolutions;"
    "Residual load [MWh] Calculated resolutions\n"
)

FORECAST_HEADER = (
    "Start date;End date;grid load [MWh] Calculated resolutions;"
    "Residual load [MWh] Calculated resolutions\n"
)


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_preserves_source_and_parses_actual_numeric_columns(tmp_path):
    source_path = tmp_path / "actual.csv"
    write_csv(
        source_path,
        ACTUAL_HEADER,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;43,500.50;43,776.25;275.75;36,890.25"
        ],
    )
    original_bytes = source_path.read_bytes()

    result = load_smard_file(
        source_path,
        country="Germany",
        dataset_type="actual",
        timezone_name="Europe/Berlin",
    )

    assert source_path.read_bytes() == original_bytes
    assert list(result["source"].columns) == [
        "Start date",
        "End date",
        "grid load [MWh] Calculated resolutions",
        "Grid load incl. hydro pumped storage [MWh] Calculated resolutions",
        "Hydro pumped storage [MWh] Calculated resolutions",
        "Residual load [MWh] Calculated resolutions",
    ]
    assert result["source"].iloc[0, 2] == "43,500.50"
    assert result["source"].iloc[0, 3] == "43,776.25"
    row = result["data"].iloc[0]
    assert row[
        "interval_start_raw"
    ] == "Jan 1, 2020 12:00 AM"
    assert row["interval_end_raw"] == "Jan 1, 2020 1:00 AM"
    assert row["interval_start_local"] == pd.Timestamp("2020-01-01 00:00")
    assert row["interval_end_local"] == pd.Timestamp("2020-01-01 01:00")
    assert row["interval_start_utc"] == pd.Timestamp("2019-12-31 23:00", tz="UTC")
    assert row["interval_end_utc"] == pd.Timestamp("2020-01-01 00:00", tz="UTC")
    assert row["is_repeated_autumn_hour"] == False
    assert row[[
        "grid_load_mwh",
        "grid_load_including_pumped_storage_mwh",
        "pumped_storage_mwh",
        "residual_load_mwh",
    ]].to_dict() == {
        "grid_load_mwh": 43500.50,
        "grid_load_including_pumped_storage_mwh": 43776.25,
        "pumped_storage_mwh": 275.75,
        "residual_load_mwh": 36890.25,
    }
    assert result["column_map"] == {
        "grid load [MWh] Calculated resolutions": "grid_load_mwh",
        "Grid load incl. hydro pumped storage [MWh] Calculated resolutions": (
            "grid_load_including_pumped_storage_mwh"
        ),
        "Hydro pumped storage [MWh] Calculated resolutions": "pumped_storage_mwh",
        "Residual load [MWh] Calculated resolutions": "residual_load_mwh",
    }


def test_load_records_country_dataset_and_timezone_metadata(tmp_path):
    source_path = tmp_path / "forecast.csv"
    write_csv(
        source_path,
        (
            "Start date;End date;grid load [MWh] Calculated resolutions;"
            "Residual load [MWh] Calculated resolutions\n"
        ),
        ["Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;5,752.00;3,730.00"],
    )

    result = load_smard_file(
        source_path,
        country="Austria",
        dataset_type="forecast",
        timezone_name="Europe/Vienna",
    )

    assert result["metadata"] == {
        "country": "Austria",
        "dataset_type": "forecast",
        "timezone": "Europe/Vienna",
    }
    assert result["column_map"] == {
        "grid load [MWh] Calculated resolutions": "grid_load_mwh",
        "Residual load [MWh] Calculated resolutions": "residual_load_mwh",
    }
    assert result["data"].iloc[0].to_dict() == {
        "interval_start_raw": "Jan 1, 2020 12:00 AM",
        "interval_end_raw": "Jan 1, 2020 1:00 AM",
        "interval_start_local": pd.Timestamp("2020-01-01 00:00"),
        "interval_end_local": pd.Timestamp("2020-01-01 01:00"),
        "interval_start_utc": pd.Timestamp("2019-12-31 23:00", tz="UTC"),
        "interval_end_utc": pd.Timestamp("2020-01-01 00:00", tz="UTC"),
        "is_repeated_autumn_hour": False,
        "grid_load_mwh": 5752.0,
        "residual_load_mwh": 3730.0,
    }


def test_load_preserves_invalid_source_values_and_reports_numeric_parsing(tmp_path):
    source_path = tmp_path / "invalid.csv"
    write_csv(
        source_path,
        ACTUAL_HEADER,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;;not-a-number;NaN;36,890.25"
        ],
    )

    result = load_smard_file(
        source_path,
        country="Germany",
        dataset_type="actual",
        timezone_name="Europe/Berlin",
    )

    assert result["source"].iloc[0, 2] == ""
    assert result["source"].iloc[0, 3] == "not-a-number"
    assert result["source"].iloc[0, 4] == "NaN"
    assert result["data"].iloc[0]["grid_load_mwh"] != result["data"].iloc[0][
        "grid_load_mwh"
    ]
    assert result["data"].iloc[0]["pumped_storage_mwh"] != result["data"].iloc[0][
        "pumped_storage_mwh"
    ]
    assert result["diagnostics"]["numeric_columns"][
        "grid load [MWh] Calculated resolutions"
    ] == {"missing_count": 1, "non_numeric_count": 0, "non_finite_count": 0}
    assert result["diagnostics"]["numeric_columns"][
        "Grid load incl. hydro pumped storage [MWh] Calculated resolutions"
    ] == {"missing_count": 0, "non_numeric_count": 1, "non_finite_count": 0}
    assert result["diagnostics"]["numeric_columns"][
        "Hydro pumped storage [MWh] Calculated resolutions"
    ] == {"missing_count": 0, "non_numeric_count": 0, "non_finite_count": 1}


def make_forecast_row(start: str, end: str, value: int) -> str:
    return f"{start};{end};{value}.00;{value - 1}.00"


@pytest.mark.parametrize(
    "country, timezone_name",
    [("Germany", "Europe/Berlin"), ("Austria", "Europe/Vienna")],
)
def test_spring_transition_preserves_missing_civil_hour_and_keeps_utc_hourly(
    tmp_path, country, timezone_name
):
    source_path = tmp_path / f"{country.lower()}_spring.csv"
    write_csv(
        source_path,
        FORECAST_HEADER,
        [
            make_forecast_row(
                "Mar 29, 2020 12:00 AM", "Mar 29, 2020 1:00 AM", 100
            ),
            make_forecast_row(
                "Mar 29, 2020 1:00 AM", "Mar 29, 2020 2:00 AM", 101
            ),
            make_forecast_row(
                "Mar 29, 2020 3:00 AM", "Mar 29, 2020 4:00 AM", 103
            ),
            make_forecast_row(
                "Mar 29, 2020 4:00 AM", "Mar 29, 2020 5:00 AM", 104
            ),
        ],
    )

    result = load_smard_file(
        source_path,
        country=country,
        dataset_type="forecast",
        timezone_name=timezone_name,
    )
    data = result["data"]

    assert list(data["interval_start_local"].dt.hour) == [0, 1, 3, 4]
    assert list(data["interval_start_utc"]) == list(
        pd.date_range("2020-03-28 23:00", periods=4, freq="h", tz="UTC")
    )
    assert data["interval_start_utc"].is_unique
    assert result["diagnostics"]["timestamps"] == {
        "utc_start_unique": True,
        "utc_start_strictly_increasing": True,
        "utc_start_hourly_continuity": True,
        "utc_interval_duration_valid": True,
        "repeated_autumn_hour_count": 0,
        "errors": [],
    }


@pytest.mark.parametrize(
    "country, timezone_name",
    [("Germany", "Europe/Berlin"), ("Austria", "Europe/Vienna")],
)
def test_autumn_transition_assigns_both_offsets_and_keeps_utc_hourly(
    tmp_path, country, timezone_name
):
    source_path = tmp_path / f"{country.lower()}_autumn.csv"
    write_csv(
        source_path,
        FORECAST_HEADER,
        [
            make_forecast_row(
                "Oct 25, 2020 12:00 AM", "Oct 25, 2020 1:00 AM", 100
            ),
            make_forecast_row(
                "Oct 25, 2020 1:00 AM", "Oct 25, 2020 2:00 AM", 101
            ),
            make_forecast_row(
                "Oct 25, 2020 2:00 AM", "Oct 25, 2020 3:00 AM", 102
            ),
            make_forecast_row(
                "Oct 25, 2020 2:00 AM", "Oct 25, 2020 3:00 AM", 103
            ),
            make_forecast_row(
                "Oct 25, 2020 3:00 AM", "Oct 25, 2020 4:00 AM", 104
            ),
        ],
    )

    result = load_smard_file(
        source_path,
        country=country,
        dataset_type="forecast",
        timezone_name=timezone_name,
    )
    data = result["data"]

    assert list(data["interval_start_utc"]) == list(
        pd.date_range("2020-10-24 22:00", periods=5, freq="h", tz="UTC")
    )
    assert list(data["interval_end_utc"]) == list(
        pd.date_range("2020-10-24 23:00", periods=5, freq="h", tz="UTC")
    )
    assert list(data["is_repeated_autumn_hour"]) == [False, False, True, True, False]
    assert data["interval_start_utc"].is_unique
    assert result["diagnostics"]["timestamps"] == {
        "utc_start_unique": True,
        "utc_start_strictly_increasing": True,
        "utc_start_hourly_continuity": True,
        "utc_interval_duration_valid": True,
        "repeated_autumn_hour_count": 2,
        "errors": [],
    }


def test_invalid_timestamp_order_is_reported_without_silent_resolution(tmp_path):
    source_path = tmp_path / "out_of_order.csv"
    write_csv(
        source_path,
        FORECAST_HEADER,
        [
            make_forecast_row(
                "Oct 25, 2020 12:00 AM", "Oct 25, 2020 1:00 AM", 100
            ),
            make_forecast_row(
                "Oct 25, 2020 2:00 AM", "Oct 25, 2020 3:00 AM", 102
            ),
            make_forecast_row(
                "Oct 25, 2020 2:00 AM", "Oct 25, 2020 3:00 AM", 103
            ),
            make_forecast_row(
                "Oct 25, 2020 1:00 AM", "Oct 25, 2020 2:00 AM", 101
            ),
        ],
    )

    result = load_smard_file(
        source_path,
        country="Germany",
        dataset_type="forecast",
        timezone_name="Europe/Berlin",
    )

    assert result["diagnostics"]["timestamps"]["utc_start_unique"] is True
    assert result["diagnostics"]["timestamps"][
        "utc_start_strictly_increasing"
    ] is False
    assert result["diagnostics"]["timestamps"][
        "utc_start_hourly_continuity"
    ] is False
    assert result["diagnostics"]["timestamps"]["errors"]
