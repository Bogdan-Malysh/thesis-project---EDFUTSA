from pathlib import Path

import pandas as pd

from apg_hourly import (
    aggregate_hourly,
    audit_apg_file,
    combine_hourly,
    load_apg_file,
)


ACTUAL_HEADER = "Time from [CET/CEST],Time to [CET/CEST],Power [MW]\n"
FORECAST_HEADER = "Time from [CET/CEST],Time to [CET/CEST],Load [MW]\n"


def write_apg(path: Path, rows: list[str], forecast: bool = False) -> None:
    header = FORECAST_HEADER if forecast else ACTUAL_HEADER
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_converts_mw_to_quarter_hour_mwh(tmp_path):
    path = tmp_path / "actual.csv"
    write_apg(
        path,
        ["2020-01-01 00:00:00,2020-01-01 00:15:00,100.000"],
    )

    result = load_apg_file(path, "actual")
    row = result["data"].iloc[0]

    assert result["source"].iloc[0, 2] == "100.000"
    assert row["value_mw"] == 100.0
    assert row["energy_mwh"] == 25.0
    assert row["interval_start_utc"] == pd.Timestamp("2019-12-31 23:00", tz="UTC")
    assert row["interval_end_utc"] == pd.Timestamp("2019-12-31 23:15", tz="UTC")


def test_load_parses_spring_transition_without_creating_0200(tmp_path):
    path = tmp_path / "spring.csv"
    write_apg(
        path,
        ["2020-03-29 01:45:00,2020-03-29 03:00:00,100.000"],
    )

    data = load_apg_file(path, "actual")["data"]

    assert data.iloc[0]["interval_start_utc"] == pd.Timestamp(
        "2020-03-29 00:45", tz="UTC"
    )
    assert data.iloc[0]["interval_end_utc"] == pd.Timestamp(
        "2020-03-29 01:00", tz="UTC"
    )
    assert data.iloc[0]["interval_start_local"].hour == 1
    assert data.iloc[0]["interval_end_local"].hour == 3


def test_load_distinguishes_autumn_2a_and_2b_in_utc(tmp_path):
    path = tmp_path / "autumn.csv"
    write_apg(
        path,
        [
            "2020-10-25 01:45:00,2020-10-25 2A:00:00,100.000",
            "2020-10-25 2A:45:00,2020-10-25 2B:00:00,100.000",
            "2020-10-25 2B:45:00,2020-10-25 03:00:00,100.000",
        ],
    )

    data = load_apg_file(path, "actual")["data"]

    assert list(data["interval_start_utc"]) == list(
        pd.to_datetime(
            [
                "2020-10-24 23:45",
                "2020-10-25 00:45",
                "2020-10-25 01:45",
            ],
            utc=True,
        )
    )
    assert list(data["interval_end_utc"]) == list(
        pd.to_datetime(
            ["2020-10-25 00:00", "2020-10-25 01:00", "2020-10-25 02:00"],
            utc=True,
        )
    )


def test_audit_checks_schema_row_count_and_15_minute_continuity(tmp_path):
    path = tmp_path / "actual.csv"
    write_apg(
        path,
        [
            "2020-01-01 00:00:00,2020-01-01 00:15:00,100.000",
            "2020-01-01 00:15:00,2020-01-01 00:30:00,101.000",
            "2020-01-01 00:30:00,2020-01-01 00:45:00,102.000",
            "2020-01-01 00:45:00,2020-01-01 01:00:00,103.000",
        ],
    )

    audit = audit_apg_file(path, "actual")

    assert audit["status"] == "PASS"
    assert audit["row_count"] == 4
    assert audit["missing_or_empty_count"] == 0
    assert audit["non_numeric_count"] == 0
    assert audit["duplicate_utc_intervals"] == 0
    assert audit["utc_15_minute_continuity"] is True


def test_aggregate_requires_four_intervals_and_combines_actual_forecast(tmp_path):
    actual_path = tmp_path / "actual.csv"
    forecast_path = tmp_path / "forecast.csv"
    rows = [
        "2020-01-01 00:00:00,2020-01-01 00:15:00,100.000",
        "2020-01-01 00:15:00,2020-01-01 00:30:00,101.000",
        "2020-01-01 00:30:00,2020-01-01 00:45:00,102.000",
        "2020-01-01 00:45:00,2020-01-01 01:00:00,103.000",
    ]
    forecast_rows = [row.rsplit(",", 1)[0] + ",80.000" for row in rows]
    write_apg(actual_path, rows)
    write_apg(forecast_path, forecast_rows, forecast=True)

    actual_hour = aggregate_hourly(load_apg_file(actual_path, "actual")["data"], "actual_load_mwh")
    forecast_hour = aggregate_hourly(
        load_apg_file(forecast_path, "forecast")["data"], "forecasted_load_mwh"
    )
    combined = combine_hourly(actual_hour, forecast_hour)

    assert len(combined) == 1
    assert combined.iloc[0]["actual_load_mwh"] == 101.5
    assert combined.iloc[0]["forecasted_load_mwh"] == 80.0
    assert combined.iloc[0]["interval_start_utc"] == pd.Timestamp(
        "2019-12-31 23:00", tz="UTC"
    )
