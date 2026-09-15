from pathlib import Path

import pandas as pd

from smard_hourly import build_country_hourly


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


def write_smard(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_build_country_preserves_utc_local_columns_auxiliary_values_and_flags(tmp_path):
    write_smard(
        tmp_path / "smard_austria_actual.csv",
        ACTUAL_HEADER,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;102.00;2.00;-5.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;not-number;101.00;0.00;0.00",
        ],
    )
    write_smard(
        tmp_path / "smard_austria_forecasted.csv",
        FORECAST_HEADER,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;99.00;-4.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;0.00;0.00",
        ],
    )

    data = build_country_hourly(tmp_path, "Austria")

    assert len(data) == 2
    assert list(data["interval_start_utc"]) == list(
        pd.to_datetime(["2019-12-31 23:00", "2020-01-01 00:00"], utc=True)
    )
    assert data.iloc[0]["interval_start_local"] == pd.Timestamp("2020-01-01 00:00")
    assert data.iloc[0]["actual_grid_load_mwh"] == 100.0
    assert pd.isna(data.iloc[1]["actual_grid_load_mwh"])
    assert bool(data.iloc[0]["actual_grid_load_valid"]) is True
    assert bool(data.iloc[1]["actual_grid_load_valid"]) is False
    assert data.iloc[0]["forecasted_grid_load_mwh"] == 99.0
    assert bool(data.iloc[1]["forecasted_grid_load_valid"]) is False
    assert data.iloc[0]["actual_residual_load_mwh"] == -5.0
    assert bool(data.iloc[0]["actual_residual_load_valid"]) is True
    assert data.iloc[0]["actual_pumped_storage_mwh"] == 2.0
    assert data.iloc[0]["actual_grid_load_including_pumped_storage_mwh"] == 102.0


def test_build_country_keeps_all_rows_when_values_are_invalid(tmp_path):
    write_smard(
        tmp_path / "smard_germany_actual.csv",
        ACTUAL_HEADER,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;;100.00;;0.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;100.00;100.00;0.00;0.00",
        ],
    )
    write_smard(
        tmp_path / "smard_germany_forecasted.csv",
        FORECAST_HEADER,
        [
            "Jan 1, 2020 12:00 AM;Jan 1, 2020 1:00 AM;100.00;0.00",
            "Jan 1, 2020 1:00 AM;Jan 1, 2020 2:00 AM;100.00;0.00",
        ],
    )

    data = build_country_hourly(tmp_path, "Germany")

    assert len(data) == 2
    assert pd.isna(data.iloc[0]["actual_grid_load_mwh"])
    assert pd.isna(data.iloc[0]["actual_pumped_storage_mwh"])
    assert bool(data.iloc[0]["actual_grid_load_valid"]) is False
    assert bool(data.iloc[0]["forecasted_grid_load_valid"]) is True
