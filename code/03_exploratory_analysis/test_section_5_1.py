from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from section_5_1 import (
    FIGURE_DPI,
    EXPECTED_DAILY_ROWS,
    EXPECTED_ROWS,
    MOVING_AVERAGE_MIN_PERIODS,
    MOVING_AVERAGE_WINDOW,
    YEAR_TICK_DATES,
    Y_AXIS_FORMAT,
    build_annual_summary,
    build_daily_load_series,
    build_descriptive_table,
    hourly_descriptive_statistics,
    load_country_data,
    run_section_5_1,
    validate_country_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
COUNTRIES = ["Germany", "Austria"]


def test_figure_uses_requested_axis_and_moving_average_contract():
    assert FIGURE_DPI == 300
    assert MOVING_AVERAGE_WINDOW == 30
    assert MOVING_AVERAGE_MIN_PERIODS == 30
    assert YEAR_TICK_DATES == tuple(
        pd.date_range("2020-01-01", "2025-01-01", freq="YS")
    )
    assert Y_AXIS_FORMAT == "{x:,.0f}"


def _small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": [
                "2019-12-31 23:00:00+00:00",
                "2020-01-01 00:00:00+00:00",
                "2021-01-01 00:00:00+00:00",
                "2021-01-01 01:00:00+00:00",
            ],
            "timestamp_local": [
                "2020-01-01T00:00:00+01:00",
                "2020-01-01T01:00:00+01:00",
                "2021-01-01T01:00:00+01:00",
                "2021-01-01T02:00:00+01:00",
            ],
            "actual_grid_load_mwh": [10.0, 20.0, 30.0, 50.0],
        }
    )


@pytest.mark.parametrize("country", COUNTRIES)
def test_real_inputs_validate_and_remain_byte_identical(country):
    path = PROCESSED / f"modelling_{country.lower()}_hourly.csv"
    before = path.read_bytes()

    data = load_country_data(country)
    validation = validate_country_data(data, country)

    assert validation["row_count"] == EXPECTED_ROWS
    assert validation["local_years"] == [2020, 2021, 2022, 2023, 2024, 2025]
    assert validation["finite_actual_load"] is True
    assert path.read_bytes() == before


def test_validation_rejects_nonfinite_actual_load():
    data = load_country_data("Germany")
    data.loc[0, "actual_grid_load_mwh"] = np.nan

    with pytest.raises(ValueError, match="finite"):
        validate_country_data(data, "Germany")


def test_hourly_descriptive_statistics_use_original_hourly_values():
    statistics = hourly_descriptive_statistics(_small_frame())

    assert statistics["Number of observations"] == 4
    assert statistics["Mean"] == pytest.approx(27.5)
    assert statistics["Standard deviation"] == pytest.approx(np.std([10, 20, 30, 50], ddof=1))
    assert statistics["Minimum"] == 10.0
    assert statistics["First quartile"] == 17.5
    assert statistics["Median"] == 25.0
    assert statistics["Third quartile"] == 35.0
    assert statistics["Maximum"] == 50.0


def test_descriptive_table_has_countries_as_columns_and_rounds_display_values():
    table = build_descriptive_table({"Germany": _small_frame(), "Austria": _small_frame()})

    assert list(table.columns) == ["statistic", "Germany", "Austria"]
    assert table.loc[table["statistic"].eq("Number of observations"), "Germany"].iloc[0] == 4
    assert table.loc[table["statistic"].eq("Mean"), "Germany"].iloc[0] == 27.50
    assert table.loc[table["statistic"].eq("Standard deviation"), "Germany"].iloc[0] == round(
        np.std([10, 20, 30, 50], ddof=1), 2
    )


def test_annual_summary_calculates_previous_year_mean_change():
    summary = build_annual_summary({"Germany": _small_frame()})

    assert summary["year"].tolist() == [2020, 2021]
    assert summary["number_of_hourly_observations"].tolist() == [2, 2]
    assert summary.loc[0, "percentage_change_mean"] != summary.loc[0, "percentage_change_mean"]
    assert summary.loc[1, "percentage_change_mean"] == pytest.approx(166.6666667)


@pytest.mark.parametrize("country", COUNTRIES)
def test_daily_series_covers_full_local_period_and_retains_dst_day_lengths(country):
    data = load_country_data(country)
    daily = build_daily_load_series(data, country)

    assert len(daily) == EXPECTED_DAILY_ROWS
    assert daily["local_date"].iloc[0] == date(2020, 1, 1)
    assert daily["local_date"].iloc[-1] == date(2025, 12, 31)
    assert data["timestamp_local"].map(lambda value: pd.Timestamp(value).date()).nunique() == EXPECTED_DAILY_ROWS

    local_dates = data["timestamp_local"].map(lambda value: pd.Timestamp(value).date())
    day_counts = local_dates.value_counts()
    assert day_counts.loc[date(2020, 3, 29)] == 23
    assert day_counts.loc[date(2020, 10, 25)] == 25
    assert daily["moving_average_30d_mwh"].iloc[:29].isna().all()
    assert daily["moving_average_30d_mwh"].iloc[29:].notna().all()


def test_section_outputs_are_generated_in_an_isolated_directory(tmp_path):
    input_paths = [
        PROCESSED / "modelling_germany_hourly.csv",
        PROCESSED / "modelling_austria_hourly.csv",
    ]
    before = {path: path.read_bytes() for path in input_paths}

    result = run_section_5_1(tmp_path)

    expected_names = {
        "table_5_1_descriptive_statistics.csv",
        "annual_load_summary.csv",
        "figure_5_1_grid_load_development.png",
        "figure_5_1_grid_load_development.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    assert result["countries"] == COUNTRIES
    assert all(path.read_bytes() == content for path, content in before.items())

    with Image.open(tmp_path / "figure_5_1_grid_load_development.png") as image:
        assert image.info["dpi"][0] >= 299
        assert image.width > 1000
        assert image.height > 500
    assert (tmp_path / "figure_5_1_grid_load_development.pdf").stat().st_size > 0
