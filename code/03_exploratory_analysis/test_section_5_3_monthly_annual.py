from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from section_5_3_monthly_annual import (
    COUNTRIES,
    EXPECTED_DAILY_ROWS,
    EXPECTED_ROWS,
    MONTHS,
    MONTH_NAMES,
    MONTHLY_COLUMNS,
    YEAR_MONTH_COLUMNS,
    YEARS,
    _render_monthly_figure,
    _render_year_month_heatmap,
    build_daily_observations,
    build_monthly_demand_profile,
    build_year_month_summary,
    load_country_data,
    run_section_5_3_monthly_annual,
    validate_country_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
SECTION_5_2 = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_2"
PROTECTED_OUTPUTS = [
    PROJECT_ROOT / "outputs" / "chapter5" / "section_5_1" / name
    for name in (
        "table_5_1_descriptive_statistics.csv",
        "annual_load_summary.csv",
        "figure_5_1_grid_load_development.png",
        "figure_5_1_grid_load_development.pdf",
    )
] + [
    SECTION_5_2 / name
    for name in (
        "hourly_demand_profile.csv",
        "day_of_week_summary.csv",
        "weekly_hourly_profile.csv",
        "figure_5_2_hourly_demand_patterns.png",
        "figure_5_2_hourly_demand_patterns.pdf",
        "figure_5_3_daily_demand_patterns.png",
        "figure_5_3_daily_demand_patterns.pdf",
        "figure_5_4_weekly_demand_patterns.png",
        "figure_5_4_weekly_demand_patterns.pdf",
    )
]


@pytest.mark.parametrize("country", COUNTRIES)
def test_real_inputs_validate_local_dates_and_dst_counts(country):
    data = load_country_data(country)
    validation = validate_country_data(data, country)

    assert validation["row_count"] == EXPECTED_ROWS
    assert validation["local_date_count"] == EXPECTED_DAILY_ROWS
    assert validation["local_years"] == list(YEARS)
    assert set(validation["daily_observation_counts"]) <= {23, 24, 25}
    assert validation["finite_actual_load"] is True


@pytest.mark.parametrize("country", COUNTRIES)
def test_daily_observations_use_local_dates_and_all_available_hours(country):
    daily = build_daily_observations(load_country_data(country), country)

    assert len(daily) == EXPECTED_DAILY_ROWS
    assert daily["local_date"].iloc[0] == date(2020, 1, 1)
    assert daily["local_date"].iloc[-1] == date(2025, 12, 31)
    assert set(daily["number_of_hourly_observations"]) <= {23, 24, 25}
    assert daily.loc[
        daily["local_date"].eq(date(2020, 3, 29)),
        "number_of_hourly_observations",
    ].iloc[0] == 23
    assert daily.loc[
        daily["local_date"].eq(date(2020, 10, 25)),
        "number_of_hourly_observations",
    ].iloc[0] == 25


def _build_real_daily_frames() -> dict[str, pd.DataFrame]:
    return {
        country: build_daily_observations(load_country_data(country), country)
        for country in COUNTRIES
    }


def _assert_summary_row_matches_values(row, values):
    assert row["number_of_days"] == values.count()
    assert row["mean_daily_mean_grid_load_mwh"] == pytest.approx(values.mean())
    assert row["median_daily_mean_grid_load_mwh"] == pytest.approx(values.median())
    assert row["standard_deviation_mwh"] == pytest.approx(values.std(ddof=1))
    assert row["first_quartile_mwh"] == pytest.approx(values.quantile(0.25))
    assert row["third_quartile_mwh"] == pytest.approx(values.quantile(0.75))


def test_monthly_profile_has_24_rows_and_reconciles_to_daily_means():
    daily_by_country = _build_real_daily_frames()
    monthly = build_monthly_demand_profile(daily_by_country)

    assert list(monthly.columns) == MONTHLY_COLUMNS
    assert len(monthly) == 24
    assert monthly["country"].tolist() == ["Germany"] * 12 + ["Austria"] * 12
    assert monthly["month"].tolist() == list(MONTHS) * 2
    assert monthly["month_name"].tolist() == list(MONTH_NAMES) * 2

    for country in COUNTRIES:
        daily = daily_by_country[country]
        for month in MONTHS:
            values = daily.loc[
                daily["month"].eq(month), "daily_mean_grid_load_mwh"
            ]
            row = monthly.loc[
                monthly["country"].eq(country) & monthly["month"].eq(month)
            ].iloc[0]
            _assert_summary_row_matches_values(row, values)


def test_year_month_summary_has_144_rows_and_reconciles_to_daily_means():
    daily_by_country = _build_real_daily_frames()
    year_month = build_year_month_summary(daily_by_country)

    assert list(year_month.columns) == YEAR_MONTH_COLUMNS
    assert len(year_month) == 144
    assert year_month.groupby("country").size().to_dict() == {
        "Germany": 72,
        "Austria": 72,
    }
    assert sorted(year_month["year"].unique()) == list(YEARS)
    assert sorted(year_month["month"].unique()) == list(MONTHS)
    assert year_month["month_name"].tolist()[:12] == list(MONTH_NAMES)

    for country in COUNTRIES:
        daily = daily_by_country[country]
        for year in YEARS:
            for month in MONTHS:
                values = daily.loc[
                    daily["year"].eq(year) & daily["month"].eq(month),
                    "daily_mean_grid_load_mwh",
                ]
                row = year_month.loc[
                    year_month["country"].eq(country)
                    & year_month["year"].eq(year)
                    & year_month["month"].eq(month)
                ].iloc[0]
                _assert_summary_row_matches_values(row, values)


def test_summary_values_are_finite_ordered_and_monthly_means_reconcile():
    daily_by_country = _build_real_daily_frames()
    monthly = build_monthly_demand_profile(daily_by_country)
    year_month = build_year_month_summary(daily_by_country)

    for summary in (monthly, year_month):
        numeric = summary.select_dtypes(include=np.number)
        assert not summary.isna().any().any()
        assert np.isfinite(numeric.to_numpy()).all()
        assert (
            summary["first_quartile_mwh"]
            <= summary["median_daily_mean_grid_load_mwh"]
        ).all()
        assert (
            summary["median_daily_mean_grid_load_mwh"]
            <= summary["third_quartile_mwh"]
        ).all()

    for country in COUNTRIES:
        for month in MONTHS:
            monthly_row = monthly.loc[
                monthly["country"].eq(country) & monthly["month"].eq(month)
            ].iloc[0]
            year_month_rows = year_month.loc[
                year_month["country"].eq(country) & year_month["month"].eq(month)
            ]
            weighted_mean = np.average(
                year_month_rows["mean_daily_mean_grid_load_mwh"],
                weights=year_month_rows["number_of_days"],
            )
            assert monthly_row["mean_daily_mean_grid_load_mwh"] == pytest.approx(
                weighted_mean
            )


def test_monthly_figure_has_shared_legend_and_requested_axis_labels(tmp_path, monkeypatch):
    rows = []
    for country in COUNTRIES:
        for month in MONTHS:
            rows.append(
                {
                    "country": country,
                    "month": month,
                    "month_name": MONTH_NAMES[month - 1],
                    "number_of_days": 10,
                    "mean_daily_mean_grid_load_mwh": 100.0 + month,
                    "median_daily_mean_grid_load_mwh": 100.0 + month,
                    "standard_deviation_mwh": 1.0,
                    "first_quartile_mwh": 99.0 + month,
                    "third_quartile_mwh": 101.0 + month,
                }
            )
    captured = []
    monkeypatch.setattr(plt, "close", captured.append)

    _render_monthly_figure(
        pd.DataFrame(rows, columns=MONTHLY_COLUMNS),
        tmp_path / "figure_5_5_monthly_demand_patterns.png",
        tmp_path / "figure_5_5_monthly_demand_patterns.pdf",
    )

    figure = captured[0]
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "Daily mean grid load",
        "Interquartile range",
    ]
    assert legend._loc == 9
    anchor = legend.get_bbox_to_anchor().transformed(figure.transFigure.inverted())
    assert anchor.x0 == pytest.approx(0.5)
    assert [axis.get_ylabel() for axis in figure.axes[:2]] == [
        "Daily mean grid load [MWh]",
        "Daily mean grid load [MWh]",
    ]
    monkeypatch.undo()
    plt.close(figure)


def test_year_month_heatmap_uses_abbreviated_centered_month_labels(tmp_path, monkeypatch):
    rows = []
    for country_index, country in enumerate(COUNTRIES):
        for year_index, year in enumerate(YEARS):
            for month in MONTHS:
                rows.append(
                    {
                        "country": country,
                        "year": year,
                        "month": month,
                        "month_name": MONTH_NAMES[month - 1],
                        "number_of_days": 28,
                        "mean_daily_mean_grid_load_mwh": 100.0
                        + country_index * 10
                        + year_index
                        + month,
                        "median_daily_mean_grid_load_mwh": 100.0
                        + country_index * 10
                        + year_index
                        + month,
                        "standard_deviation_mwh": 1.0,
                        "first_quartile_mwh": 99.0
                        + country_index * 10
                        + year_index
                        + month,
                        "third_quartile_mwh": 101.0
                        + country_index * 10
                        + year_index
                        + month,
                    }
                )
    captured = []
    monkeypatch.setattr(plt, "close", captured.append)

    _render_year_month_heatmap(
        pd.DataFrame(rows, columns=YEAR_MONTH_COLUMNS),
        tmp_path / "figure_5_6_year_month_heatmap.png",
        tmp_path / "figure_5_6_year_month_heatmap.pdf",
    )

    figure = captured[0]
    expected_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for axis in figure.axes[:2]:
        tick_labels = axis.get_xticklabels()
        assert [label.get_text() for label in tick_labels] == expected_labels
        assert [label.get_rotation() for label in tick_labels] == [0] * 12
        assert [label.get_ha() for label in tick_labels] == ["center"] * 12
        assert [label.get_position()[0] for label in tick_labels] == list(range(12))
    assert [axis.get_ylabel() for axis in figure.axes[2:]] == [
        "Daily mean grid load [MWh]",
        "Daily mean grid load [MWh]",
    ]
    monkeypatch.undo()
    plt.close(figure)


def test_section_outputs_are_generated_in_isolated_directory(tmp_path):
    input_paths = [
        PROCESSED / "modelling_germany_hourly.csv",
        PROCESSED / "modelling_austria_hourly.csv",
    ]
    before_inputs = {path: path.read_bytes() for path in input_paths}
    before_outputs = {path: path.read_bytes() for path in PROTECTED_OUTPUTS}

    result = run_section_5_3_monthly_annual(tmp_path)

    expected_names = {
        "monthly_demand_profile.csv",
        "year_month_summary.csv",
        "figure_5_5_monthly_demand_patterns.png",
        "figure_5_5_monthly_demand_patterns.pdf",
        "figure_5_6_year_month_heatmap.png",
        "figure_5_6_year_month_heatmap.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    assert len(result["monthly_demand_profile"]) == 24
    assert len(result["year_month_summary"]) == 144
    assert all(path.read_bytes() == content for path, content in before_inputs.items())
    assert all(path.read_bytes() == content for path, content in before_outputs.items())

    monthly_file = pd.read_csv(tmp_path / "monthly_demand_profile.csv")
    year_month_file = pd.read_csv(tmp_path / "year_month_summary.csv")
    assert list(monthly_file.columns) == MONTHLY_COLUMNS
    assert list(year_month_file.columns) == YEAR_MONTH_COLUMNS
    assert len(monthly_file) == 24
    assert len(year_month_file) == 144

    for name in (
        "figure_5_5_monthly_demand_patterns.png",
        "figure_5_6_year_month_heatmap.png",
    ):
        with Image.open(tmp_path / name) as image:
            assert image.info["dpi"][0] >= 299
            assert image.width > 700
            assert image.height > 500
    assert (tmp_path / "figure_5_5_monthly_demand_patterns.pdf").stat().st_size > 0
    assert (tmp_path / "figure_5_6_year_month_heatmap.pdf").stat().st_size > 0
