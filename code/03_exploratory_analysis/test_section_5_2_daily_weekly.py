from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from section_5_2_daily_weekly import (
    DAY_CODES,
    DAY_NAMES,
    EXPECTED_DAILY_ROWS,
    EXPECTED_ROWS,
    FIGURE_DPI,
    _render_daily_figure,
    build_comparison_metrics,
    build_daily_observations,
    build_day_of_week_summary,
    build_weekly_hourly_profile,
    load_country_data,
    run_section_5_2_daily_weekly,
    validate_country_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
COUNTRIES = ["Germany", "Austria"]
PROTECTED_OUTPUTS = [
    PROJECT_ROOT / "outputs" / "chapter5" / "section_5_1" / name
    for name in (
        "table_5_1_descriptive_statistics.csv",
        "annual_load_summary.csv",
        "figure_5_1_grid_load_development.png",
        "figure_5_1_grid_load_development.pdf",
    )
] + [
    PROJECT_ROOT / "outputs" / "chapter5" / "section_5_2" / name
    for name in (
        "hourly_demand_profile.csv",
        "day_of_week_summary.csv",
        "weekly_hourly_profile.csv",
        "figure_5_2_hourly_demand_patterns.png",
        "figure_5_2_hourly_demand_patterns.pdf",
    )
]


@pytest.mark.parametrize("country", COUNTRIES)
def test_real_inputs_validate_local_day_codes_and_dst_counts(country):
    data = load_country_data(country)
    validation = validate_country_data(data, country)

    assert validation["row_count"] == EXPECTED_ROWS
    assert validation["day_code_mapping"] == DAY_CODES
    assert validation["local_date_count"] == EXPECTED_DAILY_ROWS
    assert set(validation["daily_observation_counts"]) <= {23, 24, 25}
    assert validation["day_names"] == list(DAY_NAMES)
    assert validation["all_hours_present"] is True


def test_daily_observations_use_local_dates_and_record_hour_counts():
    data = load_country_data("Germany")
    daily = build_daily_observations(data, "Germany")

    assert len(daily) == EXPECTED_DAILY_ROWS
    assert daily["local_date"].iloc[0] == pd.Timestamp("2020-01-01").date()
    assert daily["local_date"].iloc[-1] == pd.Timestamp("2025-12-31").date()
    assert set(daily["number_of_hourly_observations"]) <= {23, 24, 25}
    assert daily.loc[
        daily["local_date"].eq(pd.Timestamp("2020-03-29").date()),
        "number_of_hourly_observations",
    ].iloc[0] == 23
    assert daily.loc[
        daily["local_date"].eq(pd.Timestamp("2020-10-25").date()),
        "number_of_hourly_observations",
    ].iloc[0] == 25


def test_day_of_week_summary_has_14_rows_in_monday_to_sunday_order():
    data_by_country = {country: load_country_data(country) for country in COUNTRIES}
    daily_by_country = {
        country: build_daily_observations(data, country)
        for country, data in data_by_country.items()
    }
    summary = build_day_of_week_summary(daily_by_country)

    assert len(summary) == 14
    assert summary["day_name"].tolist() == list(DAY_NAMES) * 2
    assert summary["country"].tolist() == ["Germany"] * 7 + ["Austria"] * 7
    assert not summary.isna().any().any()
    assert np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()
    assert (
        (summary["first_quartile_mwh"] <= summary["median_daily_mean_grid_load_mwh"])
        & (summary["median_daily_mean_grid_load_mwh"] <= summary["third_quartile_mwh"])
    ).all()


def test_weekly_hourly_profile_has_all_country_day_hour_combinations():
    data_by_country = {country: load_country_data(country) for country in COUNTRIES}
    profile = build_weekly_hourly_profile(data_by_country)

    assert len(profile) == 336
    assert profile.groupby("country").size().to_dict() == {"Germany": 168, "Austria": 168}
    assert profile.groupby("country")["number_of_observations"].sum().to_dict() == {
        "Germany": EXPECTED_ROWS,
        "Austria": EXPECTED_ROWS,
    }
    assert profile.groupby("country")["day_name"].nunique().to_dict() == {
        "Germany": 7,
        "Austria": 7,
    }
    assert profile.groupby(["country", "day_of_week"]).size().eq(24).all()
    assert profile["hour"].between(0, 23).all()
    assert not profile.isna().any().any()
    assert np.isfinite(profile.select_dtypes(include=np.number).to_numpy()).all()
    assert (
        (profile["first_quartile_mwh"] <= profile["median_hourly_grid_load_mwh"])
        & (profile["median_hourly_grid_load_mwh"] <= profile["third_quartile_mwh"])
    ).all()


def test_comparison_metrics_match_direct_profile_extrema():
    data_by_country = {country: load_country_data(country) for country in COUNTRIES}
    daily_by_country = {
        country: build_daily_observations(data, country)
        for country, data in data_by_country.items()
    }
    summary = build_day_of_week_summary(daily_by_country)
    profile = build_weekly_hourly_profile(data_by_country)
    metrics = build_comparison_metrics(summary, profile)

    for country in COUNTRIES:
        country_summary = summary[summary["country"].eq(country)]
        country_profile = profile[profile["country"].eq(country)]
        assert metrics[country]["highest_mean_day"] == country_summary.loc[
            country_summary["mean_daily_mean_grid_load_mwh"].idxmax(), "day_name"
        ]
        assert metrics[country]["lowest_mean_day"] == country_summary.loc[
            country_summary["mean_daily_mean_grid_load_mwh"].idxmin(), "day_name"
        ]
        assert metrics[country]["weekday_peak_hour"] in range(24)
        assert metrics[country]["saturday_peak_hour"] in range(24)
        assert metrics[country]["sunday_peak_hour"] in range(24)
        assert metrics[country]["weekday_minimum_hour"] in range(24)
        assert metrics[country]["saturday_minimum_hour"] in range(24)
        assert metrics[country]["sunday_minimum_hour"] in range(24)
        assert country_profile["number_of_observations"].sum() == EXPECTED_ROWS


def test_daily_figure_has_shared_legend_and_requested_axis_labels(tmp_path, monkeypatch):
    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "chapter5"
        / "section_5_2"
        / "day_of_week_summary.csv"
    )
    summary = pd.read_csv(summary_path)
    captured = []
    monkeypatch.setattr(plt, "close", captured.append)

    _render_daily_figure(
        summary,
        tmp_path / "figure_5_3_daily_demand_patterns.png",
        tmp_path / "figure_5_3_daily_demand_patterns.pdf",
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


def test_daily_weekly_outputs_are_generated_in_isolated_directory(tmp_path):
    input_paths = [
        PROCESSED / "modelling_germany_hourly.csv",
        PROCESSED / "modelling_austria_hourly.csv",
    ]
    before_inputs = {path: path.read_bytes() for path in input_paths}
    before_outputs = {path: path.read_bytes() for path in PROTECTED_OUTPUTS}

    result = run_section_5_2_daily_weekly(tmp_path)

    expected_names = {
        "day_of_week_summary.csv",
        "weekly_hourly_profile.csv",
        "figure_5_3_daily_demand_patterns.png",
        "figure_5_3_daily_demand_patterns.pdf",
        "figure_5_4_weekly_demand_patterns.png",
        "figure_5_4_weekly_demand_patterns.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    assert len(result["day_of_week_summary"]) == 14
    assert len(result["weekly_hourly_profile"]) == 336
    assert all(path.read_bytes() == content for path, content in before_inputs.items())
    assert all(path.read_bytes() == content for path, content in before_outputs.items())

    with Image.open(tmp_path / "figure_5_3_daily_demand_patterns.png") as image:
        assert image.info["dpi"][0] >= 299
        assert image.width > 700
        assert image.height > 500
    assert (tmp_path / "figure_5_3_daily_demand_patterns.pdf").stat().st_size > 0

    with Image.open(tmp_path / "figure_5_4_weekly_demand_patterns.png") as image:
        assert image.info["dpi"][0] >= 299
        assert image.width > 700
        assert image.height > 500
    assert (tmp_path / "figure_5_4_weekly_demand_patterns.pdf").stat().st_size > 0
