from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from section_5_4_temperature_public_holidays import (
    COUNTRIES,
    DAILY_COLUMNS,
    EXPECTED_DAILY_ROWS,
    EXPECTED_ROWS,
    HOLIDAY_SUMMARY_COLUMNS,
    TEMPERATURE_BIN_COLUMNS,
    YEARS,
    _render_public_holiday_effect,
    _render_temperature_relationship,
    align_hourly_data,
    build_daily_temperature_public_holidays,
    build_public_holiday_effect_summary,
    build_temperature_bin_summary,
    holiday_calendar,
    run_section_5_4_temperature_public_holidays,
    validate_aligned_hourly_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
SECTION_5_2 = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_2"
SECTION_5_3 = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_3"
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
] + [
    SECTION_5_3 / name
    for name in (
        "monthly_demand_profile.csv",
        "year_month_summary.csv",
        "figure_5_5_monthly_demand_patterns.png",
        "figure_5_5_monthly_demand_patterns.pdf",
        "figure_5_6_year_month_heatmap.png",
        "figure_5_6_year_month_heatmap.pdf",
    )
]
PROCESSED_INPUTS = [
    PROCESSED / "modelling_germany_hourly.csv",
    PROCESSED / "modelling_austria_hourly.csv",
    PROCESSED / "temperature_germany_hourly.csv",
    PROCESSED / "temperature_austria_hourly.csv",
]


@pytest.mark.parametrize("country", COUNTRIES)
def test_real_hourly_inputs_align_on_utc_and_cover_local_period(country):
    aligned = align_hourly_data(country)
    validation = validate_aligned_hourly_data(aligned, country)

    assert validation["row_count"] == EXPECTED_ROWS
    assert validation["utc_unique"] is True
    assert validation["utc_hourly_continuous"] is True
    assert validation["finite_load"] is True
    assert validation["finite_temperature"] is True
    assert validation["local_date_count"] == EXPECTED_DAILY_ROWS
    assert validation["daily_observation_counts"] == [23, 24, 25]


@pytest.mark.parametrize("country", COUNTRIES)
def test_daily_data_has_dst_aware_counts_and_exact_schema(country):
    daily = build_daily_temperature_public_holidays(align_hourly_data(country), country)

    assert len(daily) == EXPECTED_DAILY_ROWS
    assert list(daily[DAILY_COLUMNS].columns) == DAILY_COLUMNS
    assert daily["local_date"].iloc[0] == date(2020, 1, 1)
    assert daily["local_date"].iloc[-1] == date(2025, 12, 31)
    assert set(daily["hourly_count"]) <= {23, 24, 25}
    assert daily.loc[
        daily["local_date"].eq(date(2020, 3, 29)), "hourly_count"
    ].iloc[0] == 23
    assert daily.loc[
        daily["local_date"].eq(date(2020, 10, 25)), "hourly_count"
    ].iloc[0] == 25
    assert np.isfinite(
        daily[["daily_mean_load_mwh", "daily_mean_temperature_c"]].to_numpy()
    ).all()


def test_holiday_calendar_uses_nationwide_fixed_and_easter_rules():
    germany_2023 = holiday_calendar("Germany", [2023])
    austria_2023 = holiday_calendar("Austria", [2023])

    assert len(germany_2023) == 9
    assert len(austria_2023) == 13
    assert date(2023, 10, 3) in germany_2023
    assert date(2023, 11, 1) not in germany_2023
    assert date(2023, 10, 26) in austria_2023
    assert date(2023, 12, 8) in austria_2023


def _real_daily_frames() -> dict[str, pd.DataFrame]:
    return {
        country: build_daily_temperature_public_holidays(
            align_hourly_data(country), country
        )
        for country in COUNTRIES
    }


def test_daily_day_types_are_mutually_exclusive_and_baselines_exclude_holidays():
    daily_by_country = _real_daily_frames()
    allowed_types = {
        "ordinary weekday",
        "weekend",
        "weekday public holiday",
        "weekend public holiday",
    }
    for country, daily in daily_by_country.items():
        assert set(daily["day_type"]) <= allowed_types
        assert daily["day_type"].notna().all()
        assert daily.loc[
            daily["day_type"].eq("weekday public holiday"), "public_holiday_flag"
        ].eq(1).all()
        assert daily.loc[
            daily["day_type"].eq("weekend public holiday"), "weekday_number"
        ].ge(5).all()
        assert daily.loc[
            daily["day_type"].isin(["weekend", "weekend public holiday"]),
            "matching_baseline_mwh",
        ].isna().all()

        ordinary = daily[daily["day_type"].eq("ordinary weekday")]
        grouped = ordinary.groupby(
            ["year", "month", "weekday_number"], sort=True
        )["relative_load_deviation_pct"]
        assert grouped.mean().abs().max() < 1e-10
        assert not ordinary["public_holiday_flag"].astype(bool).any()


def test_temperature_bins_are_common_valid_and_quartiles_ordered():
    daily_by_country = _real_daily_frames()
    daily = pd.concat(daily_by_country.values(), ignore_index=True)
    bins = build_temperature_bin_summary(daily)

    assert list(bins.columns) == TEMPERATURE_BIN_COLUMNS
    assert (bins["number_of_days"] >= 10).all()
    assert bins.groupby("country").size().gt(0).all()
    assert bins["temperature_bin_lower_c"].min() == pytest.approx(
        bins["temperature_bin_lower_c"].min()
    )
    assert set(bins["temperature_bin_lower_c"]) <= set(
        bins["temperature_bin_lower_c"]
    )
    assert np.isfinite(bins.select_dtypes(include=np.number).to_numpy()).all()
    assert (
        bins["first_quartile_mwh"] <= bins["median_daily_mean_load_mwh"]
    ).all()
    assert (
        bins["median_daily_mean_load_mwh"] <= bins["third_quartile_mwh"]
    ).all()
    for _, group in bins.groupby("country"):
        assert group["temperature_bin_lower_c"].tolist() == sorted(
            group["temperature_bin_lower_c"].tolist()
        )
    assert (bins["temperature_bin_lower_c"] % 2).eq(0).all()
    assert (
        bins["temperature_bin_upper_c"] - bins["temperature_bin_lower_c"]
    ).eq(2).all()


def test_public_holiday_summary_matches_daily_deviations():
    daily = pd.concat(_real_daily_frames().values(), ignore_index=True)
    summary = build_public_holiday_effect_summary(daily)

    assert list(summary.columns) == HOLIDAY_SUMMARY_COLUMNS
    assert len(summary) > 0
    assert (summary["weekday_public_holiday_count"] > 0).all()
    assert np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()
    for _, row in summary.iterrows():
        holiday_rows = daily.loc[
            daily["country"].eq(row["country"])
            & daily["year"].eq(row["year"])
            & daily["month"].eq(row["month"])
            & daily["weekday_number"].eq(row["weekday"])
            & daily["day_type"].eq("weekday public holiday")
        ]
        assert len(holiday_rows) == row["weekday_public_holiday_count"]
        assert holiday_rows["relative_load_deviation_pct"].mean() == pytest.approx(
            row["weekday_public_holiday_mean_deviation_pct"]
        )


def _synthetic_daily() -> pd.DataFrame:
    rows = []
    for country_index, country in enumerate(COUNTRIES):
        for day_index in range(20):
            rows.append(
                {
                    "country": country,
                    "local_date": date(2020, 1, day_index + 1),
                    "daily_mean_load_mwh": 1000.0
                    + country_index * 100
                    + day_index,
                    "daily_mean_temperature_c": -10.0 + day_index,
                    "hourly_count": 24,
                    "weekday": "Monday",
                    "weekday_number": 0,
                    "year": 2020,
                    "month": 1,
                    "public_holiday_flag": 0,
                    "holiday_name": "",
                    "day_type": "ordinary weekday",
                    "matching_baseline_mwh": 1000.0 + country_index * 100,
                    "relative_load_deviation_pct": float(day_index),
                }
            )
    return pd.DataFrame(rows)


def test_figures_have_requested_layout_and_vector_page_outputs(tmp_path, monkeypatch):
    daily = _synthetic_daily()
    temperature_bins = pd.DataFrame(
        {
            "country": COUNTRIES * 2,
            "temperature_bin_lower_c": [-10.0, 0.0, -10.0, 0.0],
            "temperature_bin_upper_c": [0.0, 10.0, 0.0, 10.0],
            "temperature_bin_midpoint_c": [-5.0, 5.0, -5.0, 5.0],
            "number_of_days": [10, 10, 10, 10],
            "mean_daily_mean_load_mwh": [1000.0, 1100.0, 1100.0, 1200.0],
            "median_daily_mean_load_mwh": [1000.0, 1100.0, 1100.0, 1200.0],
            "standard_deviation_mwh": [1.0] * 4,
            "first_quartile_mwh": [999.0, 1099.0, 1099.0, 1199.0],
            "third_quartile_mwh": [1001.0, 1101.0, 1101.0, 1201.0],
        }
    )
    captured = []
    monkeypatch.setattr(plt, "close", captured.append)
    _render_temperature_relationship(
        daily,
        temperature_bins,
        tmp_path / "figure_5_7_temperature_load_relationship.png",
        tmp_path / "figure_5_7_temperature_load_relationship.pdf",
    )
    temperature_figure = captured.pop()
    assert len(temperature_figure.axes) == 2
    assert len(temperature_figure.legends) == 1
    assert temperature_figure._suptitle is None
    assert temperature_figure.axes[0].get_ylabel() == "Daily mean grid load [MWh]"
    assert temperature_figure.axes[-1].get_xlabel() == (
        "Daily mean population-weighted temperature [°C]"
    )

    _render_public_holiday_effect(
        daily,
        tmp_path / "figure_5_8_public_holiday_effect.png",
        tmp_path / "figure_5_8_public_holiday_effect.pdf",
    )
    holiday_figure = captured.pop()
    assert len(holiday_figure.axes) == 2
    assert holiday_figure._suptitle is None
    assert holiday_figure.axes[0].get_ylim() == holiday_figure.axes[1].get_ylim()
    assert any(np.allclose(line.get_ydata(), 0) for line in holiday_figure.axes[0].lines)
    monkeypatch.undo()
    plt.close(temperature_figure)
    plt.close(holiday_figure)


def test_public_holiday_figure_uses_fixed_limits_sample_labels_and_mean_markers(
    tmp_path, monkeypatch
):
    daily = pd.concat(_real_daily_frames().values(), ignore_index=True)
    captured = []
    monkeypatch.setattr(plt, "close", captured.append)

    _render_public_holiday_effect(
        daily,
        tmp_path / "figure_5_8_public_holiday_effect.png",
        tmp_path / "figure_5_8_public_holiday_effect.pdf",
    )

    figure = captured[0]
    expected_labels = [
        [
            "Ordinary weekdays (n = 1,522)",
            "Weekday public holidays (n = 44)",
        ],
        [
            "Ordinary weekdays (n = 1,504)",
            "Weekday public holidays (n = 62)",
        ],
    ]
    for axis, labels in zip(figure.axes, expected_labels):
        assert axis.get_ylim() == (-35.0, 20.0)
        assert [tick.get_text() for tick in axis.get_xticklabels()] == labels
        mean_markers = [line for line in axis.lines if line.get_marker() == "D"]
        assert len(mean_markers) == 1
        assert mean_markers[0].get_xdata().tolist() == [1, 2]
        assert len(mean_markers[0].get_ydata()) == 2
    monkeypatch.undo()
    plt.close(figure)


def test_section_outputs_are_generated_in_isolated_directory(tmp_path):
    before_inputs = {path: path.read_bytes() for path in PROCESSED_INPUTS}
    before_outputs = {path: path.read_bytes() for path in PROTECTED_OUTPUTS}
    raw_metadata = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in RAW.rglob("*")
        if path.is_file()
    }

    result = run_section_5_4_temperature_public_holidays(tmp_path)

    expected_names = {
        "daily_temperature_public_holidays.csv",
        "temperature_bin_summary.csv",
        "public_holiday_effect_summary.csv",
        "figure_5_7_temperature_load_relationship.png",
        "figure_5_7_temperature_load_relationship.pdf",
        "figure_5_8_public_holiday_effect.png",
        "figure_5_8_public_holiday_effect.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    assert len(result["daily"]) == 4_384
    assert len(result["temperature_bin_summary"]) > 0
    assert len(result["public_holiday_effect_summary"]) > 0
    assert all(path.read_bytes() == content for path, content in before_inputs.items())
    assert all(path.read_bytes() == content for path, content in before_outputs.items())
    assert all(
        path.exists()
        and (path.stat().st_size, path.stat().st_mtime_ns) == metadata
        for path, metadata in raw_metadata.items()
    )

    daily_file = pd.read_csv(tmp_path / "daily_temperature_public_holidays.csv")
    bins_file = pd.read_csv(tmp_path / "temperature_bin_summary.csv")
    holiday_file = pd.read_csv(tmp_path / "public_holiday_effect_summary.csv")
    assert list(daily_file.columns) == DAILY_COLUMNS
    assert list(bins_file.columns) == TEMPERATURE_BIN_COLUMNS
    assert list(holiday_file.columns) == HOLIDAY_SUMMARY_COLUMNS
    assert len(daily_file) == 4_384
    assert (daily_file["hourly_count"].isin([23, 24, 25])).all()

    for name in (
        "figure_5_7_temperature_load_relationship.png",
        "figure_5_8_public_holiday_effect.png",
    ):
        with Image.open(tmp_path / name) as image:
            assert image.info["dpi"][0] >= 299
            assert image.width > 700
            assert image.height > 500
    for name in (
        "figure_5_7_temperature_load_relationship.pdf",
        "figure_5_8_public_holiday_effect.pdf",
    ):
        pdf = (tmp_path / name).read_bytes()
        assert pdf.startswith(b"%PDF")
        assert pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages") == 1
