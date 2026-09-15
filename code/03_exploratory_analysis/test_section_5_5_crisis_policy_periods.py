from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from section_5_5_crisis_policy_periods import (
    ANALYSIS_METADATA,
    COUNTRIES,
    CONTEXT_COLOR,
    COVID_DAILY_COLUMNS,
    COVID_SUMMARY_COLUMNS,
    EXPECTED_DAILY_ROWS,
    EXPECTED_ROWS,
    MONTHLY_COLUMNS,
    POST_INVASION_MONTHLY_COLUMNS,
    POST_INVASION_SUMMARY_COLUMNS,
    _render_covid_event_strip,
    _render_post_invasion_fingerprint,
    build_covid_daily_deviations,
    build_covid_period_summary,
    build_daily_actual_forecast,
    build_monthly_actual_load,
    build_post_invasion_monthly_deviations,
    build_post_invasion_period_summary,
    load_country_data,
    run_section_5_5_crisis_policy_periods,
    validate_country_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"


@pytest.mark.parametrize("country", COUNTRIES)
def test_real_inputs_validate_local_dates_dst_and_forecast_denominators(country):
    data = load_country_data(country)
    validation = validate_country_data(data, country)

    assert validation["row_count"] == EXPECTED_ROWS
    assert validation["local_date_count"] == EXPECTED_DAILY_ROWS
    assert validation["daily_observation_counts"] == [23, 24, 25]
    assert validation["utc_hourly_continuous"] is True
    expected_valid_forecasts = 52_584 if country == "Germany" else 52_608
    assert validation["forecast_valid_observation_count"] == expected_valid_forecasts
    assert validation["forecast_invalid_observation_count"] == (
        24 if country == "Germany" else 0
    )
    assert validation["finite_actual_load"] is True
    assert validation["finite_forecast_load"] is True
    assert validation["positive_forecast_denominator"] is True


@pytest.mark.parametrize("country", COUNTRIES)
def test_covid_daily_deviations_use_local_dates_and_trailing_window(country):
    daily = build_daily_actual_forecast(load_country_data(country), country)
    covid = build_covid_daily_deviations(daily, country)

    assert list(covid.columns) == COVID_DAILY_COLUMNS
    assert len(covid) == (181 if country == "Germany" else 182)
    assert covid["local_date"].iloc[0] == date(2020, 1, 1)
    assert covid["local_date"].iloc[-1] == date(2020, 6, 30)
    assert covid["local_date"].nunique() == len(covid)
    assert (date(2020, 1, 31) not in set(covid["local_date"])) == (
        country == "Germany"
    )
    assert (covid["daily_mean_forecast_load_mwh"] > 0).all()
    assert np.isfinite(covid.select_dtypes(include=np.number).to_numpy()).all()

    first_seven = covid.iloc[:7]["relative_deviation_pct"]
    assert covid.iloc[0]["trailing_7_day_mean_deviation_pct"] == pytest.approx(
        first_seven.iloc[:1].mean()
    )
    assert covid.iloc[6]["trailing_7_day_mean_deviation_pct"] == pytest.approx(
        first_seven.mean()
    )
    assert covid.iloc[7]["trailing_7_day_mean_deviation_pct"] == pytest.approx(
        covid.iloc[1:8]["relative_deviation_pct"].mean()
    )


def test_covid_summaries_reconcile_to_two_event_periods():
    daily_by_country = {
        country: build_covid_daily_deviations(
            build_daily_actual_forecast(load_country_data(country), country), country
        )
        for country in COUNTRIES
    }
    covid = pd.concat(daily_by_country.values(), ignore_index=True)
    summary = build_covid_period_summary(covid)

    assert list(summary.columns) == COVID_SUMMARY_COLUMNS
    assert len(summary) == 4
    assert summary.groupby("country").size().to_dict() == {
        "Germany": 2,
        "Austria": 2,
    }
    assert summary.groupby(["country", "period"])["n"].sum().to_dict() == {
        ("Germany", "1 January-10 March 2020"): 69,
        ("Germany", "11 March-30 June 2020"): 112,
        ("Austria", "1 January-10 March 2020"): 70,
        ("Austria", "11 March-30 June 2020"): 112,
    }
    assert summary["n"].tolist() == [69, 112, 70, 112]
    assert np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()

    for _, row in summary.iterrows():
        values = covid.loc[
            covid["country"].eq(row["country"])
            & covid["period"].eq(row["period"]),
            "relative_deviation_pct",
        ]
        assert row["n"] == len(values)
        assert row["mean_deviation_pct"] == pytest.approx(values.mean())
        assert row["median_deviation_pct"] == pytest.approx(values.median())
        assert row["mean_absolute_deviation_pct"] == pytest.approx(
            values.abs().mean()
        )


def test_german_gap_is_not_imputed_and_rolling_window_is_calendar_based():
    daily = build_daily_actual_forecast(load_country_data("Germany"), "Germany")
    covid = build_covid_daily_deviations(daily, "Germany")

    gap = daily.loc[daily["local_date"].eq(date(2020, 1, 31))].iloc[0]
    assert pd.isna(gap["daily_mean_forecast_load_mwh"])
    assert gap["number_of_forecast_observations"] == 0
    assert date(2020, 1, 31) not in set(covid["local_date"])

    february_first = covid.loc[covid["local_date"].eq(date(2020, 2, 1))].iloc[0]
    calendar_window = covid.loc[
        covid["local_date"].between(date(2020, 1, 26), date(2020, 2, 1)),
        "relative_deviation_pct",
    ]
    assert len(calendar_window) == 6
    assert february_first["trailing_7_day_mean_deviation_pct"] == pytest.approx(
        calendar_window.mean()
    )
    available_row_window = covid.loc[
        covid["local_date"].between(date(2020, 1, 25), date(2020, 2, 1)),
        "relative_deviation_pct",
    ]
    assert february_first["trailing_7_day_mean_deviation_pct"] != pytest.approx(
        available_row_window.tail(7).mean()
    )


def test_monthly_post_invasion_deviations_match_month_2021_baseline():
    daily_by_country = {
        country: build_daily_actual_forecast(load_country_data(country), country)
        for country in COUNTRIES
    }
    monthly = build_monthly_actual_load(daily_by_country)
    deviations = build_post_invasion_monthly_deviations(monthly)

    assert list(monthly.columns) == MONTHLY_COLUMNS
    assert len(monthly) == 120
    assert list(deviations.columns) == POST_INVASION_MONTHLY_COLUMNS
    assert len(deviations) == 96
    assert sorted(deviations["year"].unique()) == [2022, 2023, 2024, 2025]
    assert set(deviations["month"]) == set(range(1, 13))
    assert deviations["monthly_mean_actual_load_mwh_2021"].notna().all()
    assert np.isfinite(deviations.select_dtypes(include=np.number).to_numpy()).all()

    february_2022 = deviations.loc[
        deviations["year"].eq(2022) & deviations["month"].eq(2)
    ]
    assert len(february_2022) == 2
    assert february_2022["is_transition_month"].eq(True).all()
    row = deviations.iloc[0]
    expected = (
        row["monthly_mean_actual_load_mwh"]
        / row["monthly_mean_actual_load_mwh_2021"]
        - 1.0
    ) * 100.0
    assert row["monthly_deviation_pct"] == pytest.approx(expected)


def test_post_invasion_summaries_exclude_february_from_aggregate_period():
    daily_by_country = {
        country: build_daily_actual_forecast(load_country_data(country), country)
        for country in COUNTRIES
    }
    deviations = build_post_invasion_monthly_deviations(
        build_monthly_actual_load(daily_by_country)
    )
    summary = build_post_invasion_period_summary(deviations)

    assert list(summary.columns) == POST_INVASION_SUMMARY_COLUMNS
    assert len(summary) == 10
    assert summary.groupby("country").size().to_dict() == {
        "Germany": 5,
        "Austria": 5,
    }
    assert summary["n"].tolist() == [1, 10, 12, 12, 12] * 2
    assert np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()
    assert "February 2022" not in set(summary["period"])

    for _, row in summary.iterrows():
        values = deviations.loc[
            deviations["country"].eq(row["country"])
            & deviations["summary_period"].eq(row["period"]),
            "monthly_deviation_pct",
        ]
        assert row["n"] == len(values)
        assert row["mean_monthly_deviation_pct"] == pytest.approx(values.mean())
        assert row["median_monthly_deviation_pct"] == pytest.approx(values.median())


def _real_section_frames():
    daily_by_country = {
        country: build_daily_actual_forecast(load_country_data(country), country)
        for country in COUNTRIES
    }
    covid = pd.concat(
        [
            build_covid_daily_deviations(daily_by_country[country], country)
            for country in COUNTRIES
        ],
        ignore_index=True,
    )
    monthly = build_monthly_actual_load(daily_by_country)
    deviations = build_post_invasion_monthly_deviations(monthly)
    return covid, deviations


def test_figures_have_requested_panels_marks_rows_labels_and_vector_outputs(
    tmp_path, monkeypatch
):
    covid, post_invasion = _real_section_frames()
    captured = []
    monkeypatch.setattr(plt, "close", captured.append)

    _render_covid_event_strip(
        covid,
        tmp_path / "figure_5_9_covid19_event_strip.png",
        tmp_path / "figure_5_9_covid19_event_strip.pdf",
    )
    covid_figure = captured.pop()
    assert len(covid_figure.axes) == 2
    assert [axis.get_title(loc="left").split()[-1] for axis in covid_figure.axes] == [
        "Germany",
        "Austria",
    ]
    assert covid_figure.axes[0].get_ylim() == covid_figure.axes[1].get_ylim()
    assert len(covid_figure.axes[0].patches) == 182
    assert len(covid_figure.axes[1].patches) == 182
    assert [axis.get_ylabel() for axis in covid_figure.axes] == [
        "Actual–forecast deviation [%]",
        "Actual–forecast deviation [%]",
    ]
    legend_labels = [text.get_text() for text in covid_figure.legends[0].get_texts()]
    assert legend_labels[:2] == ["Actual below forecast", "Actual above forecast"]
    assert any(
        text.get_text() == "WHO pandemic declaration, 11 Mar 2020"
        for text in covid_figure.axes[0].texts
    )
    assert [text.get_text() for text in covid_figure.axes[-1].get_xticklabels()] == [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
    ]
    assert any(
        len(line.get_ydata()) == 182 and np.isnan(line.get_ydata()).sum() == 1
        for line in covid_figure.axes[0].lines
    )
    assert any(np.isnan(patch.get_height()) for patch in covid_figure.axes[0].patches)
    assert any(np.allclose(line.get_ydata(), 0) for line in covid_figure.axes[0].lines)

    _render_post_invasion_fingerprint(
        post_invasion,
        tmp_path / "figure_5_10_post_invasion_fingerprint.png",
        tmp_path / "figure_5_10_post_invasion_fingerprint.pdf",
    )
    fingerprint = captured.pop()
    assert len(fingerprint.axes) == 8
    assert len(fingerprint.axes[0].patches) == 12
    assert fingerprint.axes[0].get_ylim() == fingerprint.axes[-1].get_ylim()
    for axis, country in zip(fingerprint.axes[0:2], COUNTRIES):
        context_faces = [patch.get_facecolor()[:3] for patch in axis.patches[:2]]
        assert all(np.allclose(face, to_rgb(CONTEXT_COLOR)) for face in context_faces)
        mean_marker = next(
            line for line in axis.lines if line.get_marker() == "D"
        )
        expected_mean = -5.57 if country == "Germany" else -2.94
        assert mean_marker.get_ydata()[0] == pytest.approx(expected_mean, abs=0.01)
    for row_index, year in enumerate((2023, 2024, 2025), start=1):
        for column_index, country in enumerate(COUNTRIES):
            axis = fingerprint.axes[row_index * 2 + column_index]
            mean_marker = next(
                line for line in axis.lines if line.get_marker() == "D"
            )
            expected = post_invasion.loc[
                post_invasion["country"].eq(country)
                & post_invasion["year"].eq(year),
                "monthly_deviation_pct",
            ].mean()
            assert mean_marker.get_ydata()[0] == pytest.approx(expected)
    assert [
        text.get_text() for text in fingerprint.axes[-2].get_xticklabels()
    ] == [
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
    assert any(line.get_marker() == "D" for axis in fingerprint.axes for line in axis.lines)
    figure_texts = [text.get_text() for text in fingerprint.texts]
    axis_texts = [
        text.get_text() for axis in fingerprint.axes for text in axis.texts
    ]
    assert any("24 Feb 2022" in text for text in axis_texts)
    assert "Deviation from corresponding month of 2021 [%]" in figure_texts
    shared_y_label = next(
        text for text in fingerprint.texts
        if text.get_text() == "Deviation from corresponding month of 2021 [%]"
    )
    assert shared_y_label.get_position()[0] == pytest.approx(0.01)
    assert not any("Monthly actual-load deviation" in text for text in figure_texts)
    assert [text.get_text() for text in fingerprint.legends[0].get_texts()] == [
        "Below 2021",
        "Above 2021",
        "Pre-event/transition context",
        "Period mean",
    ]

    for path in (
        tmp_path / "figure_5_9_covid19_event_strip.png",
        tmp_path / "figure_5_10_post_invasion_fingerprint.png",
    ):
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.info["dpi"][0] == pytest.approx(300, abs=1)
    monkeypatch.undo()
    plt.close(covid_figure)
    plt.close(fingerprint)


def test_runner_writes_only_section_5_5_outputs_and_returns_metadata(tmp_path):
    result = run_section_5_5_crisis_policy_periods(tmp_path)
    expected_names = {
        "covid19_daily_deviations.csv",
        "covid19_period_summary.csv",
        "post_invasion_monthly_deviations.csv",
        "post_invasion_period_summary.csv",
        "figure_5_9_covid19_event_strip.png",
        "figure_5_9_covid19_event_strip.pdf",
        "figure_5_10_post_invasion_fingerprint.png",
        "figure_5_10_post_invasion_fingerprint.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    assert set(result) >= {
        "metadata",
        "validation",
        "covid19_daily_deviations",
        "covid19_period_summary",
        "post_invasion_monthly_deviations",
        "post_invasion_period_summary",
    }
    assert result["metadata"] == ANALYSIS_METADATA
    assert "not a causal COVID-19 effect" in ANALYSIS_METADATA["covid_measure"]
    assert "seven-calendar-day" in ANALYSIS_METADATA["rolling_rule"]
    assert "not temperature" in ANALYSIS_METADATA["post_invasion_comparison"]
    assert "descriptive rather than causal" in ANALYSIS_METADATA["post_invasion_comparison"]
