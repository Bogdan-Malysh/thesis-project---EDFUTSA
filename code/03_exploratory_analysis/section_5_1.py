from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_1"
EXPECTED_ROWS = 52_608
EXPECTED_DAILY_ROWS = 2_192
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 31)
FIGURE_DPI = 300
MOVING_AVERAGE_WINDOW = 30
MOVING_AVERAGE_MIN_PERIODS = 30
YEAR_TICK_DATES = tuple(pd.date_range("2020-01-01", "2025-01-01", freq="YS"))
Y_AXIS_FORMAT = "{x:,.0f}"
COUNTRIES = ["Germany", "Austria"]
INPUT_COLUMNS = ["timestamp_local", "timestamp_utc", "actual_grid_load_mwh"]
COUNTRY_INPUTS = {
    "Germany": PROCESSED / "modelling_germany_hourly.csv",
    "Austria": PROCESSED / "modelling_austria_hourly.csv",
}
STATISTIC_NAMES = [
    "Number of observations",
    "Mean",
    "Standard deviation",
    "Minimum",
    "First quartile",
    "Median",
    "Third quartile",
    "Maximum",
]
ANNUAL_COLUMNS = [
    "country",
    "year",
    "number_of_hourly_observations",
    "mean_hourly_grid_load_mwh",
    "standard_deviation_mwh",
    "minimum_mwh",
    "maximum_mwh",
    "percentage_change_mean",
]
TEXT_COLOR = "#263238"
GRID_COLOR = "#D9DEE3"
DAILY_COLOR = "#A6CEE3"
MOVING_AVERAGE_COLOR = "#0072B2"


def _parse_local_dates(values: pd.Series) -> pd.Series:
    dates: list[date] = []
    for value in values:
        if not isinstance(value, str) or not re.search(r"[+-][0-9]{2}:[0-9]{2}$", value):
            raise ValueError("timestamp_local must contain offset-aware timestamp strings")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp_local must be timezone-aware")
        dates.append(timestamp.date())
    return pd.Series(dates, index=values.index)


def load_country_data(
    country: str,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    if country not in COUNTRY_INPUTS:
        raise ValueError(f"unsupported country: {country}")
    path = Path(processed_directory) / COUNTRY_INPUTS[country].name
    data = pd.read_csv(
        path,
        usecols=INPUT_COLUMNS,
        dtype={"timestamp_local": "string", "timestamp_utc": "string"},
    )
    return data[INPUT_COLUMNS]


def validate_country_data(data: pd.DataFrame, country: str) -> dict[str, object]:
    if list(data.columns) != INPUT_COLUMNS:
        raise ValueError(f"{country} input columns do not match the Section 5.1 contract")
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"{country} input must contain {EXPECTED_ROWS} rows")

    utc = pd.DatetimeIndex(
        pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")
    )
    if utc.isna().any():
        raise ValueError(f"{country} timestamp_utc contains invalid values")
    if not utc.is_unique or not utc.is_monotonic_increasing:
        raise ValueError(f"{country} timestamp_utc is not unique and ordered")
    if not (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{country} timestamp_utc is not hourly continuous")

    actual = pd.to_numeric(data["actual_grid_load_mwh"], errors="coerce")
    if actual.isna().any() or not np.isfinite(actual.to_numpy()).all():
        raise ValueError(f"{country} actual_grid_load_mwh is not complete and finite")

    local_dates = _parse_local_dates(data["timestamp_local"])
    local_years = sorted(local_dates.map(lambda value: value.year).unique().tolist())
    expected_years = list(range(2020, 2026))
    if local_years != expected_years:
        raise ValueError(f"{country} local years do not equal 2020-2025")
    if local_dates.min() != START_DATE or local_dates.max() != END_DATE:
        raise ValueError(f"{country} local dates do not cover 2020-2025 completely")

    return {
        "country": country,
        "row_count": int(len(data)),
        "local_years": local_years,
        "finite_actual_load": True,
        "utc_start": utc[0].isoformat(),
        "utc_end": utc[-1].isoformat(),
    }


def hourly_descriptive_statistics(data: pd.DataFrame) -> dict[str, float | int]:
    values = pd.to_numeric(data["actual_grid_load_mwh"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("actual_grid_load_mwh must be finite for descriptive statistics")
    return {
        "Number of observations": int(values.count()),
        "Mean": float(values.mean()),
        "Standard deviation": float(values.std(ddof=1)),
        "Minimum": float(values.min()),
        "First quartile": float(values.quantile(0.25)),
        "Median": float(values.median()),
        "Third quartile": float(values.quantile(0.75)),
        "Maximum": float(values.max()),
    }


def build_descriptive_table(data_by_country: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for statistic in STATISTIC_NAMES:
        row: dict[str, object] = {"statistic": statistic}
        for country in COUNTRIES:
            value = hourly_descriptive_statistics(data_by_country[country])[statistic]
            row[country] = int(value) if statistic == "Number of observations" else round(float(value), 2)
        rows.append(row)
    return pd.DataFrame(rows, columns=["statistic", *COUNTRIES])


def build_annual_summary(data_by_country: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for country in [country for country in COUNTRIES if country in data_by_country]:
        data = data_by_country[country].copy()
        data["local_date"] = _parse_local_dates(data["timestamp_local"])
        data["year"] = data["local_date"].map(lambda value: value.year)
        annual = data.groupby("year", sort=True)["actual_grid_load_mwh"]
        previous_mean: float | None = None
        for year, values in annual:
            mean = float(values.mean())
            percentage_change = (
                np.nan
                if previous_mean is None
                else (mean - previous_mean) / previous_mean * 100.0
            )
            rows.append(
                {
                    "country": country,
                    "year": int(year),
                    "number_of_hourly_observations": int(values.count()),
                    "mean_hourly_grid_load_mwh": mean,
                    "standard_deviation_mwh": float(values.std(ddof=1)),
                    "minimum_mwh": float(values.min()),
                    "maximum_mwh": float(values.max()),
                    "percentage_change_mean": percentage_change,
                }
            )
            previous_mean = mean
    return pd.DataFrame(rows, columns=ANNUAL_COLUMNS)


def build_daily_load_series(data: pd.DataFrame, country: str) -> pd.DataFrame:
    local_dates = _parse_local_dates(data["timestamp_local"])
    hourly = pd.DataFrame(
        {
            "local_date": local_dates,
            "actual_grid_load_mwh": pd.to_numeric(
                data["actual_grid_load_mwh"], errors="coerce"
            ),
        }
    )
    daily = (
        hourly.groupby("local_date", sort=True)["actual_grid_load_mwh"]
        .mean()
        .rename("daily_mean_mwh")
        .reset_index()
    )
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="D").date
    if len(daily) != EXPECTED_DAILY_ROWS or daily["local_date"].tolist() != list(expected_dates):
        raise ValueError(f"{country} local daily series does not cover 2020-2025 completely")
    daily["moving_average_30d_mwh"] = daily["daily_mean_mwh"].rolling(
        window=MOVING_AVERAGE_WINDOW, min_periods=MOVING_AVERAGE_MIN_PERIODS
    ).mean()
    return daily


def _write_descriptive_table(table: pd.DataFrame, path: Path) -> None:
    formatted = table.copy()
    for country in COUNTRIES:
        formatted[country] = [
            str(int(value)) if statistic == "Number of observations" else f"{float(value):.2f}"
            for statistic, value in zip(formatted["statistic"], formatted[country])
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(path, index=False)


def _write_annual_summary(summary: pd.DataFrame, path: Path) -> None:
    formatted = summary.copy()
    formatted["year"] = formatted["year"].astype(int)
    formatted["number_of_hourly_observations"] = formatted[
        "number_of_hourly_observations"
    ].astype(int)
    for column in ANNUAL_COLUMNS[3:]:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(path, index=False)


def _render_figure(
    daily_by_country: dict[str, pd.DataFrame],
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.4),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    figure.patch.set_facecolor("white")
    for axis, country in zip(axes, COUNTRIES):
        daily = daily_by_country[country]
        dates = pd.to_datetime(daily["local_date"])
        axis.set_facecolor("white")
        axis.plot(
            dates,
            daily["daily_mean_mwh"],
            color=DAILY_COLOR,
            linewidth=0.65,
            alpha=0.8,
            label="Daily mean",
            zorder=1,
        )
        axis.plot(
            dates,
            daily["moving_average_30d_mwh"],
            color=MOVING_AVERAGE_COLOR,
            linewidth=1.8,
            label="30-day moving average",
            zorder=2,
        )
        axis.set_ylabel("Average hourly grid load [MWh]", color=TEXT_COLOR)
        axis.yaxis.set_major_formatter(StrMethodFormatter(Y_AXIS_FORMAT))
        axis.set_title(
            f"({'a' if country == 'Germany' else 'b'}) {country}",
            loc="left",
            pad=7,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.grid(axis="x", visible=False)
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(GRID_COLOR)

    axes[-1].set_xlabel("Local date", color=TEXT_COLOR, labelpad=8)
    axes[-1].set_xticks(YEAR_TICK_DATES)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE) + timedelta(days=1))
    figure.legend(
        handles=[
            Line2D([], [], color=DAILY_COLOR, linewidth=0.9, label="Daily mean"),
            Line2D(
                [],
                [],
                color=MOVING_AVERAGE_COLOR,
                linewidth=1.8,
                label="30-day moving average",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        labelcolor=TEXT_COLOR,
        handlelength=2.4,
        columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.12)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_section_5_1(output_directory: str | Path = OUTPUT_DIRECTORY) -> dict[str, object]:
    output_directory = Path(output_directory)
    data_by_country = {
        country: load_country_data(country) for country in COUNTRIES
    }
    for country, data in data_by_country.items():
        validate_country_data(data, country)

    descriptive_table = build_descriptive_table(data_by_country)
    annual_summary = build_annual_summary(data_by_country)
    daily_by_country = {
        country: build_daily_load_series(data, country)
        for country, data in data_by_country.items()
    }

    _write_descriptive_table(
        descriptive_table,
        output_directory / "table_5_1_descriptive_statistics.csv",
    )
    _write_annual_summary(
        annual_summary,
        output_directory / "annual_load_summary.csv",
    )
    _render_figure(
        daily_by_country,
        output_directory / "figure_5_1_grid_load_development.png",
        output_directory / "figure_5_1_grid_load_development.pdf",
    )
    return {
        "countries": COUNTRIES,
        "descriptive_table": descriptive_table,
        "annual_summary": annual_summary,
        "daily_by_country": daily_by_country,
    }


if __name__ == "__main__":
    run_section_5_1()
