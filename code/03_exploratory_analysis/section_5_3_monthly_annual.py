from __future__ import annotations

from calendar import month_name
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_3"
EXPECTED_ROWS = 52_608
EXPECTED_DAILY_ROWS = 2_192
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 31)
FIGURE_DPI = 300
COUNTRIES = ["Germany", "Austria"]
MONTHS = tuple(range(1, 13))
MONTH_NAMES = tuple(month_name[month] for month in MONTHS)
MONTH_ABBREVIATIONS = tuple(name[:3] for name in MONTH_NAMES)
YEARS = tuple(range(2020, 2026))
INPUT_COLUMNS = ["timestamp_local", "actual_grid_load_mwh"]
COUNTRY_INPUTS = {
    "Germany": PROCESSED / "modelling_germany_hourly.csv",
    "Austria": PROCESSED / "modelling_austria_hourly.csv",
}
MONTHLY_COLUMNS = [
    "country",
    "month",
    "month_name",
    "number_of_days",
    "mean_daily_mean_grid_load_mwh",
    "median_daily_mean_grid_load_mwh",
    "standard_deviation_mwh",
    "first_quartile_mwh",
    "third_quartile_mwh",
]
YEAR_MONTH_COLUMNS = [
    "country",
    "year",
    "month",
    "month_name",
    "number_of_days",
    "mean_daily_mean_grid_load_mwh",
    "median_daily_mean_grid_load_mwh",
    "standard_deviation_mwh",
    "first_quartile_mwh",
    "third_quartile_mwh",
]
TEXT_COLOR = "#263238"
GRID_COLOR = "#D9DEE3"
Y_AXIS_FORMAT = "{x:,.0f}"
DAILY_MEAN_COLOR = "#164A7B"
INTERQUARTILE_COLOR = "#DCE6F1"


def _parse_local_dates(values: pd.Series) -> pd.Series:
    local_dates: list[date] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("timestamp_local contains missing or invalid values")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp_local must be timezone-aware")
        local_dates.append(timestamp.date())
    return pd.Series(local_dates, index=values.index, name="local_date")


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
        dtype={"timestamp_local": "string"},
    )
    return data[INPUT_COLUMNS]


def _validated_components(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if list(data.columns) != INPUT_COLUMNS:
        raise ValueError("monthly/annual input columns do not match the contract")

    actual = pd.to_numeric(data["actual_grid_load_mwh"], errors="coerce")
    if actual.isna().any() or not np.isfinite(actual.to_numpy()).all():
        raise ValueError("actual_grid_load_mwh is not complete and finite")
    local_dates = _parse_local_dates(data["timestamp_local"])
    return actual, local_dates


def validate_country_data(data: pd.DataFrame, country: str) -> dict[str, object]:
    actual, local_dates = _validated_components(data)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"{country} input must contain {EXPECTED_ROWS} rows")
    if local_dates.nunique() != EXPECTED_DAILY_ROWS:
        raise ValueError(f"{country} must contain {EXPECTED_DAILY_ROWS} local dates")
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="D").date
    if sorted(local_dates.unique().tolist()) != list(expected_dates):
        raise ValueError(f"{country} local dates do not cover 2020-2025 completely")
    daily_counts = local_dates.value_counts()
    if not set(daily_counts.tolist()) <= {23, 24, 25}:
        raise ValueError(f"{country} local daily counts are outside 23-25")
    local_years = sorted({value.year for value in local_dates})
    if local_years != list(YEARS):
        raise ValueError(f"{country} local years do not equal 2020-2025")
    return {
        "country": country,
        "row_count": int(len(data)),
        "local_date_count": int(local_dates.nunique()),
        "daily_observation_counts": sorted(daily_counts.unique().tolist()),
        "local_years": local_years,
        "finite_actual_load": bool(np.isfinite(actual.to_numpy()).all()),
    }


def build_daily_observations(data: pd.DataFrame, country: str) -> pd.DataFrame:
    actual, local_dates = _validated_components(data)
    hourly = pd.DataFrame(
        {
            "local_date": local_dates,
            "actual_grid_load_mwh": actual,
        }
    )
    daily = (
        hourly.groupby("local_date", sort=True)
        .agg(
            number_of_hourly_observations=("actual_grid_load_mwh", "count"),
            daily_mean_grid_load_mwh=("actual_grid_load_mwh", "mean"),
        )
        .reset_index()
    )
    if len(daily) != EXPECTED_DAILY_ROWS:
        raise ValueError(f"{country} daily observations do not contain 2,192 dates")
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="D").date
    if daily["local_date"].tolist() != list(expected_dates):
        raise ValueError(f"{country} local daily observations do not cover 2020-2025")
    if not set(daily["number_of_hourly_observations"]) <= {23, 24, 25}:
        raise ValueError(f"{country} daily observations have invalid hour counts")
    daily["year"] = daily["local_date"].map(lambda value: value.year).astype(int)
    daily["month"] = daily["local_date"].map(lambda value: value.month).astype(int)
    return daily[
        [
            "local_date",
            "year",
            "month",
            "number_of_hourly_observations",
            "daily_mean_grid_load_mwh",
        ]
    ]


def _validate_statistics(frame: pd.DataFrame, label: str) -> None:
    numeric = frame.select_dtypes(include=np.number)
    if frame.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    if not (
        frame["first_quartile_mwh"]
        <= frame["median_daily_mean_grid_load_mwh"]
    ).all():
        raise ValueError(f"{label} has invalid lower quartile ordering")
    if not (
        frame["median_daily_mean_grid_load_mwh"]
        <= frame["third_quartile_mwh"]
    ).all():
        raise ValueError(f"{label} has invalid upper quartile ordering")


def _summary_statistics(values: pd.Series) -> dict[str, float | int]:
    return {
        "number_of_days": int(values.count()),
        "mean_daily_mean_grid_load_mwh": float(values.mean()),
        "median_daily_mean_grid_load_mwh": float(values.median()),
        "standard_deviation_mwh": float(values.std(ddof=1)),
        "first_quartile_mwh": float(values.quantile(0.25)),
        "third_quartile_mwh": float(values.quantile(0.75)),
    }


def build_monthly_demand_profile(
    daily_by_country: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for country in COUNTRIES:
        daily = daily_by_country[country]
        for month in MONTHS:
            values = daily.loc[
                daily["month"].eq(month), "daily_mean_grid_load_mwh"
            ]
            if values.empty:
                raise ValueError(f"{country} has no daily observations for month {month}")
            rows.append(
                {
                    "country": country,
                    "month": month,
                    "month_name": MONTH_NAMES[month - 1],
                    **_summary_statistics(values),
                }
            )
    summary = pd.DataFrame(rows, columns=MONTHLY_COLUMNS)
    _validate_statistics(summary, "monthly demand profile")
    return summary


def build_year_month_summary(
    daily_by_country: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for country in COUNTRIES:
        daily = daily_by_country[country]
        for year in YEARS:
            for month in MONTHS:
                values = daily.loc[
                    daily["year"].eq(year) & daily["month"].eq(month),
                    "daily_mean_grid_load_mwh",
                ]
                if values.empty:
                    raise ValueError(
                        f"{country} has no daily observations for {year}-{month:02d}"
                    )
                rows.append(
                    {
                        "country": country,
                        "year": year,
                        "month": month,
                        "month_name": MONTH_NAMES[month - 1],
                        **_summary_statistics(values),
                    }
                )
    summary = pd.DataFrame(rows, columns=YEAR_MONTH_COLUMNS)
    _validate_statistics(summary, "year-month summary")
    return summary


def _write_formatted(frame: pd.DataFrame, path: Path) -> None:
    formatted = frame.copy()
    for column in formatted.columns:
        if column in {"country", "month_name"}:
            continue
        if pd.api.types.is_numeric_dtype(formatted[column]):
            if column in {"year", "month", "number_of_days"}:
                formatted[column] = formatted[column].astype(int)
            else:
                formatted[column] = formatted[column].map(lambda value: f"{float(value):.2f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(path, index=False)


def _render_monthly_figure(summary: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.8),
        sharex=True,
        gridspec_kw={"hspace": 0.38},
    )
    figure.patch.set_facecolor("white")
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        country_summary = summary.loc[summary["country"].eq(country)].sort_values("month")
        x = country_summary["month"].to_numpy()
        means = country_summary["mean_daily_mean_grid_load_mwh"].to_numpy()
        first_quartile = country_summary["first_quartile_mwh"].to_numpy()
        third_quartile = country_summary["third_quartile_mwh"].to_numpy()
        axis.fill_between(
            x,
            first_quartile,
            third_quartile,
            color=INTERQUARTILE_COLOR,
            alpha=0.9,
            label="Interquartile range",
        )
        axis.plot(
            x,
            means,
            color=DAILY_MEAN_COLOR,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Daily mean grid load",
        )
        axis.set_title(
            f"{panel} {country}",
            loc="left",
            pad=8,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        axis.set_ylabel("Daily mean grid load [MWh]", color=TEXT_COLOR)
        axis.set_xlim(0.75, 12.25)
        axis.set_xticks(MONTHS, MONTH_NAMES)
        axis.yaxis.set_major_formatter(StrMethodFormatter(Y_AXIS_FORMAT))
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(GRID_COLOR)
        axis.spines["bottom"].set_color(GRID_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = [labels.index("Daily mean grid load"), labels.index("Interquartile range")]
    figure.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    axes[-1].set_xlabel("Month", color=TEXT_COLOR, labelpad=8)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_year_month_heatmap(
    summary: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.2),
        gridspec_kw={"hspace": 0.42},
    )
    figure.patch.set_facecolor("white")
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        country_summary = summary.loc[summary["country"].eq(country)]
        matrix = country_summary.pivot(
            index="year",
            columns="month",
            values="mean_daily_mean_grid_load_mwh",
        ).reindex(index=YEARS, columns=MONTHS)
        image = axis.imshow(
            matrix.to_numpy(),
            aspect="auto",
            cmap="cividis",
            interpolation="nearest",
            vmin=float(matrix.min().min()),
            vmax=float(matrix.max().max()),
        )
        axis.set_title(
            f"{panel} {country}",
            loc="left",
            pad=8,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        axis.set_yticks(range(len(YEARS)), YEARS)
        axis.set_xticks(range(len(MONTHS)), MONTH_ABBREVIATIONS)
        for label in axis.get_xticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("center")
        axis.set_ylabel("Year", color=TEXT_COLOR)
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.grid(False)
        colorbar = figure.colorbar(image, ax=axis, pad=0.02, aspect=25)
        colorbar.set_label("Daily mean grid load [MWh]", color=TEXT_COLOR, labelpad=8)
        colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter(Y_AXIS_FORMAT))
        colorbar.ax.tick_params(colors=TEXT_COLOR, length=0)
        colorbar.outline.set_edgecolor(GRID_COLOR)
    axes[-1].set_xlabel("Month", color=TEXT_COLOR, labelpad=8)
    figure.subplots_adjust(left=0.13, right=0.88, top=0.96, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_section_5_3_monthly_annual(
    output_directory: str | Path = OUTPUT_DIRECTORY,
) -> dict[str, object]:
    output_directory = Path(output_directory)
    data_by_country = {country: load_country_data(country) for country in COUNTRIES}
    validation = {
        country: validate_country_data(data, country)
        for country, data in data_by_country.items()
    }
    daily_by_country = {
        country: build_daily_observations(data, country)
        for country, data in data_by_country.items()
    }
    monthly = build_monthly_demand_profile(daily_by_country)
    year_month = build_year_month_summary(daily_by_country)
    _write_formatted(monthly, output_directory / "monthly_demand_profile.csv")
    _write_formatted(year_month, output_directory / "year_month_summary.csv")
    _render_monthly_figure(
        monthly,
        output_directory / "figure_5_5_monthly_demand_patterns.png",
        output_directory / "figure_5_5_monthly_demand_patterns.pdf",
    )
    _render_year_month_heatmap(
        year_month,
        output_directory / "figure_5_6_year_month_heatmap.png",
        output_directory / "figure_5_6_year_month_heatmap.pdf",
    )
    return {
        "validation": validation,
        "daily": daily_by_country,
        "monthly_demand_profile": monthly,
        "year_month_summary": year_month,
    }


if __name__ == "__main__":
    run_section_5_3_monthly_annual()
