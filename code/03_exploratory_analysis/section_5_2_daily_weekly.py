from __future__ import annotations

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
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_2"
EXPECTED_ROWS = 52_608
EXPECTED_DAILY_ROWS = 2_192
FIGURE_DPI = 300
COUNTRIES = ["Germany", "Austria"]
DAY_CODES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
DAY_NAMES = tuple(DAY_CODES.values())
DAY_NAME_TO_CODE = {name: code for code, name in DAY_CODES.items()}
HOURS = tuple(range(24))
INPUT_COLUMNS = ["timestamp_local", "actual_grid_load_mwh", "hour", "day_of_week"]
COUNTRY_INPUTS = {
    "Germany": PROCESSED / "modelling_germany_hourly.csv",
    "Austria": PROCESSED / "modelling_austria_hourly.csv",
}
DAY_SUMMARY_COLUMNS = [
    "country",
    "day_of_week",
    "day_name",
    "number_of_days",
    "mean_daily_mean_grid_load_mwh",
    "median_daily_mean_grid_load_mwh",
    "standard_deviation_mwh",
    "first_quartile_mwh",
    "third_quartile_mwh",
]
WEEKLY_PROFILE_COLUMNS = [
    "country",
    "day_of_week",
    "day_name",
    "hour",
    "number_of_observations",
    "mean_hourly_grid_load_mwh",
    "median_hourly_grid_load_mwh",
    "first_quartile_mwh",
    "third_quartile_mwh",
]
TEXT_COLOR = "#263238"
GRID_COLOR = "#D9DEE3"
Y_AXIS_FORMAT = "{x:,.0f}"


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


def _validated_components(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if list(data.columns) != INPUT_COLUMNS:
        raise ValueError("daily/weekly input columns do not match the contract")

    actual = pd.to_numeric(data["actual_grid_load_mwh"], errors="coerce")
    if actual.isna().any() or not np.isfinite(actual.to_numpy()).all():
        raise ValueError("actual_grid_load_mwh is not complete and finite")

    hours = pd.to_numeric(data["hour"], errors="coerce")
    day_codes = pd.to_numeric(data["day_of_week"], errors="coerce")
    for values, name in ((hours, "hour"), (day_codes, "day_of_week")):
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name} is not complete and finite")
        if not np.equal(values.to_numpy(), values.to_numpy().astype(int)).all():
            raise ValueError(f"{name} contains non-integer values")
    hours = hours.astype(int)
    day_codes = day_codes.astype(int)
    if not hours.between(0, 23).all():
        raise ValueError("hour must contain values from 0 through 23")
    if not day_codes.between(0, 6).all():
        raise ValueError("day_of_week must contain values from 0 through 6")

    local_dates: list[date] = []
    local_names: list[str] = []
    for value in data["timestamp_local"]:
        if not isinstance(value, str):
            raise ValueError("timestamp_local contains missing or invalid values")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp_local must be timezone-aware")
        local_dates.append(timestamp.date())
        local_names.append(timestamp.day_name())
    dates = pd.Series(local_dates, index=data.index)
    names = pd.Series(local_names, index=data.index)
    expected_names = day_codes.map(DAY_CODES)
    if not names.equals(expected_names):
        raise ValueError("day_of_week coding does not match timestamp_local weekdays")
    return actual, hours, day_codes, dates, names


def validate_country_data(data: pd.DataFrame, country: str) -> dict[str, object]:
    actual, hours, day_codes, dates, names = _validated_components(data)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"{country} input must contain {EXPECTED_ROWS} rows")
    if dates.nunique() != EXPECTED_DAILY_ROWS:
        raise ValueError(f"{country} must contain {EXPECTED_DAILY_ROWS} local dates")
    expected_dates = pd.date_range("2020-01-01", "2025-12-31", freq="D").date
    if sorted(dates.unique().tolist()) != list(expected_dates):
        raise ValueError(f"{country} local dates do not cover 2020-2025 completely")
    daily_counts = dates.value_counts()
    if not set(daily_counts.tolist()) <= {23, 24, 25}:
        raise ValueError(f"{country} local daily counts are outside 23-25")
    if set(day_codes.unique()) != set(DAY_CODES):
        raise ValueError(f"{country} does not contain all seven weekday codes")
    if set(hours.unique()) != set(HOURS):
        raise ValueError(f"{country} does not contain all 24 local hours")
    return {
        "country": country,
        "row_count": int(len(data)),
        "day_code_mapping": DAY_CODES.copy(),
        "day_names": list(DAY_NAMES),
        "local_date_count": int(dates.nunique()),
        "daily_observation_counts": sorted(daily_counts.unique().tolist()),
        "all_hours_present": True,
        "finite_actual_load": bool(np.isfinite(actual.to_numpy()).all()),
        "local_weekday_names": sorted(names.unique().tolist()),
    }


def build_daily_observations(data: pd.DataFrame, country: str) -> pd.DataFrame:
    actual, _, day_codes, dates, names = _validated_components(data)
    working = pd.DataFrame(
        {
            "local_date": dates,
            "day_of_week": day_codes,
            "day_name": names,
            "actual_grid_load_mwh": actual,
        }
    )
    daily = (
        working.groupby("local_date", sort=True)
        .agg(
            day_of_week=("day_of_week", "first"),
            day_name=("day_name", "first"),
            number_of_hourly_observations=("actual_grid_load_mwh", "count"),
            daily_mean_grid_load_mwh=("actual_grid_load_mwh", "mean"),
        )
        .reset_index()
    )
    if len(daily) != EXPECTED_DAILY_ROWS:
        raise ValueError(f"{country} daily observations do not contain 2,192 dates")
    if not set(daily["number_of_hourly_observations"]) <= {23, 24, 25}:
        raise ValueError(f"{country} daily observations have invalid hour counts")
    return daily[
        [
            "local_date",
            "day_of_week",
            "day_name",
            "number_of_hourly_observations",
            "daily_mean_grid_load_mwh",
        ]
    ]


def build_day_of_week_summary(
    daily_by_country: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for country in COUNTRIES:
        daily = daily_by_country[country]
        for code in DAY_CODES:
            values = daily.loc[
                daily["day_of_week"].eq(code), "daily_mean_grid_load_mwh"
            ]
            if values.empty:
                raise ValueError(f"{country} has no daily observations for {DAY_CODES[code]}")
            rows.append(
                {
                    "country": country,
                    "day_of_week": code,
                    "day_name": DAY_CODES[code],
                    "number_of_days": int(values.count()),
                    "mean_daily_mean_grid_load_mwh": float(values.mean()),
                    "median_daily_mean_grid_load_mwh": float(values.median()),
                    "standard_deviation_mwh": float(values.std(ddof=1)),
                    "first_quartile_mwh": float(values.quantile(0.25)),
                    "third_quartile_mwh": float(values.quantile(0.75)),
                }
            )
    summary = pd.DataFrame(rows, columns=DAY_SUMMARY_COLUMNS)
    _validate_statistics(summary, "day-of-week summary")
    return summary


def build_weekly_hourly_profile(
    data_by_country: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for country in COUNTRIES:
        actual, hours, day_codes, _, _ = _validated_components(data_by_country[country])
        working = pd.DataFrame(
            {
                "hour": hours.to_numpy(),
                "day_of_week": day_codes.to_numpy(),
                "actual_grid_load_mwh": actual.to_numpy(),
            }
        )
        for code in DAY_CODES:
            for hour in HOURS:
                values = working.loc[
                    working["day_of_week"].eq(code) & working["hour"].eq(hour),
                    "actual_grid_load_mwh",
                ]
                if values.empty:
                    raise ValueError(f"{country} has no data for {DAY_CODES[code]} hour {hour}")
                rows.append(
                    {
                        "country": country,
                        "day_of_week": code,
                        "day_name": DAY_CODES[code],
                        "hour": hour,
                        "number_of_observations": int(values.count()),
                        "mean_hourly_grid_load_mwh": float(values.mean()),
                        "median_hourly_grid_load_mwh": float(values.median()),
                        "first_quartile_mwh": float(values.quantile(0.25)),
                        "third_quartile_mwh": float(values.quantile(0.75)),
                    }
                )
    profile = pd.DataFrame(rows, columns=WEEKLY_PROFILE_COLUMNS)
    _validate_statistics(profile, "weekly hourly profile")
    return profile


def _validate_statistics(frame: pd.DataFrame, label: str) -> None:
    numeric = frame.select_dtypes(include=np.number)
    if frame.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    if "first_quartile_mwh" in frame.columns:
        if not (
            frame["first_quartile_mwh"].le(frame["median_daily_mean_grid_load_mwh"])
            if "median_daily_mean_grid_load_mwh" in frame.columns
            else frame["first_quartile_mwh"].le(frame["median_hourly_grid_load_mwh"])
        ).all():
            raise ValueError(f"{label} has invalid lower quartile ordering")
        median_column = (
            "median_daily_mean_grid_load_mwh"
            if "median_daily_mean_grid_load_mwh" in frame.columns
            else "median_hourly_grid_load_mwh"
        )
        if not frame[median_column].le(frame["third_quartile_mwh"]).all():
            raise ValueError(f"{label} has invalid upper quartile ordering")


def _weighted_hour_means(profile: pd.DataFrame, day_codes: set[int]) -> pd.Series:
    selected = profile.loc[profile["day_of_week"].isin(day_codes)].copy()
    selected["weighted_mean"] = (
        selected["mean_hourly_grid_load_mwh"] * selected["number_of_observations"]
    )
    grouped = selected.groupby("hour")
    return grouped["weighted_mean"].sum() / grouped["number_of_observations"].sum()


def build_comparison_metrics(
    summary: pd.DataFrame,
    profile: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for country in COUNTRIES:
        country_summary = summary.loc[summary["country"].eq(country)]
        weekday = country_summary.loc[country_summary["day_of_week"].le(4)]
        weekend = country_summary.loc[country_summary["day_of_week"].ge(5)]
        weekday_mean = float(
            np.average(
                weekday["mean_daily_mean_grid_load_mwh"],
                weights=weekday["number_of_days"],
            )
        )
        weekend_mean = float(
            np.average(
                weekend["mean_daily_mean_grid_load_mwh"],
                weights=weekend["number_of_days"],
            )
        )
        country_profile = profile.loc[profile["country"].eq(country)]
        weekday_hours = _weighted_hour_means(country_profile, {0, 1, 2, 3, 4})
        saturday_hours = _weighted_hour_means(country_profile, {5})
        sunday_hours = _weighted_hour_means(country_profile, {6})
        metrics[country] = {
            "highest_mean_day": country_summary.loc[
                country_summary["mean_daily_mean_grid_load_mwh"].idxmax(), "day_name"
            ],
            "lowest_mean_day": country_summary.loc[
                country_summary["mean_daily_mean_grid_load_mwh"].idxmin(), "day_name"
            ],
            "weekday_mean_mwh": weekday_mean,
            "weekend_mean_mwh": weekend_mean,
            "weekday_to_weekend_reduction_percent": (weekday_mean - weekend_mean)
            / weekday_mean
            * 100.0,
            "weekday_peak_hour": int(weekday_hours.idxmax()),
            "saturday_peak_hour": int(saturday_hours.idxmax()),
            "sunday_peak_hour": int(sunday_hours.idxmax()),
            "weekday_minimum_hour": int(weekday_hours.idxmin()),
            "saturday_minimum_hour": int(saturday_hours.idxmin()),
            "sunday_minimum_hour": int(sunday_hours.idxmin()),
        }
    return metrics


def _write_formatted(frame: pd.DataFrame, path: Path) -> None:
    formatted = frame.copy()
    for column in formatted.columns:
        if column in {"country", "day_name"}:
            continue
        if pd.api.types.is_numeric_dtype(formatted[column]):
            if column in {"day_of_week", "hour", "number_of_days", "number_of_observations"}:
                formatted[column] = formatted[column].astype(int)
            else:
                formatted[column] = formatted[column].map(lambda value: f"{float(value):.2f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(path, index=False)


def _render_daily_figure(summary: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
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
        country_summary = summary.loc[summary["country"].eq(country)].sort_values(
            "day_of_week"
        )
        x = country_summary["day_of_week"].to_numpy()
        means = country_summary["mean_daily_mean_grid_load_mwh"].to_numpy()
        first_quartile = country_summary["first_quartile_mwh"].to_numpy()
        third_quartile = country_summary["third_quartile_mwh"].to_numpy()
        axis.fill_between(
            x,
            first_quartile,
            third_quartile,
            color="#DCE6F1",
            alpha=0.9,
            label="Interquartile range",
        )
        axis.plot(
            x,
            means,
            color="#164A7B",
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
        axis.set_xlim(-0.25, 6.25)
        axis.set_xticks(range(7), DAY_NAMES)
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
    axes[-1].set_xlabel("Day of week", color=TEXT_COLOR, labelpad=8)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.89, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_heatmaps(profile: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        gridspec_kw={"hspace": 0.38},
    )
    figure.patch.set_facecolor("white")
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        country_profile = profile.loc[profile["country"].eq(country)]
        matrix = country_profile.pivot(
            index="day_of_week", columns="hour", values="mean_hourly_grid_load_mwh"
        ).reindex(index=range(7), columns=HOURS)
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
        axis.set_yticks(range(7), DAY_NAMES)
        axis.set_xticks(range(0, 24, 2), range(0, 24, 2))
        axis.set_ylabel("Day of week", color=TEXT_COLOR)
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.grid(False)
        colorbar = figure.colorbar(image, ax=axis, pad=0.02, aspect=25)
        colorbar.set_label("Mean grid load [MWh]", color=TEXT_COLOR, labelpad=8)
        colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter(Y_AXIS_FORMAT))
        colorbar.ax.tick_params(colors=TEXT_COLOR, length=0)
        colorbar.outline.set_edgecolor(GRID_COLOR)
    axes[-1].set_xlabel("Local hour", color=TEXT_COLOR, labelpad=8)
    figure.subplots_adjust(left=0.13, right=0.88, top=0.96, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_section_5_2_daily_weekly(
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
    day_summary = build_day_of_week_summary(daily_by_country)
    weekly_profile = build_weekly_hourly_profile(data_by_country)
    metrics = build_comparison_metrics(day_summary, weekly_profile)
    _write_formatted(day_summary, output_directory / "day_of_week_summary.csv")
    _write_formatted(weekly_profile, output_directory / "weekly_hourly_profile.csv")
    _render_daily_figure(
        day_summary,
        output_directory / "figure_5_3_daily_demand_patterns.png",
        output_directory / "figure_5_3_daily_demand_patterns.pdf",
    )
    _render_heatmaps(
        weekly_profile,
        output_directory / "figure_5_4_weekly_demand_patterns.png",
        output_directory / "figure_5_4_weekly_demand_patterns.pdf",
    )
    return {
        "validation": validation,
        "daily": daily_by_country,
        "day_of_week_summary": day_summary,
        "weekly_hourly_profile": weekly_profile,
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_section_5_2_daily_weekly()
