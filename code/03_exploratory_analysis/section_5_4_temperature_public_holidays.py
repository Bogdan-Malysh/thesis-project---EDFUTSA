from __future__ import annotations

from calendar import month_name
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dateutil.easter import easter
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_4"
EXPECTED_ROWS = 52_608
EXPECTED_DAILY_ROWS = 2_192
YEARS = tuple(range(2020, 2026))
MONTH_NAMES = tuple(month_name[month] for month in range(1, 13))
WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
COUNTRIES = ["Germany", "Austria"]
COUNTRY_TIMEZONES = {
    "Germany": "Europe/Berlin",
    "Austria": "Europe/Vienna",
}
LOAD_INPUTS = {
    "Germany": PROCESSED / "modelling_germany_hourly.csv",
    "Austria": PROCESSED / "modelling_austria_hourly.csv",
}
TEMPERATURE_INPUTS = {
    "Germany": PROCESSED / "temperature_germany_hourly.csv",
    "Austria": PROCESSED / "temperature_austria_hourly.csv",
}
DAILY_COLUMNS = [
    "country",
    "local_date",
    "daily_mean_load_mwh",
    "daily_mean_temperature_c",
    "hourly_count",
    "weekday",
    "public_holiday_flag",
    "holiday_name",
    "day_type",
    "matching_baseline_mwh",
    "relative_load_deviation_pct",
]
TEMPERATURE_BIN_COLUMNS = [
    "country",
    "temperature_bin_lower_c",
    "temperature_bin_upper_c",
    "temperature_bin_midpoint_c",
    "number_of_days",
    "mean_daily_mean_load_mwh",
    "median_daily_mean_load_mwh",
    "standard_deviation_mwh",
    "first_quartile_mwh",
    "third_quartile_mwh",
]
HOLIDAY_SUMMARY_COLUMNS = [
    "country",
    "year",
    "month",
    "month_name",
    "weekday",
    "weekday_name",
    "ordinary_weekday_count",
    "weekday_public_holiday_count",
    "matching_baseline_mwh",
    "ordinary_weekday_mean_deviation_pct",
    "weekday_public_holiday_mean_deviation_pct",
    "weekday_public_holiday_median_deviation_pct",
    "weekday_public_holiday_first_quartile_pct",
    "weekday_public_holiday_third_quartile_pct",
]
TEXT_COLOR = "#263238"
GRID_COLOR = "#D9DEE3"
Y_AXIS_FORMAT = "{x:,.0f}"
DAILY_OBSERVATION_COLOR = "#A6CEE3"
DAILY_MEAN_COLOR = "#164A7B"
INTERQUARTILE_COLOR = "#DCE6F1"
HOLIDAY_COLOR = "#164A7B"
ORDINARY_COLOR = "#A6CEE3"


def _utc_index(values: pd.Series, label: str) -> pd.DatetimeIndex:
    utc = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce"))
    if utc.isna().any():
        raise ValueError(f"{label} contains invalid UTC timestamps")
    return utc


def _validate_utc_index(utc: pd.DatetimeIndex, label: str) -> None:
    utc = pd.DatetimeIndex(utc)
    if not utc.is_unique:
        raise ValueError(f"{label} contains duplicate UTC keys")
    if not utc.is_monotonic_increasing:
        raise ValueError(f"{label} is not ordered by UTC key")
    if len(utc) > 1 and not (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{label} is not hourly continuous")


def load_load_data(country: str) -> pd.DataFrame:
    if country not in LOAD_INPUTS:
        raise ValueError(f"unsupported country: {country}")
    data = pd.read_csv(
        LOAD_INPUTS[country],
        usecols=["timestamp_utc", "timestamp_local", "actual_grid_load_mwh"],
        dtype={"timestamp_utc": "string", "timestamp_local": "string"},
    )
    data["timestamp_utc"] = _utc_index(data["timestamp_utc"], "load")
    _validate_utc_index(data["timestamp_utc"], "load UTC key")
    data["actual_grid_load_mwh"] = pd.to_numeric(
        data["actual_grid_load_mwh"], errors="coerce"
    )
    if data["actual_grid_load_mwh"].isna().any() or not np.isfinite(
        data["actual_grid_load_mwh"].to_numpy()
    ).all():
        raise ValueError(f"{country} load contains missing or non-finite values")
    return data


def load_temperature_data(country: str) -> pd.DataFrame:
    if country not in TEMPERATURE_INPUTS:
        raise ValueError(f"unsupported country: {country}")
    data = pd.read_csv(
        TEMPERATURE_INPUTS[country],
        usecols=["interval_start_utc", "temperature_c", "temperature_valid"],
        dtype={"interval_start_utc": "string", "temperature_valid": "string"},
    )
    data["interval_start_utc"] = _utc_index(
        data["interval_start_utc"], "temperature"
    )
    _validate_utc_index(data["interval_start_utc"], "temperature UTC key")
    valid = data["temperature_valid"].str.lower().eq("true")
    if not valid.all():
        raise ValueError(f"{country} temperature contains invalid observations")
    data["temperature_c"] = pd.to_numeric(data["temperature_c"], errors="coerce")
    if data["temperature_c"].isna().any() or not np.isfinite(
        data["temperature_c"].to_numpy()
    ).all():
        raise ValueError(f"{country} temperature contains missing or non-finite values")
    return data


def align_hourly_data(country: str) -> pd.DataFrame:
    load = load_load_data(country)
    temperature = load_temperature_data(country)
    if len(load) != EXPECTED_ROWS or len(temperature) != EXPECTED_ROWS:
        raise ValueError(f"{country} inputs must each contain {EXPECTED_ROWS} rows")
    load_utc = pd.DatetimeIndex(load["timestamp_utc"])
    temperature_utc = pd.DatetimeIndex(temperature["interval_start_utc"])
    if not load_utc.equals(temperature_utc):
        raise ValueError(f"{country} load and temperature UTC keys do not match")

    utc = load_utc
    local = utc.tz_convert(ZoneInfo(COUNTRY_TIMEZONES[country]))
    aligned = pd.DataFrame(
        {
            "country": country,
            "timestamp_utc": utc,
            "actual_grid_load_mwh": load["actual_grid_load_mwh"].to_numpy(),
            "temperature_c": temperature["temperature_c"].to_numpy(),
            "local_date": pd.Series(local.date),
            "year": pd.Series(local.year, dtype="int64"),
            "month": pd.Series(local.month, dtype="int64"),
            "weekday_number": pd.Series(local.dayofweek, dtype="int64"),
            "weekday": pd.Series(local.dayofweek).map(
                lambda value: WEEKDAY_NAMES[int(value)]
            ),
        }
    )
    return aligned


def validate_aligned_hourly_data(
    aligned: pd.DataFrame,
    country: str,
) -> dict[str, object]:
    if len(aligned) != EXPECTED_ROWS:
        raise ValueError(f"{country} aligned data must contain {EXPECTED_ROWS} rows")
    utc = pd.DatetimeIndex(aligned["timestamp_utc"])
    _validate_utc_index(utc, f"{country} aligned UTC key")
    load = aligned["actual_grid_load_mwh"].to_numpy()
    temperature = aligned["temperature_c"].to_numpy()
    if not np.isfinite(load).all() or not np.isfinite(temperature).all():
        raise ValueError(f"{country} aligned values are not finite")
    daily_counts = aligned["local_date"].value_counts()
    if aligned["local_date"].nunique() != EXPECTED_DAILY_ROWS:
        raise ValueError(f"{country} aligned data does not contain 2,192 local dates")
    if not set(daily_counts.tolist()) <= {23, 24, 25}:
        raise ValueError(f"{country} aligned daily counts are outside 23-25")
    expected_dates = pd.date_range("2020-01-01", "2025-12-31", freq="D").date
    if sorted(aligned["local_date"].unique().tolist()) != list(expected_dates):
        raise ValueError(f"{country} aligned local dates do not cover 2020-2025")
    return {
        "country": country,
        "row_count": int(len(aligned)),
        "utc_unique": bool(utc.is_unique),
        "utc_hourly_continuous": bool(
            len(utc) < 2 or (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all()
        ),
        "finite_load": bool(np.isfinite(load).all()),
        "finite_temperature": bool(np.isfinite(temperature).all()),
        "local_date_count": int(aligned["local_date"].nunique()),
        "daily_observation_counts": sorted(daily_counts.unique().tolist()),
    }


def holiday_calendar(country: str, years: list[int] | tuple[int, ...]) -> dict[date, str]:
    if country not in COUNTRIES:
        raise ValueError(f"unsupported country: {country}")
    result: dict[date, str] = {}
    for year in sorted({int(value) for value in years}):
        easter_sunday = easter(year)
        fixed: dict[tuple[int, int], str] = {
            (1, 1): "New Year's Day",
            (5, 1): "Labour Day",
            (12, 25): "Christmas Day",
            (12, 26): "Boxing Day" if country == "Germany" else "St. Stephen's Day",
        }
        offsets: dict[int, str]
        if country == "Germany":
            fixed[(10, 3)] = "German Unity Day"
            offsets = {
                -2: "Good Friday",
                1: "Easter Monday",
                39: "Ascension Day",
                50: "Whit Monday",
            }
        else:
            fixed.update(
                {
                    (1, 6): "Epiphany",
                    (8, 15): "Assumption Day",
                    (10, 26): "Austrian National Day",
                    (11, 1): "All Saints' Day",
                    (12, 8): "Immaculate Conception",
                }
            )
            offsets = {
                1: "Easter Monday",
                39: "Ascension Day",
                50: "Whit Monday",
                60: "Corpus Christi",
            }
        for (month, day), name in fixed.items():
            result[date(year, month, day)] = name
        for offset, name in offsets.items():
            result[easter_sunday + timedelta(days=offset)] = name
    return result


def _classify_day(day: date, holiday_name: str) -> str:
    weekend = day.weekday() >= 5
    if holiday_name and weekend:
        return "weekend public holiday"
    if holiday_name:
        return "weekday public holiday"
    if weekend:
        return "weekend"
    return "ordinary weekday"


def build_daily_temperature_public_holidays(
    aligned: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    validate_aligned_hourly_data(aligned, country)
    daily = (
        aligned.groupby("local_date", sort=True)
        .agg(
            daily_mean_load_mwh=("actual_grid_load_mwh", "mean"),
            daily_mean_temperature_c=("temperature_c", "mean"),
            hourly_count=("actual_grid_load_mwh", "count"),
        )
        .reset_index()
    )
    daily["country"] = country
    daily["year"] = daily["local_date"].map(lambda value: value.year).astype(int)
    daily["month"] = daily["local_date"].map(lambda value: value.month).astype(int)
    daily["weekday_number"] = daily["local_date"].map(lambda value: value.weekday()).astype(int)
    daily["weekday"] = daily["weekday_number"].map(
        lambda value: WEEKDAY_NAMES[int(value)]
    )
    holidays_for_years = holiday_calendar(country, list(YEARS))
    daily["holiday_name"] = daily["local_date"].map(holidays_for_years).fillna("")
    daily["public_holiday_flag"] = daily["holiday_name"].ne("").astype(int)
    daily["day_type"] = daily.apply(
        lambda row: _classify_day(row["local_date"], row["holiday_name"]), axis=1
    )

    ordinary = daily[daily["day_type"].eq("ordinary weekday")]
    baseline = (
        ordinary.groupby(["year", "month", "weekday_number"])["daily_mean_load_mwh"]
        .mean()
        .rename("matching_baseline_mwh")
        .reset_index()
    )
    daily = daily.merge(
        baseline,
        on=["year", "month", "weekday_number"],
        how="left",
        validate="many_to_one",
    )
    analysis_mask = daily["day_type"].isin(
        ["ordinary weekday", "weekday public holiday"]
    )
    daily["relative_load_deviation_pct"] = np.nan
    daily.loc[analysis_mask, "relative_load_deviation_pct"] = (
        daily.loc[analysis_mask, "daily_mean_load_mwh"]
        / daily.loc[analysis_mask, "matching_baseline_mwh"]
        - 1.0
    ) * 100.0
    if daily.loc[analysis_mask, "matching_baseline_mwh"].isna().any():
        raise ValueError(f"{country} has an analysis date without a matching baseline")
    return daily.sort_values("local_date").reset_index(drop=True)


def _summary_statistics(values: pd.Series) -> dict[str, float | int]:
    return {
        "number_of_days": int(values.count()),
        "mean_daily_mean_load_mwh": float(values.mean()),
        "median_daily_mean_load_mwh": float(values.median()),
        "standard_deviation_mwh": float(values.std(ddof=1)),
        "first_quartile_mwh": float(values.quantile(0.25)),
        "third_quartile_mwh": float(values.quantile(0.75)),
    }


def _validate_quartiles(frame: pd.DataFrame, median_column: str) -> None:
    numeric = frame.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("summary contains missing or non-finite numeric values")
    if not (
        frame["first_quartile_mwh"] <= frame[median_column]
    ).all():
        raise ValueError("summary has invalid lower quartile ordering")
    if not (frame[median_column] <= frame["third_quartile_mwh"]).all():
        raise ValueError("summary has invalid upper quartile ordering")


def _common_temperature_range(daily: pd.DataFrame) -> tuple[float, float]:
    minimum = float(daily["daily_mean_temperature_c"].min())
    maximum = float(daily["daily_mean_temperature_c"].max())
    lower = float(np.floor(minimum / 2.0) * 2.0)
    upper = float(np.ceil(maximum / 2.0) * 2.0)
    if upper <= lower:
        upper = lower + 2.0
    return lower, upper


def build_temperature_bin_summary(daily: pd.DataFrame) -> pd.DataFrame:
    lower, upper = _common_temperature_range(daily)
    bin_width = 2.0
    working = daily.copy()
    bin_index = np.floor(
        (working["daily_mean_temperature_c"].to_numpy() - lower) / bin_width
    ).astype(int)
    maximum_index = int(round((upper - lower) / bin_width)) - 1
    bin_index = np.clip(bin_index, 0, maximum_index)
    working["temperature_bin_lower_c"] = lower + bin_index * bin_width
    working["temperature_bin_upper_c"] = working["temperature_bin_lower_c"] + bin_width
    working["temperature_bin_midpoint_c"] = (
        working["temperature_bin_lower_c"] + bin_width / 2.0
    )
    rows = []
    for (country, bin_lower, bin_upper, midpoint), group in working.groupby(
        [
            "country",
            "temperature_bin_lower_c",
            "temperature_bin_upper_c",
            "temperature_bin_midpoint_c",
        ],
        sort=True,
    ):
        values = group["daily_mean_load_mwh"]
        if len(values) < 10:
            continue
        rows.append(
            {
                "country": country,
                "temperature_bin_lower_c": float(bin_lower),
                "temperature_bin_upper_c": float(bin_upper),
                "temperature_bin_midpoint_c": float(midpoint),
                **_summary_statistics(values),
            }
        )
    summary = pd.DataFrame(rows, columns=TEMPERATURE_BIN_COLUMNS)
    if summary.empty:
        raise ValueError("no temperature bins contain at least 10 days")
    _validate_quartiles(summary, "median_daily_mean_load_mwh")
    return summary


def build_public_holiday_effect_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    holidays = daily[daily["day_type"].eq("weekday public holiday")]
    for (country, year, month, weekday), holiday_group in holidays.groupby(
        ["country", "year", "month", "weekday_number"], sort=True
    ):
        ordinary_group = daily.loc[
            daily["country"].eq(country)
            & daily["year"].eq(year)
            & daily["month"].eq(month)
            & daily["weekday_number"].eq(weekday)
            & daily["day_type"].eq("ordinary weekday")
        ]
        if ordinary_group.empty:
            raise ValueError(
                f"no ordinary weekday baseline for {country} {year}-{month:02d} weekday {weekday}"
            )
        ordinary_deviations = ordinary_group["relative_load_deviation_pct"]
        holiday_deviations = holiday_group["relative_load_deviation_pct"]
        rows.append(
            {
                "country": country,
                "year": int(year),
                "month": int(month),
                "month_name": MONTH_NAMES[int(month) - 1],
                "weekday": int(weekday),
                "weekday_name": WEEKDAY_NAMES[int(weekday)],
                "ordinary_weekday_count": int(len(ordinary_group)),
                "weekday_public_holiday_count": int(len(holiday_group)),
                "matching_baseline_mwh": float(
                    ordinary_group["daily_mean_load_mwh"].mean()
                ),
                "ordinary_weekday_mean_deviation_pct": float(
                    ordinary_deviations.mean()
                ),
                "weekday_public_holiday_mean_deviation_pct": float(
                    holiday_deviations.mean()
                ),
                "weekday_public_holiday_median_deviation_pct": float(
                    holiday_deviations.median()
                ),
                "weekday_public_holiday_first_quartile_pct": float(
                    holiday_deviations.quantile(0.25)
                ),
                "weekday_public_holiday_third_quartile_pct": float(
                    holiday_deviations.quantile(0.75)
                ),
            }
        )
    summary = pd.DataFrame(rows, columns=HOLIDAY_SUMMARY_COLUMNS)
    if summary.empty:
        raise ValueError("no weekday public holidays found")
    numeric = summary.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("public holiday summary contains non-finite values")
    return summary


def _render_temperature_relationship(
    daily: pd.DataFrame,
    temperature_bins: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        sharex=True,
        gridspec_kw={"hspace": 0.36},
    )
    figure.patch.set_facecolor("white")
    lower, upper = _common_temperature_range(daily)
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        country_daily = daily.loc[daily["country"].eq(country)]
        country_bins = temperature_bins.loc[temperature_bins["country"].eq(country)]
        axis.scatter(
            country_daily["daily_mean_temperature_c"],
            country_daily["daily_mean_load_mwh"],
            color=DAILY_OBSERVATION_COLOR,
            alpha=0.35,
            s=13,
            edgecolors="none",
        )
        if not country_bins.empty:
            axis.fill_between(
                country_bins["temperature_bin_midpoint_c"],
                country_bins["first_quartile_mwh"],
                country_bins["third_quartile_mwh"],
                color=INTERQUARTILE_COLOR,
                alpha=0.9,
            )
            axis.plot(
                country_bins["temperature_bin_midpoint_c"],
                country_bins["mean_daily_mean_load_mwh"],
                color=DAILY_MEAN_COLOR,
                marker="o",
                markersize=4,
                linewidth=1.8,
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
        axis.set_xlim(lower, upper)
        axis.yaxis.set_major_formatter(StrMethodFormatter(Y_AXIS_FORMAT))
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(GRID_COLOR)
        axis.spines["bottom"].set_color(GRID_COLOR)
    axes[-1].set_xlabel(
        "Daily mean population-weighted temperature [°C]",
        color=TEXT_COLOR,
        labelpad=8,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=DAILY_OBSERVATION_COLOR, marker="o", linestyle="", label="Daily observations"),
            Line2D([], [], color=DAILY_MEAN_COLOR, marker="o", linewidth=1.8, label="Mean load by temperature bin"),
            Patch(facecolor=INTERQUARTILE_COLOR, edgecolor="none", label="Interquartile range"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_public_holiday_effect(
    daily: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.0),
        gridspec_kw={"hspace": 0.38},
    )
    figure.patch.set_facecolor("white")
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        ordinary = daily.loc[
            daily["country"].eq(country)
            & daily["day_type"].eq("ordinary weekday"),
            "relative_load_deviation_pct",
        ].dropna()
        holidays = daily.loc[
            daily["country"].eq(country)
            & daily["day_type"].eq("weekday public holiday"),
            "relative_load_deviation_pct",
        ].dropna()
        category_labels = [
            f"Ordinary weekdays (n = {len(ordinary):,})",
            f"Weekday public holidays (n = {len(holidays):,})",
        ]
        box = axis.boxplot(
            [ordinary, holidays],
            tick_labels=category_labels,
            patch_artist=True,
            widths=0.68,
            boxprops={"color": TEXT_COLOR, "linewidth": 1.4},
            medianprops={"color": TEXT_COLOR, "linewidth": 1.3},
            whiskerprops={"color": TEXT_COLOR, "linewidth": 1.2},
            capprops={"color": TEXT_COLOR, "linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.35},
        )
        for patch, color in zip(box["boxes"], (ORDINARY_COLOR, HOLIDAY_COLOR)):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        axis.plot(
            [1, 2],
            [float(ordinary.mean()), float(holidays.mean())],
            linestyle="none",
            marker="D",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=TEXT_COLOR,
            markeredgewidth=1.0,
            zorder=4,
        )
        axis.axhline(0.0, color=TEXT_COLOR, linewidth=0.9, linestyle="--")
        axis.set_title(
            f"{panel} {country}",
            loc="left",
            pad=8,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        axis.set_ylabel("Relative load deviation [%]", color=TEXT_COLOR)
        axis.set_ylim(-35.0, 20.0)
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}"))
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(GRID_COLOR)
        axis.spines["bottom"].set_color(GRID_COLOR)
    axes[-1].set_xlabel("Day type", color=TEXT_COLOR, labelpad=8)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.12)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_formatted(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    output = frame.loc[:, columns].copy()
    if "local_date" in output:
        output["local_date"] = output["local_date"].map(
            lambda value: value.isoformat() if isinstance(value, date) else str(value)
        )
    for column in output.columns:
        if column in {"country", "local_date", "weekday", "holiday_name", "day_type", "month_name", "weekday_name"}:
            continue
        if pd.api.types.is_numeric_dtype(output[column]):
            if column in {
                "year",
                "month",
                "weekday",
                "weekday_number",
                "hourly_count",
                "public_holiday_flag",
                "ordinary_weekday_count",
                "weekday_public_holiday_count",
                "number_of_days",
            }:
                output[column] = output[column].astype("Int64")
            else:
                output[column] = output[column].map(
                    lambda value: "" if pd.isna(value) else f"{float(value):.2f}"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def _print_summary(
    validations: dict[str, dict[str, object]],
    daily: pd.DataFrame,
    temperature_bins: pd.DataFrame,
    holiday_summary: pd.DataFrame,
) -> None:
    coverage = ", ".join(
        f"{country}: {validations[country]['local_date_count']} dates"
        for country in COUNTRIES
    )
    temperature_min = daily["daily_mean_temperature_c"].min()
    temperature_max = daily["daily_mean_temperature_c"].max()
    holiday_count = int(
        daily["day_type"].eq("weekday public holiday").sum()
    )
    holiday_deviation = daily.loc[
        daily["day_type"].eq("weekday public holiday"),
        "relative_load_deviation_pct",
    ].mean()
    print(f"Section 5.4 coverage: {coverage}; total rows={len(daily)}")
    print(f"Temperature range: {temperature_min:.2f} to {temperature_max:.2f} °C")
    print(
        "Temperature bins: "
        f"{len(temperature_bins)} retained rows, n>=10, "
        f"{temperature_bins['temperature_bin_lower_c'].min():.0f} to "
        f"{temperature_bins['temperature_bin_upper_c'].max():.0f} °C"
    )
    print(
        "Weekday public holidays: "
        f"{holiday_count} daily rows; mean relative deviation={holiday_deviation:.2f}%"
    )
    print(f"Holiday summary rows: {len(holiday_summary)}")
    print("Protected files: Section 5.4 writes only its own output directory")


def run_section_5_4_temperature_public_holidays(
    output_directory: str | Path = OUTPUT_DIRECTORY,
) -> dict[str, object]:
    output_directory = Path(output_directory)
    aligned_by_country = {
        country: align_hourly_data(country) for country in COUNTRIES
    }
    validations = {
        country: validate_aligned_hourly_data(aligned, country)
        for country, aligned in aligned_by_country.items()
    }
    daily_by_country = {
        country: build_daily_temperature_public_holidays(aligned, country)
        for country, aligned in aligned_by_country.items()
    }
    daily = pd.concat(daily_by_country.values(), ignore_index=True)
    temperature_bins = build_temperature_bin_summary(daily)
    holiday_summary = build_public_holiday_effect_summary(daily)
    _write_formatted(
        daily,
        output_directory / "daily_temperature_public_holidays.csv",
        DAILY_COLUMNS,
    )
    _write_formatted(
        temperature_bins,
        output_directory / "temperature_bin_summary.csv",
        TEMPERATURE_BIN_COLUMNS,
    )
    _write_formatted(
        holiday_summary,
        output_directory / "public_holiday_effect_summary.csv",
        HOLIDAY_SUMMARY_COLUMNS,
    )
    _render_temperature_relationship(
        daily,
        temperature_bins,
        output_directory / "figure_5_7_temperature_load_relationship.png",
        output_directory / "figure_5_7_temperature_load_relationship.pdf",
    )
    _render_public_holiday_effect(
        daily,
        output_directory / "figure_5_8_public_holiday_effect.png",
        output_directory / "figure_5_8_public_holiday_effect.pdf",
    )
    return {
        "validation": validations,
        "aligned": aligned_by_country,
        "daily_by_country": daily_by_country,
        "daily": daily,
        "temperature_bin_summary": temperature_bins,
        "public_holiday_effect_summary": holiday_summary,
    }


if __name__ == "__main__":
    result = run_section_5_4_temperature_public_holidays()
    _print_summary(
        result["validation"],
        result["daily"],
        result["temperature_bin_summary"],
        result["public_holiday_effect_summary"],
    )
