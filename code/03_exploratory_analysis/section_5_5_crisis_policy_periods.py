from __future__ import annotations

from calendar import month_name
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_5"
EXPECTED_ROWS = 52_608
EXPECTED_DAILY_ROWS = 2_192
FIGURE_DPI = 300
COUNTRIES = ["Germany", "Austria"]
MONTHS = tuple(range(1, 13))
MONTH_NAMES = tuple(month_name[month] for month in MONTHS)
MONTH_ABBREVIATIONS = tuple(name[:3] for name in MONTH_NAMES)
YEARS = tuple(range(2020, 2026))
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 31)
COVID_START = date(2020, 1, 1)
COVID_END = date(2020, 6, 30)
PANDEMIC_DECLARATION = date(2020, 3, 11)
POST_BASELINE_YEAR = 2021
POST_START = date(2022, 1, 1)
POST_END = date(2025, 12, 31)
INVASION_DATE = date(2022, 2, 24)
INPUT_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "actual_grid_load_mwh",
    "forecasted_grid_load_mwh",
    "forecasted_grid_load_valid",
]
COUNTRY_INPUTS = {
    "Germany": PROCESSED / "modelling_germany_hourly.csv",
    "Austria": PROCESSED / "modelling_austria_hourly.csv",
}
DAILY_COLUMNS = [
    "country",
    "local_date",
    "year",
    "month",
    "number_of_hourly_observations",
    "number_of_forecast_observations",
    "daily_mean_actual_load_mwh",
    "daily_mean_forecast_load_mwh",
]
COVID_DAILY_COLUMNS = [
    "country",
    "local_date",
    "year",
    "month",
    "daily_mean_actual_load_mwh",
    "daily_mean_forecast_load_mwh",
    "relative_deviation_pct",
    "trailing_7_day_mean_deviation_pct",
    "period",
]
COVID_SUMMARY_COLUMNS = [
    "country",
    "period",
    "period_start",
    "period_end",
    "n",
    "mean_deviation_pct",
    "median_deviation_pct",
    "mean_absolute_deviation_pct",
]
MONTHLY_COLUMNS = [
    "country",
    "year",
    "month",
    "month_name",
    "number_of_days",
    "monthly_mean_actual_load_mwh",
]
POST_INVASION_MONTHLY_COLUMNS = [
    "country",
    "year",
    "month",
    "month_name",
    "monthly_mean_actual_load_mwh",
    "monthly_mean_actual_load_mwh_2021",
    "monthly_deviation_pct",
    "is_transition_month",
    "summary_period",
]
POST_INVASION_SUMMARY_COLUMNS = [
    "country",
    "period",
    "period_start",
    "period_end",
    "n",
    "mean_monthly_deviation_pct",
    "median_monthly_deviation_pct",
]
COVID_PERIODS = (
    ("1 January-10 March 2020", date(2020, 1, 1), date(2020, 3, 10)),
    ("11 March-30 June 2020", date(2020, 3, 11), date(2020, 6, 30)),
)
KNOWN_INVALID_FORECAST_DATES = {
    "Germany": frozenset({date(2020, 1, 31)}),
    "Austria": frozenset(),
}
POST_SUMMARY_PERIODS = (
    ("January 2022", date(2022, 1, 1), date(2022, 1, 31)),
    ("March-December 2022", date(2022, 3, 1), date(2022, 12, 31)),
    ("2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("2025", date(2025, 1, 1), date(2025, 12, 31)),
)
ANALYSIS_METADATA = {
    "forecast_role": "Official day-ahead forecast is an external benchmark only and is not a model input.",
    "forecast_coverage_exception": "Germany has 24 invalid official forecast intervals on local 31 January 2020; that day is excluded from forecast-deviation calculations without imputation. Austria has no invalid forecast intervals.",
    "local_date_rule": "Hourly observations are grouped by the date in timestamp_local.",
    "dst_rule": "All available local-hour observations are retained, so local dates contain 23, 24 or 25 observations around DST transitions.",
    "rolling_rule": "The COVID line is a trailing seven-calendar-day mean on a DatetimeIndex; the first rows use the available dates within the window, and no future observations are used.",
    "covid_measure": "Relative deviation measures departure from the contemporary official forecast, not a causal COVID-19 effect.",
    "covid_event": "11 March 2020 is marked as the WHO pandemic declaration date.",
    "post_invasion_comparison": "The 2021 month-matched comparison controls for month but not temperature, economic conditions or other contemporaneous factors; it is descriptive rather than causal.",
    "post_invasion_event": "24 February 2022 is marked; January and February 2022 are retained for context, while February is excluded from aggregate summaries as a transition month.",
}
TEXT_COLOR = "#263238"
GRID_COLOR = "#D9DEE3"
NEGATIVE_COLOR = "#B85C5C"
POSITIVE_COLOR = "#4F7FA8"
CONTEXT_COLOR = "#D9DEE3"
DARK_LINE_COLOR = "#263238"


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
        dtype={
            "timestamp_utc": "string",
            "timestamp_local": "string",
            "forecasted_grid_load_valid": "string",
        },
    )
    return data[INPUT_COLUMNS]


def _validated_components(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex, pd.Series]:
    if list(data.columns) != INPUT_COLUMNS:
        raise ValueError("crisis/policy input columns do not match the contract")
    utc = _utc_index(data["timestamp_utc"], "timestamp_utc")
    actual = pd.to_numeric(data["actual_grid_load_mwh"], errors="coerce")
    forecast = pd.to_numeric(data["forecasted_grid_load_mwh"], errors="coerce")
    valid = data["forecasted_grid_load_valid"].astype("string").str.lower().eq("true")
    if actual.isna().any() or not np.isfinite(actual.to_numpy()).all():
        raise ValueError("actual_grid_load_mwh is not complete and finite")
    invalid = ~valid
    if invalid.any() and forecast.loc[invalid].notna().any():
        raise ValueError("invalid forecast flags must have missing forecast values")
    valid_forecast = forecast.loc[valid]
    if valid_forecast.isna().any() or not np.isfinite(valid_forecast.to_numpy()).all():
        raise ValueError("valid forecast observations are not complete and finite")
    if (valid_forecast <= 0).any():
        raise ValueError("valid forecast observations must be positive denominators")
    local_dates: list[date] = []
    for value in data["timestamp_local"]:
        if not isinstance(value, str):
            raise ValueError("timestamp_local contains missing or invalid values")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp_local must be timezone-aware")
        local_dates.append(timestamp.date())
    return actual, forecast, utc, pd.Series(local_dates, index=data.index, name="local_date")


def validate_country_data(data: pd.DataFrame, country: str) -> dict[str, object]:
    actual, forecast, utc, local_dates = _validated_components(data)
    valid = data["forecasted_grid_load_valid"].astype("string").str.lower().eq("true")
    valid_forecast = forecast.loc[valid]
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"{country} input must contain {EXPECTED_ROWS} rows")
    _validate_utc_index(utc, f"{country} UTC key")
    if local_dates.nunique() != EXPECTED_DAILY_ROWS:
        raise ValueError(f"{country} must contain {EXPECTED_DAILY_ROWS} local dates")
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="D").date
    if sorted(local_dates.unique().tolist()) != list(expected_dates):
        raise ValueError(f"{country} local dates do not cover 2020-2025 completely")
    daily_counts = local_dates.value_counts()
    if not set(daily_counts.tolist()) <= {23, 24, 25}:
        raise ValueError(f"{country} local daily counts are outside 23-25")
    return {
        "country": country,
        "row_count": int(len(data)),
        "local_date_count": int(local_dates.nunique()),
        "daily_observation_counts": sorted(daily_counts.unique().tolist()),
        "utc_unique": bool(utc.is_unique),
        "utc_hourly_continuous": bool(
            len(utc) < 2 or (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all()
        ),
        "forecast_valid_observation_count": int(valid.sum()),
        "forecast_invalid_observation_count": int((~valid).sum()),
        "finite_actual_load": bool(np.isfinite(actual.to_numpy()).all()),
        "finite_forecast_load": bool(np.isfinite(valid_forecast.to_numpy()).all()),
        "positive_forecast_denominator": bool((valid_forecast > 0).all()),
    }


def build_daily_actual_forecast(data: pd.DataFrame, country: str) -> pd.DataFrame:
    validate_country_data(data, country)
    actual, forecast, _, local_dates = _validated_components(data)
    hourly = pd.DataFrame(
        {
            "local_date": local_dates,
            "actual_grid_load_mwh": actual,
            "forecasted_grid_load_mwh": forecast,
        }
    )
    daily = (
        hourly.groupby("local_date", sort=True)
        .agg(
            number_of_hourly_observations=("actual_grid_load_mwh", "count"),
            number_of_forecast_observations=("forecasted_grid_load_mwh", "count"),
            daily_mean_actual_load_mwh=("actual_grid_load_mwh", "mean"),
            daily_mean_forecast_load_mwh=("forecasted_grid_load_mwh", "mean"),
        )
        .reset_index()
    )
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="D").date
    if len(daily) != EXPECTED_DAILY_ROWS or daily["local_date"].tolist() != list(
        expected_dates
    ):
        raise ValueError(f"{country} daily observations do not cover 2020-2025")
    daily["country"] = country
    daily["year"] = daily["local_date"].map(lambda value: value.year).astype(int)
    daily["month"] = daily["local_date"].map(lambda value: value.month).astype(int)
    daily = daily.loc[:, DAILY_COLUMNS]
    actual_values = daily["daily_mean_actual_load_mwh"].to_numpy()
    forecast_values = daily["daily_mean_forecast_load_mwh"].dropna().to_numpy()
    if not np.isfinite(actual_values).all() or not np.isfinite(forecast_values).all():
        raise ValueError(f"{country} daily actual/forecast values are not finite")
    return daily


def _covid_period(value: date) -> str:
    for label, start, end in COVID_PERIODS:
        if start <= value <= end:
            return label
    raise ValueError(f"date outside COVID analysis period: {value}")


def build_covid_daily_deviations(
    daily: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    if list(daily.columns) != DAILY_COLUMNS:
        raise ValueError("daily actual/forecast columns do not match the contract")
    covid_all = daily.loc[
        daily["local_date"].between(COVID_START, COVID_END)
    ].copy()
    expected_dates = pd.date_range(COVID_START, COVID_END, freq="D").date
    if covid_all["local_date"].tolist() != list(expected_dates):
        raise ValueError(f"{country} COVID daily dates are incomplete or unordered")
    available_dates = set(
        covid_all.loc[covid_all["daily_mean_forecast_load_mwh"].notna(), "local_date"]
    )
    missing_dates = set(expected_dates) - available_dates
    if missing_dates != set(KNOWN_INVALID_FORECAST_DATES[country]):
        raise ValueError(
            f"{country} COVID forecast coverage differs from the validated exception: "
            f"{sorted(missing_dates)}"
        )
    covid = covid_all.loc[
        covid_all["daily_mean_forecast_load_mwh"].notna()
    ].copy()
    denominator = covid["daily_mean_forecast_load_mwh"]
    if (denominator <= 0).any() or not np.isfinite(denominator.to_numpy()).all():
        raise ValueError(f"{country} COVID forecast denominator is invalid")
    covid["country"] = country
    covid["relative_deviation_pct"] = (
        covid["daily_mean_actual_load_mwh"] / denominator - 1.0
    ) * 100.0
    deviation_by_date = covid.set_index(pd.DatetimeIndex(covid["local_date"]))[
        "relative_deviation_pct"
    ]
    covid["trailing_7_day_mean_deviation_pct"] = (
        deviation_by_date.rolling(window="7D", min_periods=1).mean().to_numpy()
    )
    covid["period"] = covid["local_date"].map(_covid_period)
    result = covid.loc[:, COVID_DAILY_COLUMNS]
    numeric = result.select_dtypes(include=np.number)
    if result.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{country} COVID deviations contain missing or non-finite values")
    return result


def build_covid_period_summary(covid: pd.DataFrame) -> pd.DataFrame:
    if list(covid.columns) != COVID_DAILY_COLUMNS:
        raise ValueError("COVID daily columns do not match the contract")
    rows = []
    for country in COUNTRIES:
        for period, start, end in COVID_PERIODS:
            values = covid.loc[
                covid["country"].eq(country) & covid["period"].eq(period),
                "relative_deviation_pct",
            ]
            if values.empty:
                raise ValueError(f"no COVID observations for {country} {period}")
            rows.append(
                {
                    "country": country,
                    "period": period,
                    "period_start": start,
                    "period_end": end,
                    "n": int(values.count()),
                    "mean_deviation_pct": float(values.mean()),
                    "median_deviation_pct": float(values.median()),
                    "mean_absolute_deviation_pct": float(values.abs().mean()),
                }
            )
    summary = pd.DataFrame(rows, columns=COVID_SUMMARY_COLUMNS)
    _validate_numeric_frame(summary, "COVID summary")
    return summary


def build_monthly_actual_load(
    daily_by_country: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for country in COUNTRIES:
        daily = daily_by_country[country]
        if list(daily.columns) != DAILY_COLUMNS:
            raise ValueError("daily actual/forecast columns do not match the contract")
        for year in range(2021, 2026):
            for month in MONTHS:
                values = daily.loc[
                    daily["year"].eq(year) & daily["month"].eq(month),
                    "daily_mean_actual_load_mwh",
                ]
                if values.empty:
                    raise ValueError(f"no daily actual load for {country} {year}-{month:02d}")
                rows.append(
                    {
                        "country": country,
                        "year": year,
                        "month": month,
                        "month_name": MONTH_NAMES[month - 1],
                        "number_of_days": int(values.count()),
                        "monthly_mean_actual_load_mwh": float(values.mean()),
                    }
                )
    monthly = pd.DataFrame(rows, columns=MONTHLY_COLUMNS)
    _validate_numeric_frame(monthly, "monthly actual load")
    return monthly


def _post_summary_period(year: int, month: int) -> str:
    if year == 2022 and month == 1:
        return "January 2022"
    if year == 2022 and month == 2:
        return "February 2022"
    if year == 2022:
        return "March-December 2022"
    return str(year)


def build_post_invasion_monthly_deviations(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    if list(monthly.columns) != MONTHLY_COLUMNS:
        raise ValueError("monthly actual-load columns do not match the contract")
    rows = []
    for country in COUNTRIES:
        baseline = monthly.loc[
            monthly["country"].eq(country) & monthly["year"].eq(POST_BASELINE_YEAR)
        ].set_index("month")
        for year in range(2022, 2026):
            for month in MONTHS:
                current = monthly.loc[
                    monthly["country"].eq(country)
                    & monthly["year"].eq(year)
                    & monthly["month"].eq(month)
                ]
                if len(current) != 1 or month not in baseline.index:
                    raise ValueError(f"missing month-matched data for {country} {year}-{month:02d}")
                current_row = current.iloc[0]
                baseline_value = float(baseline.loc[month, "monthly_mean_actual_load_mwh"])
                current_value = float(current_row["monthly_mean_actual_load_mwh"])
                rows.append(
                    {
                        "country": country,
                        "year": year,
                        "month": month,
                        "month_name": MONTH_NAMES[month - 1],
                        "monthly_mean_actual_load_mwh": current_value,
                        "monthly_mean_actual_load_mwh_2021": baseline_value,
                        "monthly_deviation_pct": (current_value / baseline_value - 1.0)
                        * 100.0,
                        "is_transition_month": bool(year == 2022 and month == 2),
                        "summary_period": _post_summary_period(year, month),
                    }
                )
    deviations = pd.DataFrame(rows, columns=POST_INVASION_MONTHLY_COLUMNS)
    _validate_numeric_frame(deviations, "post-invasion monthly deviations")
    return deviations


def build_post_invasion_period_summary(
    deviations: pd.DataFrame,
) -> pd.DataFrame:
    if list(deviations.columns) != POST_INVASION_MONTHLY_COLUMNS:
        raise ValueError("post-invasion monthly columns do not match the contract")
    rows = []
    for country in COUNTRIES:
        for period, start, end in POST_SUMMARY_PERIODS:
            values = deviations.loc[
                deviations["country"].eq(country)
                & deviations["summary_period"].eq(period),
                "monthly_deviation_pct",
            ]
            if values.empty:
                raise ValueError(f"no post-invasion observations for {country} {period}")
            rows.append(
                {
                    "country": country,
                    "period": period,
                    "period_start": start,
                    "period_end": end,
                    "n": int(values.count()),
                    "mean_monthly_deviation_pct": float(values.mean()),
                    "median_monthly_deviation_pct": float(values.median()),
                }
            )
    summary = pd.DataFrame(rows, columns=POST_INVASION_SUMMARY_COLUMNS)
    _validate_numeric_frame(summary, "post-invasion summary")
    return summary


def _validate_numeric_frame(frame: pd.DataFrame, label: str) -> None:
    numeric = frame.select_dtypes(include=np.number)
    if frame.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{label} contains missing or non-finite values")


def _common_symmetric_limit(*frames: pd.Series) -> float:
    values = np.concatenate([pd.to_numeric(frame).to_numpy() for frame in frames])
    if not np.isfinite(values).all():
        raise ValueError("figure values contain non-finite observations")
    maximum = float(np.max(np.abs(values)))
    return max(1.0, float(np.ceil(maximum * 1.12)))


def _style_axis(axis) -> None:
    axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=4)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID_COLOR)
    axis.spines["bottom"].set_color(GRID_COLOR)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.1f}"))


def _render_covid_event_strip(
    covid: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.6),
        sharex=True,
        gridspec_kw={"hspace": 0.16},
    )
    figure.patch.set_facecolor("white")
    y_limit = _common_symmetric_limit(
        covid["relative_deviation_pct"],
        covid["trailing_7_day_mean_deviation_pct"],
    )
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        country_data = covid.loc[covid["country"].eq(country)].copy()
        full_dates = pd.date_range(COVID_START, COVID_END, freq="D")
        country_data = country_data.set_index(
            pd.DatetimeIndex(country_data["local_date"])
        ).reindex(full_dates)
        country_dates = full_dates
        deviations = country_data["relative_deviation_pct"]
        colors = np.where(
            deviations.isna(), "none", np.where(deviations.to_numpy() < 0, NEGATIVE_COLOR, POSITIVE_COLOR)
        )
        axis.bar(
            country_dates,
            deviations,
            width=0.82,
            color=colors.tolist(),
            edgecolor="none",
            linewidth=0,
            alpha=0.82,
            zorder=2,
        )
        axis.plot(
            country_dates,
            country_data["trailing_7_day_mean_deviation_pct"],
            color=DARK_LINE_COLOR,
            linewidth=1.25,
            zorder=3,
        )
        for period, start, end in COVID_PERIODS:
            period_data = country_data.loc[country_data["period"].eq(period)]
            mean_value = float(period_data["relative_deviation_pct"].mean())
            axis.plot(
                [pd.Timestamp(start), pd.Timestamp(end)],
                [mean_value, mean_value],
                color=DARK_LINE_COLOR,
                linewidth=2.4,
                solid_capstyle="butt",
                zorder=4,
            )
            midpoint = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            offset = 0.06 * y_limit if mean_value >= 0 else -0.06 * y_limit
            axis.text(
                midpoint,
                mean_value + offset,
                f"{mean_value:+.1f}%",
                ha="center",
                va="bottom" if mean_value >= 0 else "top",
                color=DARK_LINE_COLOR,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
            )
        axis.axhline(0.0, color=TEXT_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        axis.axvline(
            pd.Timestamp(PANDEMIC_DECLARATION),
            color=TEXT_COLOR,
            linewidth=0.8,
            linestyle=":",
            zorder=1,
        )
        axis.set_title(
            f"{panel} {country}",
            loc="left",
            pad=6,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        axis.set_ylabel("Actual–forecast deviation [%]", color=TEXT_COLOR)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_xlim(
            pd.Timestamp(COVID_START) - pd.Timedelta(days=3),
            pd.Timestamp(COVID_END) + pd.Timedelta(days=3),
        )
        _style_axis(axis)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel("Local date", color=TEXT_COLOR, labelpad=7)
    axes[0].text(
        pd.Timestamp(PANDEMIC_DECLARATION) + pd.Timedelta(days=2),
        y_limit * 0.88,
        "WHO pandemic declaration, 11 Mar 2020",
        ha="left",
        va="top",
        color=TEXT_COLOR,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
    )
    figure.legend(
        handles=[
            Patch(facecolor=NEGATIVE_COLOR, edgecolor="none", label="Actual below forecast"),
            Patch(facecolor=POSITIVE_COLOR, edgecolor="none", label="Actual above forecast"),
            Line2D([], [], color=DARK_LINE_COLOR, linewidth=1.25, label="Trailing 7-day mean"),
            Line2D([], [], color=DARK_LINE_COLOR, linewidth=2.4, label="Period mean"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _render_post_invasion_fingerprint(
    deviations: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.labelsize": 8})
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(10.0, 7.5),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.13, "wspace": 0.12},
    )
    figure.patch.set_facecolor("white")
    y_limit = _common_symmetric_limit(deviations["monthly_deviation_pct"])
    for row_index, year in enumerate(range(2022, 2026)):
        for column_index, country in enumerate(COUNTRIES):
            axis = axes[row_index, column_index]
            country_year = deviations.loc[
                deviations["country"].eq(country) & deviations["year"].eq(year)
            ].sort_values("month")
            if len(country_year) != 12:
                raise ValueError(f"{country} {year} does not contain 12 monthly deviations")
            values = country_year["monthly_deviation_pct"].to_numpy()
            colors = np.where(values < 0, NEGATIVE_COLOR, POSITIVE_COLOR).astype(object)
            if year == 2022:
                colors[:2] = CONTEXT_COLOR
            axis.bar(
                country_year["month"],
                values,
                width=0.72,
                color=colors.tolist(),
                edgecolor="none",
                linewidth=0,
                alpha=0.85,
            )
            axis.axhline(0.0, color=TEXT_COLOR, linewidth=0.8, linestyle="--")
            mean_values = values[2:] if year == 2022 else values
            mean_value = float(mean_values.mean())
            axis.plot(
                13.25,
                mean_value,
                linestyle="none",
                marker="D",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=TEXT_COLOR,
                markeredgewidth=1.0,
                clip_on=False,
                zorder=4,
            )
            axis.text(
                13.57,
                mean_value,
                f"{mean_value:+.2f}%",
                ha="left",
                va="center",
                color=TEXT_COLOR,
                fontsize=7,
                clip_on=False,
            )
            if year == 2022:
                axis.axvline(
                    2.0,
                    color=TEXT_COLOR,
                    linewidth=0.8,
                    linestyle=":",
                    zorder=1,
                )
                if country == "Germany":
                    axis.text(
                        2.15,
                        y_limit * 0.82,
                        "24 Feb 2022",
                        ha="left",
                        va="top",
                        color=TEXT_COLOR,
                        fontsize=7,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
                    )
            if column_index == 0:
                axis.set_ylabel(str(year), rotation=0, ha="right", va="center", labelpad=15)
            else:
                axis.set_ylabel("")
            if row_index == 0:
                axis.set_title(
                    f"({'a' if column_index == 0 else 'b'}) {country}",
                    loc="left",
                    pad=6,
                    fontsize=10,
                    color=TEXT_COLOR,
                    fontweight="normal",
                )
            axis.set_ylim(-y_limit, y_limit)
            axis.set_xlim(0.45, 14.6)
            axis.set_xticks(MONTHS)
            axis.set_xticklabels(MONTH_ABBREVIATIONS)
            axis.tick_params(axis="x", labelbottom=row_index == 3)
            _style_axis(axis)
    axes[3, 0].set_xlabel("Month", color=TEXT_COLOR, labelpad=6)
    axes[3, 1].set_xlabel("Month", color=TEXT_COLOR, labelpad=6)
    figure.text(
        0.01,
        0.5,
        "Deviation from corresponding month of 2021 [%]",
        ha="center",
        va="center",
        rotation=90,
        color=TEXT_COLOR,
        fontsize=8,
    )
    figure.legend(
        handles=[
            Patch(facecolor=NEGATIVE_COLOR, edgecolor="none", label="Below 2021"),
            Patch(facecolor=POSITIVE_COLOR, edgecolor="none", label="Above 2021"),
            Patch(facecolor=CONTEXT_COLOR, edgecolor="none", label="Pre-event/transition context"),
            Line2D([], [], color=TEXT_COLOR, marker="D", linestyle="none", label="Period mean"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.10, right=0.96, top=0.89, bottom=0.10)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_formatted(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    output = frame.loc[:, columns].copy()
    date_columns = {"local_date", "period_start", "period_end"}
    text_columns = {"country", "local_date", "period", "period_start", "period_end", "month_name", "summary_period"}
    for column in date_columns:
        if column in output:
            output[column] = output[column].map(
                lambda value: value.isoformat() if isinstance(value, date) else str(value)
            )
    if "is_transition_month" in output:
        output["is_transition_month"] = output["is_transition_month"].astype(int)
    integer_columns = {
        "year",
        "month",
        "number_of_hourly_observations",
        "number_of_days",
        "n",
        "is_transition_month",
    }
    for column in output.columns:
        if column in text_columns:
            continue
        if pd.api.types.is_numeric_dtype(output[column]):
            if column in integer_columns:
                output[column] = output[column].astype(int)
            else:
                output[column] = output[column].map(lambda value: f"{float(value):.2f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def _print_summary(
    validation: dict[str, dict[str, object]],
    covid_summary: pd.DataFrame,
    post_summary: pd.DataFrame,
) -> None:
    coverage = ", ".join(
        f"{country}: {validation[country]['local_date_count']} dates"
        for country in COUNTRIES
    )
    print(f"Section 5.5 coverage: {coverage}")
    print(
        "COVID daily rows: "
        f"{covid_summary.groupby('country')['n'].sum().to_dict()}"
    )
    print(f"COVID summary rows: {len(covid_summary)}")
    print(f"Post-invasion summary rows: {len(post_summary)}")
    print("Official forecasts are used as external descriptive benchmarks only")


def run_section_5_5_crisis_policy_periods(
    output_directory: str | Path = OUTPUT_DIRECTORY,
) -> dict[str, object]:
    output_directory = Path(output_directory)
    data_by_country = {country: load_country_data(country) for country in COUNTRIES}
    validation = {
        country: validate_country_data(data_by_country[country], country)
        for country in COUNTRIES
    }
    daily_by_country = {
        country: build_daily_actual_forecast(data_by_country[country], country)
        for country in COUNTRIES
    }
    covid_by_country = {
        country: build_covid_daily_deviations(daily_by_country[country], country)
        for country in COUNTRIES
    }
    covid = pd.concat(covid_by_country.values(), ignore_index=True)
    covid_summary = build_covid_period_summary(covid)
    monthly = build_monthly_actual_load(daily_by_country)
    post_deviations = build_post_invasion_monthly_deviations(monthly)
    post_summary = build_post_invasion_period_summary(post_deviations)
    _write_formatted(
        covid,
        output_directory / "covid19_daily_deviations.csv",
        COVID_DAILY_COLUMNS,
    )
    _write_formatted(
        covid_summary,
        output_directory / "covid19_period_summary.csv",
        COVID_SUMMARY_COLUMNS,
    )
    _write_formatted(
        post_deviations,
        output_directory / "post_invasion_monthly_deviations.csv",
        POST_INVASION_MONTHLY_COLUMNS,
    )
    _write_formatted(
        post_summary,
        output_directory / "post_invasion_period_summary.csv",
        POST_INVASION_SUMMARY_COLUMNS,
    )
    _render_covid_event_strip(
        covid,
        output_directory / "figure_5_9_covid19_event_strip.png",
        output_directory / "figure_5_9_covid19_event_strip.pdf",
    )
    _render_post_invasion_fingerprint(
        post_deviations,
        output_directory / "figure_5_10_post_invasion_fingerprint.png",
        output_directory / "figure_5_10_post_invasion_fingerprint.pdf",
    )
    return {
        "metadata": ANALYSIS_METADATA,
        "validation": validation,
        "data_by_country": data_by_country,
        "daily_by_country": daily_by_country,
        "covid19_daily_deviations": covid,
        "covid19_period_summary": covid_summary,
        "monthly_actual_load": monthly,
        "post_invasion_monthly_deviations": post_deviations,
        "post_invasion_period_summary": post_summary,
    }


if __name__ == "__main__":
    result = run_section_5_5_crisis_policy_periods()
    _print_summary(
        result["validation"],
        result["covid19_period_summary"],
        result["post_invasion_period_summary"],
    )
