from __future__ import annotations

from pathlib import Path

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
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "chapter5" / "section_5_2"
EXPECTED_ROWS = 52_608
HOURS = tuple(range(24))
HOUR_TICKS = tuple(range(0, 24, 2))
FIGURE_DPI = 300
INPUT_COLUMNS = ["actual_grid_load_mwh", "hour"]
COUNTRIES = ["Germany", "Austria"]
COUNTRY_INPUTS = {
    "Germany": PROCESSED / "modelling_germany_hourly.csv",
    "Austria": PROCESSED / "modelling_austria_hourly.csv",
}
PROFILE_COLUMNS = [
    "country",
    "hour",
    "number_of_observations",
    "mean_hourly_grid_load_mwh",
    "median_hourly_grid_load_mwh",
    "standard_deviation_mwh",
    "first_quartile_mwh",
    "third_quartile_mwh",
]
TEXT_COLOR = "#263238"
GRID_COLOR = "#D9DEE3"
DAILY_COLOR = "#A6CEE3"
MEAN_COLOR = "#0072B2"
Y_AXIS_FORMAT = "{x:,.0f}"


def load_country_data(
    country: str,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    if country not in COUNTRY_INPUTS:
        raise ValueError(f"unsupported country: {country}")
    path = Path(processed_directory) / COUNTRY_INPUTS[country].name
    data = pd.read_csv(path, usecols=INPUT_COLUMNS)
    return data[INPUT_COLUMNS]


def _validated_components(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if list(data.columns) != INPUT_COLUMNS:
        raise ValueError("Section 5.2 input columns do not match the contract")
    actual = pd.to_numeric(data["actual_grid_load_mwh"], errors="coerce")
    if actual.isna().any() or not np.isfinite(actual.to_numpy()).all():
        raise ValueError("actual_grid_load_mwh is not complete and finite")
    hours = pd.to_numeric(data["hour"], errors="coerce")
    if hours.isna().any() or not np.isfinite(hours.to_numpy()).all():
        raise ValueError("hour is not complete and finite")
    if not np.equal(hours.to_numpy(), hours.to_numpy().astype(int)).all():
        raise ValueError("hour contains non-integer values")
    hours = hours.astype(int)
    if not hours.between(0, 23).all():
        raise ValueError("hour must contain values from 0 through 23")
    return actual, hours


def validate_country_data(data: pd.DataFrame, country: str) -> dict[str, object]:
    _, hours = _validated_components(data)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"{country} input must contain {EXPECTED_ROWS} rows")
    counts = hours.value_counts().reindex(HOURS, fill_value=0).astype(int)
    if (counts == 0).any():
        raise ValueError(f"{country} does not contain all local hours 0-23")
    if not counts.eq(EXPECTED_ROWS // 24).all():
        raise ValueError(f"{country} local-hour counts are not complete")
    return {
        "country": country,
        "row_count": int(len(data)),
        "hours": list(HOURS),
        "hour_counts": counts.tolist(),
        "finite_actual_load": True,
    }


def build_hourly_profile(data: pd.DataFrame, country: str) -> pd.DataFrame:
    actual, hours = _validated_components(data)
    working = pd.DataFrame(
        {"hour": hours.to_numpy(), "actual_grid_load_mwh": actual.to_numpy()}
    )
    rows = []
    for hour in sorted(hours.unique().tolist()):
        values = working.loc[working["hour"].eq(hour), "actual_grid_load_mwh"]
        if values.empty:
            raise ValueError(f"{country} has no observations for local hour {hour}")
        rows.append(
            {
                "country": country,
                "hour": hour,
                "number_of_observations": int(values.count()),
                "mean_hourly_grid_load_mwh": float(values.mean()),
                "median_hourly_grid_load_mwh": float(values.median()),
                "standard_deviation_mwh": float(values.std(ddof=1)),
                "first_quartile_mwh": float(values.quantile(0.25)),
                "third_quartile_mwh": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def build_profile_table(data_by_country: dict[str, pd.DataFrame]) -> pd.DataFrame:
    profiles = [
        build_hourly_profile(data_by_country[country], country)
        for country in COUNTRIES
        if country in data_by_country
    ]
    table = pd.concat(profiles, ignore_index=True)
    numeric_columns = PROFILE_COLUMNS[2:]
    if table[numeric_columns].isna().any().any():
        raise ValueError("hourly demand profile contains missing statistics")
    if not np.isfinite(table[numeric_columns].to_numpy()).all():
        raise ValueError("hourly demand profile contains non-finite statistics")
    return table


def _write_profile_table(profile: pd.DataFrame, path: Path) -> None:
    formatted = profile.copy()
    formatted["number_of_observations"] = formatted[
        "number_of_observations"
    ].astype(int)
    for column in PROFILE_COLUMNS[3:]:
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.2f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted.to_csv(path, index=False)


def _render_figure(profile: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
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
        figsize=(8.2, 6.2),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    figure.patch.set_facecolor("white")
    for axis, country, panel in zip(axes, COUNTRIES, ("(a)", "(b)")):
        country_profile = profile.loc[profile["country"].eq(country)].sort_values("hour")
        axis.set_facecolor("white")
        axis.fill_between(
            country_profile["hour"],
            country_profile["first_quartile_mwh"],
            country_profile["third_quartile_mwh"],
            color=DAILY_COLOR,
            alpha=0.55,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            country_profile["hour"],
            country_profile["mean_hourly_grid_load_mwh"],
            color=MEAN_COLOR,
            linewidth=1.8,
            zorder=2,
        )
        axis.set_ylabel("Hourly grid load [MWh]", color=TEXT_COLOR)
        axis.yaxis.set_major_formatter(StrMethodFormatter(Y_AXIS_FORMAT))
        axis.set_title(
            f"{panel} {country}",
            loc="left",
            pad=7,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        axis.set_xlim(-0.25, 23.25)
        axis.set_xticks(HOUR_TICKS)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.grid(axis="x", visible=False)
        axis.tick_params(axis="both", colors=TEXT_COLOR, length=0, pad=5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(GRID_COLOR)

    axes[-1].set_xlabel("Local hour", color=TEXT_COLOR, labelpad=8)
    figure.legend(
        handles=[
            Line2D([], [], color=MEAN_COLOR, linewidth=1.8, label="Mean hourly grid load"),
            Patch(facecolor=DAILY_COLOR, edgecolor="none", alpha=0.55, label="Interquartile range"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        labelcolor=TEXT_COLOR,
        handlelength=2.4,
        columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.12)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_section_5_2(output_directory: str | Path = OUTPUT_DIRECTORY) -> dict[str, object]:
    output_directory = Path(output_directory)
    data_by_country = {
        country: load_country_data(country) for country in COUNTRIES
    }
    validation = {
        country: validate_country_data(data, country)
        for country, data in data_by_country.items()
    }
    profile = build_profile_table(data_by_country)
    _write_profile_table(profile, output_directory / "hourly_demand_profile.csv")
    _render_figure(
        profile,
        output_directory / "figure_5_2_hourly_demand_patterns.png",
        output_directory / "figure_5_2_hourly_demand_patterns.pdf",
    )
    return {"profile": profile, "validation": validation}


if __name__ == "__main__":
    run_section_5_2()
