from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_APG = PROJECT_ROOT / "data" / "raw" / "apg"
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "results" / "figures"

ACTUAL_COLOR = "#1F4E79"
FORECAST_COLOR = "#C9794C"
SMARD_ACTUAL_COLOR = "#4C78A8"
SMARD_FORECAST_COLOR = "#9ECAE1"
APG_ACTUAL_COLOR = "#D17A45"
APG_FORECAST_COLOR = "#E6B17E"
GRID_COLOR = "#D9DEE3"
TEXT_COLOR = "#263238"

SMARD_WEEK_START = pd.Timestamp("2021-01-11 00:00")
SMARD_WEEK_END = pd.Timestamp("2021-01-18 00:00")
APG_DAY = pd.Timestamp("2021-01-13").date()


def _load_apg_module():
    module_path = Path(__file__).with_name("apg_hourly.py")
    spec = spec_from_file_location("chapter4_apg_hourly", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load APG preprocessing module: {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _local_coverage(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    data = pd.read_csv(path, usecols=["interval_start_local", "interval_end_local"])
    starts = pd.to_datetime(data["interval_start_local"])
    ends = pd.to_datetime(data["interval_end_local"])
    return starts.min(), ends.max()


def _apg_raw_coverage(
    module: object, filenames: Iterable[str], dataset_type: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    first_utc: pd.Timestamp | None = None
    final_utc: pd.Timestamp | None = None
    for filename in filenames:
        loaded = module.load_apg_file(RAW_APG / filename, dataset_type)
        data = loaded["data"]
        start = data["interval_start_utc"].min()
        end = data["interval_end_utc"].max()
        first_utc = start if first_utc is None else min(first_utc, start)
        final_utc = end if final_utc is None else max(final_utc, end)
    if first_utc is None or final_utc is None:
        raise ValueError(f"no APG coverage found for {dataset_type}")
    timezone = module.TIMEZONE_NAME
    return (
        first_utc.tz_convert(timezone).tz_localize(None),
        final_utc.tz_convert(timezone).tz_localize(None),
    )


def _coverage_bar(
    ax: plt.Axes,
    y: float,
    coverage: tuple[pd.Timestamp, pd.Timestamp],
    color: str,
) -> None:
    start, end = coverage
    left = mdates.date2num(start.to_pydatetime())
    width = mdates.date2num(end.to_pydatetime()) - left
    ax.barh(
        y,
        width,
        left=left,
        height=0.22,
        color=color,
        edgecolor=color,
        linewidth=0.8,
        zorder=3,
    )


def draw_data_coverage() -> None:
    germany = _local_coverage(PROCESSED / "smard_germany_hourly.csv")
    austria = _local_coverage(PROCESSED / "smard_austria_hourly.csv")
    apg_module = _load_apg_module()
    apg_actual = _apg_raw_coverage(apg_module, apg_module.ACTUAL_FILES, "actual")
    apg_forecast = _apg_raw_coverage(
        apg_module, apg_module.FORECAST_FILES, "forecast"
    )

    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    rows = [
        (3, "SMARD Germany\nhourly", germany, germany, SMARD_ACTUAL_COLOR, SMARD_FORECAST_COLOR),
        (2, "SMARD Austria\nhourly", austria, austria, SMARD_ACTUAL_COLOR, SMARD_FORECAST_COLOR),
        (1, "APG Austria actual\n15-minute raw", apg_actual, None, APG_ACTUAL_COLOR, None),
        (0, "APG Austria forecast\n15-minute raw", apg_forecast, None, APG_FORECAST_COLOR, None),
    ]
    for y, _, actual, forecast, actual_color, forecast_color in rows:
        _coverage_bar(ax, y + (0.12 if forecast is not None else 0), actual, actual_color)
        if forecast is not None and forecast_color is not None:
            _coverage_bar(ax, y - 0.12, forecast, forecast_color)

    year_ticks = pd.date_range("2020-01-01", "2026-01-01", freq="YS")
    for boundary in year_ticks:
        ax.axvline(boundary, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2026-01-01"))
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([row[1] for row in rows], color=TEXT_COLOR)
    ax.set_xticks(year_ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Calendar year", color=TEXT_COLOR, labelpad=8)
    ax.tick_params(axis="x", colors=TEXT_COLOR, length=0, pad=7)
    ax.tick_params(axis="y", length=0, pad=10, labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.legend(
        handles=[
            patches.Patch(color=SMARD_ACTUAL_COLOR, label="SMARD actual"),
            patches.Patch(color=SMARD_FORECAST_COLOR, label="SMARD forecast"),
            patches.Patch(color=APG_ACTUAL_COLOR, label="APG actual"),
            patches.Patch(color=APG_FORECAST_COLOR, label="APG forecast"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.5,
        labelcolor=TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.25, right=0.98, top=0.82, bottom=0.18)
    _save_figure(fig, "chapter4_data_coverage")


def _load_smard_week(path: Path) -> pd.DataFrame:
    columns = [
        "interval_start_local",
        "actual_grid_load_mwh",
        "forecasted_grid_load_mwh",
        "actual_grid_load_valid",
        "forecasted_grid_load_valid",
    ]
    data = pd.read_csv(path, usecols=columns)
    data["local_time"] = pd.to_datetime(data.pop("interval_start_local"))
    in_week = data["local_time"].between(
        SMARD_WEEK_START, SMARD_WEEK_END, inclusive="left"
    )
    valid = (
        data["actual_grid_load_valid"].astype(bool)
        & data["forecasted_grid_load_valid"].astype(bool)
        & data["actual_grid_load_mwh"].notna()
        & data["forecasted_grid_load_mwh"].notna()
    )
    selected = data.loc[in_week & valid].copy()
    if len(selected) != 168:
        raise ValueError(f"expected 168 valid SMARD rows, found {len(selected)}")
    return selected.sort_values("local_time")


def _format_time_axis(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.tick_params(axis="x", colors=TEXT_COLOR, length=0, pad=6)
    ax.tick_params(axis="y", colors=TEXT_COLOR, length=0, pad=5)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)


def draw_smard_week_example() -> None:
    germany = _load_smard_week(PROCESSED / "smard_germany_hourly.csv")
    austria = _load_smard_week(PROCESSED / "smard_austria_hourly.csv")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.4),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    fig.patch.set_facecolor("white")

    for ax, data, panel_label in zip(
        axes,
        (germany, austria),
        ("(a) Germany", "(b) Austria"),
    ):
        ax.set_facecolor("white")
        ax.plot(
            data["local_time"],
            data["actual_grid_load_mwh"],
            color=ACTUAL_COLOR,
            linewidth=1.15,
            label="Actual grid load",
        )
        ax.plot(
            data["local_time"],
            data["forecasted_grid_load_mwh"],
            color=FORECAST_COLOR,
            linewidth=1.05,
            linestyle="--",
            label="Official forecast",
        )
        ax.set_ylabel("MWh", color=TEXT_COLOR)
        ax.set_title(
            panel_label,
            loc="left",
            pad=7,
            fontsize=10,
            color=TEXT_COLOR,
            fontweight="normal",
        )
        _format_time_axis(ax, SMARD_WEEK_START, SMARD_WEEK_END)

    axes[-1].set_xlabel("Local time", color=TEXT_COLOR, labelpad=8)
    fig.legend(
        handles=[
            plt.Line2D([], [], color=ACTUAL_COLOR, linewidth=1.15, label="Actual grid load"),
            plt.Line2D(
                [],
                [],
                color=FORECAST_COLOR,
                linewidth=1.05,
                linestyle="--",
                label="Official forecast",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
        labelcolor=TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.14)
    _save_figure(fig, "chapter4_smard_week_example")


def _load_apg_day() -> tuple[pd.DataFrame, pd.DataFrame]:
    apg_module = _load_apg_module()
    raw = apg_module.load_apg_file(
        RAW_APG / "apg_austria_actual_2021_15min.csv", "actual"
    )["data"]
    raw_local = raw["interval_start_local"].dt.tz_localize(None)
    raw = raw.assign(local_time=raw_local)
    raw = raw.loc[raw["local_time"].dt.date == APG_DAY].copy()
    if len(raw) != 96:
        raise ValueError(f"expected 96 APG quarter-hours, found {len(raw)}")

    processed = pd.read_csv(
        PROCESSED / "apg_austria_hourly.csv",
        usecols=["interval_start_local", "actual_load_mwh"],
    )
    processed["local_time"] = (
        pd.to_datetime(
            processed.pop("interval_start_local"), format="mixed", utc=True
        )
        .dt.tz_convert("Europe/Vienna")
        .dt.tz_localize(None)
    )
    processed = processed.loc[processed["local_time"].dt.date == APG_DAY].copy()
    if len(processed) != 24:
        raise ValueError(f"expected 24 APG hourly observations, found {len(processed)}")
    return raw.sort_values("local_time"), processed.sort_values("local_time")


def draw_apg_resolution_example() -> None:
    raw, processed = _load_apg_day()
    start = pd.Timestamp("2021-01-13 00:00")
    end = pd.Timestamp("2021-01-14 00:00")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    fig.patch.set_facecolor("white")

    raw_ax, hourly_ax = axes
    raw_ax.set_facecolor("white")
    raw_ax.plot(
        raw["local_time"],
        raw["value_mw"],
        color=ACTUAL_COLOR,
        linewidth=0.8,
        marker="o",
        markersize=1.8,
        label="15-minute actual load",
    )
    raw_ax.set_ylabel("MW", color=TEXT_COLOR)
    raw_ax.set_title(
        "(a) Original 15-minute power",
        loc="left",
        pad=7,
        fontsize=10,
        color=TEXT_COLOR,
        fontweight="normal",
    )

    hourly_ax.set_facecolor("white")
    hourly_ax.step(
        list(processed["local_time"]) + [end],
        list(processed["actual_load_mwh"]) + [processed["actual_load_mwh"].iloc[-1]],
        where="post",
        color=FORECAST_COLOR,
        linewidth=1.25,
        label="Hourly actual energy",
    )
    hourly_ax.set_ylabel("MWh", color=TEXT_COLOR)
    hourly_ax.set_title(
        "(b) Processed hourly energy",
        loc="left",
        pad=7,
        fontsize=10,
        color=TEXT_COLOR,
        fontweight="normal",
    )

    for ax in axes:
        _format_time_axis(ax, start, end)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("Local time (Europe/Vienna)", color=TEXT_COLOR, labelpad=8)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.14)
    _save_figure(fig, "chapter4_apg_resolution_example")


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
        }
    )
    draw_data_coverage()
    draw_smard_week_example()
    draw_apg_resolution_example()
    print("Created three Chapter 4 figures in results/figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
