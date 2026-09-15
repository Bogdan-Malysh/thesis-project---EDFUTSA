from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIRECTORY = PROJECT_ROOT / "results" / "tables"
FIGURE_DIRECTORY = PROJECT_ROOT / "results" / "figures"
FIGURE_DPI = 300

TEXT_COLOR = "#263238"
ACTUAL_COLOR = "#1F4E79"
FORECAST_COLOR = "#C9794C"
PROCESS_COLOR = "#5B7895"
LOOP_COLOR = "#6D8A6D"
MUTED_COLOR = "#EEF1F3"
GRID_COLOR = "#D9DEE3"
WHITE = "#FFFFFF"


def _table_61() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Element": "Forecast target",
                "Germany": "Hourly national electricity load",
                "Austria": "Hourly national electricity load",
            },
            {
                "Element": "Forecast origin",
                "Germany": "18:00 on the preceding day",
                "Austria": "08:00 on the preceding day",
            },
            {
                "Element": "Training period",
                "Germany": "2020-2023",
                "Austria": "2020-2023",
            },
            {
                "Element": "Validation period",
                "Germany": "2024",
                "Austria": "2024",
            },
            {
                "Element": "Test period",
                "Germany": "2025",
                "Austria": "2025",
            },
            {
                "Element": "Timestamp handling",
                "Germany": "Preserve complete timestamps, including 23- and 25-hour daylight-saving days",
                "Austria": "Preserve complete timestamps, including 23- and 25-hour daylight-saving days",
            },
            {
                "Element": "Information constraint",
                "Germany": "Only information available at the forecast origin may be used",
                "Austria": "Only information available at the forecast origin may be used",
            },
        ]
    )


def _table_62() -> pd.DataFrame:
    calendar_predictors = (
        "Hour of day; day of week; month; country-specific public holidays. "
        "No separate weekend dummy when day-of-week variables are used."
    )
    lagged_target = "Lagged target only"
    return pd.DataFrame(
        [
            {
                "Model family": "Naïve",
                "Candidate variants": "Last available observation",
                "Seasonal structure": "None",
                "Predictor set": lagged_target,
            },
            {
                "Model family": "Daily seasonal naïve",
                "Candidate variants": "Fixed 24-hour lag",
                "Seasonal structure": "Daily (24 hours)",
                "Predictor set": lagged_target,
            },
            {
                "Model family": "Weekly seasonal naïve",
                "Candidate variants": "Fixed 168-hour lag",
                "Seasonal structure": "Weekly (168 hours)",
                "Predictor set": lagged_target,
            },
            {
                "Model family": "Hour-of-week profile",
                "Candidate variants": "Historical hour-of-week profile",
                "Seasonal structure": "Weekly hour-of-week profile",
                "Predictor set": "Historical target profile",
            },
            {
                "Model family": "Holt-Winters",
                "Candidate variants": "Additive and multiplicative; trend and damped-trend variants",
                "Seasonal structure": "Daily (24-hour) and weekly (168-hour) seasonality",
                "Predictor set": lagged_target,
            },
            {
                "Model family": "ARIMA",
                "Candidate variants": "Order (p,d,q) selected using training and validation data",
                "Seasonal structure": "None",
                "Predictor set": lagged_target,
            },
            {
                "Model family": "SARIMA",
                "Candidate variants": "Candidate non-seasonal and seasonal orders",
                "Seasonal structure": "Seasonal period is 24 hours. Weekly effects are represented through weekly baselines or calendar predictors rather than a seasonal period of 168.",
                "Predictor set": lagged_target,
            },
            {
                "Model family": "Regression with ARIMA errors",
                "Candidate variants": "Calendar-predictor regression with ARIMA errors",
                "Seasonal structure": "Model-dependent",
                "Predictor set": calendar_predictors,
            },
            {
                "Model family": "SARIMAX",
                "Candidate variants": "Calendar-predictor regression with seasonal ARIMA errors",
                "Seasonal structure": "Model-dependent",
                "Predictor set": calendar_predictors,
            },
        ]
    )


TABLE_62_NOTE = (
    "Temperature is considered only as an optional extension if genuine archived "
    "day-ahead temperature forecasts are available."
)


def _write_tables() -> None:
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table_61 = _table_61()
    table_62 = _table_62()
    table_61.to_csv(TABLE_DIRECTORY / "table_6_1_forecasting_tasks.csv", index=False)
    table_62.to_csv(TABLE_DIRECTORY / "table_6_2_candidate_models.csv", index=False)
    with (TABLE_DIRECTORY / "table_6_2_candidate_models.csv").open(
        "a", encoding="utf-8", newline=""
    ) as output:
        output.write(f'"Note: {TABLE_62_NOTE}",,,\n')


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    number: str,
    title: str,
    body: str,
    color: str,
    body_y: float | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=WHITE,
        edgecolor=color,
        linewidth=1.7,
        transform=ax.transAxes,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012,
        y + height - 0.035,
        number,
        color=color,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
        zorder=4,
    )
    ax.text(
        x + width / 2 + 0.012,
        y + height - 0.028,
        title,
        color=TEXT_COLOR,
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="top",
        transform=ax.transAxes,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.012 if body_y is None else body_y,
        body,
        color=TEXT_COLOR,
        fontsize=8.8,
        linespacing=1.35,
        ha="center",
        va="center",
        transform=ax.transAxes,
        zorder=4,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = TEXT_COLOR,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.35,
            color=color,
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
            zorder=2,
        )
    )


def _interval_ribbon(ax: plt.Axes, x: float, y: float, width: float, count: int) -> None:
    segment_width = width / 25
    for index in range(count):
        ax.add_patch(
            Rectangle(
                (x + index * segment_width, y),
                segment_width * 0.86,
                0.018,
                facecolor=FORECAST_COLOR,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=4,
            )
        )


def _draw_figure() -> None:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
        }
    )
    fig = plt.figure(figsize=(16, 9.2))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(
        FancyBboxPatch(
            (0.018, 0.425),
            0.964,
            0.43,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#FAFBFC",
            edgecolor=GRID_COLOR,
            linewidth=1.0,
            transform=ax.transAxes,
            zorder=0,
        )
    )
    ax.text(
        0.035,
        0.845,
        "ROLLING EXPANDING-WINDOW LOOP",
        fontsize=8.7,
        fontweight="bold",
        color="#6B777E",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    width = 0.274
    height = 0.145
    _box(
        ax,
        0.042,
        0.68,
        width,
        height,
        "1",
        "Set forecast origin",
        "Germany: 18:00\nAustria: 08:00\n(on the preceding day)",
        PROCESS_COLOR,
    )
    _box(
        ax,
        0.363,
        0.68,
        width,
        height,
        "2",
        "Construct information set",
        "Available load observations\n+ predictors known in advance\nNo future actual load or observed future temperature",
        PROCESS_COLOR,
    )
    _box(
        ax,
        0.684,
        0.68,
        width,
        height,
        "3",
        "Generate continuous forecast path",
        "From the first unavailable interval\nthrough the end of the target day",
        FORECAST_COLOR,
    )
    _box(
        ax,
        0.684,
        0.475,
        width,
        height,
        "4",
        "Use bridge forecasts where necessary",
        "Bridge the interval between the last available\nobservation and the target day when necessary.",
        FORECAST_COLOR,
    )
    _box(
        ax,
        0.363,
        0.475,
        width,
        height,
        "5",
        "Freeze and store",
        "Store all target-day forecasts\nwithout later revision",
        LOOP_COLOR,
    )
    _box(
        ax,
        0.042,
        0.475,
        width,
        height,
        "6",
        "Move to the next origin",
        "Add newly available actual values\nto the expanding window and repeat",
        LOOP_COLOR,
    )

    _arrow(ax, (0.316, 0.752), (0.359, 0.752), color=PROCESS_COLOR)
    _arrow(ax, (0.637, 0.752), (0.680, 0.752), color=PROCESS_COLOR)
    _arrow(ax, (0.821, 0.675), (0.821, 0.625), color=FORECAST_COLOR)
    _arrow(ax, (0.680, 0.548), (0.641, 0.548), color=FORECAST_COLOR)
    _arrow(ax, (0.359, 0.548), (0.320, 0.548), color=LOOP_COLOR)
    _arrow(
        ax,
        (0.179, 0.628),
        (0.179, 0.672),
        color=LOOP_COLOR,
        connectionstyle="arc3,rad=0.0",
    )
    ax.text(
        0.194,
        0.651,
        "repeat",
        fontsize=8.5,
        color=LOOP_COLOR,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.018, 0.065),
            0.964,
            0.32,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#FAFBFC",
            edgecolor=GRID_COLOR,
            linewidth=1.0,
            transform=ax.transAxes,
            zorder=0,
        )
    )
    ax.text(
        0.035,
        0.357,
        "EVALUATION AFTER OUTCOMES ARE OBSERVED",
        fontsize=8.7,
        fontweight="bold",
        color="#6B777E",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.332,
        "Forecasts are evaluated only after the target outcomes are observed",
        fontsize=10.2,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    _box(
        ax,
        0.048,
        0.11,
        0.255,
        0.19,
        "",
        "Target-day timestamp set",
        "Target days may contain 23, 24 or 25\ncomplete local-hour intervals",
        PROCESS_COLOR,
        body_y=0.225,
    )
    _interval_ribbon(ax, 0.078, 0.159, 0.195, 23)
    _interval_ribbon(ax, 0.078, 0.136, 0.195, 24)
    _interval_ribbon(ax, 0.078, 0.113, 0.195, 25)
    for y, label in ((0.168, "23"), (0.145, "24"), (0.122, "25")):
        ax.text(
            0.064,
            y,
            label,
            fontsize=7.6,
            color=TEXT_COLOR,
            ha="right",
            va="center",
            transform=ax.transAxes,
        )

    _box(
        ax,
        0.357,
        0.145,
        0.255,
        0.14,
        "",
        "Match complete timestamps",
        "Match the frozen target-day forecasts\nto SMARD actual load and the archived\nofficial forecast using complete timestamps.",
        PROCESS_COLOR,
    )
    _box(
        ax,
        0.692,
        0.20,
        0.255,
        0.105,
        "",
        "Observed outcome",
        "SMARD actual load",
        ACTUAL_COLOR,
    )
    _box(
        ax,
        0.692,
        0.08,
        0.255,
        0.105,
        "",
        "External benchmark",
        "Archived official forecast",
        FORECAST_COLOR,
    )
    _arrow(ax, (0.612, 0.215), (0.685, 0.252), color=ACTUAL_COLOR)
    _arrow(ax, (0.612, 0.185), (0.685, 0.132), color=FORECAST_COLOR)
    _arrow(ax, (0.50, 0.462), (0.50, 0.392), color=LOOP_COLOR)
    ax.text(
        0.516,
        0.425,
        "retain forecast archive",
        fontsize=8.1,
        color=LOOP_COLOR,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    fig.savefig(
        FIGURE_DIRECTORY / "figure_6_1_rolling_day_ahead_forecasting.png",
        dpi=FIGURE_DPI,
        bbox_inches=Bbox.from_bounds(0, 0, 16, 8.1),
        pad_inches=0.05,
        facecolor=WHITE,
    )
    plt.close(fig)


def main() -> None:
    _write_tables()
    _draw_figure()


if __name__ == "__main__":
    main()
