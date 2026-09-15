from __future__ import annotations

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from matplotlib.path import Path as MatplotlibPath
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_ERA5 = PROJECT_ROOT / "data" / "raw" / "era5_land"
RAW_GISCO = PROJECT_ROOT / "data" / "raw" / "gisco_boundaries"
FIGURES = PROJECT_ROOT / "results" / "figures"
BOUNDARY_PATH = RAW_GISCO / "CNTR_RG_01M_2024_4326.geojson"
OUTPUT_PATH = FIGURES / "temperature_population_cold_hour.png"
OUTPUT_DPI = 300
FIGURE_TITLE = "Temperature Field and Population Weighting in Germany and Austria"

COUNTRIES = ("Germany", "Austria")
COUNTRY_CODES = {"Germany": "DE", "Austria": "AT"}
COUNTRY_TIMEZONES = {"Germany": "Europe/Berlin", "Austria": "Europe/Vienna"}
TEXT_COLOR = "#263238"
GRID_COLOR = "#D9DEE3"
BUBBLE_COLOR = "#20252B"
BUBBLE_SCALE = 8000.0
MAP_ASPECT = "equal"
LEGEND_TITLE = "Share of national population"

plt.rcParams["font.family"] = "serif"


def _load_temperature_module() -> Any:
    module_path = PROJECT_ROOT / "code" / "02_preprocessing" / "temperature_pipeline.py"
    spec = spec_from_file_location("temperature_pipeline_for_map", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load temperature pipeline: {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_processed_temperature(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        usecols=["interval_start_utc", "temperature_c", "temperature_valid"],
    )
    data["interval_start_utc"] = pd.to_datetime(
        data["interval_start_utc"], utc=True
    )
    data["temperature_c"] = pd.to_numeric(data["temperature_c"], errors="coerce")
    data = data.loc[
        data["temperature_valid"].astype(bool)
        & np.isfinite(data["temperature_c"])
    ].copy()
    return data


def _parse_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def select_cold_timestamp(
    germany: pd.DataFrame,
    austria: pd.DataFrame,
    override: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, dict[str, float]]:
    """Select the common joint-coldest valid UTC timestamp from 2020-2025."""
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2025-12-31 23:00", tz="UTC")
    germany_data = germany.copy()
    austria_data = austria.copy()
    for data in (germany_data, austria_data):
        data["interval_start_utc"] = pd.to_datetime(
            data["interval_start_utc"], utc=True
        )
        data["temperature_c"] = pd.to_numeric(data["temperature_c"], errors="coerce")
    joined = germany_data[
        ["interval_start_utc", "temperature_c"]
    ].merge(
        austria_data[["interval_start_utc", "temperature_c"]],
        on="interval_start_utc",
        how="inner",
        validate="one_to_one",
        suffixes=("_germany", "_austria"),
    )
    joined = joined.loc[
        joined["interval_start_utc"].between(start, end, inclusive="both")
    ].copy()
    if joined.empty:
        raise ValueError("no common valid temperature timestamp exists in 2020-2025")
    if not np.isfinite(
        joined[["temperature_c_germany", "temperature_c_austria"]].to_numpy()
    ).all():
        raise ValueError("common temperature selection contains non-finite values")

    by_timestamp = joined.set_index("interval_start_utc")
    if override is None:
        joint_mean = by_timestamp[
            ["temperature_c_germany", "temperature_c_austria"]
        ].mean(axis=1)
        selected = joint_mean.idxmin()
    else:
        selected = _parse_utc_timestamp(override)
        if selected not in by_timestamp.index:
            raise ValueError(f"UTC override is not a common valid timestamp: {selected}")
    row = by_timestamp.loc[selected]
    return selected, {
        "Germany": float(row["temperature_c_germany"]),
        "Austria": float(row["temperature_c_austria"]),
    }


def mask_grid_to_geometry(
    values: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    geometry: Any,
) -> np.ndarray:
    if values.shape != (len(latitudes), len(longitudes)):
        raise ValueError("temperature grid shape does not match its coordinates")
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    inside = contains_xy(geometry, longitude_grid, latitude_grid)
    return np.where(inside, values, np.nan)


def mapped_grid_values(values: np.ndarray, weights: pd.DataFrame) -> np.ndarray:
    return values[
        weights["latitude_index"].to_numpy(),
        weights["longitude_index"].to_numpy(),
    ]


def validate_plot_data(
    temperatures: dict[str, np.ndarray], weights: dict[str, pd.DataFrame]
) -> None:
    for country in COUNTRIES:
        if country not in temperatures or country not in weights:
            raise ValueError(f"missing plot data for {country}")
        values = np.asarray(temperatures[country], dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"displayed {country} temperatures are not finite")
        weight_values = pd.to_numeric(weights[country]["weight"], errors="coerce")
        if weight_values.empty or not np.isfinite(weight_values).all():
            raise ValueError(f"{country} weights are not finite")
        if not np.isclose(weight_values.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"{country} weight sum is not one")


def validate_color_scale(
    temperatures: dict[str, np.ndarray], vmin: float, vmax: float
) -> dict[str, int]:
    values = np.concatenate(
        [
            np.asarray(data, dtype=float)[np.isfinite(data)]
            for data in temperatures.values()
        ]
    )
    below_min = int((values < vmin).sum())
    above_max = int((values > vmax).sum())
    return {
        "below_min": below_min,
        "above_max": above_max,
        "total": int(values.size),
    }


def bubble_area(weight: float) -> float:
    return BUBBLE_SCALE * float(weight)


def bubble_marker_size(weight: float) -> float:
    return float(np.sqrt(bubble_area(weight)))


def population_legend_values(weights: dict[str, pd.DataFrame]) -> tuple[float, ...]:
    maximum_shared_weight = min(
        float(pd.to_numeric(weights[country]["weight"], errors="coerce").max())
        for country in COUNTRIES
    )
    candidate_values = np.array([0.001, 0.005, 0.008, 0.01, 0.02, 0.05])
    supported = candidate_values[candidate_values <= maximum_shared_weight]
    if len(supported) >= 3:
        return tuple(float(value) for value in supported[-3:])
    fallback = np.linspace(maximum_shared_weight / 4, maximum_shared_weight * 0.9, 3)
    return tuple(float(value) for value in fallback)


def _load_country_boundaries() -> dict[str, Any]:
    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(BOUNDARY_PATH)
    with BOUNDARY_PATH.open(encoding="utf-8") as handle:
        document = json.load(handle)
    geometries: dict[str, list[Any]] = {code: [] for code in COUNTRY_CODES.values()}
    for feature in document.get("features", []):
        code = feature.get("properties", {}).get("CNTR_ID")
        if code in geometries:
            geometries[code].append(shape(feature["geometry"]))
    missing = [code for code, values in geometries.items() if not values]
    if missing:
        raise ValueError(f"GISCO boundary file lacks country codes: {missing}")
    return {code: unary_union(values) for code, values in geometries.items()}


def _era5_grid(
    path: Path,
    timestamp: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperature_module = _load_temperature_module()
    dataset, time_name, variable_name = temperature_module._open_era5_temperature(path)
    try:
        values = dataset[variable_name]
        latitude_name = "latitude" if "latitude" in values.dims else "lat"
        longitude_name = "longitude" if "longitude" in values.dims else "lon"
        values = values.transpose(time_name, latitude_name, longitude_name)
        if timestamp is None:
            time_index = 0
        else:
            timestamps = pd.DatetimeIndex(
                pd.to_datetime(dataset[time_name].values, utc=True)
            )
            matches = np.flatnonzero(timestamps == timestamp)
            if len(matches) != 1:
                raise ValueError(f"{timestamp} is not uniquely present in {path.name}")
            time_index = int(matches[0])
        selected = values.isel({time_name: time_index}).to_numpy().astype(float)
        return (
            dataset[latitude_name].to_numpy(),
            dataset[longitude_name].to_numpy(),
            selected - 273.15,
        )
    finally:
        dataset.close()


def _exact_weights(
    temperature_module: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, pd.DataFrame], dict[str, float]]:
    first_path = RAW_ERA5 / "era5_land_2019_boundary.nc"
    latitudes, longitudes, first_values = _era5_grid(first_path)
    finite_mask = np.isfinite(first_values)
    weights: dict[str, pd.DataFrame] = {}
    populations: dict[str, float] = {}
    for country in COUNTRIES:
        weights[country], populations[country] = temperature_module.build_weights(
            country,
            latitudes,
            longitudes,
            valid_grid_mask=finite_mask,
        )
    return latitudes, longitudes, weights, populations


def _plot_boundary(ax: plt.Axes, geometry: Any) -> None:
    polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        exterior = np.asarray(polygon.exterior.coords)
        ax.plot(
            exterior[:, 0],
            exterior[:, 1],
            color="#17212B",
            linewidth=1.0,
            zorder=5,
        )
        for interior in polygon.interiors:
            ring = np.asarray(interior.coords)
            ax.plot(ring[:, 0], ring[:, 1], color="#17212B", linewidth=0.7, zorder=5)


def _geometry_path(geometry: Any) -> MatplotlibPath:
    polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for polygon in polygons:
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coordinates = np.asarray(ring.coords)
            vertices.extend((float(x), float(y)) for x, y in coordinates)
            codes.extend(
                [MatplotlibPath.MOVETO]
                + [MatplotlibPath.LINETO] * (len(coordinates) - 2)
                + [MatplotlibPath.CLOSEPOLY]
            )
    return MatplotlibPath(np.asarray(vertices), np.asarray(codes, dtype=np.uint8))


def _country_extent(geometry: Any) -> tuple[float, float, float, float]:
    west, south, east, north = geometry.bounds
    longitude_padding = max((east - west) * 0.06, 0.25)
    latitude_padding = max((north - south) * 0.06, 0.2)
    return (
        west - longitude_padding,
        east + longitude_padding,
        south - latitude_padding,
        north + latitude_padding,
    )


def _local_timestamp(timestamp: pd.Timestamp, country: str) -> str:
    return timestamp.tz_convert(COUNTRY_TIMEZONES[country]).strftime(
        "%Y-%m-%d %H:%M %Z"
    )


def _local_time_summary(timestamp: pd.Timestamp) -> str:
    germany = timestamp.tz_convert(COUNTRY_TIMEZONES["Germany"])
    austria = timestamp.tz_convert(COUNTRY_TIMEZONES["Austria"])
    if germany != austria:
        raise ValueError("Germany and Austria local timestamps do not match")
    return (
        "Local time in Germany and Austria: "
        f"{germany.strftime('%d %B %Y, %H:%M %Z')}"
    )


def _draw_map(
    timestamp: pd.Timestamp,
    national_temperatures: dict[str, float],
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
    grid_values: dict[str, np.ndarray],
    selected_values: np.ndarray,
    weights: dict[str, pd.DataFrame],
    geometries: dict[str, Any],
    legend_values: tuple[float, ...],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    figure.patch.set_facecolor("white")
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in grid_values.values()]
    )
    lower = float(np.floor(finite_values.min()))
    upper = float(np.ceil(finite_values.max()))
    if lower == upper:
        upper = lower + 1.0
    normalization = Normalize(vmin=lower, vmax=upper)
    meshes = []
    for ax, country in zip(axes, COUNTRIES):
        ax.set_facecolor("#F7F9FA")
        mesh = ax.pcolormesh(
            grid_longitudes,
            grid_latitudes,
            grid_values[country],
            cmap="RdBu_r",
            norm=normalization,
            shading="auto",
            rasterized=True,
            zorder=1,
        )
        meshes.append(mesh)
        boundary_geometry = geometries[COUNTRY_CODES[country]]
        mesh.set_clip_path(
            PathPatch(_geometry_path(boundary_geometry), transform=ax.transData)
        )
        _plot_boundary(ax, boundary_geometry)
        country_weights = weights[country]
        latitude_indices = country_weights["latitude_index"].to_numpy()
        longitude_indices = country_weights["longitude_index"].to_numpy()
        bubble_temperatures = mapped_grid_values(selected_values, country_weights)
        if not np.isfinite(bubble_temperatures).all():
            raise ValueError(f"mapped {country} temperatures contain non-finite values")
        ax.scatter(
            grid_longitudes[longitude_indices],
            grid_latitudes[latitude_indices],
            s=[bubble_area(value) for value in country_weights["weight"]],
            facecolor=BUBBLE_COLOR,
            edgecolor="white",
            linewidth=0.18,
            alpha=0.36,
            zorder=4,
        )
        west, east, south, north = _country_extent(geometries[COUNTRY_CODES[country]])
        ax.set_xlim(west, east, auto=True)
        ax.set_ylim(south, north, auto=True)
        ax.set_aspect(MAP_ASPECT, adjustable="datalim")
        ax.set_title(
            country,
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
            pad=8,
            fontfamily="serif",
        )
        ax.set_xlabel("Longitude (°)", color=TEXT_COLOR, labelpad=5)
        ax.set_ylabel("Latitude (°)", color=TEXT_COLOR, labelpad=5)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8, length=0, pad=4)
        ax.grid(color=GRID_COLOR, linewidth=0.55, alpha=0.7, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)

    figure.suptitle(
        FIGURE_TITLE,
        fontsize=16,
        fontfamily="serif",
        color=TEXT_COLOR,
        y=0.975,
    )
    figure.text(
        0.5,
        0.265,
        _local_time_summary(timestamp),
        ha="center",
        fontsize=8.5,
        color=TEXT_COLOR,
        fontfamily="serif",
    )
    colorbar_axis = figure.add_axes([0.28, 0.215, 0.44, 0.022])
    colorbar = figure.colorbar(meshes[0], cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("ERA5-Land 2 m temperature (°C)", color=TEXT_COLOR, labelpad=3)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8, length=0, pad=3)
    colorbar.outline.set_edgecolor(GRID_COLOR)

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=bubble_marker_size(value),
            markerfacecolor=BUBBLE_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.45,
            alpha=0.55,
            label=f"{value:.1%}",
        )
        for value in legend_values
    ]
    figure.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        title=LEGEND_TITLE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=3,
        frameon=False,
        labelcolor=TEXT_COLOR,
        title_fontproperties={"family": "serif", "size": 9},
        prop={"family": "serif", "size": 8.5},
        handletextpad=0.45,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.34, wspace=0.17)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def generate_figure(override: str | pd.Timestamp | None = None) -> dict[str, object]:
    temperature_module = _load_temperature_module()
    germany = _read_processed_temperature(
        PROCESSED / "temperature_germany_hourly.csv"
    )
    austria = _read_processed_temperature(
        PROCESSED / "temperature_austria_hourly.csv"
    )
    timestamp, national_temperatures = select_cold_timestamp(
        germany, austria, override=override
    )
    era5_path = RAW_ERA5 / f"era5_land_{timestamp.year}_{timestamp.month:02d}.nc"
    if not era5_path.exists():
        raise FileNotFoundError(era5_path)
    grid_latitudes, grid_longitudes, selected_values = _era5_grid(era5_path, timestamp)
    _, _, exact_weights, populations = _exact_weights(temperature_module)
    boundaries = _load_country_boundaries()
    grid_values: dict[str, np.ndarray] = {}
    temperature_summary: dict[str, dict[str, float]] = {}
    weighted_comparison: dict[str, dict[str, float]] = {}
    for country in COUNTRIES:
        grid_values[country] = mask_grid_to_geometry(
            selected_values,
            grid_latitudes,
            grid_longitudes,
            boundaries[COUNTRY_CODES[country]],
        )
        country_weights = exact_weights[country]
        mapped_values = mapped_grid_values(selected_values, country_weights)
        weighted_value = float(
            mapped_values @ country_weights["weight"].to_numpy()
        )
        finite_values = grid_values[country][np.isfinite(grid_values[country])]
        temperature_summary[country] = {
            "minimum_c": float(finite_values.min()),
            "maximum_c": float(finite_values.max()),
            "population_weighted_c": weighted_value,
        }
        weighted_comparison[country] = {
            "plotted_cells_c": weighted_value,
            "processed_csv_c": national_temperatures[country],
            "difference_c": weighted_value - national_temperatures[country],
        }
        if not np.isclose(weighted_value, national_temperatures[country], atol=1e-6):
            raise ValueError(
                f"{country} mapped temperature disagrees with processed output: "
                f"{weighted_value} versus {national_temperatures[country]}"
            )
    displayed_values = {
        country: np.concatenate(
            [
                values[np.isfinite(values)],
                mapped_grid_values(selected_values, exact_weights[country]),
            ]
        )
        for country, values in grid_values.items()
    }
    validate_plot_data(displayed_values, exact_weights)
    color_scale_minimum = float(
        np.floor(min(summary["minimum_c"] for summary in temperature_summary.values()))
    )
    color_scale_maximum = float(
        np.ceil(max(summary["maximum_c"] for summary in temperature_summary.values()))
    )
    color_scale_validation = validate_color_scale(
        grid_values, color_scale_minimum, color_scale_maximum
    )
    if color_scale_validation["below_min"] or color_scale_validation["above_max"]:
        raise ValueError("temperature values are clipped by the shared colour scale")
    legend_values = population_legend_values(exact_weights)
    _draw_map(
        timestamp,
        national_temperatures,
        grid_latitudes,
        grid_longitudes,
        grid_values,
        selected_values,
        exact_weights,
        boundaries,
        legend_values,
    )
    return {
        "output": str(OUTPUT_PATH),
        "selected_timestamp_utc": timestamp.isoformat(),
        "selection_method": (
            "minimum of the joint mean of the two processed national temperatures "
            "across common valid hourly timestamps from 2020-01-01 through 2025-12-31"
        ),
        "local_time_summary": _local_time_summary(timestamp),
        "local_timestamps": {
            country: _local_timestamp(timestamp, country) for country in COUNTRIES
        },
        "national_temperature_c": national_temperatures,
        "temperature_summary": temperature_summary,
        "weighted_temperature_comparison": weighted_comparison,
        "population_total": populations,
        "weight_sum": {
            country: float(exact_weights[country]["weight"].sum())
            for country in COUNTRIES
        },
        "mapped_cell_count": {
            country: int(len(exact_weights[country])) for country in COUNTRIES
        },
        "maximum_population_weight": {
            country: float(exact_weights[country]["weight"].max())
            for country in COUNTRIES
        },
        "legend_values": list(legend_values),
        "color_scale": {
            "minimum_c": color_scale_minimum,
            "maximum_c": color_scale_maximum,
            **color_scale_validation,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestamp",
        help="optional common UTC timestamp override, for example 2021-02-13T06:00:00Z",
    )
    arguments = parser.parse_args()
    print(json.dumps(generate_figure(arguments.timestamp), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
