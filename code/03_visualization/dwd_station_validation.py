from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ERA5 = PROJECT_ROOT / "data" / "raw" / "era5_land"
PROCESSED = PROJECT_ROOT / "data" / "processed"
VALIDATION = PROJECT_ROOT / "data" / "validation"
TABLE_PATH = VALIDATION / "dwd_era5_station_comparison_2021-02-13_0600.csv"
METADATA_PATH = VALIDATION / "dwd_era5_station_comparison_metadata.json"
TIMESTAMP_UTC = pd.Timestamp("2021-02-13 06:00", tz="UTC")
DWD_BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/hourly/air_temperature/historical/"
)
DWD_STATION_INVENTORY = "TU_Stundenwerte_Beschreibung_Stationen.txt"

TARGETS = {
    "Cuxhaven": {
        "latitude": 53.861,
        "longitude": 8.694,
        "station_id": "00891",
    },
    "Hamburg": {
        "latitude": 53.551,
        "longitude": 9.993,
        "station_id": "01975",
    },
    "Kiel": {
        "latitude": 54.323,
        "longitude": 10.123,
        "station_id": "02564",
    },
    "Rostock": {
        "latitude": 54.092,
        "longitude": 12.099,
        "station_id": "04271",
    },
    "Göttingen": {
        "latitude": 51.541,
        "longitude": 9.916,
        "station_id": "01691",
    },
    "Kassel": {
        "latitude": 51.313,
        "longitude": 9.480,
        "station_id": "15207",
    },
    "Erfurt": {
        "latitude": 50.984,
        "longitude": 11.029,
        "station_id": "01270",
    },
    "Leipzig": {
        "latitude": 51.340,
        "longitude": 12.374,
        "station_id": "02928",
    },
}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    radius_km = 6371.0088
    lat_a = np.radians(latitude_a)
    lat_b = np.radians(latitude_b)
    delta_lat = np.radians(latitude_b - latitude_a)
    delta_lon = np.radians(longitude_b - longitude_a)
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2) ** 2
    )
    return float(2 * radius_km * np.arcsin(np.sqrt(haversine)))


def read_dwd_station_inventory(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="latin1").splitlines()[2:]:
        if not line.strip():
            continue
        try:
            rows.append(
                {
                    "station_id": line[0:5].strip(),
                    "start_date": line[6:14].strip(),
                    "end_date": line[15:23].strip(),
                    "latitude": float(line[43:50]),
                    "longitude": float(line[53:60]),
                    "station_name": line[61:102].strip(),
                }
            )
        except (IndexError, ValueError):
            continue
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise ValueError("DWD station inventory contains no rows")
    return inventory


def _select_station(
    target: dict[str, object], inventory: pd.DataFrame
) -> pd.Series:
    date_key = TIMESTAMP_UTC.strftime("%Y%m%d")
    active = inventory.loc[
        inventory["start_date"].le(date_key) & inventory["end_date"].ge(date_key)
    ].copy()
    if active.empty:
        raise ValueError("DWD inventory contains no active stations")
    target_latitude = float(target["latitude"])
    target_longitude = float(target["longitude"])
    active["distance_to_target_km"] = active.apply(
        lambda row: _haversine_km(
            target_latitude,
            target_longitude,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )
    selected_id = str(target["station_id"])
    selected = active.loc[active["station_id"].eq(selected_id)]
    if len(selected) != 1:
        raise ValueError(f"selected DWD station is not active: {selected_id}")
    nearest = active.sort_values("distance_to_target_km").iloc[0]
    selected_row = selected.iloc[0]
    if selected_row["station_id"] != nearest["station_id"]:
        raise ValueError(
            f"selected station {selected_id} is not nearest active station "
            f"({nearest['station_id']})"
        )
    return selected_row


def select_exact_dwd_observation(
    observations: pd.DataFrame, timestamp: pd.Timestamp
) -> dict[str, float | int]:
    timestamp = timestamp.tz_convert("UTC")
    key = timestamp.strftime("%Y%m%d%H")
    rows = observations.loc[observations["MESS_DATUM"].astype(str).str.strip().eq(key)].copy()
    if rows.empty:
        raise ValueError(f"no exact UTC observation at {timestamp}")
    rows["temperature_c"] = pd.to_numeric(rows["TT_TU"], errors="coerce")
    rows["quality_flag"] = pd.to_numeric(rows["QN_9"], errors="coerce")
    valid = rows.loc[
        rows["temperature_c"].notna()
        & np.isfinite(rows["temperature_c"])
        & rows["quality_flag"].notna()
        & rows["quality_flag"].gt(0)
    ]
    if len(valid) != 1:
        raise ValueError(f"exact UTC observation at {timestamp} is missing or invalid")
    row = valid.iloc[0]
    return {
        "temperature_c": float(row["temperature_c"]),
        "quality_flag": int(row["quality_flag"]),
    }


def find_dwd_archive(source_dir: Path, station_id: str) -> Path:
    archives = sorted(source_dir.glob(f"stundenwerte_TU_{station_id}_*.zip"))
    if len(archives) != 1:
        raise FileNotFoundError(
            f"expected one DWD historical archive for station {station_id}"
        )
    return archives[0]


def source_filename_for_url(filename: str) -> str:
    if filename.endswith("_hist.zip"):
        return filename
    if filename.endswith(".zip"):
        return f"{filename[:-4]}_hist.zip"
    raise ValueError(f"unexpected DWD archive filename: {filename}")


def _read_dwd_observation(archive: Path) -> dict[str, float | int]:
    with ZipFile(archive) as handle:
        data_name = next(
            name for name in handle.namelist() if name.startswith("produkt_tu_stunde_")
        )
        observations = pd.read_csv(
            handle.open(data_name),
            sep=";",
            encoding="latin1",
            dtype={"MESS_DATUM": str},
            na_values=["-999", "-999.0"],
        )
    return select_exact_dwd_observation(observations, TIMESTAMP_UTC)


def nearest_finite_grid_cell(
    requested_latitude: float,
    requested_longitude: float,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | int | bool]:
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    distance = (
        (latitude_grid - requested_latitude) ** 2
        + (
            np.cos(np.radians(requested_latitude))
            * (longitude_grid - requested_longitude)
        )
        ** 2
    )
    raw_index = np.unravel_index(np.argmin(distance), distance.shape)
    finite_distance = np.where(np.isfinite(values), distance, np.inf)
    if not np.isfinite(finite_distance).any():
        raise ValueError("ERA5 grid has no finite cell near the requested location")
    index = np.unravel_index(np.argmin(finite_distance), distance.shape)
    return {
        "latitude_index": int(index[0]),
        "longitude_index": int(index[1]),
        "latitude": float(latitudes[index[0]]),
        "longitude": float(longitudes[index[1]]),
        "temperature_c": float(values[index]),
        "requested_nearest_value_valid": bool(np.isfinite(values[raw_index])),
        "requested_nearest_latitude": float(latitudes[raw_index[0]]),
        "requested_nearest_longitude": float(longitudes[raw_index[1]]),
    }


def _era5_snapshot() -> dict[str, Any]:
    path = RAW_ERA5 / "era5_land_2021_02.nc"
    dataset = xr.open_dataset(path)
    try:
        variable_name = "t2m"
        time_name = "valid_time"
        if variable_name not in dataset.data_vars:
            raise ValueError("ERA5 file does not contain t2m")
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(dataset[time_name].values, utc=True)
        )
        matches = np.flatnonzero(timestamps == TIMESTAMP_UTC)
        if len(matches) != 1:
            raise ValueError("ERA5 file does not contain exactly one requested timestamp")
        values = dataset[variable_name].transpose(
            time_name, "latitude", "longitude"
        ).isel({time_name: int(matches[0])})
        latitude = dataset["latitude"].to_numpy()
        longitude = dataset["longitude"].to_numpy()
        values_kelvin = values.to_numpy().astype(float)
        if dataset[variable_name].attrs.get("units") != "K":
            raise ValueError("ERA5 t2m unit is not Kelvin")
        if not np.all(np.diff(latitude) < 0):
            raise ValueError("ERA5 latitude coordinate is not descending")
        if not np.all(np.diff(longitude) > 0):
            raise ValueError("ERA5 longitude coordinate is not ascending")
        return {
            "path": path,
            "values_kelvin": values_kelvin,
            "latitude": latitude,
            "longitude": longitude,
            "variable": variable_name,
            "unit": dataset[variable_name].attrs["units"],
            "latitude_direction": "descending",
            "longitude_direction": "ascending",
            "metadata": {
                "name": dataset[variable_name].attrs.get("long_name"),
                "standard_name": dataset[variable_name].attrs.get("standard_name"),
            },
        }
    finally:
        dataset.close()


def _temperature_pipeline_module() -> Any:
    return _load_module(
        PROJECT_ROOT / "code" / "02_preprocessing" / "temperature_pipeline.py",
        "temperature_pipeline_for_dwd_validation",
    )


def _figure_module() -> Any:
    return _load_module(
        PROJECT_ROOT / "code" / "03_visualization" / "temperature_population_map.py",
        "temperature_population_map_for_dwd_validation",
    )


def _population_weight_validation(snapshot: dict[str, Any]) -> dict[str, Any]:
    temperature_module = _temperature_pipeline_module()
    first_path = RAW_ERA5 / "era5_land_2019_boundary.nc"
    first_dataset, time_name, variable_name = temperature_module._open_era5_temperature(
        first_path
    )
    try:
        latitudes = first_dataset["latitude"].to_numpy()
        longitudes = first_dataset["longitude"].to_numpy()
        first_values = first_dataset[variable_name].transpose(
            time_name, "latitude", "longitude"
        ).isel({time_name: 0}).to_numpy()
    finally:
        first_dataset.close()
    weights = {}
    results = {}
    for country, filename in {
        "Germany": "temperature_germany_hourly.csv",
        "Austria": "temperature_austria_hourly.csv",
    }.items():
        country_weights, _ = temperature_module.build_weights(
            country,
            latitudes,
            longitudes,
            valid_grid_mask=np.isfinite(first_values),
        )
        weights[country] = country_weights
        indices = (
            country_weights["latitude_index"].to_numpy(),
            country_weights["longitude_index"].to_numpy(),
        )
        plotted_kelvin = snapshot["values_kelvin"][indices]
        plotted_celsius = plotted_kelvin - 273.15
        weighted_celsius = float(
            plotted_celsius @ country_weights["weight"].to_numpy()
        )
        processed = pd.read_csv(PROCESSED / filename)
        processed["interval_start_utc"] = pd.to_datetime(
            processed["interval_start_utc"], utc=True
        )
        row = processed.loc[processed["interval_start_utc"].eq(TIMESTAMP_UTC)]
        if len(row) != 1:
            raise ValueError(f"processed {country} CSV lacks the requested timestamp")
        processed_celsius = float(row.iloc[0]["temperature_c"])
        results[country] = {
            "weight_sum": float(country_weights["weight"].sum()),
            "plotted_cells_weighted_c": weighted_celsius,
            "processed_csv_c": processed_celsius,
            "difference_c": weighted_celsius - processed_celsius,
            "maximum_weight": float(country_weights["weight"].max()),
        }
    return results


def _colour_scale_validation(snapshot: dict[str, Any]) -> dict[str, Any]:
    figure_module = _figure_module()
    boundaries = figure_module._load_country_boundaries()
    values_celsius = snapshot["values_kelvin"] - 273.15
    country_values = {}
    for country in ("Germany", "Austria"):
        masked = figure_module.mask_grid_to_geometry(
            values_celsius,
            snapshot["latitude"],
            snapshot["longitude"],
            boundaries[figure_module.COUNTRY_CODES[country]],
        )
        country_values[country] = masked[np.isfinite(masked)]
    minimum = float(np.floor(min(values.min() for values in country_values.values())))
    maximum = float(np.ceil(max(values.max() for values in country_values.values())))
    all_values = np.concatenate(list(country_values.values()))
    below_min = int((all_values < minimum).sum())
    above_max = int((all_values > maximum).sum())
    cmap = plt.get_cmap("RdBu_r")
    return {
        "minimum_c": minimum,
        "maximum_c": maximum,
        "below_min": below_min,
        "above_max": above_max,
        "total_values": int(all_values.size),
        "cold_colour": tuple(float(value) for value in cmap(0.0)),
        "warm_colour": tuple(float(value) for value in cmap(1.0)),
    }


def run_validation(source_dir: Path) -> dict[str, Any]:
    inventory_path = source_dir / DWD_STATION_INVENTORY
    inventory = read_dwd_station_inventory(inventory_path)
    snapshot = _era5_snapshot()
    weight_validation = _population_weight_validation(snapshot)
    colour_validation = _colour_scale_validation(snapshot)
    rows = []
    source_files = [
        {
            "filename": DWD_STATION_INVENTORY,
            "source_url": DWD_BASE_URL + DWD_STATION_INVENTORY,
            "sha256": _sha256_file(inventory_path),
        }
    ]
    for location, target in TARGETS.items():
        station = _select_station(target, inventory)
        station_id = station["station_id"]
        archive = find_dwd_archive(source_dir, station_id)
        observation = _read_dwd_observation(archive)
        source_files.append(
            {
                "filename": source_filename_for_url(archive.name),
                "local_filename": archive.name,
                "source_url": DWD_BASE_URL + source_filename_for_url(archive.name),
                "sha256": _sha256_file(archive),
            }
        )
        target_latitude = float(target["latitude"])
        target_longitude = float(target["longitude"])
        grid = nearest_finite_grid_cell(
            target_latitude,
            target_longitude,
            snapshot["latitude"],
            snapshot["longitude"],
            snapshot["values_kelvin"] - 273.15,
        )
        grid_kelvin = float(
            snapshot["values_kelvin"][
                int(grid["latitude_index"]), int(grid["longitude_index"])
            ]
        )
        grid_celsius = grid_kelvin - 273.15
        dwd_celsius = float(observation["temperature_c"])
        rows.append(
            {
                "location": location,
                "requested_latitude": target_latitude,
                "requested_longitude": target_longitude,
                "era5_grid_latitude": grid["latitude"],
                "era5_grid_longitude": grid["longitude"],
                "era5_requested_nearest_value_valid": grid[
                    "requested_nearest_value_valid"
                ],
                "era5_requested_nearest_latitude": grid[
                    "requested_nearest_latitude"
                ],
                "era5_requested_nearest_longitude": grid[
                    "requested_nearest_longitude"
                ],
                "era5_temperature_kelvin": grid_kelvin,
                "era5_temperature_c": grid_celsius,
                "dwd_station_name": station["station_name"],
                "dwd_station_id": station_id,
                "dwd_station_latitude": float(station["latitude"]),
                "dwd_station_longitude": float(station["longitude"]),
                "dwd_distance_from_era5_grid_km": _haversine_km(
                    grid["latitude"],
                    grid["longitude"],
                    float(station["latitude"]),
                    float(station["longitude"]),
                ),
                "dwd_observation_timestamp_utc": TIMESTAMP_UTC.isoformat(),
                "dwd_quality_flag_qn9": observation["quality_flag"],
                "dwd_observed_temperature_c": dwd_celsius,
                "dwd_minus_era5_c": dwd_celsius - grid_celsius,
            }
        )
    table = pd.DataFrame(rows)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    coastal = table.loc[table["location"].isin(["Cuxhaven", "Hamburg", "Kiel", "Rostock"])]
    central = table.loc[table["location"].isin(["Göttingen", "Kassel", "Erfurt", "Leipzig"])]
    coastal_era5_mean = float(coastal["era5_temperature_c"].mean())
    central_era5_mean = float(central["era5_temperature_c"].mean())
    coastal_dwd_mean = float(coastal["dwd_observed_temperature_c"].mean())
    central_dwd_mean = float(central["dwd_observed_temperature_c"].mean())
    metadata = {
        "timestamp_utc": TIMESTAMP_UTC.isoformat(),
        "local_time": "13 February 2021, 07:00 CET in Germany and Austria",
        "era5_dataset": "ERA5-Land reanalysis",
        "era5_source_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview",
        "era5_variable": snapshot["variable"],
        "era5_variable_description": snapshot["metadata"],
        "era5_original_unit": snapshot["unit"],
        "era5_conversion": "temperature_celsius = temperature_kelvin - 273.15, exactly once",
        "era5_latitude_direction": snapshot["latitude_direction"],
        "era5_longitude_direction": snapshot["longitude_direction"],
        "colour_scale_validation": colour_validation,
        "population_weight_validation": weight_validation,
        "dwd_source_url": DWD_BASE_URL,
        "dwd_variable": "TT_TU",
        "dwd_variable_description": "quality-controlled hourly 2 m air temperature",
        "dwd_unit": "°C",
        "dwd_quality_rule": "exact MESS_DATUM in UTC, finite TT_TU, and positive QN_9",
        "dwd_source_files": source_files,
        "storage_note": "DWD source archives were used from the temporary validation workspace only; observations are not modelling inputs.",
        "pattern_summary": {
            "era5_coastal_mean_c": coastal_era5_mean,
            "era5_central_mean_c": central_era5_mean,
            "era5_coastal_minus_central_c": coastal_era5_mean - central_era5_mean,
            "dwd_coastal_mean_c": coastal_dwd_mean,
            "dwd_central_mean_c": central_dwd_mean,
            "dwd_coastal_minus_central_c": coastal_dwd_mean - central_dwd_mean,
            "conclusion": "Both ERA5-Land and DWD support the broad pattern that the selected northern/coastal locations were generally less cold than the selected central-Germany locations, although Hamburg was colder than Kassel in the DWD station observations.",
        },
        "limitations": [
            "Cuxhaven's geometrically nearest ERA5-Land grid cell was non-finite at this timestamp; the table records the nearest finite grid cell instead.",
            "DWD stations are point observations and can differ from a 0.1-degree reanalysis grid cell because of local exposure, elevation, and spatial representativeness.",
            "The eight named locations use representative city-centre coordinates; they are not a population sample or a station-based climate climatology.",
        ],
        "table_filename": TABLE_PATH.name,
    }
    metadata["table_sha256"] = _sha256_file(TABLE_PATH)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"table": str(TABLE_PATH), "metadata": str(METADATA_PATH), **metadata}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_validation(args.source_dir), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
