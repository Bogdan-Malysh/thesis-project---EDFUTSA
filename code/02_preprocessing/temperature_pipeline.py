from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW_ERA5 = PROJECT_ROOT / "data" / "raw" / "era5_land"
RAW_EUROSTAT = PROJECT_ROOT / "data" / "raw" / "eurostat_population"

SMARD_FILES = {
    "Germany": PROCESSED / "smard_germany_hourly.csv",
    "Austria": PROCESSED / "smard_austria_hourly.csv",
}
COUNTRY_TIMEZONES = {"Germany": "Europe/Berlin", "Austria": "Europe/Vienna"}

ERA5_DATASET = "reanalysis-era5-land"
ERA5_SOURCE_URL = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview"
ERA5_API_URL = "https://cds.climate.copernicus.eu/api/retrieve/v1/processes/reanalysis-era5-land"
ERA5_DOI = "10.24381/cds.e2161bac"
ERA5_LICENSE = "CC-BY"
ERA5_VERSION = "1.0.0"
ERA5_VARIABLE = "2m_temperature"
ERA5_AREA = [56.0, 4.0, 45.0, 19.0]

EUROSTAT_ZIP = RAW_EUROSTAT / "Eurostat_Census-GRID_2021_V3.zip"
EUROSTAT_SOURCE_URL = (
    "https://gisco-services.ec.europa.eu/census/2021/"
    "Eurostat_Census-GRID_2021_V3.zip"
)
EUROSTAT_PAGE_URL = (
    "https://ec.europa.eu/eurostat/web/gisco/geodata/"
    "population-distribution/population-grids"
)
EUROSTAT_VERSION = "V3"
EUROSTAT_RELEASE_DATE = "2026-05-30"
EUROSTAT_LICENSE = "Eurostat GISCO download rules"
EUROSTAT_DOI: str | None = None
EUROSTAT_INNER_ZIP = (
    "Eurostat_Census-GRID_2021_V3/ESTAT_Census_2021_country.csv.zip"
)
EUROSTAT_COUNTRY_FILES = {
    "Germany": "CENSUS_GRID_N_DE_2021.csv",
    "Austria": "CENSUS_GRID_N_AT_2021.csv",
}

OUTPUT_FILES = {
    "Germany": PROCESSED / "temperature_germany_hourly.csv",
    "Austria": PROCESSED / "temperature_austria_hourly.csv",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_smard_file(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        usecols=["interval_start_utc", "interval_end_utc"],
    )
    data["interval_start_utc"] = pd.to_datetime(
        data["interval_start_utc"], utc=True
    )
    data["interval_end_utc"] = pd.to_datetime(data["interval_end_utc"], utc=True)
    return data


def read_required_smard_keys() -> pd.DataFrame:
    country_frames = {country: _read_smard_file(path) for country, path in SMARD_FILES.items()}
    germany = country_frames["Germany"]
    austria = country_frames["Austria"]
    if not germany["interval_start_utc"].equals(austria["interval_start_utc"]):
        raise ValueError("Germany and Austria SMARD UTC keys do not match")
    if not germany["interval_end_utc"].equals(austria["interval_end_utc"]):
        raise ValueError("Germany and Austria SMARD UTC interval ends do not match")
    if germany["interval_start_utc"].duplicated().any():
        raise ValueError("SMARD UTC keys contain duplicates")
    differences = germany["interval_start_utc"].diff().dropna()
    if not differences.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("SMARD UTC keys are not continuous hourly observations")
    return germany


def required_key_groups(keys: pd.DataFrame) -> dict[tuple[int, int], pd.DatetimeIndex]:
    groups: dict[tuple[int, int], pd.DatetimeIndex] = {}
    years = keys["interval_start_utc"].dt.year
    months = keys["interval_start_utc"].dt.month
    for year, month in sorted(zip(years, months)):
        key = (int(year), int(month))
        if key not in groups:
            mask = years.eq(year) & months.eq(month)
            groups[key] = pd.DatetimeIndex(
                keys.loc[mask, "interval_start_utc"]
            )
    return groups


def _era5_target_path(year: int, month: int, month_keys: pd.DatetimeIndex) -> Path:
    if year == 2019 and month == 12 and len(month_keys) == 1:
        return RAW_ERA5 / "era5_land_2019_boundary.nc"
    return RAW_ERA5 / f"era5_land_{year}_{month:02d}.nc"


def _date_request_values(keys: pd.DatetimeIndex) -> tuple[list[str], list[str], list[str], list[str]]:
    return (
        sorted({f"{value.month:02d}" for value in keys}),
        sorted({f"{value.day:02d}" for value in keys}),
        sorted({value.strftime("%H:%M") for value in keys}),
        sorted({value.strftime("%Y-%m-%dT%H:%M:%S") for value in keys}),
    )


def _open_era5_temperature(path: Path) -> tuple[xr.Dataset, str, str]:
    dataset = xr.open_dataset(path)
    time_name = "valid_time" if "valid_time" in dataset.coords else "time"
    variable_name = "t2m" if "t2m" in dataset.data_vars else "2m_temperature"
    if variable_name not in dataset.data_vars:
        dataset.close()
        raise ValueError(f"{path.name} does not contain ERA5 2 m temperature")
    for dimension in ("number", "expver"):
        if dimension in dataset[variable_name].dims:
            dataset = dataset.isel({dimension: 0})
    return dataset, time_name, variable_name


def _validate_era5_keys(path: Path, required: pd.DatetimeIndex) -> None:
    dataset, time_name, variable_name = _open_era5_temperature(path)
    try:
        times = pd.DatetimeIndex(pd.to_datetime(dataset[time_name].values, utc=True))
        if times.duplicated().any():
            raise ValueError(f"{path.name} contains duplicate ERA5 times")
        if not required.isin(times).all():
            missing = required[~required.isin(times)]
            raise ValueError(f"{path.name} is missing required ERA5 times: {missing[0]}")
        values = dataset[variable_name]
        if values.ndim != 3:
            raise ValueError(f"{path.name} temperature variable is not time-lat-lon")
    finally:
        dataset.close()


def download_era5_files(
    groups: dict[tuple[int, int], pd.DatetimeIndex]
) -> list[tuple[Path, pd.DatetimeIndex]]:
    RAW_ERA5.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True)
    sources = []
    for (year, month), keys in groups.items():
        target = _era5_target_path(year, month, keys)
        if not target.exists():
            _, days, times, _ = _date_request_values(keys)
            request = {
                "variable": [ERA5_VARIABLE],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": times,
                "area": ERA5_AREA,
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            client.retrieve(ERA5_DATASET, request, str(target))
        _validate_era5_keys(target, keys)
        sources.append((target, keys))
    return sources


def _read_population_country(country: str) -> pd.DataFrame:
    if not EUROSTAT_ZIP.exists():
        raise FileNotFoundError(EUROSTAT_ZIP)
    with ZipFile(EUROSTAT_ZIP) as outer:
        inner_bytes = outer.read(EUROSTAT_INNER_ZIP)
    with ZipFile(io.BytesIO(inner_bytes)) as inner:
        filename = EUROSTAT_COUNTRY_FILES[country]
        with inner.open(filename) as handle:
            data = pd.read_csv(handle, encoding="utf-8-sig")
    required_columns = {"SPATIAL", "TIME_PERIOD", "OBS_VALUE", "MEASURE"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"{country} population file lacks required columns")
    data = data.loc[
        data["TIME_PERIOD"].astype(str).eq("2021")
        & data["MEASURE"].eq("populationAtResidencePlace")
    ].copy()
    data["population"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
    data = data.loc[data["population"].gt(0)].copy()
    coordinates = data["SPATIAL"].str.extract(
        r"CRS3035RES1000mN(?P<northing>\d+)E(?P<easting>\d+)"
    )
    if coordinates.isna().any().any():
        raise ValueError(f"{country} population grid contains unparseable cell IDs")
    data["northing"] = coordinates["northing"].astype(float) + 500.0
    data["easting"] = coordinates["easting"].astype(float) + 500.0
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    data["longitude"], data["latitude"] = transformer.transform(
        data["easting"].to_numpy(), data["northing"].to_numpy()
    )
    return data[["SPATIAL", "population", "longitude", "latitude"]]


def build_weights(
    country: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    valid_grid_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, float]:
    population = _read_population_country(country)
    latitude_index = np.abs(
        latitudes[:, None] - population["latitude"].to_numpy()[None, :]
    ).argmin(axis=0)
    longitude_index = np.abs(
        longitudes[:, None] - population["longitude"].to_numpy()[None, :]
    ).argmin(axis=0)
    if valid_grid_mask is not None:
        if valid_grid_mask.shape != (len(latitudes), len(longitudes)):
            raise ValueError("ERA5 finite-grid mask does not match coordinate dimensions")
        valid_coordinates = np.argwhere(valid_grid_mask)
        if len(valid_coordinates) == 0:
            raise ValueError("ERA5 grid contains no finite land cells")
        invalid = ~valid_grid_mask[latitude_index, longitude_index]
        invalid_pairs = np.unique(
            np.column_stack([latitude_index[invalid], longitude_index[invalid]]),
            axis=0,
        )
        replacement = {}
        valid_latitudes = latitudes[valid_coordinates[:, 0]]
        valid_longitudes = longitudes[valid_coordinates[:, 1]]
        for invalid_latitude, invalid_longitude in invalid_pairs:
            distances = (
                (valid_latitudes - latitudes[invalid_latitude]) ** 2
                + (valid_longitudes - longitudes[invalid_longitude]) ** 2
            )
            nearest = valid_coordinates[int(np.argmin(distances))]
            replacement[(int(invalid_latitude), int(invalid_longitude))] = (
                int(nearest[0]),
                int(nearest[1]),
            )
        for old_pair, new_pair in replacement.items():
            remap = (latitude_index == old_pair[0]) & (longitude_index == old_pair[1])
            latitude_index[remap] = new_pair[0]
            longitude_index[remap] = new_pair[1]
    mapped = pd.DataFrame(
        {
            "latitude_index": latitude_index,
            "longitude_index": longitude_index,
            "population": population["population"].to_numpy(),
        }
    )
    mapped = mapped.groupby(
        ["latitude_index", "longitude_index"], as_index=False
    )["population"].sum()
    total_population = float(mapped["population"].sum())
    if total_population <= 0:
        raise ValueError(f"{country} has no positive population weights")
    mapped["weight"] = mapped["population"] / total_population
    return mapped, total_population


def _weighted_temperature(
    dataset: xr.Dataset,
    time_name: str,
    variable_name: str,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    values = dataset[variable_name]
    lat_name = "latitude" if "latitude" in values.dims else "lat"
    lon_name = "longitude" if "longitude" in values.dims else "lon"
    values = values.transpose(time_name, lat_name, lon_name).to_numpy()
    lat_indices = weights["latitude_index"].to_numpy()
    lon_indices = weights["longitude_index"].to_numpy()
    population_weights = weights["weight"].to_numpy()
    selected = values[:, lat_indices, lon_indices]
    valid = np.isfinite(selected).all(axis=1)
    temperature_k = np.full(selected.shape[0], np.nan, dtype=float)
    temperature_k[valid] = selected[valid] @ population_weights
    timestamps = pd.to_datetime(dataset[time_name].values, utc=True)
    return pd.DataFrame(
        {"interval_start_utc": timestamps, "temperature_k": temperature_k}
    )


def _historical_reanalysis_output(
    country: str,
    smard_keys: pd.DataFrame,
    era5_paths: list[Path],
) -> tuple[pd.DataFrame, dict[str, object]]:
    first_dataset, time_name, variable_name = _open_era5_temperature(era5_paths[0])
    try:
        latitudes = first_dataset["latitude"].to_numpy()
        longitudes = first_dataset["longitude"].to_numpy()
        first_values = first_dataset[variable_name].transpose(
            time_name,
            "latitude",
            "longitude",
        ).isel({time_name: 0}).to_numpy()
    finally:
        first_dataset.close()
    weights, total_population = build_weights(
        country,
        latitudes,
        longitudes,
        valid_grid_mask=np.isfinite(first_values),
    )
    annual = []
    for path in era5_paths:
        dataset, time_name, variable_name = _open_era5_temperature(path)
        try:
            annual.append(
                _weighted_temperature(dataset, time_name, variable_name, weights)
            )
        finally:
            dataset.close()
    temperatures = pd.concat(annual, ignore_index=True)
    if temperatures["interval_start_utc"].duplicated().any():
        raise ValueError(f"{country} ERA5 aggregation contains duplicate timestamps")
    merged = smard_keys.merge(temperatures, on="interval_start_utc", how="left", validate="one_to_one")
    merged["temperature_valid"] = merged["temperature_k"].notna() & np.isfinite(
        merged["temperature_k"]
    )
    merged["temperature_c"] = merged["temperature_k"] - 273.15
    merged.loc[~merged["temperature_valid"], "temperature_c"] = np.nan
    timezone = COUNTRY_TIMEZONES[country]
    merged["interval_start_local"] = merged["interval_start_utc"].dt.tz_convert(timezone)
    merged["interval_end_local"] = merged["interval_end_utc"].dt.tz_convert(timezone)
    output = merged[
        [
            "interval_start_utc",
            "interval_end_utc",
            "interval_start_local",
            "interval_end_local",
            "temperature_c",
            "temperature_valid",
        ]
    ].copy()
    validation = validate_temperature_output(output, smard_keys, country)
    validation["total_population_weighted"] = total_population
    validation["weight_sum"] = float(weights["weight"].sum())
    return output, validation


def validate_temperature_output(
    output: pd.DataFrame, smard_keys: pd.DataFrame, country: str
) -> dict[str, object]:
    expected = smard_keys["interval_start_utc"]
    starts = output["interval_start_utc"]
    if len(output) != len(expected):
        raise ValueError(f"{country} output row count does not match SMARD")
    if not starts.equals(expected):
        raise ValueError(f"{country} output timestamps do not match SMARD exactly")
    if starts.duplicated().any():
        raise ValueError(f"{country} output contains duplicate timestamps")
    if not starts.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError(f"{country} output timestamps are not hourly continuous")
    valid = output["temperature_valid"].astype(bool)
    finite = np.isfinite(output.loc[valid, "temperature_c"])
    if not finite.all():
        raise ValueError(f"{country} valid temperature values contain non-finite data")
    minimum = float(output.loc[valid, "temperature_c"].min()) if valid.any() else np.nan
    maximum = float(output.loc[valid, "temperature_c"].max()) if valid.any() else np.nan
    if valid.any() and (minimum < -90 or maximum > 60):
        raise ValueError(f"{country} temperature range is outside physical validation bounds")
    return {
        "country": country,
        "row_count": int(len(output)),
        "valid_count": int(valid.sum()),
        "missing_count": int((~valid).sum()),
        "duplicate_count": int(starts.duplicated().sum()),
        "first_interval_start_utc": starts.iloc[0].isoformat(),
        "final_interval_end_utc": output["interval_end_utc"].iloc[-1].isoformat(),
        "minimum_temperature_c": minimum,
        "maximum_temperature_c": maximum,
    }


def _metadata_entry(path: Path, source_url: str, doi: str | None, licence: str, version: str, **extra: object) -> dict[str, object]:
    return {
        "filename": path.name,
        "source_url": source_url,
        "doi": doi,
        "licence": licence,
        "version": version,
        "access_date": datetime.now(timezone.utc).date().isoformat(),
        "sha256": sha256_file(path),
        **extra,
    }


def write_metadata(
    era5_sources: list[tuple[Path, pd.DatetimeIndex]], area: list[float]
) -> None:
    era5_entries = []
    for path, required_keys in era5_sources:
        match = re.search(r"(\d{4})_(\d{2})", path.name)
        year = 2019 if "2019_boundary" in path.name else int(match.group(1))
        month = 12 if "2019_boundary" in path.name else int(match.group(2))
        era5_entries.append(
            _metadata_entry(
                path,
                ERA5_SOURCE_URL,
                ERA5_DOI,
                ERA5_LICENSE,
                ERA5_VERSION,
                api_process_url=ERA5_API_URL,
                dataset=ERA5_DATASET,
                variable=ERA5_VARIABLE,
                area=area,
                year=year,
                month=month,
                required_timestamp_count=len(required_keys),
                required_first_timestamp=required_keys[0].isoformat(),
                required_last_timestamp=required_keys[-1].isoformat(),
                data_format="netcdf",
                interpretation="realised historical reanalysis temperature; not a day-ahead forecast",
            )
        )
    RAW_ERA5.mkdir(parents=True, exist_ok=True)
    (RAW_ERA5 / "metadata.json").write_text(
        json.dumps({"source_files": era5_entries}, indent=2), encoding="utf-8"
    )
    eurostat_entry = _metadata_entry(
        EUROSTAT_ZIP,
        EUROSTAT_SOURCE_URL,
        EUROSTAT_DOI,
        EUROSTAT_LICENSE,
        EUROSTAT_VERSION,
        release_date=EUROSTAT_RELEASE_DATE,
        metadata_page_url=EUROSTAT_PAGE_URL,
        coordinate_reference_system="EPSG:3035",
        population_variable="OBS_VALUE where MEASURE=populationAtResidencePlace",
    )
    RAW_EUROSTAT.mkdir(parents=True, exist_ok=True)
    (RAW_EUROSTAT / "metadata.json").write_text(
        json.dumps({"source_files": [eurostat_entry]}, indent=2), encoding="utf-8"
    )


def run_pipeline() -> dict[str, object]:
    smard_keys = read_required_smard_keys()
    groups = required_key_groups(smard_keys)
    era5_sources = download_era5_files(groups)
    era5_paths = [path for path, _ in era5_sources]
    outputs = {}
    validations = {}
    for country in SMARD_FILES:
        output, validation = _historical_reanalysis_output(country, smard_keys, era5_paths)
        output.to_csv(OUTPUT_FILES[country], index=False)
        outputs[country] = OUTPUT_FILES[country].name
        validations[country] = validation
    write_metadata(era5_sources, ERA5_AREA)
    return {
        "required_row_count": len(smard_keys),
        "required_years": sorted({year for year, _ in groups}),
        "required_month_count": len(groups),
        "outputs": outputs,
        "validation": validations,
    }


def main() -> int:
    result = run_pipeline()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
