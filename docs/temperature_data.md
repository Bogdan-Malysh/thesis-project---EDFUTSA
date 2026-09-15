# Historical Temperature Data

The Section 4.5.1 temperature covariate is a realised historical reanalysis,
not a day-ahead weather forecast. It is not used as a forecasting-model input
in the current phase; future model use will be defined separately to prevent
future-information leakage.

## Sources

ERA5-Land is downloaded from the Copernicus Climate Data Store dataset
`reanalysis-era5-land`, variable `2m_temperature`, in NetCDF format. The
dataset DOI is `10.24381/cds.e2161bac`, the licence is CC-BY, and the recorded
dataset version is `1.0.0`.

Eurostat's current Census Grid 2021 V3 release is used for fixed spatial
weights:

`https://gisco-services.ec.europa.eu/census/2021/Eurostat_Census-GRID_2021_V3.zip`

The release date is 30 May 2026, the grid uses EPSG:3035, and the total
resident population is `OBS_VALUE` where
`MEASURE=populationAtResidencePlace`. Eurostat does not list a DOI for this
download; the metadata records that explicitly rather than inventing one.

Complete source URLs, access dates, SHA-256 hashes, request parameters and
licence/version fields are stored in:

- `data/raw/era5_land/metadata.json`
- `data/raw/eurostat_population/metadata.json`

## Processing

The required UTC timestamps are read directly from the processed SMARD files.
This includes the 2019-12-31 23:00 UTC boundary hour. ERA5-Land grid values are
weighted separately for Germany and Austria using fixed 2021 Eurostat total
resident population weights. Population cells are assigned to the nearest
finite ERA5-Land grid cell after transforming their EPSG:3035 centroids to
WGS84. Temperatures are converted with:

```text
temperature_c = temperature_k - 273.15
```

Outputs:

- `data/processed/temperature_germany_hourly.csv`
- `data/processed/temperature_austria_hourly.csv`

Each output contains UTC and country-local interval timestamps,
`temperature_c`, and `temperature_valid`.
