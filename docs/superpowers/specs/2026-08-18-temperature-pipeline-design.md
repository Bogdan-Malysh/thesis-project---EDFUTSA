# Section 4.5.1 Temperature Pipeline Design

## Purpose

Create reproducible population-weighted historical national hourly temperature
series for Germany and Austria using Copernicus ERA5-Land 2 m temperature and
the current Eurostat Census Grid 2021 V3 total resident population variable.
The output is a realised historical reanalysis covariate, not a day-ahead
forecast or a model input at this stage.

## Inputs and Coverage

- SMARD processed files define the required UTC timestamp keys. Their exact
  keys are read at runtime, including any boundary timestamp outside the
  nominal calendar years.
- ERA5-Land uses the CDS dataset `reanalysis-era5-land`, variable
  `2m_temperature`, NetCDF format, and a fixed extraction area covering
  Germany and Austria.
- Eurostat Census Grid 2021 V3 is downloaded from
  `https://gisco-services.ec.europa.eu/census/2021/Eurostat_Census-GRID_2021_V3.zip`.
  It is an ETRS89-LAEA (EPSG:3035) 1 km grid. The total resident population
  field supplies fixed 2021 spatial weights.

## Processing

1. Download ERA5-Land in yearly source files for every year present in the
   SMARD UTC keys, with the exact required month/day/hour selections.
2. Extract the Eurostat population grid and identify the total resident
   population variable and grid-cell coordinates.
3. Transform population-cell centroids to WGS84 and assign cells to the
   corresponding ERA5 grid cell. Normalize positive population weights
   separately within Germany and Austria.
4. Calculate each hourly national weighted temperature in Kelvin, convert with
   `temperature_c = temperature_k - 273.15`, and align exactly to SMARD UTC
   keys.
5. Preserve UTC and country-local timestamps and set `temperature_valid` only
   when the weighted value is finite and the timestamp aligns.

## Provenance and Validation

Every downloaded raw file receives source URL, DOI, licence, version, access
date and SHA-256 metadata. Validation checks row counts, missing values,
duplicates, hourly continuity, temperature range, country weight totals and
exact SMARD timestamp alignment. Existing APG and SMARD files are read-only.
