# Temperature-Population Map Design

## Goal

Create one reproducible thesis figure showing the ERA5-Land 2 m temperature field and the final Census Grid population weights for Germany and Austria at one common cold historical UTC timestamp. The figure generation must not modify any existing raw or processed source data.

## Data Flow

The new visualization script will read the two processed temperature CSVs to select the timestamp with the lowest joint national mean over 2020-2025. It will open the corresponding existing ERA5-Land NetCDF file, rebuild the final population-to-ERA5 mapping by calling the existing `temperature_pipeline.build_weights()` function with the same first-file finite-grid mask, and use the selected grid values for both panels. It will read a locally stored official GISCO Countries 2024 1:1M GeoJSON boundary file, retain Germany (`DE`) and Austria (`AT`), and mask grid-cell centres outside each country geometry.

The script will accept an optional UTC timestamp override. Without an override it will select the joint coldest timestamp automatically. It will fail if the selected grid values or plotted weights are non-finite, if a country boundary is missing, or if either country weight sum differs from one beyond floating-point tolerance.

## Figure

The script will create a 12 x 6.5 inch white-background figure with serif typography and two side-by-side panels. Each panel will show a masked temperature grid with a shared blue-to-red normalization, an official country outline, and circles centred on mapped ERA5 grid cells. Circle area will be exactly proportional to the normalized population weight. A shared horizontal colorbar, shared population-share bubble legend, title, timestamp/local-time subtitle, and source note will be included. The source note will identify ERA5-Land, Eurostat Census Grid 2021 V3, GISCO Countries 2024, and the required `© EuroGeographics for the administrative boundaries` attribution.

Only `results/figures/temperature_population_cold_hour.png` will be written by the script, at 300 dpi. No SVG or PDF will be produced.

## Provenance and Dependencies

The GISCO GeoJSON will be stored at `data/raw/gisco_boundaries/CNTR_RG_01M_2024_4326.geojson` and accompanied by `metadata.json` containing the dataset version, URL, access date, licence/attribution, CRS, scale, countries, and SHA-256 hash. The minimum new dependency is `shapely`, used for GeoJSON geometry parsing and point-in-polygon masking; it will be pinned in `requirements.txt`.

## Tests

Tests will cover joint timestamp selection and override behaviour, country-grid masking, exact weight-sum and finite-value validation, and the 300-dpi PNG output contract. Existing temperature pipeline tests will remain unchanged and will be run together with the new visualization tests.
