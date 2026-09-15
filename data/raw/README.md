# SMARD Raw Data

These files are local downloads from **SMARD**, the German Federal Network Agency's electricity-market data platform. They are stored in `data/raw/smard/` and contain hourly national grid-load observations and official forecast benchmark values for Germany and Austria.

## Files

| Path relative to `data/raw/` | Country | Role | Columns |
|---|---|---|---|
| `smard/smard_austria_actual.csv` | Austria | Actual observations | `grid load [MWh] Calculated resolutions`; `Grid load incl. hydro pumped storage [MWh] Calculated resolutions`; `Hydro pumped storage [MWh] Calculated resolutions`; `Residual load [MWh] Calculated resolutions` |
| `smard/smard_austria_forecasted.csv` | Austria | Official forecast benchmark | `grid load [MWh] Calculated resolutions`; `Residual load [MWh] Calculated resolutions` |
| `smard/smard_germany_actual.csv` | Germany | Actual observations | `grid load [MWh] Calculated resolutions`; `Grid load incl. hydro pumped storage [MWh] Calculated resolutions`; `Hydro pumped storage [MWh] Calculated resolutions`; `Residual load [MWh] Calculated resolutions` |
| `smard/smard_germany_forecasted.csv` | Germany | Official forecast benchmark | `grid load [MWh] Calculated resolutions`; `Residual load [MWh] Calculated resolutions` |

All load series use **MWh** as the source unit. The selected thesis target is actual `grid load [MWh]`; the matching forecast `grid load [MWh]` series is an external benchmark, not a model input. The actual files also retain pumped-storage and residual-load series for audit and later analysis.

## Confirmed Coverage

The final audit confirmed 52,608 hourly observations in each file, covering calendar years 2020 through 2025:

- First interval start: `2020-01-01 00:00:00` local time
- Final interval end: `2026-01-01 00:00:00` local time
- First interval start: `2019-12-31 23:00:00+00:00` UTC
- Final interval end: `2025-12-31 23:00:00+00:00` UTC
- Germany timezone: `Europe/Berlin`
- Austria timezone: `Europe/Vienna`

The audit confirmed unique, strictly increasing UTC interval keys, one-hour UTC durations and continuity, and expected spring and autumn daylight-saving-time effects.

## Download Date

The original download date was not recorded.

## Immutability

These four CSV files are immutable raw inputs. Do not edit, rename, move, overwrite, clean, delete, impute, or replace them. All parsing, validation and later preprocessing must occur in memory or in approved derived locations without changing `data/raw/`.
