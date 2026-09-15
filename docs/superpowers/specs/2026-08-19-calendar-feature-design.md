# Country-Specific Calendar Feature Design

## Purpose

Create reproducible, country-specific calendar feature tables for Germany and
Austria. The tables will be joined to modelling datasets later and will not
rewrite existing demand or temperature datasets.

## Existing Project Context

The existing processed SMARD and temperature files each contain exactly 52,608
hourly rows for Germany and Austria. Their `interval_start_utc` values are
unique, ordered, and continuous at one-hour intervals. Their local timestamp
columns already preserve timezone-aware offsets in the source pipeline, and the
SMARD files retain both autumn repeated-hour rows.

No existing processed dataset or preprocessing script currently contains the
requested calendar variables.

## Inputs and Outputs

The calendar preprocessing script reads the existing SMARD processed country
files only for their validated UTC key columns:

- `data/processed/smard_germany_hourly.csv`
- `data/processed/smard_austria_hourly.csv`

It creates two independent outputs:

- `data/processed/calendar_germany_hourly.csv`
- `data/processed/calendar_austria_hourly.csv`

Each output contains exactly these columns in this order:

1. `timestamp_utc`
2. `timestamp_local`
3. `hour`
4. `day_of_week`
5. `month`
6. `is_weekend`
7. `is_public_holiday`

`timestamp_utc` is the unique join key. `timestamp_local` is timezone-aware
and is written as ISO 8601 text including the UTC offset, for example
`2020-10-25T02:00:00+02:00` and `2020-10-25T02:00:00+01:00`. The two strings
must remain distinguishable during the autumn daylight-saving transition.

## Feature Rules

For each country, UTC timestamps are converted before any calendar component
is calculated:

- Germany uses `Europe/Berlin`.
- Austria uses `Europe/Vienna`.
- `hour` is the local hour in the range 0 through 23.
- `day_of_week` uses Monday=0 through Sunday=6.
- `month` is the local calendar month in the range 1 through 12.
- `is_weekend` is 1 when `day_of_week` is 5 or 6, otherwise 0.
- `is_public_holiday` is 1 when the local calendar date is in the country
  holiday set, otherwise 0.

No dummy variables, one-hot variables, holiday names, or model-specific
transformations are created.

## Holiday Calendar

The implementation uses `holidays==0.103`, with the public-holiday category,
no country subdivision, and observed/substitute dates disabled. This makes the
calendar represent statutory holiday dates only:

- Germany: New Year's Day, Good Friday, Easter Monday, Labour Day, Ascension
  Day, Whit Monday, German Unity Day, Christmas Day, and Boxing Day.
- Austria: New Year's Day, Epiphany, Easter Monday, Labour Day, Ascension Day,
  Whit Monday, Corpus Christi, Assumption, National Day, All Saints' Day,
  Immaculate Conception, Christmas Day, and St. Stephen's Day.

Independent verification uses:

- Germany: the Federal Ministry of the Interior official Feiertage page
  identifying the nine holidays observed throughout Germany, recorded in the
  metadata as the primary German source:
  `https://www.bmi.bund.de/DE/themen/verfassung/staatliche-symbole/feiertage/feiertage-node.html`
- Austria: the Austrian Legal Information System provision
  `Feiertagsruhegesetz 1957, Art. 1 Section 1`, which explicitly lists the 13
  nationwide holidays:
  `https://www.ris.bka.gv.at/eli/bgbl/1957/153/A1P1/NOR40213432`

The metadata also records the package name/version, source names and URLs,
calendar parameters, and the verified holiday-date counts for each year.

## Validation Artifacts

The script writes:

- `data/validation/calendar_holiday_summary.csv`
- `data/validation/calendar_feature_metadata.json`

The summary contains one row per country and local year, with separate counts
for `holiday_date_count` and `holiday_hour_observation_count`. Hourly counts
are not inferred as dates multiplied by 24; they are counted from the hourly
table so that DST dates with 23 or 25 local observations are represented
correctly.

The metadata records source-file hashes, output hashes, package versions,
official-source references, timezones, holiday parameters, row counts, and all
validation results.

## Validation Requirements

The implementation and tests must verify, separately for Germany and Austria:

- exactly 52,608 output rows;
- exact column names and order;
- UTC timestamps are timezone-aware, complete, ordered, and unique;
- local timestamps equal the correct country timezone conversion;
- both occurrences of every autumn repeated local hour are retained and have
  distinct UTC offsets in `timestamp_local`;
- no missing calendar variables;
- integer/binary types and defined value ranges;
- `is_weekend` agrees with `day_of_week`;
- holiday flags agree with the local calendar date and country holiday set;
- exactly nine German holiday dates per local year;
- exactly thirteen Austrian holiday dates per local year;
- holiday dates and holiday-hour observations are reported separately;
- source SMARD row order and UTC key values are unchanged;
- existing demand and temperature files are byte-for-byte unchanged.

Tests cover representative regular dates, Germany-only and Austria-only
holidays, excluded state or non-statutory holidays, spring and autumn DST
boundaries, and the full 2020-2025 project coverage.

## Files

- Create `code/02_preprocessing/calendar_features.py` for generation,
  validation, metadata, and the command-line entry point.
- Create `code/02_preprocessing/test_calendar_features.py` for unit and
  integration tests.
- Modify `requirements.txt` to pin `holidays==0.103`.
- Create the two processed country calendar tables and two validation artifacts
  listed above.

No raw file, existing processed demand file, or existing processed temperature
file is modified.
