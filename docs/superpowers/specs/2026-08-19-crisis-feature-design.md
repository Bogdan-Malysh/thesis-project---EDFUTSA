# Crisis Feature Design

## Purpose

Create an independent crisis-feature preprocessing step for Germany and
Austria. It reads the completed country-specific calendar tables, preserves
their timestamp strings exactly, derives two local-date intervention
indicators, and writes separate crisis tables without modifying any existing
raw or processed data.

## Existing Project Context

The completed calendar preprocessing step produces these authoritative inputs:

- `data/processed/calendar_germany_hourly.csv`
- `data/processed/calendar_austria_hourly.csv`

Each input contains 52,608 rows, a unique ordered UTC key, and an ISO 8601
offset-aware `timestamp_local` string. The crisis step must not regenerate or
rewrite either timestamp column.

## Inputs and Outputs

The implementation is contained in:

- `code/02_preprocessing/crisis_features.py`
- `code/02_preprocessing/test_crisis_features.py`

It creates:

- `data/processed/crisis_germany_hourly.csv`
- `data/processed/crisis_austria_hourly.csv`

Each output contains exactly these columns in this order:

1. `timestamp_utc`
2. `timestamp_local`
3. `is_covid_period`
4. `is_post_invasion`

The two timestamp columns are copied as raw CSV strings from the corresponding
calendar input. The flags are integer values restricted to 0 and 1.

The validation artifacts are:

- `data/validation/crisis_feature_summary.csv`
- `data/validation/crisis_feature_metadata.json`

The summary has one row per country and contains:

- `total_observations`
- `covid_period_hours`
- `post_invasion_hours`
- `overlap_hours`
- `covid_first_local_timestamp`
- `covid_last_local_timestamp`
- `post_invasion_first_local_timestamp`
- `post_invasion_last_local_timestamp`

## Indicator Rules

The implementation parses the local timestamp strings and uses only their
local calendar dates. The country timezone is already encoded in each source
string and is not reconstructed from UTC.

- `is_covid_period = 1` for local dates from 2020-03-11 through 2023-05-05,
  inclusive; otherwise 0.
- `is_post_invasion = 1` for local dates from 2022-02-24 through the final
  available local date, inclusive; otherwise 0.
- The indicators are calculated identically for Germany and Austria.
- The overlap from 2022-02-24 through 2023-05-05 is retained in both flags.

The implementation must fail if timestamps cannot be parsed, if source rows are
not exactly 52,608, or if either source timestamp column contains missing,
duplicated, reordered, or non-hourly UTC keys.

## Official Source Metadata

The metadata JSON records each source title, URL, publication date, and the
local access date. The source definitions are:

- World Health Organization, `WHO Director-General's opening remarks at the
  media briefing on COVID-19 - 11 March 2020`, published 2020-03-11:
  `https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020`
- World Health Organization, `WHO Director-General's opening remarks at the
  media briefing - 5 May 2023`, published 2023-05-05:
  `https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing---5-may-2023`
- European Council, `Statement by the members of the European Council on the
  Russian military aggression against Ukraine`, published 2022-02-24:
  `https://www.consilium.europa.eu/en/press/press-releases/2022/02/24/statement-by-the-members-of-the-european-council-on-the-russian-military-aggression-against-ukraine/`
- International Energy Agency, `World Energy Outlook 2022`, published
  2022-10-27:
  `https://www.iea.org/reports/world-energy-outlook-2022`

The IEA report is recorded as contextual energy-crisis source metadata only;
it is not used as a timestamp or indicator input.

## Validation Requirements

Tests and the generation script must verify separately for Germany and Austria:

- exactly 52,608 output rows;
- exact preservation of every `timestamp_utc` string from the calendar input;
- exact preservation of every offset-aware `timestamp_local` string;
- UTC timestamps remain complete, ordered, unique, and hourly continuous;
- local timestamp strings retain both seasonal UTC offsets;
- no missing values in timestamps or indicators;
- both indicators contain only binary integer values;
- values immediately before, on, and after 2020-03-11;
- values immediately before, on, and after 2022-02-24;
- values immediately before, on, and after 2023-05-05;
- identical rules produce identical flag series for both countries;
- overlap is 1 for both flags throughout the inclusive overlap period;
- the summary counts match the generated rows and local timestamp bounds;
- source calendar files, existing demand files, existing temperature files, and
  every raw source file remain byte-for-byte unchanged.

The tests must cover both the first and final available local timestamps for
each flagged period and must check the boundary at the local-date level rather
than by UTC date.

## Files

- Create `code/02_preprocessing/crisis_features.py` for source loading, local
  date classification, validation, summary generation, metadata generation,
  and the command-line entry point.
- Create `code/02_preprocessing/test_crisis_features.py` for unit and
  integration tests.
- Create the two processed crisis tables and the two validation artifacts
  listed above.

No raw file, demand file, temperature file, or completed calendar table is
modified.
