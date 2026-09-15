# Country-Specific Calendar Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and validate separate Germany and Austria hourly calendar-feature tables without modifying existing demand, temperature, or raw datasets.

**Architecture:** A focused preprocessing module reads only the validated `interval_start_utc` keys from the two processed SMARD files, converts them to the country timezone, derives basic integer calendar features, and writes independent country tables. The module uses `holidays==0.103` for statutory holiday dates with `subdiv=None` and `observed=False`, while an independent rule set in tests verifies the package output against the approved official sources. Validation metadata and a country/year summary are written separately under `data/validation/`.

**Tech Stack:** Python 3, pandas 3.0.5, `zoneinfo`, `holidays==0.103`, existing `python-dateutil==2.9.0.post0`, pytest 9.1.1, SHA-256 file hashing.

---

## File Map

- Create `code/02_preprocessing/calendar_features.py`: country configuration, holiday lookup, feature construction, validation, CSV writing, summary writing, metadata writing, and CLI entry point.
- Create `code/02_preprocessing/test_calendar_features.py`: unit tests for holiday rules, local calendar derivation, DST offset preservation, and integration tests against the existing processed key files.
- Modify `requirements.txt`: add the exact dependency `holidays==0.103`.
- Create `data/processed/calendar_germany_hourly.csv`: generated Germany calendar table.
- Create `data/processed/calendar_austria_hourly.csv`: generated Austria calendar table.
- Create `data/validation/calendar_holiday_summary.csv`: country/year holiday-date and holiday-hour counts.
- Create `data/validation/calendar_feature_metadata.json`: package/source configuration, hashes, and validation results.
- Do not modify any file under `data/raw/`, `data/processed/smard_*.csv`, or `data/processed/temperature_*.csv`.

### Task 1: Pin the dependency and write the failing test suite

**Files:**
- Modify: `requirements.txt`
- Create: `code/02_preprocessing/test_calendar_features.py`

- [ ] **Step 1: Add the exact holiday-calendar dependency.**

Add this line to `requirements.txt` in the existing alphabetized dependency list:

```text
holidays==0.103
```

Install it only into the project interpreter:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install holidays==0.103
```

Verify the installed package version:

```powershell
& ".\.venv\Scripts\python.exe" -c "from importlib.metadata import version; print(version('holidays'))"
```

Expected output: `0.103`.

- [ ] **Step 2: Write the failing tests before creating the implementation module.**

Create tests with these concrete behaviors:

```python
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from calendar_features import (
    COUNTRY_CONFIG,
    OUTPUT_COLUMNS,
    build_calendar_features,
    holiday_dates,
    validate_calendar_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "country, expected_count, excluded",
    [
        (
            "Germany",
            9,
            {date(2020, 6, 11), date(2020, 10, 31)},
        ),
        (
            "Austria",
            13,
            {date(2020, 4, 10), date(2020, 11, 2)},
        ),
    ],
)
def test_holiday_dates_use_only_the_countrywide_statutory_set(
    country, expected_count, excluded
):
    dates = holiday_dates(country, [2020])
    assert len(dates) == expected_count
    assert excluded.isdisjoint(dates)


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_calendar_features_derive_from_local_date_and_preserve_dst_offsets(country):
    timestamps = pd.date_range(
        "2020-10-25 00:00:00+00:00", periods=5, freq="h"
    )
    result = build_calendar_features(timestamps, country)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["timestamp_utc"].is_unique
    assert result["timestamp_local"].dt.tz is not None
    assert result.loc[2, "timestamp_local"].hour == 2
    assert result.loc[3, "timestamp_local"].hour == 2
    assert result.loc[2, "timestamp_local"].utcoffset() != result.loc[3, "timestamp_local"].utcoffset()
    assert result.loc[2, "timestamp_local"].strftime("%z") != result.loc[3, "timestamp_local"].strftime("%z")


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_calendar_features_use_local_holiday_date(country):
    timestamps = pd.to_datetime(
        [
            "2020-04-13 21:00:00+00:00",
            "2020-04-13 22:00:00+00:00",
            "2020-04-14 22:00:00+00:00",
        ],
        utc=True,
    )
    result = build_calendar_features(timestamps, country)

    assert result["timestamp_local"].dt.date.tolist() == [
        date(2020, 4, 13),
        date(2020, 4, 14),
        date(2020, 4, 15),
    ]
    assert result["is_public_holiday"].tolist() == [1, 0, 0]


def test_calendar_features_have_basic_integer_ranges_and_weekend_consistency():
    timestamps = pd.date_range("2020-01-03", periods=72, freq="h", tz="UTC")
    result = build_calendar_features(timestamps, "Germany")

    assert result["hour"].between(0, 23).all()
    assert result["day_of_week"].between(0, 6).all()
    assert result["month"].between(1, 12).all()
    assert result["is_weekend"].isin([0, 1]).all()
    assert result["is_public_holiday"].isin([0, 1]).all()
    assert (result["is_weekend"] == result["day_of_week"].ge(5).astype(int)).all()
    assert not result.drop(columns=["timestamp_utc", "timestamp_local"]).isna().any().any()


@pytest.mark.parametrize(
    "country, source_name",
    [
        ("Germany", "smard_germany_hourly.csv"),
        ("Austria", "smard_austria_hourly.csv"),
    ],
)
def test_full_project_key_set_validates_without_changing_source(country, source_name):
    source = pd.read_csv(PROJECT_ROOT / "data" / "processed" / source_name)
    keys = pd.to_datetime(source["interval_start_utc"], utc=True)
    result = build_calendar_features(keys, country)
    validation = validate_calendar_features(result, keys, country)

    assert validation["row_count"] == 52_608
    assert validation["timestamp_utc_unique"] is True
    assert validation["timestamp_utc_hourly_continuous"] is True
    assert validation["missing_feature_count"] == 0
    assert validation["autumn_repeated_local_rows"] == 12
```

- [ ] **Step 3: Run the new tests and confirm the expected missing-module failure.**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/02_preprocessing/test_calendar_features.py -q
```

Expected result: collection fails because `calendar_features.py` does not yet
exist. This confirms the tests are exercising the new feature rather than an
existing implementation.

### Task 2: Implement the pure calendar feature builder

**Files:**
- Create: `code/02_preprocessing/calendar_features.py`

- [ ] **Step 1: Add configuration and constants.**

Use this module-level structure:

```python
from __future__ import annotations

from datetime import date
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
VALIDATION = PROJECT_ROOT / "data" / "validation"
EXPECTED_ROWS = 52_608
HOLIDAYS_VERSION = version("holidays")
HOLIDAY_CATEGORY = holidays.constants.PUBLIC
OUTPUT_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday",
]
COUNTRY_CONFIG = {
    "Germany": {
        "code": "DE",
        "timezone": "Europe/Berlin",
        "source_file": PROCESSED / "smard_germany_hourly.csv",
        "output_file": PROCESSED / "calendar_germany_hourly.csv",
        "expected_holiday_dates": 9,
    },
    "Austria": {
        "code": "AT",
        "timezone": "Europe/Vienna",
        "source_file": PROCESSED / "smard_austria_hourly.csv",
        "output_file": PROCESSED / "calendar_austria_hourly.csv",
        "expected_holiday_dates": 13,
    },
}
```

Add official-source metadata constants:

```python
OFFICIAL_SOURCES = {
    "Germany": {
        "name": "Federal Ministry of the Interior, nationwide public holidays",
        "url": "https://www.bmi.bund.de/DE/themen/verfassung/staatliche-symbole/feiertage/feiertage-node.html",
    },
    "Austria": {
        "name": "RIS Feiertagsruhegesetz 1957, Art. 1 Section 1",
        "url": "https://www.ris.bka.gv.at/eli/bgbl/1957/153/A1P1/NOR40213432",
    },
}
```

- [ ] **Step 2: Implement package-backed holiday lookup.**

Implement this function so production flags come only from the pinned
`holidays` package and only from local calendar dates:

```python
def holiday_dates(country: str, years: Iterable[int]) -> set[date]:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    requested_years = sorted({int(year) for year in years})
    calendar = holidays.country_holidays(
        COUNTRY_CONFIG[country]["code"],
        years=requested_years,
        subdiv=None,
        observed=False,
        categories=HOLIDAY_CATEGORY,
    )
    return {
        holiday_date
        for holiday_date in calendar
        if holiday_date.year in requested_years
    }
```

- [ ] **Step 3: Implement the pure UTC-to-local feature builder.**

Use an aware UTC index, convert it to the country timezone before extracting
calendar values, and retain timezone-aware timestamp objects in memory:

```python
def build_calendar_features(
    timestamps_utc: pd.Series | pd.DatetimeIndex,
    country: str,
) -> pd.DataFrame:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    utc = pd.DatetimeIndex(pd.to_datetime(timestamps_utc, utc=True))
    if utc.isna().any():
        raise ValueError("UTC timestamps contain missing values")
    if not utc.is_unique:
        raise ValueError("UTC timestamps contain duplicates")
    if len(utc) > 1 and not (utc[1:] > utc[:-1]).all():
        raise ValueError("UTC timestamps are not strictly ordered")
    if len(utc) > 1 and not (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("UTC timestamps are not hourly continuous")

    local = utc.tz_convert(ZoneInfo(COUNTRY_CONFIG[country]["timezone"]))
    local_dates = pd.Series(local.date)
    holidays_for_years = holiday_dates(country, local_dates.map(lambda value: value.year))
    day_of_week = local.dayofweek.astype("int8")
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local,
            "hour": local.hour.astype("int8"),
            "day_of_week": day_of_week,
            "month": local.month.astype("int8"),
            "is_weekend": day_of_week.ge(5).astype("int8"),
            "is_public_holiday": local_dates.isin(holidays_for_years).astype("int8"),
        }
    )
```

- [ ] **Step 4: Run the focused tests and confirm they pass.**

Run:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/02_preprocessing/test_calendar_features.py -q
```

Expected result: the holiday-set, local-date, DST, range, weekend, and full-key
tests pass. Fix implementation defects without weakening the tests.

### Task 3: Add independent official-rule verification and validation

**Files:**
- Modify: `code/02_preprocessing/calendar_features.py`
- Modify: `code/02_preprocessing/test_calendar_features.py`

- [ ] **Step 1: Add independent expected-date rules to the test module.**

Use the already pinned `python-dateutil` package only for an independent check;
do not use this helper to generate production flags:

```python
from dateutil.easter import easter


def official_reference_dates(country: str, year: int) -> set[date]:
    easter_sunday = easter(year)
    fixed = {
        date(year, 1, 1),
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
    }
    if country == "Germany":
        fixed |= {date(year, 10, 3)}
        movable = {
            easter_sunday - pd.Timedelta(days=2),
            easter_sunday + pd.Timedelta(days=1),
            easter_sunday + pd.Timedelta(days=39),
            easter_sunday + pd.Timedelta(days=50),
        }
    elif country == "Austria":
        fixed |= {
            date(year, 1, 6), date(year, 8, 15), date(year, 10, 26),
            date(year, 11, 1), date(year, 12, 8),
        }
        movable = {
            easter_sunday + pd.Timedelta(days=1),
            easter_sunday + pd.Timedelta(days=39),
            easter_sunday + pd.Timedelta(days=50),
            easter_sunday + pd.Timedelta(days=60),
        }
    else:
        raise ValueError(country)
    return fixed | {value.date() for value in movable}


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_package_holiday_dates_match_independent_official_rules(country):
    for year in range(2020, 2026):
        assert holiday_dates(country, [year]) == official_reference_dates(country, year)
```

The production module may use `datetime.timedelta` rather than pandas timedeltas
for its own code; the test must convert every result to `datetime.date` before
comparison.

- [ ] **Step 2: Implement `validate_calendar_features`.**

The function must compare the generated frame to the exact source key series
and return JSON-serializable validation evidence:

```python
def validate_calendar_features(
    frame: pd.DataFrame,
    source_keys: pd.Series | pd.DatetimeIndex,
    country: str,
) -> dict[str, object]:
    config = COUNTRY_CONFIG[country]
    expected_utc = pd.DatetimeIndex(pd.to_datetime(source_keys, utc=True))
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{country} calendar columns do not match the schema")
    if len(frame) != EXPECTED_ROWS or len(frame) != len(expected_utc):
        raise ValueError(f"{country} calendar row count is not {EXPECTED_ROWS}")
    utc = pd.DatetimeIndex(frame["timestamp_utc"])
    local = pd.DatetimeIndex(frame["timestamp_local"])
    if not utc.equals(expected_utc):
        raise ValueError(f"{country} UTC keys changed")
    if utc.tz is None or str(utc.tz) != "UTC":
        raise ValueError(f"{country} UTC key is not timezone-aware UTC")
    expected_local = utc.tz_convert(config["timezone"])
    if not local.equals(expected_local):
        raise ValueError(f"{country} local timestamps do not match the timezone conversion")
    feature_columns = OUTPUT_COLUMNS[2:]
    if frame[feature_columns].isna().any().any():
        raise ValueError(f"{country} calendar features contain missing values")
    if not frame["hour"].between(0, 23).all():
        raise ValueError(f"{country} hour is outside 0..23")
    if not frame["day_of_week"].between(0, 6).all():
        raise ValueError(f"{country} day_of_week is outside 0..6")
    if not frame["month"].between(1, 12).all():
        raise ValueError(f"{country} month is outside 1..12")
    for column in ("is_weekend", "is_public_holiday"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{country} {column} is not binary")
    expected_weekend = frame["day_of_week"].ge(5).astype("int8")
    if not frame["is_weekend"].eq(expected_weekend).all():
        raise ValueError(f"{country} weekend flags disagree with day_of_week")

    local_dates = pd.Series(local.date)
    years = sorted(local_dates.map(lambda value: value.year).unique().tolist())
    expected_holidays = holiday_dates(country, years)
    expected_flags = local_dates.isin(expected_holidays).astype("int8")
    if not frame["is_public_holiday"].eq(expected_flags).all():
        raise ValueError(f"{country} holiday flags disagree with local dates")
    wall_times = local.tz_localize(None)
    repeated_rows = int(wall_times.duplicated(keep=False).sum())
    if repeated_rows != 12:
        raise ValueError(f"{country} autumn repeated-hour rows are not preserved")
    holiday_summary = {}
    for year in years:
        year_mask = local_dates.map(lambda value: value.year).eq(year)
        holiday_mask = year_mask & frame["is_public_holiday"].eq(1)
        holiday_summary[str(year)] = {
            "holiday_date_count": int(local_dates[holiday_mask].nunique()),
            "holiday_hour_observation_count": int(holiday_mask.sum()),
        }
        if holiday_summary[str(year)]["holiday_date_count"] != config["expected_holiday_dates"]:
            raise ValueError(f"{country} has the wrong annual holiday-date count")
    return {
        "country": country,
        "row_count": int(len(frame)),
        "timestamp_utc_unique": bool(utc.is_unique),
        "timestamp_utc_hourly_continuous": True,
        "timestamp_local_offset_preserved": True,
        "autumn_repeated_local_rows": repeated_rows,
        "missing_feature_count": int(frame[feature_columns].isna().sum().sum()),
        "holiday_summary": holiday_summary,
    }
```

The implementation should use explicit integer dtypes (`int8` or `int64`) for
the five derived integer/binary columns and should assert that the UTC series
is unique, ordered, and one-hour continuous before building features.

- [ ] **Step 3: Add tests for the full official rule set and validation failures.**

Add tests that deliberately change one holiday flag, one weekend flag, and one
UTC key in a copy of a valid frame, then assert `validate_calendar_features`
raises `ValueError`. Add a test that serializes both repeated local timestamps
with `Timestamp.isoformat()` and asserts the resulting strings contain both
`+02:00` and `+01:00`.

- [ ] **Step 4: Run focused tests again.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/02_preprocessing/test_calendar_features.py -q
```

Expected result: all focused tests pass, including independent Germany and
Austria holiday-rule comparisons for 2020-2025.

### Task 4: Implement file generation, summary, metadata, and CLI

**Files:**
- Modify: `code/02_preprocessing/calendar_features.py`

- [ ] **Step 1: Implement source-key loading and ISO serialization.**

Read only `interval_start_utc` from each configured SMARD file, parse it with
`utc=True`, and preserve the exact source bytes for the later immutability
check. Serialize timestamps with `Timestamp.isoformat()` before writing so the
local offset is explicit:

```python
def _read_source_keys(path: Path) -> tuple[pd.Series, str]:
    source_hash = sha256_file(path)
    data = pd.read_csv(path, usecols=["interval_start_utc"])
    keys = pd.to_datetime(data["interval_start_utc"], utc=True)
    return keys, source_hash


def _write_calendar_table(frame: pd.DataFrame, path: Path) -> str:
    output = frame.copy()
    output["timestamp_utc"] = output["timestamp_utc"].map(pd.Timestamp.isoformat)
    output["timestamp_local"] = output["timestamp_local"].map(pd.Timestamp.isoformat)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return sha256_file(path)
```

- [ ] **Step 2: Implement summary and metadata writers.**

Write one summary row for each country/year from the validation result:

```python
def _write_summary(validations: dict[str, dict[str, object]]) -> str:
    rows = []
    for country, validation in validations.items():
        for year, counts in validation["holiday_summary"].items():
            rows.append(
                {
                    "country": country,
                    "year": int(year),
                    **counts,
                }
            )
    summary = pd.DataFrame(rows).sort_values(["country", "year"])
    path = VALIDATION / "calendar_holiday_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return sha256_file(path)
```

The metadata JSON must include:

```python
{
    "holiday_package": "holidays",
    "holiday_package_version": HOLIDAYS_VERSION,
    "holiday_category": "PUBLIC",
    "observed_dates": False,
    "subdivision": None,
    "official_sources": OFFICIAL_SOURCES,
    "source_files": {},
    "output_files": {},
    "summary_file": "calendar_holiday_summary.csv",
    "validation": {},
}
```

Populate `source_files`, `output_files`, and `validation` for both countries;
include the summary hash and the exact generation date. Do not write any source
SMARD or temperature file.

- [ ] **Step 3: Implement `run_pipeline` and the command-line entry point.**

Use this flow:

```python
def run_pipeline() -> dict[str, object]:
    validations = {}
    metadata = {
        "holiday_package": "holidays",
        "holiday_package_version": HOLIDAYS_VERSION,
        "holiday_category": "PUBLIC",
        "observed_dates": False,
        "subdivision": None,
        "official_sources": OFFICIAL_SOURCES,
        "source_files": {},
        "output_files": {},
        "validation": {},
    }
    for country, config in COUNTRY_CONFIG.items():
        keys, source_hash = _read_source_keys(config["source_file"])
        frame = build_calendar_features(keys, country)
        validation = validate_calendar_features(frame, keys, country)
        output_hash = _write_calendar_table(frame, config["output_file"])
        metadata["source_files"][country] = {
            "filename": config["source_file"].name,
            "sha256_before": source_hash,
            "sha256_after": sha256_file(config["source_file"]),
        }
        metadata["output_files"][country] = {
            "filename": config["output_file"].name,
            "sha256": output_hash,
        }
        validations[country] = validation
    metadata["validation"] = validations
    metadata["summary_file_sha256"] = _write_summary(validations)
    metadata_path = VALIDATION / "calendar_feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
```

- [ ] **Step 4: Run the generation script.**

```powershell
& ".\.venv\Scripts\python.exe" code/02_preprocessing/calendar_features.py
```

Expected result: two 52,608-row calendar files, one 12-row summary covering
Germany and Austria for 2020-2025, and metadata JSON with matching source
hashes before and after generation.

### Task 5: Full validation and artifact review

**Files:**
- Read: `data/processed/calendar_germany_hourly.csv`
- Read: `data/processed/calendar_austria_hourly.csv`
- Read: `data/validation/calendar_holiday_summary.csv`
- Read: `data/validation/calendar_feature_metadata.json`
- Read: all existing processed demand and temperature CSV files

- [ ] **Step 1: Run the complete test suite.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code -q
```

Expected result: all existing and new tests pass with no warnings or errors.

- [ ] **Step 2: Verify dependency consistency.**

```powershell
& ".\.venv\Scripts\python.exe" -m pip check
```

Expected result: `No broken requirements found.`

- [ ] **Step 3: Independently inspect both generated CSVs.**

Run a project-interpreter check that reads both tables and asserts:

```python
assert len(table) == 52_608
assert list(table.columns) == OUTPUT_COLUMNS
assert table["timestamp_utc"].is_unique
assert table["timestamp_utc"].is_monotonic_increasing
assert table["timestamp_local"].str.contains(r"[+-][0-9]{2}:[0-9]{2}$", regex=True).all()
assert table["hour"].between(0, 23).all()
assert table["day_of_week"].between(0, 6).all()
assert table["month"].between(1, 12).all()
assert table["is_weekend"].isin([0, 1]).all()
assert table["is_public_holiday"].isin([0, 1]).all()
```

Also confirm the autumn wall-clock value appears twice with different offset
strings, not just twice with identical serialized timestamps.

- [ ] **Step 4: Verify all annual holiday counts and DST-sensitive hourly counts.**

Read the summary CSV and assert:

```python
summary = pd.read_csv("data/validation/calendar_holiday_summary.csv")
assert set(summary["country"]) == {"Germany", "Austria"}
assert set(summary["year"]) == set(range(2020, 2026))
assert summary.loc[summary["country"].eq("Germany"), "holiday_date_count"].eq(9).all()
assert summary.loc[summary["country"].eq("Austria"), "holiday_date_count"].eq(13).all()
assert summary["holiday_hour_observation_count"].gt(0).all()
```

Report date counts separately from hourly counts. Explain any annual hourly
count that differs from `holiday_date_count * 24` by identifying a holiday on
an autumn 25-hour date or spring 23-hour date.

- [ ] **Step 5: Verify existing file immutability.**

Compare the metadata `sha256_before` and `sha256_after` values for both SMARD
inputs and independently hash these unchanged files:

- `data/processed/smard_germany_hourly.csv`
- `data/processed/smard_austria_hourly.csv`
- `data/processed/temperature_germany_hourly.csv`
- `data/processed/temperature_austria_hourly.csv`

Confirm no file under `data/raw/` has a changed modification time or hash. The
final report must list every changed file and explicitly state that no existing
load, temperature, timestamp value, or row was changed.

- [ ] **Step 6: Run the final verification command set and record exact output.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code -q
& ".\.venv\Scripts\python.exe" -m pip check
& ".\.venv\Scripts\python.exe" code/02_preprocessing/calendar_features.py
```

Use the actual command output and generated metadata for the final report. Do
not claim completion until all tests, package checks, row counts, holiday
counts, DST-offset checks, source hashes, and existing-data immutability checks
have passed.

No git commit is included because the project instructions prohibit
initializing or modifying git state without explicit user approval.
