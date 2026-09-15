# Crisis Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate independent Germany and Austria crisis-indicator tables from the completed calendar tables while preserving all source timestamp strings and protected datasets.

**Architecture:** A small preprocessing module reads each calendar CSV as strings, validates the UTC key and offset-aware local timestamp columns, derives both indicators from parsed local dates, and writes new crisis CSVs. It also writes a one-row-per-country validation summary and source metadata. No existing calendar, demand, temperature, or raw file is opened for writing.

**Tech Stack:** Python 3, pandas 3.0.5, `zoneinfo`/`datetime`, pytest 9.1.1, SHA-256 hashing. No new dependency is required.

---

## File Map

- Create `code/02_preprocessing/crisis_features.py`: configuration, source loading, local-date rules, validation, output writing, summary, metadata, and CLI.
- Create `code/02_preprocessing/test_crisis_features.py`: unit and integration tests.
- Create `data/processed/crisis_germany_hourly.csv`.
- Create `data/processed/crisis_austria_hourly.csv`.
- Create `data/validation/crisis_feature_summary.csv`.
- Create `data/validation/crisis_feature_metadata.json`.
- Do not modify `data/processed/calendar_*.csv`, demand files, temperature files, or any `data/raw/` file.

### Task 1: Write the failing crisis-feature tests

**Files:**
- Create: `code/02_preprocessing/test_crisis_features.py`

- [x] **Step 1: Add tests for boundary rules, source copying, and binary flags.**

Create these tests before creating the production module:

```python
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from crisis_features import (
    OUTPUT_COLUMNS,
    build_crisis_features,
    summarize_crisis_features,
    validate_crisis_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def source_for_utc(values: list[str], timezone_name: str) -> pd.DataFrame:
    utc = pd.to_datetime(values, utc=True)
    local = utc.tz_convert(timezone_name)
    return pd.DataFrame(
        {
            "timestamp_utc": [value.isoformat() for value in utc],
            "timestamp_local": [value.isoformat() for value in local],
        }
    )


def test_boundary_dates_use_the_local_calendar_date():
    source = source_for_utc(
        [
            "2020-03-10 22:00:00+00:00",
            "2020-03-10 23:00:00+00:00",
            "2022-02-23 22:00:00+00:00",
            "2022-02-23 23:00:00+00:00",
            "2023-05-05 21:00:00+00:00",
            "2023-05-05 22:00:00+00:00",
            "2023-05-05 23:00:00+00:00",
            "2023-05-06 00:00:00+00:00",
        ],
        "Europe/Berlin",
    )

    result = build_crisis_features(source)

    assert result["is_covid_period"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert result["is_post_invasion"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert result.loc[3, "is_covid_period"] == 1
    assert result.loc[3, "is_post_invasion"] == 1


@pytest.mark.parametrize("country, timezone_name", [
    ("Germany", "Europe/Berlin"),
    ("Austria", "Europe/Vienna"),
])
def test_builder_preserves_timestamp_strings_and_offset_values(country, timezone_name):
    source = source_for_utc(
        [
            "2020-10-24 22:00:00+00:00",
            "2020-10-24 23:00:00+00:00",
            "2020-10-25 00:00:00+00:00",
            "2020-10-25 01:00:00+00:00",
            "2020-10-25 02:00:00+00:00",
        ],
        timezone_name,
    )

    result = build_crisis_features(source)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert result["timestamp_utc"].tolist() == source["timestamp_utc"].tolist()
    assert result["timestamp_local"].tolist() == source["timestamp_local"].tolist()
    assert any(value.endswith("+02:00") for value in result["timestamp_local"])
    assert any(value.endswith("+01:00") for value in result["timestamp_local"])
    assert result["is_covid_period"].isin([0, 1]).all()
    assert result["is_post_invasion"].isin([0, 1]).all()
    assert str(result["is_covid_period"].dtype).startswith("int")
    assert str(result["is_post_invasion"].dtype).startswith("int")


def test_overlap_is_one_for_both_indicators():
    source = source_for_utc(
        [
            "2022-02-24 00:00:00+00:00",
            "2023-05-05 21:00:00+00:00",
        ],
        "Europe/Berlin",
    )

    result = build_crisis_features(source)

    assert (result["is_covid_period"] == 1).all()
    assert (result["is_post_invasion"] == 1).all()


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_full_calendar_input_validates_without_source_changes(country):
    source_path = PROJECT_ROOT / "data" / "processed" / f"calendar_{country.lower()}_hourly.csv"
    before = source_path.read_bytes()
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False)
    source = source[["timestamp_utc", "timestamp_local"]].copy()

    result = build_crisis_features(source)
    validation = validate_crisis_features(result, source, country)

    assert source_path.read_bytes() == before
    assert validation["row_count"] == 52_608
    assert validation["timestamp_utc_unique"] is True
    assert validation["timestamp_utc_hourly_continuous"] is True
    assert validation["missing_value_count"] == 0
    assert validation["timestamp_local_offset_count"] == 2


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_summary_contains_counts_and_local_bounds(country):
    source_path = PROJECT_ROOT / "data" / "processed" / f"calendar_{country.lower()}_hourly.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False)
    source = source[["timestamp_utc", "timestamp_local"]].copy()
    result = build_crisis_features(source)

    summary = summarize_crisis_features(result, country)

    assert summary["country"] == country
    assert summary["total_observations"] == 52_608
    assert summary["covid_period_hours"] > 0
    assert summary["post_invasion_hours"] > 0
    assert summary["overlap_hours"] > 0
    assert summary["covid_first_local_timestamp"].startswith("2020-03-11")
    assert summary["covid_last_local_timestamp"].startswith("2023-05-05")
    assert summary["post_invasion_first_local_timestamp"].startswith("2022-02-24")


def test_validation_rejects_non_hourly_utc_keys():
    source_path = PROJECT_ROOT / "data" / "processed" / "calendar_germany_hourly.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False)
    source = source[["timestamp_utc", "timestamp_local"]].copy()
    source.loc[1, "timestamp_utc"] = "2020-01-01T00:30:00+00:00"
    result = build_crisis_features(source)

    with pytest.raises(ValueError, match="hourly"):
        validate_crisis_features(result, source, "Germany")
```

- [x] **Step 2: Run the new tests and confirm the expected missing-module failure.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/02_preprocessing/test_crisis_features.py -q
```

Expected result: collection fails because `crisis_features.py` does not yet
exist.

### Task 2: Implement the pure crisis-feature builder and validator

**Files:**
- Create: `code/02_preprocessing/crisis_features.py`

- [x] **Step 1: Add constants and source configuration.**

Use this structure:

```python
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
VALIDATION = PROJECT_ROOT / "data" / "validation"
EXPECTED_ROWS = 52_608
OUTPUT_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "is_covid_period",
    "is_post_invasion",
]
COVID_START = date(2020, 3, 11)
COVID_END = date(2023, 5, 5)
INVASION_START = date(2022, 2, 24)
COUNTRY_CONFIG = {
    "Germany": {
        "input_file": PROCESSED / "calendar_germany_hourly.csv",
        "output_file": PROCESSED / "crisis_germany_hourly.csv",
    },
    "Austria": {
        "input_file": PROCESSED / "calendar_austria_hourly.csv",
        "output_file": PROCESSED / "crisis_austria_hourly.csv",
    },
}
CRISIS_SOURCES = [
    {
        "institution": "World Health Organization",
        "title": "WHO Director-General's opening remarks at the media briefing on COVID-19 - 11 March 2020",
        "url": "https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020",
        "publication_date": "2020-03-11",
    },
    {
        "institution": "World Health Organization",
        "title": "WHO Director-General's opening remarks at the media briefing - 5 May 2023",
        "url": "https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-the-media-briefing---5-may-2023",
        "publication_date": "2023-05-05",
    },
    {
        "institution": "European Council",
        "title": "Statement by the members of the European Council on the Russian military aggression against Ukraine",
        "url": "https://www.consilium.europa.eu/en/press/press-releases/2022/02/24/statement-by-the-members-of-the-european-council-on-the-russian-military-aggression-against-ukraine/",
        "publication_date": "2022-02-24",
    },
    {
        "institution": "International Energy Agency",
        "title": "World Energy Outlook 2022",
        "url": "https://www.iea.org/reports/world-energy-outlook-2022",
        "publication_date": "2022-10-27",
    },
]
DEFINITIONS = {
    "is_covid_period": {
        "start_local_date": "2020-03-11",
        "end_local_date": "2023-05-05",
        "inclusive": True,
    },
    "is_post_invasion": {
        "start_local_date": "2022-02-24",
        "end_local_date": "available_dataset_end",
        "inclusive": True,
    },
}
```

- [x] **Step 2: Implement source parsing without rewriting timestamps.**

Read calendar files with `dtype=str`, `keep_default_na=False`, and
`na_filter=False`. Keep a raw two-column frame and parse copies only:

```python
def _parse_local_values(values: pd.Series) -> list[pd.Timestamp]:
    parsed = []
    for value in values:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("local timestamps must contain UTC offsets")
        if not re.search(r"[+-][0-9]{2}:[0-9]{2}$", value):
            raise ValueError("local timestamps are not ISO 8601 offset-aware strings")
        parsed.append(timestamp)
    return parsed


def _validate_timestamp_source(source: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[pd.Timestamp]]:
    if list(source.columns) != ["timestamp_utc", "timestamp_local"]:
        raise ValueError("calendar source must contain only the two timestamp columns")
    if len(source) != EXPECTED_ROWS:
        raise ValueError(f"calendar source must contain {EXPECTED_ROWS} rows")
    if source.isna().any().any() or (source == "").any().any():
        raise ValueError("calendar source contains missing timestamp values")
    utc = pd.DatetimeIndex(pd.to_datetime(source["timestamp_utc"], utc=True))
    if utc.isna().any() or not utc.is_unique or not utc.is_monotonic_increasing:
        raise ValueError("calendar UTC timestamps are incomplete or unordered")
    if not (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("calendar UTC timestamps are not hourly continuous")
    local = _parse_local_values(source["timestamp_local"])
    return utc, local


def build_crisis_features(source: pd.DataFrame) -> pd.DataFrame:
    if list(source.columns) != ["timestamp_utc", "timestamp_local"]:
        raise ValueError("calendar source must contain only the two timestamp columns")
    if source.isna().any().any() or (source == "").any().any():
        raise ValueError("calendar source contains missing timestamp values")
    if pd.to_datetime(source["timestamp_utc"], utc=True).isna().any():
        raise ValueError("calendar UTC timestamps contain missing values")
    local = _parse_local_values(source["timestamp_local"])
    local_dates = pd.Series([timestamp.date() for timestamp in local])
    covid = local_dates.between(COVID_START, COVID_END).astype("int8")
    post_invasion = local_dates.ge(INVASION_START).astype("int8")
    result = source[["timestamp_utc", "timestamp_local"]].copy()
    result["is_covid_period"] = covid
    result["is_post_invasion"] = post_invasion
    return result[OUTPUT_COLUMNS]
```

The production builder must use the raw source strings in the result and never
call `isoformat()` on the timestamp columns.

- [x] **Step 3: Implement crisis validation and summary functions.**

Use exact string equality for timestamp preservation and parse-only copies for
key validation:

```python
def validate_crisis_features(
    result: pd.DataFrame,
    source: pd.DataFrame,
    country: str,
) -> dict[str, object]:
    if list(result.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{country} crisis columns do not match the schema")
    if len(result) != EXPECTED_ROWS or len(source) != EXPECTED_ROWS:
        raise ValueError(f"{country} crisis table must contain {EXPECTED_ROWS} rows")
    if not result[["timestamp_utc", "timestamp_local"]].equals(
        source[["timestamp_utc", "timestamp_local"]]
    ):
        raise ValueError(f"{country} timestamps changed")
    utc, local = _validate_timestamp_source(result[["timestamp_utc", "timestamp_local"]])
    for column in ("is_covid_period", "is_post_invasion"):
        if not pd.api.types.is_integer_dtype(result[column]):
            raise ValueError(f"{country} {column} is not an integer")
        if result[column].isna().any() or not result[column].isin([0, 1]).all():
            raise ValueError(f"{country} {column} is not binary and complete")
    local_dates = pd.Series([timestamp.date() for timestamp in local])
    expected_covid = local_dates.between(COVID_START, COVID_END).astype("int8")
    expected_post = local_dates.ge(INVASION_START).astype("int8")
    if not result["is_covid_period"].equals(expected_covid):
        raise ValueError(f"{country} COVID flag is incorrect")
    if not result["is_post_invasion"].equals(expected_post):
        raise ValueError(f"{country} post-invasion flag is incorrect")
    return {
        "country": country,
        "row_count": int(len(result)),
        "timestamp_utc_unique": bool(utc.is_unique),
        "timestamp_utc_hourly_continuous": True,
        "timestamp_local_offset_count": len({timestamp.strftime("%z") for timestamp in local}),
        "missing_value_count": int(result.isna().sum().sum()),
        "covid_period_hours": int(result["is_covid_period"].sum()),
        "post_invasion_hours": int(result["is_post_invasion"].sum()),
        "overlap_hours": int((result["is_covid_period"] & result["is_post_invasion"]).sum()),
    }


def summarize_crisis_features(result: pd.DataFrame, country: str) -> dict[str, object]:
    local_values = result["timestamp_local"].tolist()
    summary = {
        "country": country,
        "total_observations": int(len(result)),
        "covid_period_hours": int(result["is_covid_period"].sum()),
        "post_invasion_hours": int(result["is_post_invasion"].sum()),
        "overlap_hours": int((result["is_covid_period"] & result["is_post_invasion"]).sum()),
    }
    for column, prefix in (
        ("is_covid_period", "covid"),
        ("is_post_invasion", "post_invasion"),
    ):
        indices = result.index[result[column].eq(1)]
        summary[f"{prefix}_first_local_timestamp"] = local_values[indices[0]]
        summary[f"{prefix}_last_local_timestamp"] = local_values[indices[-1]]
    return summary
```

- [x] **Step 4: Run focused tests and make them pass.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code/02_preprocessing/test_crisis_features.py -q
```

Expected result: all unit and full-calendar integration tests pass.

### Task 3: Implement output generation, validation summary, and metadata

**Files:**
- Modify: `code/02_preprocessing/crisis_features.py`

- [x] **Step 1: Add file hashing and source loading.**

Implement `sha256_file` and `_read_calendar_source` so each input hash is
recorded before and after processing:

```python
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_calendar_source(path: Path) -> tuple[pd.DataFrame, str]:
    source_hash = sha256_file(path)
    source = pd.read_csv(
        path,
        usecols=["timestamp_utc", "timestamp_local"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    return source, source_hash
```

- [x] **Step 2: Add output, summary, and metadata writers.**

Write timestamps without transformation. The summary must use the exact local
timestamp strings from each output:

```python
def _write_output(result: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return sha256_file(path)


def _write_summary(rows: list[dict[str, object]]) -> str:
    summary = pd.DataFrame(rows).sort_values("country")
    path = VALIDATION / "crisis_feature_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return sha256_file(path)
```

Metadata must include:

```python
metadata = {
    "definitions": {
        "is_covid_period": {
            "start_local_date": "2020-03-11",
            "end_local_date": "2023-05-05",
            "inclusive": True,
        },
        "is_post_invasion": {
            "start_local_date": "2022-02-24",
            "end_local_date": "available_dataset_end",
            "inclusive": True,
        },
    },
    "official_sources": CRISIS_SOURCES,
    "access_date": date.today().isoformat(),
    "source_files": {},
    "output_files": {},
    "validation": {},
}
```

Record each calendar input hash before and after, each crisis output hash, the
summary hash, and the validation dictionary for both countries.

- [x] **Step 3: Add the CLI pipeline.**

Implement this execution flow:

```python
def run_pipeline() -> dict[str, object]:
    rows = []
    metadata = {
        "definitions": DEFINITIONS,
        "official_sources": CRISIS_SOURCES,
        "access_date": date.today().isoformat(),
        "source_files": {},
        "output_files": {},
        "validation": {},
    }
    for country, config in COUNTRY_CONFIG.items():
        source, source_hash = _read_calendar_source(config["input_file"])
        result = build_crisis_features(source)
        validation = validate_crisis_features(result, source, country)
        summary = summarize_crisis_features(result, country)
        output_hash = _write_output(result, config["output_file"])
        metadata["source_files"][country] = {
            "filename": config["input_file"].name,
            "sha256_before": source_hash,
            "sha256_after": sha256_file(config["input_file"]),
        }
        metadata["output_files"][country] = {
            "filename": config["output_file"].name,
            "sha256": output_hash,
        }
        metadata["validation"][country] = validation
        rows.append(summary)
    metadata["summary_file_sha256"] = _write_summary(rows)
    metadata_path = VALIDATION / "crisis_feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
```

- [x] **Step 4: Generate the two crisis tables and validation artifacts.**

```powershell
& ".\.venv\Scripts\python.exe" code/02_preprocessing/crisis_features.py
```

Expected result: two 52,608-row crisis tables, a two-row validation summary,
and metadata recording all four official sources.

### Task 4: Full verification and protected-data audit

**Files:**
- Read: both crisis outputs and both validation artifacts
- Read: both calendar inputs, demand files, temperature files, and raw metadata

- [x] **Step 1: Run all tests and dependency checks.**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest code -q
& ".\.venv\Scripts\python.exe" -m pip check
```

Expected result: every existing and new test passes and pip reports no broken
requirements.

- [x] **Step 2: Independently verify generated crisis artifacts.**

For each country assert:

```python
assert len(output) == 52_608
assert list(output.columns) == OUTPUT_COLUMNS
assert output["timestamp_utc"].tolist() == source["timestamp_utc"].tolist()
assert output["timestamp_local"].tolist() == source["timestamp_local"].tolist()
assert pd.to_datetime(output["timestamp_utc"], utc=True).is_unique
assert output["is_covid_period"].isin([0, 1]).all()
assert output["is_post_invasion"].isin([0, 1]).all()
assert not output.isna().any().any()
```

Check the local timestamp strings end in an explicit offset and include both
`+01:00` and `+02:00`.

- [x] **Step 3: Verify summary boundaries and overlap.**

Assert both countries have the same summary counts and local bounds. Assert
the overlap is positive and that every row where `is_covid_period == 1` and
`is_post_invasion == 1` has a local date from 2022-02-24 through 2023-05-05.

- [x] **Step 4: Verify no existing data changed.**

Compare metadata `sha256_before` and `sha256_after` for both calendar inputs.
Hash the existing calendar, demand, temperature, and raw files independently
and confirm they match the hashes captured before generation. Confirm no raw
file was opened for writing.

- [x] **Step 5: Record the final verification output.**

Do not claim completion until the full test output, pip check output, artifact
checks, summary checks, source hashes, and protected-data checks all pass. No
git commit or remote operation is included because project instructions forbid
initializing or modifying git state without explicit approval.
