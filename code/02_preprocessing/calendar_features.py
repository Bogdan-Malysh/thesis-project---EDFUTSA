from __future__ import annotations

from datetime import date, datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import holidays
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
OFFICIAL_SOURCES = {
    "Germany": {
        "name": "Federal Ministry of the Interior, nationwide public holidays",
        "url": (
            "https://www.bmi.bund.de/DE/themen/verfassung/"
            "staatliche-symbole/feiertage/feiertage-node.html"
        ),
    },
    "Austria": {
        "name": "RIS Feiertagsruhegesetz 1957, Art. 1 Section 1",
        "url": "https://www.ris.bka.gv.at/eli/bgbl/1957/153/A1P1/NOR40213432",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if len(utc) > 1 and not (
        utc[1:] - utc[:-1] == pd.Timedelta(hours=1)
    ).all():
        raise ValueError("UTC timestamps are not hourly continuous")

    local = utc.tz_convert(ZoneInfo(COUNTRY_CONFIG[country]["timezone"]))
    local_dates = pd.Series(local.date)
    years = local_dates.map(lambda value: value.year)
    holidays_for_years = holiday_dates(country, years)
    day_of_week = pd.Series(local.dayofweek, dtype="int8")
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local,
            "hour": pd.Series(local.hour, dtype="int8"),
            "day_of_week": day_of_week,
            "month": pd.Series(local.month, dtype="int8"),
            "is_weekend": day_of_week.ge(5).astype("int8"),
            "is_public_holiday": local_dates.isin(holidays_for_years).astype("int8"),
        }
    )


def validate_calendar_features(
    frame: pd.DataFrame,
    source_keys: pd.Series | pd.DatetimeIndex,
    country: str,
) -> dict[str, object]:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    config = COUNTRY_CONFIG[country]
    expected_utc = pd.DatetimeIndex(pd.to_datetime(source_keys, utc=True))
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{country} calendar columns do not match the schema")
    if len(frame) != len(expected_utc):
        raise ValueError(f"{country} calendar row count does not match source keys")
    full_coverage = len(expected_utc) == EXPECTED_ROWS
    if full_coverage and len(frame) != EXPECTED_ROWS:
        raise ValueError(f"{country} calendar row count is not {EXPECTED_ROWS}")

    utc = pd.DatetimeIndex(frame["timestamp_utc"])
    local = pd.DatetimeIndex(frame["timestamp_local"])
    if not utc.equals(expected_utc):
        raise ValueError(f"{country} UTC keys changed")
    if utc.tz is None or str(utc.tz) != "UTC":
        raise ValueError(f"{country} UTC key is not timezone-aware UTC")
    expected_local = utc.tz_convert(config["timezone"])
    if not local.equals(expected_local):
        raise ValueError(
            f"{country} local timestamps do not match the timezone conversion"
        )

    feature_columns = OUTPUT_COLUMNS[2:]
    if frame[feature_columns].isna().any().any():
        raise ValueError(f"{country} calendar features contain missing values")
    for column, minimum, maximum in (
        ("hour", 0, 23),
        ("day_of_week", 0, 6),
        ("month", 1, 12),
    ):
        if not pd.api.types.is_integer_dtype(frame[column]):
            raise ValueError(f"{country} {column} is not an integer")
        if not frame[column].between(minimum, maximum).all():
            raise ValueError(f"{country} {column} is outside its allowed range")
    for column in ("is_weekend", "is_public_holiday"):
        if not pd.api.types.is_integer_dtype(frame[column]):
            raise ValueError(f"{country} {column} is not an integer")
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
    if full_coverage and repeated_rows != 12:
        raise ValueError(f"{country} autumn repeated-hour rows are not preserved")

    holiday_summary: dict[str, dict[str, int]] = {}
    for year in years:
        year_mask = local_dates.map(lambda value: value.year).eq(year)
        holiday_mask = year_mask & frame["is_public_holiday"].eq(1)
        holiday_summary[str(year)] = {
            "holiday_date_count": int(local_dates[holiday_mask].nunique()),
            "holiday_hour_observation_count": int(holiday_mask.sum()),
        }
        if full_coverage and (
            holiday_summary[str(year)]["holiday_date_count"]
            != config["expected_holiday_dates"]
        ):
            raise ValueError(f"{country} has the wrong annual holiday-date count")

    return {
        "country": country,
        "row_count": int(len(frame)),
        "timestamp_utc_unique": bool(utc.is_unique),
        "timestamp_utc_hourly_continuous": bool(
            len(utc) < 2
            or (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all()
        ),
        "timestamp_local_offset_preserved": True,
        "autumn_repeated_local_rows": repeated_rows,
        "missing_feature_count": int(frame[feature_columns].isna().sum().sum()),
        "holiday_summary": holiday_summary,
    }


def _read_source_keys(path: Path) -> tuple[pd.Series, str]:
    source_hash = sha256_file(path)
    data = pd.read_csv(path, usecols=["interval_start_utc"])
    keys = pd.to_datetime(data["interval_start_utc"], utc=True)
    return keys, source_hash


def _write_calendar_table(frame: pd.DataFrame, path: Path) -> str:
    output = frame.copy()
    output["timestamp_utc"] = output["timestamp_utc"].map(
        pd.Timestamp.isoformat
    )
    output["timestamp_local"] = output["timestamp_local"].map(
        pd.Timestamp.isoformat
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return sha256_file(path)


def _write_summary(validations: dict[str, dict[str, object]]) -> str:
    rows = []
    for country, validation in validations.items():
        for year, counts in validation["holiday_summary"].items():
            rows.append({"country": country, "year": int(year), **counts})
    summary = pd.DataFrame(rows).sort_values(["country", "year"])
    path = VALIDATION / "calendar_holiday_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return sha256_file(path)


def run_pipeline() -> dict[str, object]:
    validations: dict[str, dict[str, object]] = {}
    metadata: dict[str, object] = {
        "holiday_package": "holidays",
        "holiday_package_version": HOLIDAYS_VERSION,
        "holiday_category": "PUBLIC",
        "observed_dates": False,
        "subdivision": None,
        "official_sources": OFFICIAL_SOURCES,
        "source_files": {},
        "output_files": {},
        "validation": {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for country, config in COUNTRY_CONFIG.items():
        keys, source_hash = _read_source_keys(config["source_file"])
        if len(keys) != EXPECTED_ROWS:
            raise ValueError(f"{country} expected {EXPECTED_ROWS} source rows")
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
