from __future__ import annotations

from datetime import date
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
        "url": "https://www.who.int/news-room/speeches/item/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020",
        "publication_date": "2020-03-11",
        "automated_retrieval": {
            "status": "retrieved",
            "http_status": 200,
            "final_url": "https://www.who.int/news-room/speeches/item/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020",
            "checked_with": "curl -L",
            "content_validated": True,
            "limitation": None,
        },
    },
    {
        "institution": "World Health Organization",
        "title": "Statement on the fifteenth meeting of the IHR (2005) Emergency Committee on the COVID-19 pandemic",
        "url": "https://www.who.int/news/item/05-05-2023-statement-on-the-fifteenth-meeting-of-the-international-health-regulations-%282005%29-emergency-committee-regarding-the-coronavirus-disease-%28covid-19%29-pandemic",
        "publication_date": "2023-05-05",
        "automated_retrieval": {
            "status": "retrieved",
            "http_status": 200,
            "final_url": "https://www.who.int/news/item/05-05-2023-statement-on-the-fifteenth-meeting-of-the-international-health-regulations-%282005%29-emergency-committee-regarding-the-coronavirus-disease-%28covid-19%29-pandemic",
            "checked_with": "curl -L",
            "content_validated": True,
            "limitation": None,
        },
    },
    {
        "institution": "European Council",
        "title": "Special meeting of the European Council, 24 February 2022",
        "url": "https://www.consilium.europa.eu/en/meetings/european-council/2022/02/24/",
        "publication_date": "2022-02-24",
        "automated_retrieval": {
            "status": "blocked",
            "http_status": 403,
            "final_url": "https://www.consilium.europa.eu/en/meetings/european-council/2022/02/24/",
            "checked_with": "curl -L",
            "content_validated": False,
            "limitation": "Automated curl -L retrieval returned HTTP 403 Forbidden; the official page was not treated as content-validated and no source hash was generated.",
        },
    },
    {
        "institution": "International Energy Agency",
        "title": "World Energy Outlook 2022",
        "url": "https://www.iea.org/reports/world-energy-outlook-2022",
        "publication_date": "2022-10-27",
        "automated_retrieval": {
            "status": "retrieved",
            "http_status": 200,
            "final_url": "https://www.iea.org/reports/world-energy-outlook-2022",
            "checked_with": "curl -L",
            "content_validated": True,
            "limitation": None,
        },
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


def _validate_timestamp_source(
    source: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, list[pd.Timestamp]]:
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
    local_dates = pd.Series([timestamp.date() for timestamp in local], index=source.index)
    covid = local_dates.between(COVID_START, COVID_END).astype("int8")
    post_invasion = local_dates.ge(INVASION_START).astype("int8")
    result = source[["timestamp_utc", "timestamp_local"]].copy()
    result["is_covid_period"] = covid
    result["is_post_invasion"] = post_invasion
    return result[OUTPUT_COLUMNS]


def validate_crisis_features(
    result: pd.DataFrame,
    source: pd.DataFrame,
    country: str,
) -> dict[str, object]:
    if list(result.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{country} crisis columns do not match the schema")
    if list(source.columns) != ["timestamp_utc", "timestamp_local"]:
        raise ValueError(f"{country} source columns do not match the schema")
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
    local_dates = pd.Series(
        [timestamp.date() for timestamp in local], index=result.index
    )
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
        "timestamp_local_offset_count": len(
            {timestamp.strftime("%z") for timestamp in local}
        ),
        "missing_value_count": int(result.isna().sum().sum()),
        "covid_period_hours": int(result["is_covid_period"].sum()),
        "post_invasion_hours": int(result["is_post_invasion"].sum()),
        "overlap_hours": int(
            (result["is_covid_period"] & result["is_post_invasion"]).sum()
        ),
    }


def summarize_crisis_features(result: pd.DataFrame, country: str) -> dict[str, object]:
    summary = {
        "country": country,
        "total_observations": int(len(result)),
        "covid_period_hours": int(result["is_covid_period"].sum()),
        "post_invasion_hours": int(result["is_post_invasion"].sum()),
        "overlap_hours": int(
            (result["is_covid_period"] & result["is_post_invasion"]).sum()
        ),
    }
    for column, prefix in (
        ("is_covid_period", "covid"),
        ("is_post_invasion", "post_invasion"),
    ):
        indices = result.index[result[column].eq(1)]
        if len(indices) == 0:
            raise ValueError(f"{country} {column} has no active observations")
        active_timestamps = result.loc[indices, "timestamp_local"]
        summary[f"{prefix}_first_local_timestamp"] = active_timestamps.iloc[0]
        summary[f"{prefix}_last_local_timestamp"] = active_timestamps.iloc[-1]
    return summary


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
        source_hash_after = sha256_file(config["input_file"])
        if source_hash_after != source_hash:
            raise RuntimeError(f"{country} calendar source changed during processing")
        metadata["source_files"][country] = {
            "filename": config["input_file"].name,
            "sha256_before": source_hash,
            "sha256_after": source_hash_after,
        }
        metadata["output_files"][country] = {
            "filename": config["output_file"].name,
            "sha256": output_hash,
        }
        metadata["validation"][country] = validation
        rows.append(summary)
    metadata["summary_file_sha256"] = _write_summary(rows)
    metadata_path = VALIDATION / "crisis_feature_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
