from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
EXPECTED_ROWS = 52_608
OUTPUT_COLUMNS = [
    "timestamp_utc",
    "timestamp_local",
    "actual_grid_load_mwh",
    "forecasted_grid_load_mwh",
    "forecasted_grid_load_valid",
    "temperature_c",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday",
    "is_covid_period",
    "is_post_invasion",
]
BINARY_COLUMNS = [
    "forecasted_grid_load_valid",
    "is_weekend",
    "is_public_holiday",
    "is_covid_period",
    "is_post_invasion",
]
NUMERIC_COLUMNS = OUTPUT_COLUMNS[2:]
COUNTRY_CONFIG = {
    "Germany": {
        "smard": "smard_germany_hourly.csv",
        "temperature": "temperature_germany_hourly.csv",
        "calendar": "calendar_germany_hourly.csv",
        "crisis": "crisis_germany_hourly.csv",
        "output": "modelling_germany_hourly.csv",
    },
    "Austria": {
        "smard": "smard_austria_hourly.csv",
        "temperature": "temperature_austria_hourly.csv",
        "calendar": "calendar_austria_hourly.csv",
        "crisis": "crisis_austria_hourly.csv",
        "output": "modelling_austria_hourly.csv",
    },
}


def _prepare_frame(
    frame: pd.DataFrame,
    key_column: str,
    selected_columns: list[str],
    label: str,
) -> pd.DataFrame:
    missing = [column for column in selected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

    prepared = frame[selected_columns].copy()
    parsed = pd.to_datetime(prepared[key_column], utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} contains missing or invalid timestamp_utc values")
    prepared[key_column] = pd.DatetimeIndex(parsed)
    prepared = prepared.rename(columns={key_column: "timestamp_utc"})
    if prepared["timestamp_utc"].duplicated().any():
        raise ValueError(f"{label} timestamp_utc contains duplicates")
    return prepared.sort_values("timestamp_utc", ignore_index=True)


def _require_same_keys(
    target_keys: pd.DatetimeIndex,
    candidate_keys: pd.DatetimeIndex,
    label: str,
) -> None:
    if target_keys.equals(candidate_keys):
        return
    missing = target_keys.difference(candidate_keys)
    extra = candidate_keys.difference(target_keys)
    raise ValueError(
        f"{label} timestamp_utc keys do not match target "
        f"(missing={len(missing)}, extra={len(extra)})"
    )


def _validate_target_keys(keys: pd.DatetimeIndex, country: str) -> None:
    if len(keys) != EXPECTED_ROWS:
        raise ValueError(f"{country} target must contain {EXPECTED_ROWS} rows")
    if not keys.is_unique or not keys.is_monotonic_increasing:
        raise ValueError(f"{country} target timestamp_utc keys are not unique and ordered")
    if not (keys[1:] - keys[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{country} target timestamp_utc keys are not hourly continuous")


def _merge_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    target_keys: pd.DatetimeIndex,
    label: str,
) -> pd.DataFrame:
    candidate_keys = pd.DatetimeIndex(right["timestamp_utc"])
    _require_same_keys(target_keys, candidate_keys, label)
    right_columns = [column for column in right.columns if column != "timestamp_utc"]
    merged = left.merge(
        right,
        on="timestamp_utc",
        how="left",
        validate="one_to_one",
        sort=True,
    )
    if merged[right_columns].isna().any().any():
        raise ValueError(f"{label} merge contains unmatched timestamp_utc values")
    return merged


def _validate_output(result: pd.DataFrame, country: str) -> pd.DataFrame:
    if list(result.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"{country} modelling columns do not match the schema")
    if len(result) != EXPECTED_ROWS:
        raise ValueError(f"{country} modelling output must contain {EXPECTED_ROWS} rows")

    utc = pd.DatetimeIndex(pd.to_datetime(result["timestamp_utc"], utc=True))
    _validate_target_keys(utc, country)
    complete_columns = [
        column for column in result.columns if column != "forecasted_grid_load_mwh"
    ]
    if result[complete_columns].isna().any().any():
        raise ValueError(f"{country} modelling output contains missing non-benchmark values")

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        if column == "forecasted_grid_load_mwh":
            values = values.loc[values.notna()]
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{country} {column} contains non-finite values")
    for column in BINARY_COLUMNS:
        if not result[column].isin([0, 1]).all():
            raise ValueError(f"{country} {column} is not binary")
    missing_benchmark = result["forecasted_grid_load_mwh"].isna()
    if not result.loc[missing_benchmark, "forecasted_grid_load_valid"].eq(False).all():
        raise ValueError(f"{country} missing benchmark values are not marked invalid")
    if not result["timestamp_local"].map(
        lambda value: isinstance(value, str)
        and re.search(r"[+-][0-9]{2}:[0-9]{2}$", value) is not None
    ).all():
        raise ValueError(f"{country} timestamp_local is not offset-aware")
    return result


def build_modelling_dataset(
    target: pd.DataFrame,
    temperature: pd.DataFrame,
    calendar: pd.DataFrame,
    crisis: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")

    target_frame = _prepare_frame(
        target,
        "interval_start_utc",
        [
            "interval_start_utc",
            "actual_grid_load_mwh",
            "forecasted_grid_load_mwh",
            "forecasted_grid_load_valid",
        ],
        f"{country} SMARD target",
    )
    temperature_frame = _prepare_frame(
        temperature,
        "interval_start_utc",
        ["interval_start_utc", "temperature_c"],
        f"{country} temperature",
    )
    calendar_frame = _prepare_frame(
        calendar,
        "timestamp_utc",
        [
            "timestamp_utc",
            "timestamp_local",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_public_holiday",
        ],
        f"{country} calendar",
    )
    crisis_frame = _prepare_frame(
        crisis,
        "timestamp_utc",
        ["timestamp_utc", "is_covid_period", "is_post_invasion"],
        f"{country} crisis",
    )

    target_keys = pd.DatetimeIndex(target_frame["timestamp_utc"])
    _validate_target_keys(target_keys, country)
    result = _merge_one_to_one(target_frame, temperature_frame, target_keys, "temperature")
    result = _merge_one_to_one(result, calendar_frame, target_keys, "calendar")
    result = _merge_one_to_one(result, crisis_frame, target_keys, "crisis")
    return _validate_output(result[OUTPUT_COLUMNS], country)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_country_modelling_dataset(
    country: str,
    processed_directory: str | Path = PROCESSED,
) -> pd.DataFrame:
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"unsupported country: {country}")
    processed_directory = Path(processed_directory)
    config = COUNTRY_CONFIG[country]
    return build_modelling_dataset(
        _read_csv(processed_directory / config["smard"]),
        _read_csv(processed_directory / config["temperature"]),
        _read_csv(processed_directory / config["calendar"]),
        _read_csv(processed_directory / config["crisis"]),
        country,
    )


def run_pipeline(processed_directory: str | Path = PROCESSED) -> None:
    processed_directory = Path(processed_directory)
    frames = {
        country: build_country_modelling_dataset(country, processed_directory)
        for country in COUNTRY_CONFIG
    }
    for country, frame in frames.items():
        output_path = processed_directory / COUNTRY_CONFIG[country]["output"]
        frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    run_pipeline()
