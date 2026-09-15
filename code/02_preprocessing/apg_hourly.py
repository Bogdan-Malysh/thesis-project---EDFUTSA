from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TIMEZONE_NAME = "Europe/Vienna"
TIMEZONE = ZoneInfo(TIMEZONE_NAME)
EXPECTED_HOURLY_ROWS = 26_304
ACTUAL_FILES = (
    "apg_austria_actual_2020_15min.csv",
    "apg_austria_actual_2021_15min.csv",
    "apg_austria_actual_2022_15min.csv",
)
FORECAST_FILES = (
    "apg_austria_forecasted_2020_15min.csv",
    "apg_austria_forecasted_2021_15min.csv",
    "apg_austria_forecasted_2022_15min.csv",
)
EXPECTED_FILES = ACTUAL_FILES + FORECAST_FILES
ACTUAL_COLUMNS = ["Time from [CET/CEST]", "Time to [CET/CEST]", "Power [MW]"]
FORECAST_COLUMNS = ["Time from [CET/CEST]", "Time to [CET/CEST]", "Load [MW]"]


def _parse_timestamp(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    fold = 0
    normalized = value
    if " 2A:" in normalized:
        normalized = normalized.replace(" 2A:", " 02:")
        fold = 0
    elif " 2B:" in normalized:
        normalized = normalized.replace(" 2B:", " 02:")
        fold = 1
    naive = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
    local = pd.Timestamp(naive.replace(tzinfo=TIMEZONE, fold=fold))
    utc = local.tz_convert("UTC")
    round_trip = utc.tz_convert(TIMEZONE).tz_localize(None)
    if round_trip != pd.Timestamp(naive):
        raise ValueError(f"nonexistent local timestamp: {value}")
    return local, utc


def _parse_timestamp_column(values: pd.Series) -> tuple[pd.Series, pd.Series, list[str]]:
    local_values = []
    utc_values = []
    errors = []
    for row_number, value in enumerate(values):
        try:
            local, utc = _parse_timestamp(value)
        except (TypeError, ValueError) as error:
            local_values.append(pd.NaT)
            utc_values.append(pd.NaT)
            errors.append(f"row {row_number}: {error}")
        else:
            local_values.append(local)
            utc_values.append(utc)
    return (
        pd.Series(pd.to_datetime(local_values), index=values.index),
        pd.Series(pd.to_datetime(utc_values, utc=True), index=values.index),
        errors,
    )


def load_apg_file(path: str | Path, dataset_type: str) -> dict[str, object]:
    path = Path(path)
    expected_columns = ACTUAL_COLUMNS if dataset_type == "actual" else FORECAST_COLUMNS
    value_column = "Power [MW]" if dataset_type == "actual" else "Load [MW]"
    source = pd.read_csv(
        path,
        sep=",",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
    )
    start_local, start_utc, start_errors = _parse_timestamp_column(
        source["Time from [CET/CEST]"]
    )
    end_local, end_utc, end_errors = _parse_timestamp_column(
        source["Time to [CET/CEST]"]
    )
    raw_values = source[value_column]
    stripped_values = raw_values.str.strip()
    missing_count = int(stripped_values.eq("").sum())
    normalized_values = stripped_values.str.replace(",", "", regex=False)
    values = pd.to_numeric(normalized_values, errors="coerce")
    non_numeric_count = int(
        ((~stripped_values.eq("")) & values.isna()).sum()
    )
    non_finite_count = int(
        (values.notna() & ~np.isfinite(values)).sum()
    )
    data = pd.DataFrame(
        {
            "interval_start_utc": start_utc,
            "interval_end_utc": end_utc,
            "interval_start_local": start_local,
            "interval_end_local": end_local,
            "value_mw": values,
            "energy_mwh": values * 0.25,
        },
        index=source.index,
    )
    return {
        "source": source,
        "data": data,
        "metadata": {
            "filename": path.name,
            "dataset_type": dataset_type,
            "timezone": TIMEZONE_NAME,
            "value_column": value_column,
            "expected_columns": expected_columns,
        },
        "diagnostics": {
            "start_timestamp_errors": start_errors,
            "end_timestamp_errors": end_errors,
            "missing_count": missing_count,
            "non_numeric_count": non_numeric_count,
            "non_finite_count": non_finite_count,
        },
    }


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def audit_apg_file(path: str | Path, dataset_type: str) -> dict[str, object]:
    loaded = load_apg_file(path, dataset_type)
    source = loaded["source"]
    data = loaded["data"]
    expected_columns = ACTUAL_COLUMNS if dataset_type == "actual" else FORECAST_COLUMNS
    schema_ok = list(source.columns) == expected_columns
    valid_timestamps = data["interval_start_utc"].notna() & data["interval_end_utc"].notna()
    durations = data["interval_end_utc"] - data["interval_start_utc"]
    duration_ok = bool(valid_timestamps.all() and durations.eq(pd.Timedelta(minutes=15)).all())
    starts = data["interval_start_utc"]
    differences = starts.diff().dropna()
    continuity_ok = bool(
        valid_timestamps.all()
        and (len(differences) == 0 or differences.eq(pd.Timedelta(minutes=15)).all())
    )
    duplicate_utc = int(starts.duplicated().sum())
    raw_duplicate_rows = int(source.duplicated().sum())
    start_values = source["Time from [CET/CEST]"]
    autumn_2a_count = int(start_values.str.contains(" 2A:", regex=False).sum())
    autumn_2b_count = int(start_values.str.contains(" 2B:", regex=False).sum())
    spring_jump_count = int(
        (
            start_values.str.endswith("01:45:00")
            & source["Time to [CET/CEST]"].str.endswith("03:00:00")
        ).sum()
    )
    errors = [
        *loaded["diagnostics"]["start_timestamp_errors"],
        *loaded["diagnostics"]["end_timestamp_errors"],
    ]
    if not schema_ok:
        errors.append("unexpected columns")
    if loaded["diagnostics"]["missing_count"]:
        errors.append("missing or empty values")
    if loaded["diagnostics"]["non_numeric_count"]:
        errors.append("non-numeric values")
    if loaded["diagnostics"]["non_finite_count"]:
        errors.append("non-finite values")
    if duplicate_utc or raw_duplicate_rows:
        errors.append("duplicate intervals or rows")
    if not duration_ok or not continuity_ok:
        errors.append("invalid 15-minute continuity")
    return {
        "filename": path.name,
        "dataset_type": dataset_type,
        "row_count": len(source),
        "column_count": len(source.columns),
        "status": "ERROR" if errors else "PASS",
        "errors": errors,
        "missing_or_empty_count": loaded["diagnostics"]["missing_count"],
        "non_numeric_count": loaded["diagnostics"]["non_numeric_count"],
        "non_finite_count": loaded["diagnostics"]["non_finite_count"],
        "duplicate_utc_intervals": duplicate_utc,
        "duplicate_rows": raw_duplicate_rows,
        "utc_15_minute_continuity": continuity_ok,
        "interval_duration_valid": duration_ok,
        "first_interval_start_utc": _iso(starts.iloc[0]) if len(data) else None,
        "final_interval_end_utc": _iso(data["interval_end_utc"].iloc[-1])
        if len(data)
        else None,
        "spring_transition_intervals": spring_jump_count,
        "autumn_2a_intervals": autumn_2a_count,
        "autumn_2b_intervals": autumn_2b_count,
    }


def audit_apg_files(raw_directory: str | Path) -> dict[str, object]:
    raw_directory = Path(raw_directory)
    audits = []
    for filename in ACTUAL_FILES:
        audits.append(audit_apg_file(raw_directory / filename, "actual"))
    for filename in FORECAST_FILES:
        audits.append(audit_apg_file(raw_directory / filename, "forecast"))
    return {
        "status": "ERROR" if any(audit["status"] == "ERROR" for audit in audits) else "PASS",
        "files": audits,
        "total_rows": sum(audit["row_count"] for audit in audits),
    }


def aggregate_hourly(data: pd.DataFrame, output_column: str) -> pd.DataFrame:
    valid = data.dropna(subset=["interval_start_utc", "interval_end_utc", "energy_mwh"])
    if len(valid) != len(data):
        raise ValueError("cannot aggregate invalid APG intervals")
    hour_start = valid["interval_start_utc"].dt.floor("h")
    counts = valid.groupby(hour_start, sort=True).size()
    if (counts != 4).any():
        raise ValueError("every UTC hour must contain exactly four valid intervals")
    hourly = (
        valid.assign(hour_start_utc=hour_start)
        .groupby("hour_start_utc", sort=True)["energy_mwh"]
        .sum()
        .rename(output_column)
        .to_frame()
        .reset_index()
    )
    hourly["interval_start_utc"] = hourly.pop("hour_start_utc")
    hourly["interval_end_utc"] = hourly["interval_start_utc"] + pd.Timedelta(hours=1)
    hourly["interval_start_local"] = hourly["interval_start_utc"].dt.tz_convert(TIMEZONE_NAME)
    hourly["interval_end_local"] = hourly["interval_end_utc"].dt.tz_convert(TIMEZONE_NAME)
    hourly[output_column] = hourly[output_column].round(6)
    return hourly[
        [
            "interval_start_utc",
            "interval_end_utc",
            "interval_start_local",
            "interval_end_local",
            output_column,
        ]
    ]


def combine_hourly(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    merged = actual.merge(
        forecast,
        on="interval_start_utc",
        how="outer",
        suffixes=("_actual", "_forecast"),
        validate="one_to_one",
    )
    if merged[["actual_load_mwh", "forecasted_load_mwh"]].isna().any().any():
        raise ValueError("actual and forecast hourly UTC intervals do not align")
    if not (merged["interval_end_utc_actual"] == merged["interval_end_utc_forecast"]).all():
        raise ValueError("actual and forecast interval ends do not align")
    merged["interval_end_utc"] = merged["interval_end_utc_actual"]
    merged["interval_start_local"] = merged["interval_start_local_actual"]
    merged["interval_end_local"] = merged["interval_end_local_actual"]
    return merged[
        [
            "interval_start_utc",
            "interval_end_utc",
            "interval_start_local",
            "interval_end_local",
            "actual_load_mwh",
            "forecasted_load_mwh",
        ]
    ].sort_values("interval_start_utc", ignore_index=True)


def preprocess_apg(
    raw_directory: str | Path,
    output_path: str | Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    raw_directory = Path(raw_directory)
    output_path = Path(output_path)
    audit = audit_apg_files(raw_directory)
    if audit["status"] == "ERROR":
        raise ValueError("APG audit failed: " + "; ".join(
            error for file_audit in audit["files"] for error in file_audit["errors"]
        ))
    actual_data = pd.concat(
        [load_apg_file(raw_directory / filename, "actual")["data"] for filename in ACTUAL_FILES],
        ignore_index=True,
    )
    forecast_data = pd.concat(
        [load_apg_file(raw_directory / filename, "forecast")["data"] for filename in FORECAST_FILES],
        ignore_index=True,
    )
    actual_hourly = aggregate_hourly(actual_data, "actual_load_mwh")
    forecast_hourly = aggregate_hourly(forecast_data, "forecasted_load_mwh")
    hourly = combine_hourly(actual_hourly, forecast_hourly)
    if len(hourly) != EXPECTED_HOURLY_ROWS:
        raise ValueError(f"expected {EXPECTED_HOURLY_ROWS} hourly rows, found {len(hourly)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = hourly.copy()
    for column in (
        "interval_start_utc",
        "interval_end_utc",
        "interval_start_local",
        "interval_end_local",
    ):
        output[column] = output[column].map(_iso)
    output.to_csv(output_path, index=False)
    return audit, hourly


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    audit, hourly = preprocess_apg(
        project_root / "data" / "raw" / "apg",
        project_root / "data" / "processed" / "apg_austria_hourly.csv",
    )
    print(
        f"APG audit: {audit['status']} | files={len(audit['files'])} | "
        f"rows={audit['total_rows']}"
    )
    print(
        f"APG hourly preprocessing: rows={len(hourly)} | "
        f"first_utc={_iso(hourly['interval_start_utc'].iloc[0])} | "
        f"final_utc={_iso(hourly['interval_end_utc'].iloc[-1])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
