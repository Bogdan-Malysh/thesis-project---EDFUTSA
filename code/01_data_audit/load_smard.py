from collections import Counter
from datetime import timezone as datetime_timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NUMERIC_COLUMN_MAP = {
    "grid load [MWh] Calculated resolutions": "grid_load_mwh",
    "Grid load incl. hydro pumped storage [MWh] Calculated resolutions": (
        "grid_load_including_pumped_storage_mwh"
    ),
    "Hydro pumped storage [MWh] Calculated resolutions": "pumped_storage_mwh",
    "Residual load [MWh] Calculated resolutions": "residual_load_mwh",
}
TIMESTAMP_FORMAT = "%b %d, %Y %I:%M %p"


def _parse_local_labels(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format=TIMESTAMP_FORMAT, errors="coerce")


def _utc_candidates(local_timestamp: pd.Timestamp, timezone_name: str) -> list[pd.Timestamp]:
    if pd.isna(local_timestamp):
        return []

    timezone = ZoneInfo(timezone_name)
    naive = local_timestamp.to_pydatetime().replace(tzinfo=None)
    candidates = []
    for fold in (0, 1):
        aware = naive.replace(tzinfo=timezone, fold=fold)
        utc_value = aware.astimezone(datetime_timezone.utc)
        round_tripped = utc_value.astimezone(timezone).replace(tzinfo=None)
        if round_tripped == naive:
            candidate = pd.Timestamp(utc_value)
            if candidate not in candidates:
                candidates.append(candidate)
    return sorted(candidates)


def _parse_timestamps(
    source: pd.DataFrame, timezone_name: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    local_starts = _parse_local_labels(source["Start date"])
    local_ends = _parse_local_labels(source["End date"])
    local_counts = Counter(local_starts.dropna().tolist())
    local_occurrences = Counter()
    utc_starts = []
    repeated_flags = []
    errors = []

    for row_number, local_start in enumerate(local_starts):
        candidates = _utc_candidates(local_start, timezone_name)
        if pd.isna(local_start):
            utc_starts.append(pd.NaT)
            repeated_flags.append(False)
            errors.append(f"row {row_number}: interval start could not be parsed")
            continue
        if not candidates:
            utc_starts.append(pd.NaT)
            repeated_flags.append(False)
            errors.append(f"row {row_number}: interval start is nonexistent in local time")
            continue
        if len(candidates) == 1:
            utc_starts.append(candidates[0])
            repeated_flags.append(False)
            continue

        occurrence = local_occurrences[local_start]
        local_occurrences[local_start] += 1
        repeated_flags.append(True)
        if local_counts[local_start] != 2:
            utc_starts.append(pd.NaT)
            errors.append(
                f"row {row_number}: ambiguous interval start occurs "
                f"{local_counts[local_start]} times"
            )
        else:
            utc_starts.append(candidates[occurrence])

    utc_start_series = pd.Series(
        pd.to_datetime(utc_starts, utc=True), index=source.index, name="interval_start_utc"
    )
    utc_end_values = [
        start + pd.Timedelta(hours=1) if pd.notna(start) else pd.NaT
        for start in utc_start_series
    ]
    utc_end_series = pd.Series(
        pd.to_datetime(utc_end_values, utc=True), index=source.index, name="interval_end_utc"
    )

    valid_starts = utc_start_series.dropna()
    all_starts_present = len(valid_starts) == len(utc_start_series)
    unique = all_starts_present and valid_starts.is_unique
    differences = valid_starts.diff().dropna()
    strictly_increasing = all_starts_present and (
        len(differences) == 0 or differences.gt(pd.Timedelta(hours=0)).all()
    )
    hourly_continuity = all_starts_present and (
        len(differences) == 0 or differences.eq(pd.Timedelta(hours=1)).all()
    )
    durations = utc_end_series - utc_start_series
    duration_valid = (
        utc_start_series.notna().all()
        and utc_end_series.notna().all()
        and durations.eq(pd.Timedelta(hours=1)).all()
    )
    start_parse_errors = int(local_starts.isna().sum())
    end_parse_errors = int(local_ends.isna().sum())
    if end_parse_errors:
        errors.append(f"{end_parse_errors} interval end labels could not be parsed")
    if not unique:
        errors.append("UTC interval starts are not unique")
    if not strictly_increasing:
        errors.append("UTC interval starts are not strictly increasing")
    if not hourly_continuity:
        errors.append("UTC interval starts are not separated by one hour")
    if not duration_valid:
        errors.append("UTC intervals are not all exactly one hour long")

    timestamp_data = pd.DataFrame(
        {
            "interval_start_raw": source["Start date"],
            "interval_end_raw": source["End date"],
            "interval_start_local": local_starts,
            "interval_end_local": local_ends,
            "interval_start_utc": utc_start_series,
            "interval_end_utc": utc_end_series,
            "is_repeated_autumn_hour": pd.Series(
                repeated_flags, index=source.index, dtype=bool
            ),
        },
        index=source.index,
    )
    diagnostics = {
        "start_parse_error_count": start_parse_errors,
        "end_parse_error_count": end_parse_errors,
        "timestamps": {
            "utc_start_unique": bool(unique),
            "utc_start_strictly_increasing": bool(strictly_increasing),
            "utc_start_hourly_continuity": bool(hourly_continuity),
            "utc_interval_duration_valid": bool(duration_valid),
            "repeated_autumn_hour_count": int(sum(repeated_flags)),
            "errors": errors,
        },
    }
    return timestamp_data, diagnostics


def _parse_numeric_column(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    stripped = values.str.strip()
    missing = stripped.eq("")
    normalized = stripped.str.replace(",", "", regex=False)
    parsed = pd.to_numeric(normalized, errors="coerce")
    special_non_finite = normalized.str.lower().isin(
        {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
    )
    non_numeric = (~missing) & parsed.isna() & (~special_non_finite)
    non_finite = (~missing) & (~non_numeric) & (~np.isfinite(parsed.fillna(np.nan)))

    diagnostics = {
        "missing_count": int(missing.sum()),
        "non_numeric_count": int(non_numeric.sum()),
        "non_finite_count": int(non_finite.sum()),
    }
    return parsed, diagnostics


def load_smard_file(
    path: str | Path,
    country: str,
    dataset_type: str,
    timezone_name: str,
) -> dict[str, object]:
    """Read one SMARD CSV while preserving the source values unchanged."""
    if country not in {"Germany", "Austria"}:
        raise ValueError("country must be Germany or Austria")
    if dataset_type not in {"actual", "forecast"}:
        raise ValueError("dataset_type must be actual or forecast")
    if not timezone_name:
        raise ValueError("timezone_name must not be empty")

    source = pd.read_csv(
        Path(path),
        sep=";",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
    )
    timestamp_data, timestamp_diagnostics = _parse_timestamps(source, timezone_name)
    data = timestamp_data.copy()
    column_map = {
        original: standardized
        for original, standardized in NUMERIC_COLUMN_MAP.items()
        if original in source.columns
    }
    numeric_diagnostics = {}
    for original, standardized in column_map.items():
        data[standardized], numeric_diagnostics[original] = _parse_numeric_column(
            source[original]
        )

    return {
        "source": source,
        "data": data,
        "metadata": {
            "country": country,
            "dataset_type": dataset_type,
            "timezone": timezone_name,
        },
        "column_map": column_map,
        "diagnostics": {
            "numeric_columns": numeric_diagnostics,
            **timestamp_diagnostics,
        },
    }
