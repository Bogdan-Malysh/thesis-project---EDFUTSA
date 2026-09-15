import codecs
import csv
import hashlib
import io
import json
import os
import platform
import struct
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from load_smard import NUMERIC_COLUMN_MAP, load_smard_file


EXPECTED_FILES = {
    "smard_austria_actual.csv": {
        "country": "Austria",
        "dataset_type": "actual",
        "timezone": "Europe/Vienna",
    },
    "smard_austria_forecasted.csv": {
        "country": "Austria",
        "dataset_type": "forecast",
        "timezone": "Europe/Vienna",
    },
    "smard_germany_actual.csv": {
        "country": "Germany",
        "dataset_type": "actual",
        "timezone": "Europe/Berlin",
    },
    "smard_germany_forecasted.csv": {
        "country": "Germany",
        "dataset_type": "forecast",
        "timezone": "Europe/Berlin",
    },
}

TIMESTAMP_COLUMNS = ["Start date", "End date"]
ACTUAL_NUMERIC_COLUMNS = list(NUMERIC_COLUMN_MAP)
FORECAST_NUMERIC_COLUMNS = [
    "grid load [MWh] Calculated resolutions",
    "Residual load [MWh] Calculated resolutions",
]
SUMMARY_PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
REPORT_FILENAMES = (
    "smard_data_audit.md",
    "smard_data_audit.json",
    "smard_column_summary.csv",
)
REPORT_PACKAGES = (
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "pytest",
    "tzdata",
)
SMARD_DIRECTORY = "smard"


def _check(
    check_id: str,
    scope: str,
    status: str,
    summary: str,
    evidence: dict[str, object],
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "scope": scope,
        "status": status,
        "summary": summary,
        "evidence": evidence,
    }


def _iso_timestamp(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_csv_format(path: Path) -> tuple[dict[str, object], list[str]]:
    raw = path.read_bytes()
    bom_present = raw.startswith(codecs.BOM_UTF8)
    text = raw.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    header = rows[0] if rows else []
    field_counts = [len(row) for row in rows]
    delimiter_present = len(header) > 1
    fields_consistent = bool(header) and all(
        count == len(header) for count in field_counts
    )
    return (
        {
            "delimiter": ";" if delimiter_present else None,
            "delimiter_present": delimiter_present,
            "field_counts": field_counts,
            "header_field_count": len(header),
            "fields_consistent": fields_consistent,
            "bom_present": bom_present,
        },
        header,
    )


def _expected_columns(dataset_type: str) -> list[str]:
    numeric_columns = (
        ACTUAL_NUMERIC_COLUMNS
        if dataset_type == "actual"
        else FORECAST_NUMERIC_COLUMNS
    )
    return TIMESTAMP_COLUMNS + numeric_columns


def _dst_check(data: pd.DataFrame) -> dict[str, object]:
    local_starts = data["interval_start_local"]
    utc_starts = data["interval_start_utc"]
    repeated = data["is_repeated_autumn_hour"]
    spring_rows = []
    autumn_rows = []
    unexpected_rows = []

    for row_number in range(1, len(data)):
        local_difference = local_starts.iloc[row_number] - local_starts.iloc[row_number - 1]
        utc_difference = utc_starts.iloc[row_number] - utc_starts.iloc[row_number - 1]
        if pd.isna(local_difference) or pd.isna(utc_difference):
            continue
        if local_difference == pd.Timedelta(hours=2) and utc_difference == pd.Timedelta(hours=1):
            spring_rows.append(row_number)
        elif (
            local_difference == pd.Timedelta(0)
            and repeated.iloc[row_number - 1]
            and repeated.iloc[row_number]
            and utc_difference == pd.Timedelta(hours=1)
        ):
            autumn_rows.append(row_number)
        elif local_difference != pd.Timedelta(hours=1):
            unexpected_rows.append(row_number)

    status = "PASS" if not unexpected_rows else "ERROR"
    return _check(
        "interval.dst",
        "file",
        status,
        "Expected daylight-saving-time effects are represented correctly."
        if status == "PASS"
        else "Unexpected local-time transition detected.",
        {
            "spring_transition_rows": spring_rows,
            "autumn_transition_rows": autumn_rows,
            "unexpected_transition_rows": unexpected_rows,
        },
    )


def _duplicate_check(source: pd.DataFrame, data: pd.DataFrame) -> dict[str, object]:
    groups: dict[tuple[object, ...], list[int]] = {}
    for row_number, row in enumerate(source.itertuples(index=False, name=None)):
        groups.setdefault(tuple(row), []).append(row_number)

    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    expected_autumn_rows = 0
    unexpected_duplicate_rows = 0
    for indices in duplicate_groups:
        repeated_rows = data.iloc[indices]["is_repeated_autumn_hour"]
        utc_rows = data.iloc[indices]["interval_start_utc"]
        is_expected_autumn = (
            repeated_rows.all()
            and utc_rows.notna().all()
            and utc_rows.is_unique
        )
        if is_expected_autumn:
            expected_autumn_rows += len(indices)
        else:
            unexpected_duplicate_rows += len(indices)

    status = "PASS" if unexpected_duplicate_rows == 0 else "ERROR"
    return _check(
        "duplicates.rows",
        "file",
        status,
        "No unexplained exact duplicate source rows found."
        if status == "PASS"
        else "Unexplained exact duplicate source rows found.",
        {
            "duplicate_group_count": len(duplicate_groups),
            "expected_autumn_rows": expected_autumn_rows,
            "unexpected_duplicate_rows": unexpected_duplicate_rows,
        },
    )


def _completeness_check(
    source: pd.DataFrame,
    loaded: dict[str, object],
    dataset_type: str,
) -> dict[str, object]:
    column_map = loaded["column_map"]
    diagnostics = loaded["diagnostics"]["numeric_columns"]
    evidence = {}
    error_found = False
    warning_found = False
    target_original = "grid load [MWh] Calculated resolutions"

    for original, standardized in column_map.items():
        raw_values = source[original]
        empty_count = int(raw_values.str.strip().eq("").sum())
        values = diagnostics[original]
        column_evidence = {
            "missing_count": values["missing_count"],
            "empty_count": empty_count,
            "non_numeric_count": values["non_numeric_count"],
            "non_finite_count": values["non_finite_count"],
        }
        evidence[standardized] = column_evidence
        has_issue = any(value > 0 for value in column_evidence.values())
        if has_issue:
            if dataset_type == "actual" and original == target_original:
                error_found = True
            else:
                warning_found = True

    status = "ERROR" if error_found else "WARNING" if warning_found else "PASS"
    summary = {
        "PASS": "No missing, empty, non-numeric, or non-finite numeric values found.",
        "WARNING": "Non-target or benchmark numeric completeness issues found.",
        "ERROR": "Primary actual target contains incomplete or invalid numeric values.",
    }[status]
    return _check("completeness.values", "file", status, summary, evidence)


def _numeric_summary(values: pd.Series, data: pd.DataFrame) -> dict[str, object]:
    finite_values = values.where(np.isfinite(values)).dropna()
    if finite_values.empty:
        summary = {
            "valid_count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "percentiles": {str(percentile): None for percentile in SUMMARY_PERCENTILES},
        }
    else:
        summary = {
            "valid_count": int(finite_values.size),
            "minimum": float(finite_values.min()),
            "maximum": float(finite_values.max()),
            "mean": float(finite_values.mean()),
            "percentiles": {
                str(percentile): float(finite_values.quantile(percentile / 100))
                for percentile in SUMMARY_PERCENTILES
            },
        }

    finite_series = values.where(np.isfinite(values))
    absolute_changes = finite_series.diff().abs()
    if absolute_changes.dropna().empty:
        summary["largest_abs_one_hour_change"] = None
        summary["largest_change_interval_start_utc"] = None
    else:
        row_number = absolute_changes.idxmax()
        summary["largest_abs_one_hour_change"] = float(absolute_changes.loc[row_number])
        summary["largest_change_interval_start_utc"] = _iso_timestamp(
            data.loc[row_number, "interval_start_utc"]
        )
    return summary


def _numerical_summary_check(
    data: pd.DataFrame, column_map: dict[str, str]
) -> dict[str, object]:
    evidence = {
        standardized: _numeric_summary(data[standardized], data)
        for standardized in column_map.values()
    }
    return _check(
        "numerical.summaries",
        "file",
        "PASS",
        "Descriptive numerical summaries were calculated for every numeric source column.",
        evidence,
    )


def _target_observation_issue(
    raw_value: str, parsed_value: float | None
) -> list[str]:
    stripped = raw_value.strip()
    if stripped == "":
        return ["missing"]
    if pd.isna(parsed_value):
        if stripped.lower() in {
            "nan",
            "+nan",
            "-nan",
            "inf",
            "+inf",
            "-inf",
            "infinity",
            "+infinity",
            "-infinity",
        }:
            return ["non_finite"]
        return ["non_numeric"]
    if not np.isfinite(parsed_value):
        return ["non_finite"]
    if parsed_value <= 0:
        return ["non_positive"]
    return []


def _primary_target_check(
    source: pd.DataFrame, data: pd.DataFrame, dataset_type: str
) -> dict[str, object]:
    original_name = "grid load [MWh] Calculated resolutions"
    parsed_name = "grid_load_mwh"
    observations = []
    if original_name in source.columns and parsed_name in data.columns:
        for row_number, (raw_value, parsed_value) in enumerate(
            zip(source[original_name], data[parsed_name])
        ):
            reasons = _target_observation_issue(raw_value, parsed_value)
            if reasons:
                observations.append(
                    {
                        "row": row_number,
                        "interval_start_utc": _iso_timestamp(
                            data.iloc[row_number]["interval_start_utc"]
                        ),
                        "value": None
                        if pd.isna(parsed_value) or not np.isfinite(parsed_value)
                        else float(parsed_value),
                        "reasons": reasons,
                    }
                )

    invalid_count = len(observations)
    status = "PASS" if invalid_count == 0 else (
        "ERROR" if dataset_type == "actual" else "WARNING"
    )
    summary = (
        "All primary grid-load observations are valid."
        if status == "PASS"
        else "Invalid actual target observations found."
        if status == "ERROR"
        else "Invalid forecast benchmark observations are unusable."
    )
    return _check(
        "primary_target.validity",
        "file",
        status,
        summary,
        {
            "target": parsed_name,
            "invalid_observation_count": invalid_count,
            "invalid_observations": observations,
            "unusable_benchmark_observations": observations
            if dataset_type == "forecast"
            else [],
        },
    )


def _discrepancy_summary(discrepancies: pd.Series) -> dict[str, object]:
    finite = discrepancies.where(np.isfinite(discrepancies)).dropna()
    if finite.empty:
        return {
            "mismatch_count": 0,
            "maximum_discrepancy_mwh": None,
            "discrepancy_percentiles": {
                str(percentile): None for percentile in SUMMARY_PERCENTILES
            },
        }
    return {
        "mismatch_count": int((finite > 0.02).sum()),
        "maximum_discrepancy_mwh": float(finite.max()),
        "discrepancy_percentiles": {
            str(percentile): float(finite.quantile(percentile / 100))
            for percentile in SUMMARY_PERCENTILES
        },
    }


def _pumped_storage_check(
    data: pd.DataFrame, dataset_type: str
) -> dict[str, object]:
    if dataset_type != "actual":
        return _check(
            "pumped_storage.relationship",
            "file",
            "PASS",
            "Pumped-storage relationship is not applicable to forecast files.",
            {"not_applicable": True},
        )

    required = {
        "grid_load_mwh",
        "grid_load_including_pumped_storage_mwh",
        "pumped_storage_mwh",
    }
    if not required <= set(data.columns):
        return _check(
            "pumped_storage.relationship",
            "file",
            "WARNING",
            "Pumped-storage relationship could not be tested because columns are missing.",
            {"not_applicable": False, "convention": None},
        )

    complete = data[list(required)].notna().all(axis=1)
    complete &= data[list(required)].apply(np.isfinite).all(axis=1)
    grid = data["grid_load_mwh"]
    included = data["grid_load_including_pumped_storage_mwh"]
    pumped = data["pumped_storage_mwh"]
    candidate_results = {}
    mismatch_timestamps = {}
    for name, expected in {
        "plus": grid + pumped,
        "minus": grid - pumped,
    }.items():
        discrepancies = (included - expected).abs().where(complete)
        candidate_results[name] = _discrepancy_summary(discrepancies)
        mismatch_timestamps[name] = [
            _iso_timestamp(data.iloc[row_number]["interval_start_utc"])
            for row_number in discrepancies.index[discrepancies > 0.02]
        ]

    matching = [
        name
        for name, result in candidate_results.items()
        if result["mismatch_count"] == 0 and int(complete.sum()) > 0
    ]
    convention = matching[0] if len(matching) == 1 else None
    negative_rows = [
        {
            "row": row_number,
            "interval_start_utc": _iso_timestamp(
                data.iloc[row_number]["interval_start_utc"]
            ),
            "value": float(value),
        }
        for row_number, value in pumped.items()
        if pd.notna(value) and np.isfinite(value) and value < 0
    ]
    if convention is None:
        status = "WARNING"
        summary = "Pumped-storage sign convention could not be established."
    elif convention == "plus" and negative_rows:
        status = "ERROR"
        summary = "Positive-consumption convention confirmed with negative pumping values."
    else:
        status = "PASS"
        summary = f"Pumped-storage {convention} convention confirmed."

    return _check(
        "pumped_storage.relationship",
        "file",
        status,
        summary,
        {
            "tolerance_mwh": 0.02,
            "complete_row_count": int(complete.sum()),
            "convention": convention,
            "candidate_results": candidate_results,
            "mismatch_timestamps": mismatch_timestamps,
            "negative_pumped_storage_rows": negative_rows,
        },
    )


def _key_strings(values: set[pd.Timestamp]) -> list[str]:
    return [_iso_timestamp(value) for value in sorted(values)]


def audit_country_alignment(
    actual_path: str | Path,
    forecast_path: str | Path,
    country: str,
    timezone_name: str,
) -> dict[str, object]:
    actual = load_smard_file(actual_path, country, "actual", timezone_name)
    forecast = load_smard_file(forecast_path, country, "forecast", timezone_name)
    actual_data = actual["data"]
    forecast_data = forecast["data"]
    actual_keys = set(actual_data["interval_start_utc"].dropna())
    forecast_keys = set(forecast_data["interval_start_utc"].dropna())
    missing_forecast = actual_keys - forecast_keys
    missing_actual = forecast_keys - actual_keys
    actual_duplicate_keys = actual_data["interval_start_utc"].duplicated(keep=False)
    forecast_duplicate_keys = forecast_data["interval_start_utc"].duplicated(keep=False)
    interval_end_mismatches = []
    for key in sorted(actual_keys & forecast_keys):
        actual_rows = actual_data.index[actual_data["interval_start_utc"] == key]
        forecast_rows = forecast_data.index[forecast_data["interval_start_utc"] == key]
        if len(actual_rows) != 1 or len(forecast_rows) != 1:
            continue
        actual_end = actual_data.loc[actual_rows[0], "interval_end_utc"]
        forecast_end = forecast_data.loc[forecast_rows[0], "interval_end_utc"]
        if actual_end != forecast_end:
            interval_end_mismatches.append(
                {
                    "interval_start_utc": _iso_timestamp(key),
                    "actual_end_utc": _iso_timestamp(actual_end),
                    "forecast_end_utc": _iso_timestamp(forecast_end),
                }
            )

    forecast_invalid = _primary_target_check(
        forecast["source"], forecast_data, "forecast"
    )["evidence"]["unusable_benchmark_observations"]
    invalid_forecast_intervals = [
        {
            "row": row_number,
            "interval_start_utc": _iso_timestamp(
                forecast_data.iloc[row_number]["interval_start_utc"]
            ),
        }
        for row_number in forecast_data.index[
            forecast_data["interval_start_utc"].isna()
        ]
    ]
    actual_invalid = _primary_target_check(
        actual["source"], actual_data, "actual"
    )["evidence"]["invalid_observations"]
    actual_continuity_ok = actual["diagnostics"]["timestamps"][
        "utc_start_hourly_continuity"
    ]
    actual_errors = bool(
        missing_actual
        or actual_duplicate_keys.any()
        or actual_invalid
        or not actual_continuity_ok
    )
    forecast_warnings = bool(
        missing_forecast
        or forecast_duplicate_keys.any()
        or forecast_invalid
        or invalid_forecast_intervals
        or interval_end_mismatches
        or not forecast["diagnostics"]["timestamps"]["utc_start_hourly_continuity"]
    )
    status = "ERROR" if actual_errors else "WARNING" if forecast_warnings else "PASS"
    return _check(
        "alignment.country",
        f"country:{country}",
        status,
        "Actual and official forecast intervals align as one-to-one hourly benchmarks."
        if status == "PASS"
        else "Actual-target alignment contains errors."
        if status == "ERROR"
        else "Forecast benchmark alignment has unusable or missing intervals.",
        {
            "country": country,
            "timezone": timezone_name,
            "forecast_role": "external_benchmark_only",
            "actual_interval_count": len(actual_keys),
            "forecast_interval_count": len(forecast_keys),
            "matched_interval_count": len(actual_keys & forecast_keys),
            "missing_forecast_intervals": _key_strings(missing_forecast),
            "missing_actual_intervals": _key_strings(missing_actual),
            "duplicate_actual_intervals": [
                _iso_timestamp(value)
                for value in actual_data.loc[actual_duplicate_keys, "interval_start_utc"]
            ],
            "duplicate_forecast_intervals": [
                _iso_timestamp(value)
                for value in forecast_data.loc[
                    forecast_duplicate_keys, "interval_start_utc"
                ]
            ],
            "interval_end_mismatches": interval_end_mismatches,
            "invalid_forecast_intervals": invalid_forecast_intervals,
            "invalid_forecast_observations": forecast_invalid,
            "invalid_actual_observations": actual_invalid,
        },
    )


def audit_file(
    path: str | Path,
    country: str,
    dataset_type: str,
    timezone_name: str,
    audit_timestamp: str | None = None,
) -> dict[str, object]:
    path = Path(path)
    audit_timestamp = audit_timestamp or datetime.now(timezone.utc).isoformat()
    format_evidence, raw_headers = _inspect_csv_format(path)
    loaded = load_smard_file(path, country, dataset_type, timezone_name)
    source = loaded["source"]
    data = loaded["data"]
    expected_headers = _expected_columns(dataset_type)
    expected_mapping = {
        column: NUMERIC_COLUMN_MAP[column]
        for column in expected_headers
        if column in NUMERIC_COLUMN_MAP
    }
    expected_target = "grid load [MWh] Calculated resolutions"
    provenance = {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "country": country,
        "dataset_type": dataset_type,
        "timezone": timezone_name,
        "audit_timestamp": audit_timestamp,
    }
    timestamp_diagnostics = loaded["diagnostics"]["timestamps"]
    first_row = data.iloc[0]
    final_row = data.iloc[-1]
    coverage_evidence = {
        "first_interval_start_utc": _iso_timestamp(first_row["interval_start_utc"]),
        "first_interval_end_utc": _iso_timestamp(first_row["interval_end_utc"]),
        "final_interval_start_utc": _iso_timestamp(final_row["interval_start_utc"]),
        "final_interval_end_utc": _iso_timestamp(final_row["interval_end_utc"]),
    }
    utc_key_evidence = {
        "utc_start_unique": timestamp_diagnostics["utc_start_unique"],
        "utc_start_strictly_increasing": timestamp_diagnostics[
            "utc_start_strictly_increasing"
        ],
        "duplicate_utc_rows": int(data["interval_start_utc"].duplicated().sum()),
    }
    duration_evidence = {
        "utc_interval_duration_valid": timestamp_diagnostics[
            "utc_interval_duration_valid"
        ],
        "interval_count": len(data),
    }
    continuity_evidence = {
        "utc_start_hourly_continuity": timestamp_diagnostics[
            "utc_start_hourly_continuity"
        ],
        "expected_elapsed_hours": max(len(data) - 1, 0),
    }

    checks = [
        _check(
            "provenance.file",
            "file",
            "PASS",
            "File provenance was recorded.",
            provenance,
        ),
        _check(
            "format.delimiter",
            "file",
            "PASS" if format_evidence["delimiter_present"] else "ERROR",
            "Semicolon delimiter detected."
            if format_evidence["delimiter_present"]
            else "Semicolon delimiter was not detected.",
            {
                "delimiter": format_evidence["delimiter"],
                "header_field_count": format_evidence["header_field_count"],
            },
        ),
        _check(
            "format.field_counts",
            "file",
            "PASS" if format_evidence["fields_consistent"] else "ERROR",
            "All CSV records have consistent field counts."
            if format_evidence["fields_consistent"]
            else "CSV records have inconsistent field counts.",
            {
                "field_counts": format_evidence["field_counts"],
                "header_field_count": format_evidence["header_field_count"],
            },
        ),
        _check(
            "format.numeric_parsing",
            "file",
            "PASS",
            "Numeric parsing diagnostics were recorded.",
            loaded["diagnostics"]["numeric_columns"],
        ),
        _check(
            "format.bom",
            "file",
            "PASS",
            "UTF-8 BOM presence was recorded.",
            {"present": format_evidence["bom_present"]},
        ),
        _check(
            "schema.headers",
            "file",
            "PASS" if raw_headers == expected_headers else "ERROR",
            "Original headers match the expected SMARD schema."
            if raw_headers == expected_headers
            else "Original headers do not match the expected SMARD schema.",
            {"original": raw_headers, "expected": expected_headers},
        ),
        _check(
            "schema.mapping",
            "file",
            "PASS" if loaded["column_map"] == expected_mapping else "ERROR",
            "Standardized column mapping is documented."
            if loaded["column_map"] == expected_mapping
            else "Standardized column mapping is incomplete or unexpected.",
            {"actual": loaded["column_map"], "expected": expected_mapping},
        ),
        _check(
            "schema.shape",
            "file",
            "PASS",
            "Source row and column counts were recorded.",
            {"row_count": len(source), "column_count": len(source.columns)},
        ),
        _check(
            "schema.required_target",
            "file",
            "PASS"
            if expected_target in source.columns
            and "grid_load_mwh" in data.columns
            else "ERROR",
            "Primary grid-load target column is present."
            if expected_target in source.columns and "grid_load_mwh" in data.columns
            else "Primary grid-load target column is missing.",
            {
                "original_target": expected_target,
                "standardized_target": "grid_load_mwh",
                "present": expected_target in source.columns
                and "grid_load_mwh" in data.columns,
            },
        ),
        _check(
            "interval.coverage",
            "file",
            "PASS" if len(data) > 0 else "ERROR",
            "First and final intervals were recorded."
            if len(data) > 0
            else "No observations were available for interval coverage.",
            coverage_evidence,
        ),
        _check(
            "interval.utc_keys",
            "file",
            "PASS"
            if timestamp_diagnostics["utc_start_unique"]
            and timestamp_diagnostics["utc_start_strictly_increasing"]
            else "ERROR",
            "UTC interval starts are unique and chronological."
            if timestamp_diagnostics["utc_start_unique"]
            and timestamp_diagnostics["utc_start_strictly_increasing"]
            else "UTC interval starts are duplicated or not chronological.",
            utc_key_evidence,
        ),
        _check(
            "interval.duration",
            "file",
            "PASS" if timestamp_diagnostics["utc_interval_duration_valid"] else "ERROR",
            "Every UTC interval lasts one hour."
            if timestamp_diagnostics["utc_interval_duration_valid"]
            else "One or more UTC intervals do not last one hour.",
            duration_evidence,
        ),
        _check(
            "interval.continuity",
            "file",
            "PASS" if timestamp_diagnostics["utc_start_hourly_continuity"] else "ERROR",
            "UTC interval starts are continuously hourly."
            if timestamp_diagnostics["utc_start_hourly_continuity"]
            else "Unexpected UTC gap or non-hourly step detected.",
            continuity_evidence,
        ),
        _dst_check(data),
        _completeness_check(source, loaded, dataset_type),
        _duplicate_check(source, data),
        _numerical_summary_check(data, loaded["column_map"]),
        _primary_target_check(source, data, dataset_type),
        _pumped_storage_check(data, dataset_type),
    ]
    return {"provenance": provenance, "checks": checks}


def audit_raw_directory(
    raw_directory: str | Path,
    audit_timestamp: str | None = None,
) -> list[dict[str, object]]:
    raw_directory = Path(raw_directory)
    return [
        audit_file(
            raw_directory / SMARD_DIRECTORY / filename,
            metadata["country"],
            metadata["dataset_type"],
            metadata["timezone"],
            audit_timestamp=audit_timestamp,
        )
        for filename, metadata in EXPECTED_FILES.items()
    ]


def _environment_metadata() -> dict[str, object]:
    package_versions = {}
    for package in REPORT_PACKAGES:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "architecture_bits": struct.calcsize("P") * 8,
        "platform": platform.platform(),
        "packages": package_versions,
    }


def _verify_runtime() -> None:
    if platform.python_implementation() != "CPython":
        raise RuntimeError("the audit requires CPython 3.12")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("the audit requires Python 3.12")
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("the audit requires 64-bit Python")


def _column_role(standardized: str, dataset_type: str) -> str:
    if standardized == "grid_load_mwh" and dataset_type == "actual":
        return "primary_actual_target"
    if standardized == "grid_load_mwh" and dataset_type == "forecast":
        return "official_forecast_benchmark"
    return {
        "grid_load_including_pumped_storage_mwh": "auxiliary_grid_load_including_pumped_storage",
        "pumped_storage_mwh": "auxiliary_pumped_storage",
        "residual_load_mwh": "auxiliary_residual_load",
    }[standardized]


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARNING": 1, "ERROR": 2}[status]


def _column_summary_rows(
    filename: str,
    country: str,
    dataset_type: str,
    loaded: dict[str, object],
    checks: list[dict[str, object]],
) -> list[dict[str, object]]:
    check_map = {check["check_id"]: check for check in checks}
    completeness = check_map["completeness.values"]["evidence"]
    primary_status = check_map["primary_target.validity"]["status"]
    pumping_status = check_map["pumped_storage.relationship"]["status"]
    rows = []
    for original, standardized in loaded["column_map"].items():
        values = loaded["data"][standardized]
        summary = _numeric_summary(values, loaded["data"])
        numeric_diagnostics = loaded["diagnostics"]["numeric_columns"][original]
        statuses = [
            "WARNING"
            if any(value > 0 for value in completeness[standardized].values())
            else "PASS"
        ]
        if standardized == "grid_load_mwh":
            statuses.append(primary_status)
        if standardized == "pumped_storage_mwh":
            statuses.append(pumping_status)
        status = max(statuses, key=_status_rank)
        rows.append(
            {
                "file": filename,
                "country": country,
                "dataset_type": dataset_type,
                "original_column": original,
                "standardized_column": standardized,
                "role": _column_role(standardized, dataset_type),
                "unit": "MWh",
                "source_dtype": str(loaded["source"][original].dtype),
                "parsed_dtype": str(values.dtype),
                "row_count": len(loaded["source"]),
                "valid_value_count": summary["valid_count"],
                "missing_count": numeric_diagnostics["missing_count"],
                "non_numeric_count": numeric_diagnostics["non_numeric_count"],
                "non_finite_count": numeric_diagnostics["non_finite_count"],
                "minimum": summary["minimum"],
                "p1": summary["percentiles"]["1"],
                "p5": summary["percentiles"]["5"],
                "p25": summary["percentiles"]["25"],
                "p50": summary["percentiles"]["50"],
                "p75": summary["percentiles"]["75"],
                "p95": summary["percentiles"]["95"],
                "p99": summary["percentiles"]["99"],
                "mean": summary["mean"],
                "maximum": summary["maximum"],
                "status": status,
            }
        )
    return rows


def _status_counts(checks: list[dict[str, object]]) -> dict[str, int]:
    return {
        status: sum(check["status"] == status for check in checks)
        for status in ("PASS", "WARNING", "ERROR")
    }


def _overall_status(checks: list[dict[str, object]]) -> str:
    counts = _status_counts(checks)
    if counts["ERROR"]:
        return "ERROR"
    if counts["WARNING"]:
        return "WARNING"
    return "PASS"


def _build_run(
    project_root: Path, audit_timestamp: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    raw_directory = project_root / "data" / "raw" / SMARD_DIRECTORY
    dataset_results = []
    column_rows = []
    for filename, metadata in EXPECTED_FILES.items():
        path = raw_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"required raw file not found: {path}")
        result = audit_file(
            path,
            metadata["country"],
            metadata["dataset_type"],
            metadata["timezone"],
            audit_timestamp=audit_timestamp,
        )
        loaded = load_smard_file(
            path,
            metadata["country"],
            metadata["dataset_type"],
            metadata["timezone"],
        )
        dataset_results.append(
            {
                "filename": filename,
                "country": metadata["country"],
                "dataset_type": metadata["dataset_type"],
                "timezone": metadata["timezone"],
                "provenance": result["provenance"],
                "checks": result["checks"],
            }
        )
        column_rows.extend(
            _column_summary_rows(
                filename,
                metadata["country"],
                metadata["dataset_type"],
                loaded,
                result["checks"],
            )
        )

    alignments = []
    for country in ("Austria", "Germany"):
        metadata = next(
            value
            for value in EXPECTED_FILES.values()
            if value["country"] == country and value["dataset_type"] == "actual"
        )
        alignments.append(
            audit_country_alignment(
                raw_directory / f"smard_{country.lower()}_actual.csv",
                raw_directory / f"smard_{country.lower()}_forecasted.csv",
                country,
                metadata["timezone"],
            )
        )

    all_checks = [
        check
        for dataset in dataset_results
        for check in dataset["checks"]
    ] + alignments
    counts = _status_counts(all_checks)
    report = {
        "report_schema_version": "1.0",
        "audit_timestamp": audit_timestamp,
        "environment": _environment_metadata(),
        "overall_status": _overall_status(all_checks),
        "check_counts": counts,
        "datasets": dataset_results,
        "alignments": alignments,
        "target_and_benchmark": {
            "actual_target": "grid_load_mwh",
            "official_forecast_benchmark": "grid_load_mwh",
            "benchmark_role": "external_benchmark_only",
            "countries": ["Austria", "Germany"],
        },
        "column_summary": column_rows,
    }
    return report, column_rows


def _limited(value: object, limit: int = 5) -> object:
    if isinstance(value, list):
        limited = [_limited(item, limit) for item in value[:limit]]
        if len(value) > limit:
            limited.append("... additional items omitted")
        return limited
    if isinstance(value, dict):
        return {key: _limited(item, limit) for key, item in value.items()}
    return value


def _markdown_report(report: dict[str, object]) -> str:
    counts = report["check_counts"]
    lines = [
        "# SMARD Data Audit",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Checks: {counts['PASS']} PASS, {counts['WARNING']} WARNING, {counts['ERROR']} ERROR",
        f"- Audit timestamp: `{report['audit_timestamp']}`",
        "",
        "## Target and Benchmark",
        "",
        "- Actual target: `grid_load_mwh` from actual SMARD grid load.",
        "- Official forecast: `grid_load_mwh` as an external benchmark only.",
        "- Countries are analysed separately: Austria and Germany.",
        "",
        "## Environment",
        "",
        f"- Python: `{report['environment']['python_version'].splitlines()[0]}`",
        f"- Implementation: `{report['environment']['python_implementation']}`",
        f"- Architecture: `{report['environment']['architecture_bits']}-bit`",
        "",
        "## Datasets",
        "",
    ]
    for dataset in report["datasets"]:
        lines.extend(
            [
                f"### `{dataset['filename']}`",
                "",
                f"- Country: `{dataset['country']}`; type: `{dataset['dataset_type']}`; timezone: `{dataset['timezone']}`",
                f"- SHA-256: `{dataset['provenance']['sha256']}`",
                f"- File size: `{dataset['provenance']['file_size_bytes']}` bytes",
            ]
        )
        for check in dataset["checks"]:
            evidence = json.dumps(
                _limited(check["evidence"]), ensure_ascii=False, sort_keys=True
            )
            lines.append(
                f"- **{check['status']}** `{check['check_id']}`: {check['summary']} Evidence: `{evidence}`"
            )
        lines.append("")
    lines.extend(["## Actual/Forecast Alignment", ""])
    for alignment in report["alignments"]:
        lines.append(
            f"- **{alignment['status']}** `{alignment['scope']}`: {alignment['summary']} Evidence: `"
            + json.dumps(_limited(alignment["evidence"]), sort_keys=True)
            + "`"
        )
    lines.append("")
    return "\n".join(lines)


def _column_csv(report_rows: list[dict[str, object]]) -> str:
    fieldnames = [
        "file",
        "country",
        "dataset_type",
        "original_column",
        "standardized_column",
        "role",
        "unit",
        "source_dtype",
        "parsed_dtype",
        "row_count",
        "valid_value_count",
        "missing_count",
        "non_numeric_count",
        "non_finite_count",
        "minimum",
        "p1",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "mean",
        "maximum",
        "status",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report_rows)
    return output.getvalue()


def write_report_set(
    output_directory: str | Path,
    markdown: str,
    json_text: str,
    column_csv: str,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    json.loads(json_text)
    list(csv.reader(io.StringIO(column_csv)))
    contents = {
        "smard_data_audit.md": markdown,
        "smard_data_audit.json": json_text,
        "smard_column_summary.csv": column_csv,
    }
    previous = {
        filename: (output_directory / filename).read_bytes()
        if (output_directory / filename).exists()
        else None
        for filename in REPORT_FILENAMES
    }
    temporary_paths = []
    try:
        for filename, text in contents.items():
            handle, temporary_name = tempfile.mkstemp(
                prefix=".smard_report_", suffix=".tmp", dir=output_directory
            )
            temporary_path = Path(temporary_name)
            temporary_paths.append(temporary_path)
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
                file.write(text)
        for filename, temporary_path in zip(contents, temporary_paths):
            os.replace(temporary_path, output_directory / filename)
    except Exception:
        for filename, old_contents in previous.items():
            target = output_directory / filename
            if old_contents is None:
                if target.exists():
                    target.unlink()
            else:
                target.write_bytes(old_contents)
        raise
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()


def run_audit(
    project_root: str | Path | None = None,
    audit_timestamp: str | None = None,
) -> int:
    try:
        _verify_runtime()
        project_root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        audit_timestamp = audit_timestamp or datetime.now(timezone.utc).isoformat()
        report, column_rows = _build_run(project_root, audit_timestamp)
        markdown = _markdown_report(report)
        json_text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
        column_csv = _column_csv(column_rows)
        json.loads(json_text)
        list(csv.DictReader(io.StringIO(column_csv)))
        write_report_set(project_root / "results" / "tables", markdown, json_text, column_csv)
    except Exception as error:
        print(f"SMARD audit failed: {type(error).__name__}: {error}")
        return 2

    counts = report["check_counts"]
    print(
        f"SMARD audit: {report['overall_status']} | Files: {len(report['datasets'])} | "
        f"Checks: {counts['PASS']} PASS, {counts['WARNING']} WARNING, "
        f"{counts['ERROR']} ERROR | Reports: results/tables/smard_data_audit.md, "
        "results/tables/smard_data_audit.json, results/tables/smard_column_summary.csv"
    )
    return 1 if report["overall_status"] == "ERROR" else 0


def main() -> int:
    return run_audit()


if __name__ == "__main__":
    raise SystemExit(main())
