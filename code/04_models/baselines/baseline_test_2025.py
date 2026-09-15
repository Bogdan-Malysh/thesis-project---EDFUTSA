from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import (
    BASELINE_MODELS,
    COUNTRY_CONFIG,
    PROCESSED,
    InformationContext,
    build_information_context,
    evaluate_target_day,
    extract_target_day,
    forecast_path,
    load_country_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_FAMILY = "baselines"
SPECIFICATION_ID = "Weekly seasonal naive"
VALIDATION_YEAR = 2025
EXPECTED_DAYS = 365
EXPECTED_OBSERVATIONS = 8760
FORECAST_PATH = (
    PROJECT_ROOT / "results" / "baselines" / "forecasts" / "baseline_forecasts_2025.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT / "results" / "baselines" / "tables" / "baseline_validation_2025.csv"
)
ALL_MODELS_PATH = PROJECT_ROOT / "results" / "model_test_2025_all_models.csv"
OVERVIEW_PATH = PROJECT_ROOT / "results" / "model_test_2025_overview.csv"
TABLE_COLUMNS = [
    "country",
    "model_family",
    "specification_id",
    "mae",
    "rmse",
    "mape",
    "n_observations",
    "coverage",
]
TABLE_KEY_COLUMNS = ["country", "model_family", "specification_id"]
COUNTRY_ORDER = {country: index for index, country in enumerate(COUNTRY_CONFIG)}

if SPECIFICATION_ID not in BASELINE_MODELS:
    raise RuntimeError(f"unsupported authoritative baseline: {SPECIFICATION_ID}")


def validation_dates(frame: pd.DataFrame) -> list[date]:
    values = sorted(
        {
            date.fromisoformat(value)
            for value in frame.loc[
                frame["local_date"].astype(str).str.startswith(f"{VALIDATION_YEAR:04d}-"),
                "local_date",
            ]
        }
    )
    expected = [date(VALIDATION_YEAR, 1, 1) + timedelta(days=offset) for offset in range(365)]
    if values != expected:
        raise ValueError("2025 baseline test requires every local date from 2025-01-01 through 2025-12-31")
    return values


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def validate_forecast_path(
    path: pd.DataFrame,
    country: str,
    target_date: date | str | pd.Timestamp,
    context: InformationContext,
) -> None:
    target_text = pd.Timestamp(target_date).date().isoformat()
    if path.empty:
        raise ValueError(f"empty weekly seasonal naive path for {country} {target_text}")
    if not path["model_family"].eq(SPECIFICATION_ID).all():
        raise ValueError("forecast path contains a non-authoritative baseline")
    if not path["target_date"].eq(target_text).all():
        raise ValueError("forecast path contains an incorrect target date")
    timestamps = pd.to_datetime(path["timestamp_utc"], utc=True, errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("forecast path contains duplicate or invalid UTC timestamps")
    forecasts = pd.to_numeric(path["forecast_mwh"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(forecasts).all():
        raise ValueError("forecast path contains non-finite forecasts")
    if not path["forecast_origin_utc"].eq(path["information_cutoff_utc"]).all():
        raise ValueError("forecast origin and information cutoff differ")
    if not path["forecast_origin_utc"].eq(context.cutoff_utc.isoformat()).all():
        raise ValueError("forecast path information cutoff does not match the context")

    target = path.loc[path["is_target_day"]]
    if target.empty or not target["local_date"].eq(target_text).all():
        raise ValueError("forecast path does not contain the complete target day")
    if len(target) not in (23, 24, 25):
        raise ValueError("target day has an invalid DST interval count")
    if target["timestamp_utc"].duplicated().any():
        raise ValueError("target day contains duplicate UTC timestamps")

    cutoff = _utc_timestamp(context.cutoff_utc)
    for row_index, row in path.reset_index(drop=True).iterrows():
        timestamp = _utc_timestamp(row["timestamp_utc"])
        source = _utc_timestamp(row["source_timestamp_utc"])
        expected_source = timestamp - pd.Timedelta(hours=168)
        if source != expected_source:
            raise ValueError("weekly source timestamp does not use the 168-hour lag")
        source_kind = str(row["source_kind"])
        if source_kind == "observed":
            if source + pd.Timedelta(hours=1) > cutoff:
                raise ValueError("weekly source timestamp is after the information cutoff")
        elif source_kind == "forecast":
            prior_rows = path.reset_index(drop=True).iloc[:row_index]
            if not (pd.to_datetime(prior_rows["timestamp_utc"], utc=True) == source).any():
                raise ValueError("weekly forecast source does not precede the forecast row")
        else:
            raise ValueError(f"weekly source timestamp has unsupported source kind: {source_kind}")


def summarize_forecasts(
    country: str,
    paths: list[pd.DataFrame],
    expected_observations: int,
) -> dict[str, object]:
    if not paths:
        raise ValueError(f"no weekly seasonal naive paths for {country}")
    combined = pd.concat(paths, ignore_index=True)
    metrics = evaluate_target_day(combined, expected_observations=expected_observations)
    return {
        "country": country,
        "model_family": SPECIFICATION_ID,
        "specification_id": SPECIFICATION_ID,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "evaluated_observations": metrics["evaluated_observations"],
        "expected_observations": expected_observations,
        "coverage": metrics["coverage"],
        "validation_days": len(paths),
    }


def _read_existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TABLE_COLUMNS)
    frame = pd.read_csv(path)
    missing = sorted(set(TABLE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required 2025 table columns: {missing}")
    return frame


def _table_rows(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "country",
        "mae",
        "rmse",
        "mape",
        "evaluated_observations",
        "coverage",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"2025 baseline summary is missing required columns: {missing}")
    rows = summary.copy()
    rows["model_family"] = MODEL_FAMILY
    rows["specification_id"] = SPECIFICATION_ID
    rows = rows.rename(columns={"evaluated_observations": "n_observations"})
    return rows.loc[:, TABLE_COLUMNS]


def _merge_table(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    extra_columns = sorted(set(existing.columns).difference(TABLE_COLUMNS))
    columns = TABLE_COLUMNS + extra_columns
    existing_aligned = existing.reindex(columns=columns)
    incoming_aligned = incoming.reindex(columns=columns)
    merged = pd.concat([existing_aligned, incoming_aligned], ignore_index=True)
    merged = merged.drop_duplicates(TABLE_KEY_COLUMNS, keep="last")
    merged["_country_order"] = merged["country"].map(COUNTRY_ORDER).fillna(len(COUNTRY_ORDER))
    merged = merged.sort_values(
        ["_country_order", "model_family", "specification_id"],
        kind="mergesort",
    ).drop(columns="_country_order")
    return merged.reset_index(drop=True).loc[:, columns]


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def update_test_tables(
    summary_path: str | Path = SUMMARY_PATH,
    all_models_path: str | Path = ALL_MODELS_PATH,
    overview_path: str | Path = OVERVIEW_PATH,
) -> dict[str, Path]:
    summary = pd.read_csv(summary_path)
    incoming = _table_rows(summary)
    all_path = Path(all_models_path)
    overview_path = Path(overview_path)
    all_models = _merge_table(_read_existing_table(all_path), incoming)
    overview = _merge_table(_read_existing_table(overview_path), incoming)
    _atomic_write(all_models, all_path)
    _atomic_write(overview, overview_path)
    return {"all_models": all_path, "overview": overview_path}


def run_validation(
    processed_directory: str | Path = PROCESSED,
    forecast_path_output: str | Path = FORECAST_PATH,
    summary_path_output: str | Path = SUMMARY_PATH,
) -> dict[str, object]:
    all_paths: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    per_country_target_counts: dict[str, dict[str, int]] = {}

    for country in COUNTRY_CONFIG:
        frame = load_country_data(country, processed_directory)
        dates = validation_dates(frame)
        country_paths: list[pd.DataFrame] = []
        for target_date in dates:
            context = build_information_context(frame, country, target_date)
            path = forecast_path(
                frame,
                country,
                target_date,
                SPECIFICATION_ID,
                context=context,
            )
            validate_forecast_path(path, country, target_date, context)
            country_paths.append(path)

        target_rows = pd.concat(
            [path.loc[path["is_target_day"]] for path in country_paths],
            ignore_index=True,
        )
        if target_rows["timestamp_utc"].duplicated().any():
            raise ValueError(f"duplicate scored UTC timestamps for {country}")
        if len(target_rows) != EXPECTED_OBSERVATIONS:
            raise ValueError(f"{country} has {len(target_rows)} scored observations, expected {EXPECTED_OBSERVATIONS}")
        for dst_date, expected_count in (("2025-03-30", 23), ("2025-10-26", 25)):
            dst_rows = target_rows.loc[target_rows["local_date"].eq(dst_date)]
            if len(dst_rows) != expected_count:
                raise ValueError(f"{country} {dst_date} has {len(dst_rows)} intervals, expected {expected_count}")

        all_paths.extend(country_paths)
        summaries.append(summarize_forecasts(country, country_paths, EXPECTED_OBSERVATIONS))
        per_country_target_counts[country] = {
            "validation_days": len(dates),
            "evaluated_observations": len(target_rows),
        }

    forecasts = pd.concat(all_paths, ignore_index=True)
    forecast_output = Path(forecast_path_output)
    forecast_output.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(forecast_output, index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_output = Path(summary_path_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_output, index=False)
    table_paths = update_test_tables(summary_output)
    completed_jobs = sum(
        values["validation_days"] for values in per_country_target_counts.values()
    )
    return {
        "total_jobs": completed_jobs,
        "completed_jobs": completed_jobs,
        "countries": per_country_target_counts,
        "forecast_rows": len(forecasts),
        "table_paths": {name: str(path) for name, path in table_paths.items()},
        "summary": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final 2025 Weekly seasonal naive test only"
    )
    parser.add_argument("--processed-directory", type=Path, default=PROCESSED)
    parser.add_argument("--forecast-output", type=Path, default=FORECAST_PATH)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    result = run_validation(
        processed_directory=args.processed_directory,
        forecast_path_output=args.forecast_output,
        summary_path_output=args.summary_output,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
