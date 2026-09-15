import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOADER_PATH = PROJECT_ROOT / "code" / "01_data_audit" / "load_smard.py"
LOADER_SPEC = importlib.util.spec_from_file_location("audited_smard_loader", LOADER_PATH)
AUDITED_LOADER = importlib.util.module_from_spec(LOADER_SPEC)
LOADER_SPEC.loader.exec_module(AUDITED_LOADER)
load_smard_file = AUDITED_LOADER.load_smard_file

EXPECTED_ROWS = 52_608
COUNTRY_CONFIG = {
    "Austria": "Europe/Vienna",
    "Germany": "Europe/Berlin",
}


def _valid(values: pd.Series, positive: bool = False) -> pd.Series:
    flags = values.notna() & np.isfinite(values)
    if positive:
        flags &= values > 0
    return flags.astype(bool)


def _actual_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[
        [
            "interval_start_raw",
            "interval_end_raw",
            "interval_start_local",
            "interval_end_local",
            "interval_start_utc",
            "interval_end_utc",
            "is_repeated_autumn_hour",
            "grid_load_mwh",
            "grid_load_including_pumped_storage_mwh",
            "pumped_storage_mwh",
            "residual_load_mwh",
        ]
    ].copy()
    return frame.rename(
        columns={
            "grid_load_mwh": "actual_grid_load_mwh",
            "grid_load_including_pumped_storage_mwh": (
                "actual_grid_load_including_pumped_storage_mwh"
            ),
            "pumped_storage_mwh": "actual_pumped_storage_mwh",
            "residual_load_mwh": "actual_residual_load_mwh",
        }
    )


def _forecast_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        ["interval_start_utc", "interval_end_utc", "grid_load_mwh", "residual_load_mwh"]
    ].rename(
        columns={
            "grid_load_mwh": "forecasted_grid_load_mwh",
            "residual_load_mwh": "forecasted_residual_load_mwh",
        }
    )


def build_country_hourly(raw_directory: str | Path, country: str) -> pd.DataFrame:
    raw_directory = Path(raw_directory)
    timezone_name = COUNTRY_CONFIG[country]
    country_name = country.lower()
    actual = load_smard_file(
        raw_directory / f"smard_{country_name}_actual.csv",
        country,
        "actual",
        timezone_name,
    )
    forecast = load_smard_file(
        raw_directory / f"smard_{country_name}_forecasted.csv",
        country,
        "forecast",
        timezone_name,
    )
    actual_frame = _actual_frame(actual["data"])
    forecast_frame = _forecast_frame(forecast["data"])
    merged = actual_frame.merge(
        forecast_frame,
        on="interval_start_utc",
        how="outer",
        validate="one_to_one",
        suffixes=("", "_forecast"),
    )
    if not (
        merged["interval_end_utc"] == merged["interval_end_utc_forecast"]
    ).fillna(False).all():
        raise ValueError(f"{country} actual and forecast intervals do not align")
    merged = merged.drop(columns=["interval_end_utc_forecast"])
    merged["actual_grid_load_valid"] = _valid(
        merged["actual_grid_load_mwh"], positive=True
    )
    merged["forecasted_grid_load_valid"] = _valid(
        merged["forecasted_grid_load_mwh"], positive=True
    )
    merged["actual_grid_load_including_pumped_storage_valid"] = _valid(
        merged["actual_grid_load_including_pumped_storage_mwh"]
    )
    merged["actual_pumped_storage_valid"] = _valid(
        merged["actual_pumped_storage_mwh"]
    )
    merged["actual_residual_load_valid"] = _valid(merged["actual_residual_load_mwh"])
    merged["forecasted_residual_load_valid"] = _valid(
        merged["forecasted_residual_load_mwh"]
    )
    return merged.sort_values("interval_start_utc", ignore_index=True)


def preprocess_smard(
    raw_directory: str | Path,
    processed_directory: str | Path,
) -> dict[str, pd.DataFrame]:
    processed_directory = Path(processed_directory)
    processed_directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for country in COUNTRY_CONFIG:
        data = build_country_hourly(raw_directory, country)
        if len(data) != EXPECTED_ROWS:
            raise ValueError(f"{country} expected {EXPECTED_ROWS} rows, found {len(data)}")
        output_path = processed_directory / f"smard_{country.lower()}_hourly.csv"
        output = data.copy()
        output.to_csv(output_path, index=False)
        results[country] = data
    return results


def main() -> int:
    preprocess_smard(
        PROJECT_ROOT / "data" / "raw" / "smard",
        PROJECT_ROOT / "data" / "processed",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
