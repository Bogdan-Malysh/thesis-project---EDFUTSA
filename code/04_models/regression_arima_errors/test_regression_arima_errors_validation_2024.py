from __future__ import annotations

from datetime import date

import pandas as pd

import regression_arima_errors_validation_2024 as validation
from regression_arima_errors_validation_2024 import (
    MINI_DATES,
    build_manifest,
    build_regression_forecast_path,
    upsert_frame,
)


def test_mini_manifest_has_24_jobs_and_no_2025_dates() -> None:
    manifest = build_manifest(MINI_DATES)

    assert len(manifest) == 24
    assert manifest["target_date"].str.startswith("2024-").all()
    assert not manifest.duplicated(
        ["country", "target_date", "specification_id"]
    ).any()
    assert manifest.groupby("country").size().to_dict() == {
        "Austria": 12,
        "Germany": 12,
    }


def test_validation_loader_reads_only_through_2024(monkeypatch) -> None:
    observed: dict[str, object] = {}
    original_read_csv = validation.pd.read_csv

    def read_csv(*args, **kwargs):
        observed["nrows"] = kwargs.get("nrows")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(validation.pd, "read_csv", read_csv)
    frame = validation._validation_frame("Germany", "data/processed")

    assert observed["nrows"] == 43_848
    assert len(frame) == 43_848
    assert frame["local_date"].max() == "2024-12-31"


def test_forecast_path_requires_exact_exog_timestamp_alignment() -> None:
    timestamps = pd.date_range(
        "2024-02-13 00:00:00+00:00", periods=72, freq="h"
    )
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "timestamp_local": timestamps.tz_convert("Europe/Berlin").map(
                pd.Timestamp.isoformat
            ),
            "actual_grid_load_mwh": range(len(timestamps)),
            "hour": timestamps.tz_convert("Europe/Berlin").hour,
            "day_of_week": timestamps.tz_convert("Europe/Berlin").dayofweek,
        }
    )

    class Fitted:
        def get_forecast(self, steps: int, exog: pd.DataFrame):
            assert steps == len(exog)
            return type("Forecast", (), {"predicted_mean": [1.0] * steps})()

    path = build_regression_forecast_path(
        frame,
        "Germany",
        date(2024, 2, 15),
        "reg_arima_h",
        Fitted(),
    )

    assert path["timestamp_utc"].is_unique
    assert path["is_target_day"].sum() == 24
    assert path["bridge_used"].sum() > 0
    assert path["forecast_mwh"].notna().all()


def test_crisis_forecast_paths_cover_ordinary_and_dst_dates() -> None:
    frame = validation._validation_frame("Germany", "data/processed")

    class Fitted:
        def get_forecast(self, steps: int, exog: pd.DataFrame):
            assert steps == len(exog)
            return type("Forecast", (), {"predicted_mean": [1.0] * steps})()

    expected_lengths = {
        date(2024, 2, 15): 24,
        date(2024, 3, 31): 23,
        date(2024, 10, 27): 25,
    }
    for target_date, expected_length in expected_lengths.items():
        path = build_regression_forecast_path(
            frame,
            "Germany",
            target_date,
            "reg_arima_h_d_m_hol_crisis",
            Fitted(),
        )
        assert path["timestamp_utc"].is_unique
        assert int(path["is_target_day"].sum()) == expected_length
        assert path["forecast_origin_utc"].eq(path["information_cutoff_utc"]).all()
        assert path["forecast_mwh"].notna().all()


def test_checkpoint_upsert_prevents_duplicate_crisis_job_keys() -> None:
    existing = pd.DataFrame(
        [{"country": "Germany", "target_date": "2024-02-15", "specification_id": "crisis", "status": "started"}]
    )
    replacement = pd.DataFrame(
        [{"country": "Germany", "target_date": "2024-02-15", "specification_id": "crisis", "status": "completed"}]
    )

    updated = upsert_frame(
        existing,
        replacement,
        ["country", "target_date", "specification_id"],
    )

    assert len(updated) == 1
    assert updated.iloc[0]["status"] == "completed"
