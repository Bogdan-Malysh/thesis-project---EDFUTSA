from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd

import sarimax_validation_2024 as validation
from sarimax_models import SARIMAX_SPECIFICATION_IDS


MINI_DATES = (date(2024, 2, 15), date(2024, 3, 31), date(2024, 10, 27))


def test_sarimax_manifest_has_twelve_unique_jobs() -> None:
    manifest = validation.build_manifest(MINI_DATES)

    assert len(manifest) == 12
    assert set(manifest["specification_id"]) == set(SARIMAX_SPECIFICATION_IDS)
    assert not manifest.duplicated(list(validation.JOB_KEY_COLUMNS)).any()
    assert manifest.groupby("country").size().to_dict() == {
        "Germany": 6,
        "Austria": 6,
    }


def test_sarimax_forecast_path_preserves_ordinary_and_dst_lengths() -> None:
    frame = validation.load_validation_country_data("Germany", "data/processed")

    class Fitted:
        def get_forecast(self, steps: int, exog: pd.DataFrame):
            assert steps == len(exog)
            return type("Forecast", (), {"predicted_mean": [1.0] * steps})()

    for target_date, expected_length in (
        (date(2024, 2, 15), 24),
        (date(2024, 3, 31), 23),
        (date(2024, 10, 27), 25),
    ):
        path = validation.build_sarimax_forecast_path(
            frame,
            "Germany",
            target_date,
            "sarimax_calendar_crisis",
            Fitted(),
        )
        assert int(path["is_target_day"].sum()) == expected_length
        assert path["timestamp_utc"].is_unique
        assert path["forecast_mwh"].notna().all()
        assert path["forecast_origin_utc"].eq(path["information_cutoff_utc"]).all()


def test_sarimax_checkpoint_upsert_prevents_duplicate_job_keys() -> None:
    existing = pd.DataFrame(
        [{"country": "Germany", "target_date": "2024-02-15", "specification_id": "sarimax_calendar", "status": "started"}]
    )
    replacement = pd.DataFrame(
        [{"country": "Germany", "target_date": "2024-02-15", "specification_id": "sarimax_calendar", "status": "completed"}]
    )

    updated = validation.upsert_frame(
        existing,
        replacement,
        validation.JOB_KEY_COLUMNS,
    )

    assert len(updated) == 1
    assert updated.iloc[0]["status"] == "completed"


def test_sarimax_outputs_include_optimizer_iteration_metadata() -> None:
    assert "optimizer_iterations" in validation.JOB_COLUMNS
    assert "optimizer_function_calls" in validation.JOB_COLUMNS
    assert "optimizer_iterations" in validation.DIAGNOSTIC_COLUMNS
    assert "optimizer_function_calls" in validation.DIAGNOSTIC_COLUMNS


def test_full_year_cli_forwards_requested_output_directory(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_run_full_validation(**kwargs):
        calls.update(kwargs)
        return {"output_directory": str(kwargs["output_directory"])}

    monkeypatch.setattr(validation, "run_full_validation", fake_run_full_validation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sarimax_validation_2024.py",
            "--full-year",
            "--output-directory",
            str(tmp_path),
            "--processed-directory",
            "processed",
            "--workers",
            "4",
        ],
    )

    validation.main()

    assert calls == {
        "output_directory": tmp_path,
        "workers": 4,
        "processed_directory": Path("processed"),
    }
