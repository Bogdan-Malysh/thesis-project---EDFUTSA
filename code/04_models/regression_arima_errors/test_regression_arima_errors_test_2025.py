from __future__ import annotations

from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

import regression_arima_errors_test_2025 as validation
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    country_forecast_origin,
    information_set,
    load_country_data,
)
from regression_arima_errors_models import specification_columns
from regression_arima_errors_validation_2024 import build_regression_forecast_path


def test_authoritative_selection_and_frozen_specification() -> None:
    selected = validation.verify_authoritative_selection()

    assert selected.set_index("country")["selected_specification_id"].to_dict() == {
        "Germany": "reg_arima_h_d_m_hol",
        "Austria": "reg_arima_h_d_m_hol",
    }
    assert selected["completed_jobs"].eq(366).all()
    assert selected["failed_jobs"].eq(0).all()
    assert selected["coverage"].eq(1.0).all()

    specification = validation.load_frozen_specification()
    row = specification.iloc[0]
    assert row["specification_id"] == validation.SPECIFICATION_ID
    assert int(row["specification_order"]) == 4
    assert json.loads(row["error_order"]) == [2, 1, 2]
    assert row["trend"] == "n"
    assert json.loads(row["exog_columns"]) == list(
        specification_columns(validation.SPECIFICATION_ID)
    )


def test_2025_manifest_has_730_unique_jobs_and_country_origins() -> None:
    frames = {country: load_country_data(country) for country in COUNTRY_CONFIG}
    dates = validation.test_dates(frames["Germany"])
    manifest = validation.build_test_manifest(frames, dates)

    assert len(dates) == 365
    assert len(manifest) == 730
    assert manifest[["country", "target_date", "specification_id"]].drop_duplicates().shape[0] == 730
    assert set(manifest["specification_id"]) == {validation.SPECIFICATION_ID}
    for row in manifest.to_dict("records"):
        origin = country_forecast_origin(row["target_date"], row["country"])
        expected_hour = 18 if row["country"] == "Germany" else 8
        timezone = COUNTRY_CONFIG[row["country"]]["timezone"]
        assert origin.tz_convert(timezone).hour == expected_hour


def test_training_counts_follow_expanding_origin_bounded_information_set() -> None:
    frames = {country: load_country_data(country) for country in COUNTRY_CONFIG}
    manifest = validation.build_test_manifest(
        frames,
        [date(2025, 1, 2), date(2025, 7, 1)],
    )
    germany = manifest.loc[manifest["country"].eq("Germany")]
    early = germany.iloc[0]
    late = germany.iloc[1]

    early_expected = len(
        information_set(
            frames["Germany"],
            country_forecast_origin(early["target_date"], "Germany"),
        )
    )
    late_expected = len(
        information_set(
            frames["Germany"],
            country_forecast_origin(late["target_date"], "Germany"),
        )
    )
    assert validation.expected_training_observations(early, frames["Germany"]) == early_expected
    assert validation.expected_training_observations(late, frames["Germany"]) == late_expected
    assert late_expected > early_expected


class _Fitted:
    def get_forecast(self, steps: int, exog: pd.DataFrame):
        assert steps == len(exog)
        return type("Forecast", (), {"predicted_mean": np.full(steps, 1.0)})()


def _target_path(country: str, target_date: date) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = load_country_data(country)
    path = build_regression_forecast_path(
        frame,
        country,
        target_date,
        validation.SPECIFICATION_ID,
        _Fitted(),
    )
    origin = country_forecast_origin(target_date, country)
    job = {
        "country": country,
        "target_date": target_date.isoformat(),
        "specification_id": validation.SPECIFICATION_ID,
        "status": "completed",
        "forecast_origin_utc": origin.isoformat(),
        "information_cutoff_utc": origin.isoformat(),
        "training_observations": len(information_set(frame, origin)),
        "expected_observations": int(path["is_target_day"].sum()),
    }
    return path, job


@pytest.mark.parametrize(
    ("target_date", "expected_intervals"),
    [(date(2025, 3, 30), 23), (date(2025, 10, 26), 25)],
)
def test_forecast_validation_preserves_dst_and_finite_predictions(
    target_date: date,
    expected_intervals: int,
) -> None:
    path, job = _target_path("Germany", target_date)

    validation.validate_job_result(job, path, load_country_data("Germany"))

    assert int(path["is_target_day"].sum()) == expected_intervals
    assert path["timestamp_utc"].is_unique
    assert np.isfinite(path["forecast_mwh"].to_numpy(dtype=float)).all()


def test_forecast_validation_rejects_training_information_after_origin() -> None:
    path, job = _target_path("Austria", date(2025, 3, 30))
    job["training_observations"] = int(job["training_observations"]) + 1

    with pytest.raises(ValueError, match="training observations"):
        validation.validate_job_result(job, path, load_country_data("Austria"))


def test_forecast_validation_rejects_future_path_date() -> None:
    path, job = _target_path("Germany", date(2025, 3, 30))
    path = path.copy()
    bridge = ~path["is_target_day"]
    path.loc[bridge.idxmax(), "local_date"] = "2025-03-31"

    with pytest.raises(ValueError, match="future local date"):
        validation.validate_job_result(job, path, load_country_data("Germany"))


def test_checkpoint_store_requires_forecasts_and_tracks_completed_keys(tmp_path) -> None:
    store = validation.TestValidationStore(tmp_path)
    job = {
        "country": "Germany",
        "target_date": "2025-01-01",
        "specification_id": validation.SPECIFICATION_ID,
        "status": "completed",
    }

    with pytest.raises(ValueError, match="forecast rows"):
        store.persist_result(job, pd.DataFrame())

    forecast = pd.DataFrame(
        [
            {
                "country": "Germany",
                "model_family": validation.MODEL_FAMILY,
                "target_date": "2025-01-01",
                "specification_id": validation.SPECIFICATION_ID,
                "timestamp_utc": "2025-01-01T00:00:00+00:00",
            }
        ]
    )
    store.persist_result(job, forecast)

    assert store.completed_keys() == {
        ("Germany", "2025-01-01", validation.SPECIFICATION_ID)
    }


def test_2025_table_merge_preserves_existing_rows_and_is_deterministic(tmp_path) -> None:
    all_models_path = tmp_path / "model_test_2025_all_models.csv"
    overview_path = tmp_path / "model_test_2025_overview.csv"
    existing = pd.read_csv(validation.ALL_MODELS_PATH)
    existing.to_csv(all_models_path, index=False)
    existing.to_csv(overview_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "country": country,
                "model_family": validation.MODEL_FAMILY,
                "specification_id": validation.SPECIFICATION_ID,
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 3.0,
                "evaluated_observations": 8760,
                "coverage": 1.0,
            }
            for country in ("Germany", "Austria")
        ]
    )
    summary_path = tmp_path / validation.SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)

    validation.update_test_tables(summary_path, all_models_path, overview_path)
    first_all = all_models_path.read_bytes()
    first_overview = overview_path.read_bytes()
    updated = pd.read_csv(all_models_path)
    validation.update_test_tables(summary_path, all_models_path, overview_path)

    assert len(updated) == len(existing) + 2
    assert set(updated["model_family"]) >= set(existing["model_family"])
    assert (updated["model_family"] == validation.MODEL_FAMILY).sum() == 2
    assert all_models_path.read_bytes() == first_all
    assert overview_path.read_bytes() == first_overview


def test_run_requires_positive_workers() -> None:
    with pytest.raises(ValueError, match="positive"):
        validation.run_test(workers=0)
