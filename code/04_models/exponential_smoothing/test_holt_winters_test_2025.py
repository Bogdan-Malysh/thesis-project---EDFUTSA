from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import holt_winters_test_2025 as validation
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    country_forecast_origin,
    extract_target_day,
    information_set,
    load_country_data,
)


def test_authoritative_2024_selection_is_hw_mul_s168_damped_for_both_countries() -> None:
    selected = validation.verify_authoritative_selection()

    assert selected.set_index("country")["specification_id"].to_dict() == {
        "Germany": "hw_mul_s168_damped",
        "Austria": "hw_mul_s168_damped",
    }
    assert selected["completed_jobs"].eq(366).all()
    assert selected["failed_jobs"].eq(0).all()
    assert selected["coverage"].eq(1.0).all()


def test_frozen_shortlist_rows_preserve_authoritative_model_settings() -> None:
    shortlists = validation.load_test_shortlists()

    assert set(shortlists["country"]) == set(COUNTRY_CONFIG)
    assert shortlists["specification_id"].eq("hw_mul_s168_damped").all()
    assert shortlists["seasonal_form"].eq("mul").all()
    assert shortlists["seasonal_periods"].eq(168).all()
    assert shortlists["damped_trend"].astype(str).str.lower().eq("true").all()
    assert shortlists["optimizer_attempts_json"].astype(str).str.len().gt(2).all()


def test_2025_manifest_has_730_unique_jobs_and_country_origins() -> None:
    shortlists = validation.load_test_shortlists()
    dates = validation.test_dates(load_country_data("Germany"))
    jobs = validation.build_test_jobs(shortlists, dates)

    assert len(dates) == 365
    assert len(jobs) == 730
    assert len({job.job_key for job in jobs}) == 730
    assert {job.specification_id for job in jobs} == {"hw_mul_s168_damped"}
    for job in jobs:
        origin = country_forecast_origin(job.target_date, job.country)
        expected_hour = 18 if job.country == "Germany" else 8
        assert origin.tz_convert(COUNTRY_CONFIG[job.country]["timezone"]).hour == expected_hour


def test_training_counts_follow_the_expanding_origin_bounded_information_set() -> None:
    frame = load_country_data("Germany")
    shortlists = validation.load_test_shortlists()
    jobs = validation.build_test_jobs(shortlists, [date(2025, 1, 2), date(2025, 7, 1)])
    germany_jobs = [job for job in jobs if job.country == "Germany"]
    early, late = germany_jobs[0], germany_jobs[1]

    assert validation.expected_training_observations(early, frame) == len(
        information_set(frame, country_forecast_origin(early.target_date, "Germany"))
    )
    assert validation.expected_training_observations(late, frame) == len(
        information_set(frame, country_forecast_origin(late.target_date, "Germany"))
    )
    assert validation.expected_training_observations(late, frame) > validation.expected_training_observations(
        early, frame
    )


def _target_only_forecast(country: str, target_date: date) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = load_country_data(country)
    target = extract_target_day(frame, target_date)
    origin = country_forecast_origin(target_date, country)
    path = target[
        ["timestamp_utc", "interval_end_utc", "timestamp_local", "local_date", "actual_load_mwh"]
    ].copy()
    path["country"] = country
    path["model_family"] = "Holt-Winters"
    path["specification_id"] = validation.SPECIFICATION_ID
    path["target_date"] = target_date.isoformat()
    path["forecast_origin_utc"] = origin.isoformat()
    path["information_cutoff_utc"] = origin.isoformat()
    path["forecast_mwh"] = path["actual_load_mwh"].astype(float)
    path["is_target_day"] = True
    path["evaluated"] = True
    job_row = {
        "country": country,
        "target_date": target_date.isoformat(),
        "specification_id": validation.SPECIFICATION_ID,
        "training_observations": len(information_set(frame, origin)),
        "status": "completed",
    }
    return path, job_row


@pytest.mark.parametrize(
    ("target_date", "expected_intervals"),
    [(date(2025, 3, 30), 23), (date(2025, 10, 26), 25)],
)
def test_forecast_validation_preserves_dst_and_finite_target_predictions(
    target_date: date,
    expected_intervals: int,
) -> None:
    path, job_row = _target_only_forecast("Germany", target_date)

    validation.validate_job_result(job_row, path, load_country_data("Germany"))

    assert len(path) == expected_intervals
    assert path["timestamp_utc"].is_unique
    assert np.isfinite(path["forecast_mwh"].to_numpy(dtype=float)).all()


def test_forecast_validation_rejects_training_information_after_origin() -> None:
    path, job_row = _target_only_forecast("Austria", date(2025, 3, 30))
    job_row["training_observations"] = int(job_row["training_observations"]) + 1

    with pytest.raises(ValueError, match="training observations"):
        validation.validate_job_result(job_row, path, load_country_data("Austria"))


def test_summary_delegates_to_existing_holt_winters_summary_builder(monkeypatch) -> None:
    called: list[tuple[object, object, object, object]] = []

    def fake_build_summary(manifest, jobs, forecasts, expected_observations):
        called.append((manifest, jobs, forecasts, expected_observations))
        return pd.DataFrame([{"country": "Germany", "mae": 1.0}])

    monkeypatch.setattr(validation, "build_summary", fake_build_summary)
    result = validation.build_test_summary([], pd.DataFrame(), pd.DataFrame())

    assert called
    assert result.iloc[0]["mae"] == 1.0


def test_2025_table_merge_preserves_weekly_rows_and_is_deterministic(tmp_path) -> None:
    columns = [
        "country",
        "model_family",
        "specification_id",
        "mae",
        "rmse",
        "mape",
        "n_observations",
        "coverage",
    ]
    weekly = pd.DataFrame(
        [
            {
                "country": country,
                "model_family": "baselines",
                "specification_id": "Weekly seasonal naive",
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 3.0,
                "n_observations": 8760,
                "coverage": 1.0,
            }
            for country in ("Germany", "Austria")
        ],
        columns=columns,
    )
    summary = pd.DataFrame(
        [
            {
                "country": country,
                "specification_id": validation.SPECIFICATION_ID,
                "mae": 10.0,
                "rmse": 11.0,
                "mape": 12.0,
                "evaluated_observations": 8760,
                "coverage": 1.0,
            }
            for country in ("Germany", "Austria")
        ]
    )
    summary_path = tmp_path / "holt_winters_test_2025_summary.csv"
    all_models_path = tmp_path / "model_test_2025_all_models.csv"
    overview_path = tmp_path / "model_test_2025_overview.csv"
    summary.to_csv(summary_path, index=False)
    weekly.to_csv(all_models_path, index=False)
    weekly.to_csv(overview_path, index=False)

    validation.update_test_tables(summary_path, all_models_path, overview_path)
    first_all = all_models_path.read_bytes()
    first_overview = overview_path.read_bytes()
    updated = pd.read_csv(all_models_path)
    validation.update_test_tables(summary_path, all_models_path, overview_path)

    assert len(updated) == 4
    assert set(updated["model_family"]) == {"baselines", "holt_winters"}
    assert set(updated["specification_id"]) == {
        "Weekly seasonal naive",
        validation.SPECIFICATION_ID,
    }
    assert list(updated.columns) == columns
    assert all_models_path.read_bytes() == first_all
    assert overview_path.read_bytes() == first_overview
