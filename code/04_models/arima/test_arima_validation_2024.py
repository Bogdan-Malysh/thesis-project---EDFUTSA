from datetime import date

import numpy as np
import pandas as pd

import arima_validation_2024 as validation
from arima_validation_2024 import (
    ARIMA_JOB_KEY,
    FORECAST_KEY,
    build_arima_forecast_path,
    build_manifest,
    load_completed_job_keys,
    sort_job_results,
    upsert_frame,
)
from common.forecasting_framework import extract_target_day, load_country_data


def _shortlists():
    return pd.DataFrame(
        [
            {
                "country": "Germany",
                "specification_id": "arima_p1_d1_q0",
                "specification_order": 1,
                "p": 1,
                "d": 1,
                "q": 0,
                "trend": "n",
            },
            {
                "country": "Austria",
                "specification_id": "arima_p0_d1_q1",
                "specification_order": 2,
                "p": 0,
                "d": 1,
                "q": 1,
                "trend": "n",
            },
        ]
    )


def test_manifest_is_unique_and_sorted_by_country_date_and_specification():
    manifest = build_manifest(
        _shortlists(),
        {
            "Germany": [date(2024, 3, 31), date(2024, 2, 15)],
            "Austria": [date(2024, 10, 27)],
        },
    )

    assert len(manifest) == 3
    assert list(manifest.columns[:3]) == ["country", "target_date", "specification_id"]
    assert not manifest.duplicated(list(ARIMA_JOB_KEY)).any()
    assert manifest.iloc[0]["country"] == "Germany"
    assert manifest.iloc[0]["target_date"] == "2024-02-15"


def test_forecast_path_does_not_change_when_future_actuals_are_altered():
    frame = load_country_data("Germany")
    target_date = date(2024, 2, 15)

    class FakeResult:
        def get_forecast(self, steps):
            return type("Forecast", (), {"predicted_mean": np.arange(steps, dtype=float)})()

    original = build_arima_forecast_path(
        frame, "Germany", target_date, "arima_p1_d1_q0", FakeResult()
    )
    altered = frame.copy()
    cutoff = pd.Timestamp(original["information_cutoff_utc"].iloc[0])
    altered.loc[altered["interval_end_utc"] > cutoff, "actual_load_mwh"] = -999999.0
    changed = build_arima_forecast_path(
        altered, "Germany", target_date, "arima_p1_d1_q0", FakeResult()
    )

    assert original["forecast_mwh"].tolist() == changed["forecast_mwh"].tolist()


def test_forecast_path_preserves_ordinary_and_dst_target_lengths():
    frame = load_country_data("Germany")
    class FakeResult:
        def get_forecast(self, steps):
            return type("Forecast", (), {"predicted_mean": np.ones(steps)})()

    for target_date, expected in (
        (date(2024, 2, 15), 24),
        (date(2024, 3, 31), 23),
        (date(2024, 10, 27), 25),
    ):
        path = build_arima_forecast_path(
            frame, "Germany", target_date, "arima_p1_d1_q0", FakeResult()
        )
        assert len(extract_target_day(frame, target_date)) == expected
        assert int(path["is_target_day"].sum()) == expected
        assert path.loc[path["is_target_day"], "forecast_mwh"].notna().all()


def test_checkpoint_keys_and_upserts_are_duplicate_free():
    jobs = pd.DataFrame(
        [{"country": "Germany", "target_date": "2024-02-15", "specification_id": "arima_p1_d1_q0", "status": "completed"}]
    )
    assert load_completed_job_keys(jobs) == {("Germany", "2024-02-15", "arima_p1_d1_q0")}

    existing = pd.DataFrame(
        [{"country": "Germany", "target_date": "2024-02-15", "specification_id": "arima_p1_d1_q0", "timestamp_utc": "2024-02-15T00:00:00Z", "forecast_mwh": 1.0}]
    )
    replacement = existing.assign(forecast_mwh=2.0)
    updated = upsert_frame(existing, replacement, list(FORECAST_KEY))

    assert len(updated) == 1
    assert updated.iloc[0]["forecast_mwh"] == 2.0


def test_parent_sorting_is_deterministic():
    rows = [
        {"country": "Austria", "target_date": "2024-10-27", "specification_id": "b"},
        {"country": "Germany", "target_date": "2024-02-15", "specification_id": "a"},
    ]

    ordered = sort_job_results(rows)

    assert [(row["country"], row["target_date"]) for row in ordered] == [
        ("Germany", "2024-02-15"),
        ("Austria", "2024-10-27"),
    ]


def test_run_validation_converts_manifest_rows_to_worker_dicts(monkeypatch, tmp_path):
    shortlist = _shortlists().loc[[0]].copy()

    monkeypatch.setattr(validation, "load_country_data", lambda country, directory: pd.DataFrame())

    def fake_execute(job, frame):
        return {
            "job": {
                **job,
                "status": "failed",
                "error_message": "test failure",
                "fit_status": "failed",
                "convergence_status": "failed",
                "converged": False,
                "parameters_finite": False,
                "standard_errors_finite": False,
                "stationarity_ok": False,
                "invertibility_ok": False,
                "nobs": 0,
                "n_params": 0,
                "log_likelihood": np.nan,
                "aic": np.nan,
                "aicc": np.nan,
                "bic": np.nan,
                "ar_root_minimum": np.nan,
                "ma_root_minimum": np.nan,
                "warning_messages": "[]",
                "mae": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "evaluated_observations": 0,
                "coverage": 0.0,
            },
            "forecasts": [],
        }

    monkeypatch.setattr(validation, "_execute_job", fake_execute)
    result = validation.run_validation(
        shortlist,
        {"Germany": [date(2024, 2, 15)]},
        tmp_path,
        workers=1,
    )

    assert result["total_jobs"] == 1
    assert result["failed_jobs"] == 1
