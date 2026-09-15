from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import arima_test_2025 as validation
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    country_forecast_origin,
    extract_target_day,
    information_set,
    load_country_data,
)


def test_authoritative_selection_and_frozen_arima_configuration() -> None:
    selected = validation.verify_authoritative_selection()

    assert selected.set_index("country")["selected_specification_id"].to_dict() == {
        "Germany": "arima_p2_d1_q2",
        "Austria": "arima_p2_d1_q2",
    }
    assert selected["completed_jobs"].eq(366).all()
    assert selected["failed_jobs"].eq(0).all()
    assert selected["coverage"].eq(1.0).all()
    assert validation.SPECIFICATION_ID == "arima_p2_d1_q2"
    assert validation.ORDER == (2, 1, 2)
    assert validation.TREND == "n"


def test_frozen_shortlist_rows_match_arima_2_1_2() -> None:
    shortlist = validation.load_test_shortlist()

    assert set(shortlist["country"]) == set(COUNTRY_CONFIG)
    assert shortlist["specification_id"].eq("arima_p2_d1_q2").all()
    assert shortlist["p"].eq(2).all()
    assert shortlist["d"].eq(1).all()
    assert shortlist["q"].eq(2).all()
    assert shortlist["trend"].eq("n").all()


def test_2025_manifest_has_730_unique_jobs_and_country_origins() -> None:
    frames = {country: load_country_data(country) for country in COUNTRY_CONFIG}
    dates = validation.test_dates(frames["Germany"])
    manifest = validation.build_test_manifest(frames, dates)

    assert len(dates) == 365
    assert len(manifest) == 730
    assert manifest[["country", "target_date", "specification_id"]].drop_duplicates().shape[0] == 730
    assert set(manifest["specification_id"]) == {"arima_p2_d1_q2"}
    for row in manifest.to_dict("records"):
        origin = country_forecast_origin(row["target_date"], row["country"])
        expected_hour = 18 if row["country"] == "Germany" else 8
        assert origin.tz_convert(COUNTRY_CONFIG[row["country"]]["timezone"]).hour == expected_hour


def test_training_counts_follow_expanding_origin_bounded_information_set() -> None:
    frame = load_country_data("Germany")
    manifest = validation.build_test_manifest(
        {"Germany": frame, "Austria": load_country_data("Austria")},
        [date(2025, 1, 2), date(2025, 7, 1)],
    )
    germany = manifest.loc[manifest["country"].eq("Germany")]
    early = germany.iloc[0]
    late = germany.iloc[1]

    assert validation.expected_training_observations(early, frame) == len(
        information_set(frame, country_forecast_origin(early["target_date"], "Germany"))
    )
    assert validation.expected_training_observations(late, frame) == len(
        information_set(frame, country_forecast_origin(late["target_date"], "Germany"))
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
    path["model_family"] = "ARIMA"
    path["specification_id"] = validation.SPECIFICATION_ID
    path["target_date"] = target_date.isoformat()
    path["forecast_origin_local"] = origin.tz_convert(COUNTRY_CONFIG[country]["timezone"]).isoformat()
    path["forecast_origin_utc"] = origin.isoformat()
    path["information_cutoff_utc"] = origin.isoformat()
    path["forecast_mwh"] = path["actual_load_mwh"].astype(float)
    path["bridge_used"] = False
    path["source_kind"] = "arima_forecast"
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
def test_forecast_validation_preserves_dst_and_finite_predictions(
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


def test_summary_delegates_to_existing_arima_summary_builder(monkeypatch) -> None:
    called: list[tuple[object, object, object]] = []

    def fake_summary(jobs, manifest, forecasts):
        called.append((jobs, manifest, forecasts))
        return pd.DataFrame([{"country": "Germany", "mae": 1.0}])

    monkeypatch.setattr(validation, "_summary_frame", fake_summary)
    result = validation.build_test_summary(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert called
    assert result.iloc[0]["mae"] == 1.0


def test_2025_table_merge_preserves_existing_rows_and_is_deterministic(tmp_path) -> None:
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
    existing = pd.DataFrame(
        [
            {
                "country": country,
                "model_family": family,
                "specification_id": specification,
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 3.0,
                "n_observations": 8760,
                "coverage": 1.0,
            }
            for country in ("Germany", "Austria")
            for family, specification in (
                ("baselines", "Weekly seasonal naive"),
                ("holt_winters", "hw_mul_s168_damped"),
            )
        ],
        columns=columns,
    )
    summary = pd.DataFrame(
        [
            {
                "country": country,
                "model_family": "ARIMA",
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
    summary_path = tmp_path / "arima_test_2025_summary.csv"
    all_models_path = tmp_path / "model_test_2025_all_models.csv"
    overview_path = tmp_path / "model_test_2025_overview.csv"
    summary.to_csv(summary_path, index=False)
    existing.to_csv(all_models_path, index=False)
    existing.to_csv(overview_path, index=False)

    validation.update_test_tables(summary_path, all_models_path, overview_path)
    first_all = all_models_path.read_bytes()
    first_overview = overview_path.read_bytes()
    updated = pd.read_csv(all_models_path)
    validation.update_test_tables(summary_path, all_models_path, overview_path)

    assert len(updated) == 6
    assert set(updated["model_family"]) == {"baselines", "holt_winters", "arima"}
    assert set(updated["specification_id"]) == {
        "Weekly seasonal naive",
        "hw_mul_s168_damped",
        validation.SPECIFICATION_ID,
    }
    assert list(updated.columns) == columns
    assert all_models_path.read_bytes() == first_all
    assert overview_path.read_bytes() == first_overview


def test_checkpoint_store_requires_forecasts_for_completed_jobs(tmp_path) -> None:
    store = validation.TestValidationStore(tmp_path)
    job = {
        "country": "Germany",
        "target_date": "2025-01-01",
        "specification_id": validation.SPECIFICATION_ID,
        "status": "completed",
    }

    with pytest.raises(ValueError, match="forecast rows"):
        store.persist_result(job, pd.DataFrame())


def test_run_requires_exactly_four_workers() -> None:
    with pytest.raises(ValueError, match="exactly 4 workers"):
        validation.run_test(workers=2)
