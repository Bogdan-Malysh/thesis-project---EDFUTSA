from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import sarima_test_2025 as validation
from common.forecasting_framework import (
    COUNTRY_CONFIG,
    country_forecast_origin,
    extract_target_day,
    information_set,
    load_country_data,
)


EXPECTED_SPECIFICATIONS = {
    "Germany": "sarima_p2_d0_q0_P0_D1_Q0_s24",
    "Austria": "sarima_p1_d0_q1_P1_D1_Q0_s24",
}


def test_authoritative_selection_registry_and_shortlist_are_frozen() -> None:
    selected = validation.verify_authoritative_selection()

    assert selected.set_index("country")["selected_specification_id"].to_dict() == EXPECTED_SPECIFICATIONS
    assert validation.load_test_shortlists().set_index("country")["specification_id"].to_dict() == EXPECTED_SPECIFICATIONS
    assert validation.ORDER_BY_COUNTRY == {
        "Germany": (2, 0, 0, 0, 1, 0, 24),
        "Austria": (1, 0, 1, 1, 1, 0, 24),
    }
    assert validation.TREND == "n"


def test_2025_manifest_has_730_unique_country_specific_jobs() -> None:
    frames = {country: load_country_data(country) for country in COUNTRY_CONFIG}
    dates = validation.test_dates(frames["Germany"])
    manifest = validation.build_test_manifest(frames, dates)

    assert len(dates) == 365
    assert len(manifest) == 730
    assert not manifest.duplicated(validation.JOB_KEY_COLUMNS).any()
    assert manifest.groupby("country").size().to_dict() == {"Germany": 365, "Austria": 365}
    observed = {
        country: values.tolist()
        for country, values in manifest.groupby("country")["specification_id"].unique().items()
    }
    assert observed == {
        country: [specification] for country, specification in EXPECTED_SPECIFICATIONS.items()
    }


def test_manifest_uses_exact_country_origins_and_expanding_history() -> None:
    frames = {country: load_country_data(country) for country in COUNTRY_CONFIG}
    manifest = validation.build_test_manifest(
        frames, [date(2025, 1, 2), date(2025, 7, 1)]
    )

    for row in manifest.to_dict("records"):
        origin = country_forecast_origin(row["target_date"], row["country"])
        expected_hour = 18 if row["country"] == "Germany" else 8
        assert origin.tz_convert(COUNTRY_CONFIG[row["country"]]["timezone"]).hour == expected_hour
        assert row["expected_observations"] in (23, 24, 25)

    germany = manifest.loc[manifest["country"].eq("Germany")].sort_values("target_date")
    early, late = germany.iloc[0], germany.iloc[1]
    frame = frames["Germany"]
    assert validation.expected_training_observations(early, frame) == len(
        information_set(frame, country_forecast_origin(early["target_date"], "Germany"))
    )
    assert validation.expected_training_observations(late, frame) == len(
        information_set(frame, country_forecast_origin(late["target_date"], "Germany"))
    )
    assert int(late["expected_observations"]) == 24


def _target_path_and_job(country: str, target_date: date) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = load_country_data(country)
    prepared = frame
    target_date_string = target_date.isoformat()
    origin = country_forecast_origin(target_date, country)
    path = prepared.loc[
        prepared["timestamp_utc"].ge(origin)
        & prepared["local_date"].le(target_date_string)
    ][
        [
            "timestamp_utc",
            "interval_end_utc",
            "timestamp_local",
            "local_date",
            "actual_load_mwh",
        ]
    ].copy()
    order = validation.ORDER_BY_COUNTRY[country]
    path["country"] = country
    path["model_family"] = "SARIMA"
    path["target_date"] = target_date_string
    path["specification_id"] = EXPECTED_SPECIFICATIONS[country]
    path["p"], path["d"], path["q"], path["P"], path["D"], path["Q"], path["seasonal_period"] = order
    path["trend"] = "n"
    path["forecast_origin_local"] = origin.tz_convert(COUNTRY_CONFIG[country]["timezone"]).isoformat()
    path["forecast_origin_utc"] = origin.isoformat()
    path["information_cutoff_utc"] = origin.isoformat()
    path["forecast_mwh"] = path["actual_load_mwh"].astype(float)
    path["bridge_used"] = ~path["local_date"].eq(target_date_string)
    path["source_kind"] = "sarima_forecast"
    path["is_target_day"] = path["local_date"].eq(target_date_string)
    path["evaluated"] = path["is_target_day"]
    job = {
        "country": country,
        "target_date": target_date_string,
        "specification_id": EXPECTED_SPECIFICATIONS[country],
        "training_observations": len(information_set(frame, origin)),
        "status": "completed",
    }
    return path, job


@pytest.mark.parametrize(
    ("country", "target_date", "expected_intervals"),
    [
        ("Germany", date(2025, 3, 30), 23),
        ("Austria", date(2025, 10, 26), 25),
    ],
)
def test_forecast_validation_preserves_dst_finite_values_and_target_samples(
    country: str, target_date: date, expected_intervals: int
) -> None:
    path, job = _target_path_and_job(country, target_date)

    validation.validate_job_result(job, path, load_country_data(country))

    target = path.loc[path["is_target_day"]]
    assert len(target) == expected_intervals
    assert target["timestamp_utc"].is_unique
    assert np.isfinite(path["forecast_mwh"].to_numpy(dtype=float)).all()
    assert target["evaluated"].all()


def test_forecast_validation_rejects_future_training_count() -> None:
    path, job = _target_path_and_job("Germany", date(2025, 3, 30))
    job["training_observations"] = int(job["training_observations"]) + 1

    with pytest.raises(ValueError, match="training observations"):
        validation.validate_job_result(job, path, load_country_data("Germany"))


def test_2025_execute_job_delegates_to_2024_execute_job(monkeypatch) -> None:
    frame = load_country_data("Germany")
    job = {
        "country": "Germany",
        "target_date": "2025-01-02",
        "specification_id": EXPECTED_SPECIFICATIONS["Germany"],
        "p": 2,
        "d": 0,
        "q": 0,
        "P": 0,
        "D": 1,
        "Q": 0,
        "seasonal_period": 24,
        "trend": "n",
    }
    called: list[tuple[object, object]] = []

    def fake_execute(received_job, received_frame):
        called.append((received_job, received_frame))
        return {"job": {**received_job, "status": "failed"}, "forecasts": [], "diagnostics": {}}

    monkeypatch.setattr(validation.validation_2024, "execute_job", fake_execute)

    result = validation.execute_job(job, frame)

    assert result["job"]["status"] == "failed"
    assert called == [(job, frame)]


def test_checkpoint_store_requires_diagnostics_and_forecasts_for_completion(tmp_path) -> None:
    store = validation.TestValidationStore(tmp_path)
    result = {
        "job": {
            "country": "Germany",
            "target_date": "2025-01-01",
            "specification_id": EXPECTED_SPECIFICATIONS["Germany"],
            "status": "completed",
        },
        "forecasts": [],
        "diagnostics": {},
    }

    with pytest.raises(ValueError, match="forecast rows"):
        store.persist_result(result)


def test_2025_table_merge_preserves_existing_families_and_is_deterministic(tmp_path) -> None:
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
                ("arima", "arima_p2_d1_q2"),
            )
        ],
        columns=columns,
    )
    summary = pd.DataFrame(
        [
            {
                "country": country,
                "specification_id": EXPECTED_SPECIFICATIONS[country],
                "mae": 10.0,
                "rmse": 11.0,
                "mape": 12.0,
                "evaluated_observations": 8760,
                "coverage": 1.0,
            }
            for country in ("Germany", "Austria")
        ]
    )
    summary_path = tmp_path / "sarima_test_2025_summary.csv"
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

    assert len(updated) == 8
    assert set(updated["model_family"]) == {"baselines", "holt_winters", "arima", "sarima"}
    assert list(updated.columns) == columns
    assert all_models_path.read_bytes() == first_all
    assert overview_path.read_bytes() == first_overview


def test_run_requires_an_approved_worker_count() -> None:
    with pytest.raises(ValueError, match="approved worker counts"):
        validation.run_test(workers=3)
