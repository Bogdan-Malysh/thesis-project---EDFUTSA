from __future__ import annotations

from datetime import date

import pandas as pd

import sarima_intervention_validation_2024 as validation


def test_intervention_manifest_has_six_country_specific_jobs() -> None:
    manifest = validation.build_manifest(validation.MINI_DATES)

    assert len(manifest) == 6
    assert not manifest.duplicated(list(validation.JOB_KEY_COLUMNS)).any()
    assert manifest.groupby("country").size().to_dict() == {
        "Germany": 3,
        "Austria": 3,
    }
    assert set(manifest["specification_id"]) == {
        "sarima_p2_d0_q0_P0_D1_Q0_s24_crisis",
        "sarima_p1_d0_q1_P1_D1_Q0_s24_crisis",
    }


def test_intervention_forecast_path_preserves_dst_lengths() -> None:
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
        path = validation.build_intervention_forecast_path(
            frame,
            "Germany",
            target_date,
            "sarima_p2_d0_q0_P0_D1_Q0_s24_crisis",
            Fitted(),
        )
        assert path["model_family"].eq("sarima_intervention").all()
        assert path["source_kind"].eq("sarima_intervention_forecast").all()
        assert int(path["is_target_day"].sum()) == expected_length
        assert path["timestamp_utc"].is_unique
        assert path["forecast_mwh"].notna().all()


def test_intervention_output_names_are_separate_from_sarimax() -> None:
    assert validation.JOB_FILENAME == "sarima_intervention_validation_2024_jobs.csv"
    assert validation.FORECAST_FILENAME == "sarima_intervention_validation_2024_forecasts.csv"
    assert validation.SUMMARY_FILENAME == "sarima_intervention_validation_2024_summary.csv"
