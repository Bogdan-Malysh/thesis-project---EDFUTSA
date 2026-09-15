from __future__ import annotations

from regression_arima_errors_intervention_validation_2024 import (
    INTERVENTION_MINI_DATES,
    INTERVENTION_MODEL_FAMILY,
    INTERVENTION_SPECIFICATION_IDS,
    _full_target_dates,
    build_intervention_manifest,
)


def test_intervention_manifest_has_only_six_unique_mini_jobs() -> None:
    manifest = build_intervention_manifest(INTERVENTION_MINI_DATES)

    assert INTERVENTION_MODEL_FAMILY == "regression_arima_errors_intervention"
    assert INTERVENTION_SPECIFICATION_IDS == ("reg_arima_h_d_m_hol_crisis",)
    assert len(manifest) == 6
    assert manifest["specification_id"].tolist() == [
        "reg_arima_h_d_m_hol_crisis"
    ] * 6
    assert not manifest.duplicated(
        ["country", "target_date", "specification_id"]
    ).any()


def test_full_target_dates_are_bounded_to_2024() -> None:
    dates = _full_target_dates("data/processed")

    assert all(date_value.year == 2024 for values in dates.values() for date_value in values)
    assert {country: len(values) for country, values in dates.items()} == {
        "Germany": 366,
        "Austria": 366,
    }
