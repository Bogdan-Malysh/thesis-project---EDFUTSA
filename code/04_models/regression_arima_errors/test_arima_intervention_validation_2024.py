from __future__ import annotations

from datetime import date

from arima_intervention_validation_2024 import (
    MINI_DATES,
    MODEL_FAMILY,
    SPECIFICATION_IDS,
    _full_target_dates,
    build_manifest,
)


def test_crisis_only_runner_contract_and_six_mini_jobs() -> None:
    manifest = build_manifest(MINI_DATES)

    assert MODEL_FAMILY == "arima_intervention"
    assert SPECIFICATION_IDS == ("arima_p2_d1_q2_crisis",)
    assert len(manifest) == 6
    assert not manifest.duplicated(["country", "target_date", "specification_id"]).any()
    assert set(manifest["specification_id"]) == {"arima_p2_d1_q2_crisis"}
    assert set(zip(manifest["p"], manifest["d"], manifest["q"], manifest["trend"])) == {(2, 1, 2, "n")}


def test_crisis_only_full_target_dates_are_local_2024() -> None:
    dates = _full_target_dates("data/processed")

    assert {country: len(values) for country, values in dates.items()} == {
        "Germany": 366,
        "Austria": 366,
    }
    assert all(value.year == 2024 for values in dates.values() for value in values)
