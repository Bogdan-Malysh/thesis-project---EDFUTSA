from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import baseline_test_2025 as validation
from common.forecasting_framework import (
    build_information_context,
    forecast_path,
    load_country_data,
)


def _weekly_path(country: str, target_date: date) -> tuple[pd.DataFrame, object]:
    frame = load_country_data(country)
    context = build_information_context(frame, country, target_date)
    path = forecast_path(
        frame,
        country,
        target_date,
        validation.SPECIFICATION_ID,
        context=context,
    )
    return path, context


def test_2025_runner_uses_the_authoritative_weekly_identity() -> None:
    assert validation.MODEL_FAMILY == "baselines"
    assert validation.SPECIFICATION_ID == "Weekly seasonal naive"

    dates = validation.validation_dates(load_country_data("Germany"))

    assert len(dates) == 365
    assert dates[0] == date(2025, 1, 1)
    assert dates[-1] == date(2025, 12, 31)


@pytest.mark.parametrize("country", ["Germany", "Austria"])
@pytest.mark.parametrize(
    ("target_date", "expected_intervals"),
    [(date(2025, 3, 30), 23), (date(2025, 10, 26), 25)],
)
def test_2025_paths_preserve_dst_and_origin_rules(
    country: str,
    target_date: date,
    expected_intervals: int,
) -> None:
    path, context = _weekly_path(country, target_date)

    validation.validate_forecast_path(path, country, target_date, context)
    target = path.loc[path["is_target_day"]]

    assert len(target) == expected_intervals
    assert target["timestamp_utc"].is_unique
    assert np.isfinite(path["forecast_mwh"].to_numpy(dtype=float)).all()
    assert path["model_family"].eq("Weekly seasonal naive").all()
    assert path["forecast_origin_utc"].eq(path["information_cutoff_utc"]).all()
    expected_hour = 18 if country == "Germany" else 8
    assert context.origin_local.startswith(
        f"{target_date.replace(day=target_date.day - 1).isoformat()}T{expected_hour:02d}:00:"
    )


def test_2025_path_validation_rejects_future_weekly_source() -> None:
    target_date = date(2025, 2, 15)
    path, context = _weekly_path("Germany", target_date)
    invalid = path.copy()
    invalid.loc[invalid.index[-1], "source_timestamp_utc"] = (
        invalid["timestamp_utc"].max() + pd.Timedelta(hours=168)
    )

    with pytest.raises(ValueError, match="weekly source timestamp"):
        validation.validate_forecast_path(invalid, "Germany", target_date, context)


def test_country_summary_delegates_metrics_to_existing_evaluator(monkeypatch) -> None:
    path, _ = _weekly_path("Germany", date(2025, 2, 15))
    calls: list[tuple[pd.DataFrame, int]] = []

    def fake_evaluate_target_day(
        forecast: pd.DataFrame,
        expected_observations: int | None = None,
    ) -> dict[str, float | int]:
        calls.append((forecast.copy(), int(expected_observations or 0)))
        return {
            "mae": 1.0,
            "rmse": 2.0,
            "mape": 3.0,
            "evaluated_observations": 1,
            "coverage": 1.0,
        }

    monkeypatch.setattr(validation, "evaluate_target_day", fake_evaluate_target_day)

    summary = validation.summarize_forecasts(
        "Germany", [path], expected_observations=23
    )

    assert summary["mae"] == 1.0
    assert summary["rmse"] == 2.0
    assert summary["mape"] == 3.0
    assert calls[-1][0]["is_target_day"].any()
    assert calls[-1][1] == 23


def test_2025_tables_merge_new_rows_without_dropping_completed_families(tmp_path) -> None:
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
    prior = pd.DataFrame(
        [
            {
                "country": "Germany",
                "model_family": "arima",
                "specification_id": "arima_selected",
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 3.0,
                "n_observations": 8760,
                "coverage": 1.0,
            }
        ],
        columns=columns,
    )
    summary = pd.DataFrame(
        [
            {
                "country": country,
                "model_family": "baselines",
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
    summary_path = tmp_path / "baseline_validation_2025.csv"
    all_models_path = tmp_path / "model_test_2025_all_models.csv"
    overview_path = tmp_path / "model_test_2025_overview.csv"
    summary.to_csv(summary_path, index=False)
    prior.to_csv(all_models_path, index=False)
    prior.to_csv(overview_path, index=False)

    validation.update_test_tables(summary_path, all_models_path, overview_path)
    first_all = all_models_path.read_bytes()
    first_overview = overview_path.read_bytes()
    updated = pd.read_csv(all_models_path)

    validation.update_test_tables(summary_path, all_models_path, overview_path)

    assert len(updated) == 3
    assert set(updated["model_family"]) == {"arima", "baselines"}
    assert set(updated["specification_id"]) == {
        "arima_selected",
        validation.SPECIFICATION_ID,
    }
    assert list(updated.columns) == columns
    assert all_models_path.read_bytes() == first_all
    assert overview_path.read_bytes() == first_overview
