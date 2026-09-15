from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import enhanced_arima_test_2025 as validation
from common.forecasting_framework import COUNTRY_CONFIG
from enhanced_arima_models import enhanced_specifications


def _freeze_frame() -> pd.DataFrame:
    ordinary = enhanced_specifications()[0]
    weekly = next(
        specification
        for specification in enhanced_specifications()
        if specification.branch == "weekly_differenced"
    )
    rows = []
    for country, specification in (
        ("Germany", ordinary),
        ("Austria", weekly),
    ):
        rows.append(
            {
                "country": country,
                "branch": specification.branch,
                "specification_id": specification.specification_id,
                "specification_order": 1,
                "p": specification.order[0],
                "d": specification.order[1],
                "q": specification.order[2],
                "trend": specification.trend,
                "transform": specification.transform,
                "selection_status": "frozen",
                "adequate": True,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_frame(country: str, missing_source_for: str | None = None) -> pd.DataFrame:
    timezone = COUNTRY_CONFIG[country]["timezone"]
    local = pd.date_range(
        "2024-12-24 00:00:00",
        "2026-01-01 00:00:00",
        inclusive="left",
        freq="h",
        tz=timezone,
    )
    utc = local.tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": 1000.0 + np.arange(len(local), dtype=float),
            "hour": local.hour,
            "day_of_week": local.dayofweek,
        }
    )
    if missing_source_for is not None:
        source_date = (pd.Timestamp(missing_source_for).date() - pd.Timedelta(days=7)).isoformat()
        remove = frame["timestamp_local"].str[:10].eq(source_date) & frame["hour"].eq(1)
        frame = frame.loc[~remove].reset_index(drop=True)
    return frame


def test_verify_freeze_requires_one_frozen_adequate_valid_row_per_country(tmp_path):
    path = tmp_path / "frozen_specifications_2024.csv"
    _freeze_frame().to_csv(path, index=False)

    verified = validation.verify_freeze(path)

    assert list(verified["country"]) == ["Germany", "Austria"]
    assert verified["selection_status"].eq("frozen").all()
    assert verified["adequate"].astype(str).str.lower().eq("true").all()

    duplicate = pd.concat([_freeze_frame(), _freeze_frame().iloc[[0]]], ignore_index=True)
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one"):
        validation.verify_freeze(path)


@pytest.mark.parametrize("column, value", [("selection_status", "candidate"), ("adequate", False)])
def test_verify_freeze_rejects_non_publishable_status(column, value, tmp_path):
    freeze = _freeze_frame()
    freeze.loc[0, column] = value
    path = tmp_path / "frozen_specifications_2024.csv"
    freeze.to_csv(path, index=False)

    with pytest.raises(ValueError, match="frozen|adequate"):
        validation.verify_freeze(path)


def test_2025_manifest_derives_weekly_invalidity_and_has_one_job_per_day():
    freeze = _freeze_frame()
    frames = {
        "Germany": _synthetic_frame("Germany"),
        "Austria": _synthetic_frame("Austria", missing_source_for="2025-05-10"),
    }

    manifest = validation.build_2025_manifest(freeze, frames)

    assert len(manifest) == 2 * 365
    assert not manifest.duplicated(
        ["country", "target_date", "branch", "specification_id"]
    ).any()
    assert manifest.groupby("country")["target_date"].nunique().to_dict() == {
        "Germany": 365,
        "Austria": 365,
    }
    austria_weekly = manifest.loc[
        (manifest["country"] == "Austria")
        & (manifest["branch"] == "weekly_differenced")
    ]
    assert austria_weekly.loc[
        austria_weekly["target_date"] == "2025-05-10", "weekly_invalid"
    ].item()
    assert austria_weekly.loc[
        austria_weekly["target_date"] == "2025-05-10", "weekly_invalid_reason"
    ].item() == "missing_source"
    assert manifest.loc[manifest["country"] == "Germany", "branch"].eq("ordinary").all()
    assert set(manifest["target_date"]) == {
        (pd.Timestamp("2025-01-01") + pd.Timedelta(days=offset)).date().isoformat()
        for offset in range(365)
    }


def test_2025_manifest_refuses_unverified_freeze_rows():
    freeze = _freeze_frame().copy()
    freeze.loc[0, "specification_id"] = "not_a_frozen_specification"

    with pytest.raises(ValueError, match="specification"):
        validation.build_2025_manifest(
            freeze,
            {country: _synthetic_frame(country) for country in COUNTRY_CONFIG},
        )


def test_derived_invalid_weekly_job_cannot_pass_with_forecast_rows():
    manifest = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2025-05-10",
                "branch": "weekly_differenced",
                "specification_id": "weekly_differenced_arima_p1_d0_q1",
                "weekly_invalid": True,
                "weekly_invalid_reason": "missing_source",
            }
        ]
    )
    jobs = pd.DataFrame(
        [
            {
                **manifest.iloc[0].to_dict(),
                "status": "weekly_lag_invalid",
                "error_message": "missing_source",
            }
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2025-05-10",
                "branch": "weekly_differenced",
                "specification_id": "weekly_differenced_arima_p1_d0_q1",
                "timestamp_utc": "2025-05-10T00:00:00+00:00",
            }
        ]
    )

    with pytest.raises(ValueError, match="no forecast"):
        validation._validate_run(jobs, forecasts, manifest, {})


def _fake_result(job: dict[str, object], frame: pd.DataFrame) -> dict[str, object]:
    path = validation.screening._expected_forecast_path_index(
        str(job["country"]), str(job["target_date"])
    )
    prepared = validation._as_prepared(frame)
    source = prepared.set_index("timestamp_utc")
    selected = source.loc[path].reset_index()
    local = pd.DatetimeIndex(path).tz_convert(
        COUNTRY_CONFIG[str(job["country"])]["timezone"]
    )
    target = local.strftime("%Y-%m-%d") == str(job["target_date"])
    origin = pd.Timestamp(job["forecast_origin_utc"])
    source_timestamp = []
    source_actual = []
    source_kind = []
    for timestamp, is_target in zip(path, target):
        if str(job["branch"]) == "weekly_differenced":
            local_timestamp = timestamp.tz_convert(
                COUNTRY_CONFIG[str(job["country"])]["timezone"]
            )
            source_local = local_timestamp - pd.Timedelta(days=7)
            source_utc = source_local.tz_convert("UTC")
            source_timestamp.append(source_utc)
            source_actual.append(float(source.loc[source_utc, "actual_load_mwh"]))
            source_kind.append("observed")
        else:
            source_timestamp.append(pd.NaT)
            source_actual.append(np.nan)
            source_kind.append("arima_forecast")
    forecasts = pd.DataFrame(
        {
            "country": job["country"],
            "model_family": "enhanced_arima",
            "target_date": job["target_date"],
            "branch": job["branch"],
            "specification_id": job["specification_id"],
            "forecast_origin_local": job["forecast_origin_local"],
            "forecast_origin_utc": job["forecast_origin_utc"],
            "information_cutoff_utc": job["information_cutoff_utc"],
            "timestamp_utc": path,
            "interval_end_utc": path + pd.Timedelta(hours=1),
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "local_date": local.strftime("%Y-%m-%d"),
            "actual_load_mwh": selected["actual_load_mwh"],
            "forecast_mwh": selected["actual_load_mwh"],
            "bridge_used": ~target,
            "source_timestamp_utc": source_timestamp,
            "source_actual_mwh": source_actual,
            "source_kind": source_kind,
            "is_target_day": target,
            "evaluated": target,
            "status": "completed",
            "mapping_issue_count": 0,
        }
    )
    job_row = {
        **job,
        "status": "completed",
        "error_message": "",
        "fit_status": "success",
        "convergence_status": "converged",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "log_likelihood": -1.0,
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "residual_n_effective": 100,
        "residual_rejected": False,
        "residual_rejection_reason": "",
        "residual_acf_values": '{"1": 0.1}',
        "residual_acf_flagged_lags": "[]",
        "residual_ljung_box_statistics": '{"24": 1.0, "48": 2.0}',
        "residual_ljung_box_pvalues": '{"24": 0.5, "48": 0.4}',
        "mae": 0.0,
        "rmse": 0.0,
        "mape": 0.0,
        "evaluated_observations": int(target.sum()),
        "coverage": 1.0,
    }
    return {"job": job_row, "forecasts": forecasts.to_dict("records"), "diagnostics": []}


def test_runner_uses_synthetic_frames_and_resumable_atomic_branch_stores(tmp_path, monkeypatch):
    freeze_directory = tmp_path / "freeze_2024"
    freeze_directory.mkdir()
    _freeze_frame().to_csv(
        freeze_directory / "frozen_specifications_2024.csv", index=False
    )
    frames = {country: _synthetic_frame(country) for country in COUNTRY_CONFIG}
    monkeypatch.setattr(validation, "load_country_data", lambda country, _: frames[country])
    def fake_job_result(job, frame):
        return {
            "job": {
                **job,
                "status": "completed",
                "error_message": "",
                "fit_status": "success",
                "convergence_status": "converged",
                "converged": True,
                "parameters_finite": True,
                "standard_errors_finite": True,
                "stationarity_ok": True,
                "invertibility_ok": True,
                "log_likelihood": -1.0,
                "aic": 1.0,
                "aicc": 1.0,
                "bic": 1.0,
            },
            "forecasts": [],
            "diagnostics": [],
        }
    monkeypatch.setattr(
        validation,
        "_execute_job_2025",
        fake_job_result,
    )
    monkeypatch.setattr(validation, "_validate_run", lambda *args: None)
    monkeypatch.setattr(validation.screening, "_forecast_is_complete", lambda *args: True)
    monkeypatch.setattr(
        validation,
        "_load_comparator_forecasts",
        lambda: {
            "original_arima": pd.DataFrame(),
            "original_sarima": pd.DataFrame(),
            "weekly_seasonal_naive": pd.DataFrame(),
            "holt_winters": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(validation, "_load_smard_benchmark", lambda: pd.DataFrame())

    result = validation.run_descriptive_2025_test(tmp_path, processed_directory=tmp_path)

    assert result["total_jobs"] == 730
    assert result["completed_jobs"] == 730
    assert (tmp_path / "test_2025" / "ordinary" / "jobs.csv").exists()
    assert (tmp_path / "test_2025" / "weekly_differenced" / "forecasts.csv").exists()
    assert not list(tmp_path.rglob("*.tmp"))

    second = validation.run_descriptive_2025_test(tmp_path, processed_directory=tmp_path)
    assert second["completed_jobs"] == 730


def _comparison_forecasts(country: str, model: str, timestamps: list[str], error: float):
    return pd.DataFrame(
        {
            "country": country,
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "actual_load_mwh": 100.0,
            "forecast_mwh": 100.0 + error,
            "evaluated": True,
            "is_target_day": True,
            "model_family": model,
        }
    )


def test_normalise_comparison_frame_applies_masks_without_length_mismatch():
    frame = _comparison_forecasts(
        "Germany",
        "enhanced_arima",
        ["2025-01-01 00:00", "2025-01-01 01:00"],
        1.0,
    )
    frame.loc[1, "evaluated"] = False

    result = validation._normalise_comparison_frame(frame, "Germany")

    assert len(result) == 1
    assert result["timestamp_utc"].iloc[0] == pd.Timestamp(
        "2025-01-01 00:00", tz="UTC"
    )


def test_normalise_comparison_frame_uses_local_year_at_utc_boundaries():
    frame = _comparison_forecasts(
        "Germany",
        "enhanced_arima",
        ["2024-12-31 23:00", "2025-12-31 23:00"],
        1.0,
    )
    frame["local_date"] = ["2025-01-01", "2026-01-01"]

    result = validation._normalise_comparison_frame(frame, "Germany")

    assert len(result) == 1
    assert result["timestamp_utc"].iloc[0] == pd.Timestamp(
        "2024-12-31 23:00", tz="UTC"
    )


def test_final_comparison_has_six_descriptive_rows_and_common_timestamp_metrics():
    enhanced = pd.concat(
        [
            _comparison_forecasts(
                country,
                "enhanced_arima",
                ["2025-01-01 00:00", "2025-01-01 01:00"],
                1.0,
            )
            for country in COUNTRY_CONFIG
        ],
        ignore_index=True,
    )
    summary = pd.DataFrame(
        [
            {
                "country": country,
                "branch": "ordinary",
                "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
                "p": 1,
                "d": 1,
                "q": 1,
                "trend": "n",
                "transform": "identity",
                "native_coverage": 1.0,
                "valid_dates": 365,
                "invalid_dates": 0,
                "invalid_reasons": "",
            }
            for country in COUNTRY_CONFIG
        ]
    )
    comparator = {
        name: pd.concat(
            [
                _comparison_forecasts(
                    country,
                    name,
                    ["2025-01-01 01:00", "2025-01-01 02:00"],
                    error,
                )
                for country in COUNTRY_CONFIG
            ],
            ignore_index=True,
        )
        for name, error in {
            "original_arima": 2.0,
            "original_sarima": 3.0,
            "weekly_seasonal_naive": 4.0,
            "holt_winters": 5.0,
        }.items()
    }
    smard = pd.DataFrame(
        {
            "country": ["Germany", "Austria"],
            "n_observations": [2, 2],
            "coverage": [1.0, 1.0],
            "mae": [6.0, 7.0],
            "rmse": [6.0, 7.0],
            "mape": [6.0, 7.0],
            "directly_comparable_to_2024_models": ["yes", "yes"],
        }
    )

    comparison = validation.build_final_comparison(
        summary, enhanced, comparator, smard
    )

    assert len(comparison) == 12
    assert comparison.groupby("country").size().to_dict() == {
        "Germany": 6,
        "Austria": 6,
    }
    assert set(comparison["model"]) == {
        "original ARIMA (2,1,2)",
        "enhanced ARIMA",
        "original SARIMA",
        "weekly seasonal naive",
        "Holt-Winters",
        "SMARD",
    }
    enhanced_row = comparison.loc[
        (comparison["country"] == "Germany")
        & (comparison["model"] == "enhanced ARIMA")
    ].iloc[0]
    assert enhanced_row["native_coverage"] == 1.0
    assert enhanced_row["common_timestamp_count"] == 1
    assert enhanced_row["common_mae"] == pytest.approx(1.0)
    assert enhanced_row["comparison_role"] == "descriptive_post_hoc_robustness"
    assert enhanced_row["selection_role"] == "not_used_for_selection"
    smard_row = comparison.loc[
        (comparison["country"] == "Germany")
        & (comparison["model"] == "SMARD")
    ].iloc[0]
    assert smard_row["mae"] == pytest.approx(6.0)
    assert smard_row["native_observations"] == 2
    assert smard_row["common_mae"] == pytest.approx(6.0)
