from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import COUNTRY_CONFIG, country_forecast_origin
from enhanced_arima_models import enhanced_specifications
from enhanced_arima_validation_2024 import (
    build_adequacy_review,
    build_common_timestamp_set,
    build_full_validation_manifest,
    build_native_and_common_summary,
    freeze_specifications,
    run_full_validation,
)
import enhanced_arima_validation_2024 as validation_module
import enhanced_arima_screening_2024 as screening_module


def _year_frame(country: str, *, invalid_date: str | None = None) -> pd.DataFrame:
    timezone = COUNTRY_CONFIG[country]["timezone"]
    utc = pd.date_range(
        "2023-12-24 00:00:00",
        "2024-12-31 23:00:00",
        freq="h",
        tz="UTC",
    )
    local = utc.tz_convert(timezone)
    actual = np.arange(len(utc), dtype=float) + 1000.0
    frame = pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": actual,
            "hour": local.hour,
            "day_of_week": local.dayofweek,
        }
    )
    if invalid_date is not None:
        target = frame["timestamp_local"].str[:10].eq(invalid_date)
        frame = frame.loc[~target | ~frame["hour"].eq(0)].copy()
    return frame


def _shortlist() -> pd.DataFrame:
    rows = []
    for country in COUNTRY_CONFIG:
        for branch, count in (("ordinary", 3), ("weekly_differenced", 2)):
            specifications = [
                specification
                for specification in enhanced_specifications()
                if specification.branch == branch
            ][:count]
            for specification_order, specification in enumerate(specifications, 1):
                rows.append(
                    {
                        "country": country,
                        "branch": specification.branch,
                        "specification_id": specification.specification_id,
                        "specification_order": specification_order,
                        "p": specification.order[0],
                        "d": specification.order[1],
                        "q": specification.order[2],
                        "trend": specification.trend,
                        "transform": specification.transform,
                    }
                )
    return pd.DataFrame(rows)


def test_full_manifest_has_366_dates_five_candidates_and_expanding_origins():
    frames = {country: _year_frame(country) for country in COUNTRY_CONFIG}

    manifest = build_full_validation_manifest(_shortlist(), frames)

    assert len(manifest) == 2 * 366 * 5
    assert not manifest.duplicated(
        ["country", "target_date", "branch", "specification_id"]
    ).any()
    assert manifest.groupby("country")["target_date"].nunique().to_dict() == {
        "Germany": 366,
        "Austria": 366,
    }
    counts = manifest.groupby(["country", "target_date", "branch"]).size()
    assert set(counts.loc[counts.index.get_level_values("branch") == "ordinary"]) == {3}
    assert set(counts.loc[counts.index.get_level_values("branch") == "weekly_differenced"]) == {2}
    first = manifest.loc[
        manifest["country"].eq("Germany")
        & manifest["target_date"].eq("2024-01-01")
    ].iloc[0]
    last = manifest.loc[
        manifest["country"].eq("Germany")
        & manifest["target_date"].eq("2024-12-31")
    ].iloc[0]
    assert pd.Timestamp(first["forecast_origin_utc"]) == country_forecast_origin(
        "2024-01-01", "Germany"
    )
    assert pd.Timestamp(last["forecast_origin_utc"]) > pd.Timestamp(
        first["forecast_origin_utc"]
    )
    assert not manifest["target_date"].astype(str).str.startswith("2025").any()


def test_full_manifest_rejects_non_branch_local_specification_order():
    shortlist = _shortlist()
    weekly_index = shortlist.index[shortlist["branch"].eq("weekly_differenced")][0]
    shortlist.loc[weekly_index, "specification_order"] = 99

    with pytest.raises(ValueError, match="specification order"):
        validation_module._normalise_shortlist(shortlist)


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("p", 99),
        ("d", 99),
        ("q", 99),
        ("trend", "n"),
        ("transform", "identity"),
    ),
)
def test_full_manifest_rejects_shortlist_settings_mismatched_to_approved_specification(
    column, value
):
    shortlist = _shortlist()
    weekly_index = shortlist.index[shortlist["branch"].eq("weekly_differenced")][0]
    shortlist.loc[weekly_index, column] = value

    with pytest.raises(ValueError, match="specification settings"):
        build_full_validation_manifest(
            shortlist,
            {country: _year_frame(country) for country in COUNTRY_CONFIG},
        )


def test_run_full_validation_uses_only_origin_bounded_information_and_resumes_checkpoints(
    tmp_path, monkeypatch
):
    frames = {country: _year_frame(country) for country in COUNTRY_CONFIG}
    captured: list[tuple[str, int, pd.Timestamp, pd.Timestamp]] = []
    calls: list[tuple[str, str, str, str]] = []
    shortlist = _shortlist()
    real_manifest = validation_module.build_full_validation_manifest

    monkeypatch.setattr(
        validation_module,
        "_load_validation_frames",
        lambda processed_directory: frames,
    )
    monkeypatch.setattr(validation_module, "_load_reusable_candidates", lambda: {})
    monkeypatch.setattr(
        validation_module,
        "build_full_validation_manifest",
        lambda candidates, source_frames: real_manifest(candidates, source_frames).loc[
            real_manifest(candidates, source_frames)["target_date"].isin(
                ["2024-01-01", "2024-01-02"]
            )
        ].copy(),
    )

    def fake_execute(job, frame):
        key = tuple(str(job[column]) for column in validation_module.JOB_KEY)
        calls.append(key)
        cutoff = pd.Timestamp(job["forecast_origin_utc"])
        interval_end = pd.to_datetime(frame["timestamp_utc"], utc=True) + pd.Timedelta(hours=1)
        available = frame.loc[interval_end <= cutoff]
        captured.append(
            (
                key[0],
                len(available),
                pd.Timestamp(available["timestamp_utc"].max()),
                cutoff,
            )
        )
        target = frame.loc[frame["timestamp_local"].str[:10].eq(key[1])].copy()
        path = frame.loc[
            pd.to_datetime(frame["timestamp_utc"], utc=True).ge(cutoff)
            & frame["timestamp_local"].str[:10].le(key[1])
        ].copy()
        result_job = {
            **job,
            "status": "completed",
            "error_message": None,
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
            "evaluated_observations": len(target),
            "coverage": 1.0,
        }
        weekly_path = key[2] == "weekly_differenced"
        path = path.copy()
        source_timestamps = (
            pd.to_datetime(path["timestamp_utc"], utc=True)
            - pd.Timedelta(days=7)
            if weekly_path
            else pd.Series(pd.NaT, index=path.index)
        )
        source_lookup = pd.Series(
            frame["actual_grid_load_mwh"].to_numpy(dtype=float),
            index=pd.to_datetime(frame["timestamp_utc"], utc=True),
        )
        forecasts = path.assign(
            country=key[0],
            model_family="enhanced_arima",
            target_date=key[1],
            branch=key[2],
            specification_id=key[3],
            forecast_origin_local=job["forecast_origin_local"],
            forecast_origin_utc=job["forecast_origin_utc"],
            information_cutoff_utc=job["information_cutoff_utc"],
            interval_end_utc=path["timestamp_utc"] + pd.Timedelta(hours=1),
            local_date=path["timestamp_local"].str[:10],
            actual_load_mwh=path["actual_grid_load_mwh"],
            forecast_mwh=path["actual_grid_load_mwh"],
            bridge_used=~path["timestamp_local"].str[:10].eq(key[1]),
            source_timestamp_utc=source_timestamps,
            source_actual_mwh=(
                source_timestamps.map(source_lookup)
                if weekly_path
                else np.nan
            ),
            source_kind="observed" if weekly_path else "arima_forecast",
            is_target_day=path["timestamp_local"].str[:10].eq(key[1]),
            evaluated=path["timestamp_local"].str[:10].eq(key[1]),
            status="completed",
            mapping_issue_count=0,
        )
        return {"job": result_job, "forecasts": forecasts.to_dict("records"), "diagnostics": []}

    monkeypatch.setattr(validation_module, "_execute_validation_job", fake_execute)
    first = run_full_validation(shortlist, tmp_path, workers=1, processed_directory=tmp_path)

    assert first["total_jobs"] == 2 * 2 * 5
    assert len(calls) == first["total_jobs"]
    assert all(row[2] + pd.Timedelta(hours=1) <= row[3] for row in captured)
    assert all("2025" not in str(value) for row in captured for value in row)

    calls.clear()
    second = run_full_validation(shortlist, tmp_path, workers=1, processed_directory=tmp_path)

    assert second["total_jobs"] == first["total_jobs"]
    assert calls == []

    jobs_path = tmp_path / "ordinary" / "jobs.csv"
    jobs = pd.read_csv(jobs_path)
    jobs.loc[0, "status"] = "failed"
    jobs.to_csv(jobs_path, index=False)
    run_full_validation(shortlist, tmp_path, workers=1, processed_directory=tmp_path)
    assert len(calls) == 1


def test_checkpoint_requires_exact_country_date_dst_target_index_and_valid_timestamps():
    target_index = pd.date_range(
        "2024-03-31 00:00:00",
        "2024-04-01 00:00:00",
        inclusive="left",
        freq="h",
        tz="Europe/Berlin",
    ).tz_convert("UTC")
    job = {
        "country": "Germany",
        "target_date": "2024-03-31",
        "branch": "ordinary",
        "specification_id": "ordinary_a",
        "expected_observations": 23,
    }
    forecasts = pd.DataFrame(
        {
            "country": "Germany",
            "target_date": "2024-03-31",
            "branch": "ordinary",
            "specification_id": "ordinary_a",
            "timestamp_utc": target_index,
            "local_date": "2024-03-30",
            "actual_load_mwh": 100.0,
            "forecast_mwh": 100.0,
            "is_target_day": True,
            "evaluated": True,
            "status": "completed",
        }
    )

    assert not validation_module._forecast_complete_for_job(job, forecasts)

    malformed = forecasts.copy()
    malformed.loc[0, "local_date"] = "2024-03-31"
    malformed["timestamp_utc"] = malformed["timestamp_utc"].astype(object)
    malformed.loc[0, "timestamp_utc"] = "not-a-timestamp"
    assert not validation_module._forecast_complete_for_job(job, malformed)


def test_checkpoint_rows_foreign_to_manifest_are_excluded_from_validation_inputs():
    manifest = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
            }
        ]
    )
    jobs = pd.DataFrame(
        [
            manifest.iloc[0].to_dict(),
            {
                "country": "Austria",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
            },
        ]
    )
    forecasts = pd.DataFrame(
        [
            {**manifest.iloc[0].to_dict(), "timestamp_utc": "2024-01-01T00:00:00+00:00"},
            {
                "country": "Austria",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "timestamp_utc": "2024-01-01T00:00:00+00:00",
            },
        ]
    )
    diagnostics = jobs.assign(diagnostic="residual_acf", lag=1)

    filtered_jobs, filtered_forecasts, filtered_diagnostics = (
        validation_module._reconcile_checkpoint_frames(
            jobs, forecasts, diagnostics, manifest
        )
    )

    assert set(filtered_jobs["country"]) == {"Germany"}
    assert set(filtered_forecasts["country"]) == {"Germany"}
    assert set(filtered_diagnostics["country"]) == {"Germany"}
    assert list(filtered_jobs.columns) == validation_module.JOB_COLUMNS
    assert list(filtered_forecasts.columns) == validation_module.FORECAST_COLUMNS
    assert list(filtered_diagnostics.columns) == validation_module.DIAGNOSTICS_COLUMNS


def test_checkpoint_reconciliation_rejects_stale_same_key_job_metadata():
    manifest = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "p": 1,
                "d": 1,
                "q": 1,
                "trend": "n",
                "expected_observations": 24,
            }
        ]
    )
    stale_job = manifest.iloc[0].to_dict()
    stale_job.update(
        {
            "p": 9,
            "expected_observations": 23,
            "status": "completed",
        }
    )

    jobs, _, _ = validation_module._reconcile_checkpoint_frames(
        pd.DataFrame([stale_job]),
        pd.DataFrame(),
        pd.DataFrame(),
        manifest,
    )

    assert jobs.empty


def test_checkpoint_reconciliation_rejects_all_stale_manifest_metadata():
    manifest = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "category": "holiday",
                "forecast_origin_local": "2023-12-31T18:00:00+01:00",
                "forecast_origin_utc": "2023-12-31T17:00:00+00:00",
                "information_cutoff_utc": "2023-12-31T17:00:00+00:00",
                "expected_observations": 24,
                "branch": "weekly_differenced",
                "specification_id": "weekly_a",
                "p": 1,
                "d": 0,
                "q": 1,
                "trend": "c",
                "transform": "weekly_difference",
                "weekly_invalid": False,
                "weekly_invalid_reason": "",
            }
        ]
    )
    stale_values = {
        "category": "ordinary",
        "forecast_origin_local": "2023-12-31T19:00:00+01:00",
        "forecast_origin_utc": "2023-12-31T18:00:00+00:00",
        "information_cutoff_utc": "2023-12-31T18:00:00+00:00",
        "expected_observations": 23,
        "p": 2,
        "d": 1,
        "q": 2,
        "trend": "n",
        "transform": "identity",
        "weekly_invalid": True,
        "weekly_invalid_reason": "missing_source",
    }

    for column, stale_value in stale_values.items():
        stale_job = manifest.iloc[0].to_dict()
        stale_job[column] = stale_value
        jobs, _, _ = validation_module._reconcile_checkpoint_frames(
            pd.DataFrame([stale_job]),
            pd.DataFrame(),
            pd.DataFrame(),
            manifest,
        )
        assert jobs.empty, column

    missing_metadata = manifest.iloc[0].to_dict()
    missing_metadata["category"] = np.nan
    jobs, _, _ = validation_module._reconcile_checkpoint_frames(
        pd.DataFrame([missing_metadata]),
        pd.DataFrame(),
        pd.DataFrame(),
        manifest,
    )
    assert jobs.empty


def test_completed_checkpoint_missing_fit_metadata_is_not_terminal():
    job = _adequacy_jobs().iloc[0].to_dict()
    job["expected_observations"] = 24
    job["fit_status"] = np.nan

    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]), _adequacy_forecasts()
    ) == set()


def test_completed_checkpoint_requires_finite_fit_diagnostics_and_full_bridge_path():
    job = _adequacy_jobs().iloc[0].to_dict()
    forecasts = _adequacy_forecasts()

    key = ("Germany", "2024-01-01", "ordinary", "ordinary_a")
    assert validation_module._terminal_job_keys(pd.DataFrame([job]), forecasts) == {
        key
    }

    for column, value in (
        ("aic", np.nan),
        ("residual_ljung_box_pvalues", np.nan),
        ("residual_acf_flagged_lags", "not-json"),
    ):
        incomplete = job.copy()
        incomplete[column] = value
        assert validation_module._terminal_job_keys(
            pd.DataFrame([incomplete]), forecasts
        ) == set()

    incomplete_path = forecasts.iloc[1:].copy()
    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]), incomplete_path
    ) == set()


def test_weekly_completed_checkpoint_requires_source_mapping_metadata():
    job = _adequacy_jobs().iloc[0].to_dict()
    job.update(
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_a",
            "transform": "weekly_difference",
            "weekly_invalid": False,
        }
    )
    forecasts = _adequacy_forecasts().copy()
    forecasts["branch"] = "weekly_differenced"
    forecasts["specification_id"] = "weekly_a"
    timestamps = pd.to_datetime(forecasts["timestamp_utc"], utc=True)
    forecasts["source_timestamp_utc"] = timestamps - pd.Timedelta(days=7)
    forecasts["source_actual_mwh"] = 99.0
    forecasts["mapping_issue_count"] = 0
    forecasts["source_kind"] = "observed"
    key = ("Germany", "2024-01-01", "weekly_differenced", "weekly_a")

    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]), forecasts
    ) == {key}

    for column, stale_value in (
        ("source_timestamp_utc", pd.NaT),
        ("source_actual_mwh", np.nan),
        ("mapping_issue_count", 1),
    ):
        incomplete = forecasts.copy()
        incomplete.loc[incomplete.index[0], column] = stale_value
        assert validation_module._terminal_job_keys(
            pd.DataFrame([job]), incomplete
        ) == set(), column

    missing_columns = forecasts.drop(
        columns=["source_timestamp_utc", "source_actual_mwh", "mapping_issue_count"]
    )
    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]), missing_columns
    ) == set()


def test_weekly_validation_checkpoint_rejects_non_observed_or_misaligned_sources():
    job = _adequacy_jobs().iloc[0].to_dict()
    job.update(
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_a",
            "transform": "weekly_difference",
            "weekly_invalid": False,
        }
    )
    forecasts = _adequacy_forecasts().copy()
    forecasts["branch"] = "weekly_differenced"
    forecasts["specification_id"] = "weekly_a"
    timestamps = pd.to_datetime(forecasts["timestamp_utc"], utc=True)
    forecasts["source_timestamp_utc"] = timestamps - pd.Timedelta(days=7)
    forecasts["source_actual_mwh"] = 99.0
    forecasts["mapping_issue_count"] = 0
    forecasts["source_kind"] = "observed"

    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]), forecasts
    ) == {("Germany", "2024-01-01", "weekly_differenced", "weekly_a")}

    non_observed = forecasts.copy()
    non_observed.loc[non_observed.index[0], "source_kind"] = "forecast"
    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]), non_observed
    ) == set()

    misaligned = forecasts.copy()
    misaligned.loc[misaligned.index[0], "source_timestamp_utc"] += pd.Timedelta(hours=1)
    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]), misaligned
    ) == set()


def test_weekly_validation_checkpoint_rejects_source_actual_mismatch_to_prepared_frame():
    job = _adequacy_jobs().iloc[0].to_dict()
    job.update(
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_a",
            "transform": "weekly_difference",
            "weekly_invalid": False,
        }
    )
    forecasts = _adequacy_forecasts().copy()
    forecasts["branch"] = "weekly_differenced"
    forecasts["specification_id"] = "weekly_a"
    timestamps = pd.to_datetime(forecasts["timestamp_utc"], utc=True)
    forecasts["source_timestamp_utc"] = timestamps - pd.Timedelta(days=7)
    forecasts["source_actual_mwh"] = 99.0
    forecasts["mapping_issue_count"] = 0
    forecasts["source_kind"] = "observed"
    source_frame = pd.DataFrame(
        {
            "timestamp_utc": forecasts["source_timestamp_utc"],
            "actual_load_mwh": 99.0,
        }
    )
    key = ("Germany", "2024-01-01", "weekly_differenced", "weekly_a")

    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]),
        forecasts,
        source_frames={"Germany": source_frame},
    ) == {key}

    mismatch = source_frame.copy()
    mismatch.loc[0, "actual_load_mwh"] = 98.0
    assert validation_module._terminal_job_keys(
        pd.DataFrame([job]),
        forecasts,
        source_frames={"Germany": mismatch},
    ) == set()


def test_common_set_rejects_completed_jobs_with_invalid_fit_metadata():
    job = _adequacy_jobs().iloc[0].to_dict()
    job["fit_status"] = "failed"
    job["error_message"] = "fit failed"
    candidates = pd.DataFrame(
        [{"country": "Germany", "branch": "ordinary", "specification_id": "ordinary_a"}]
    )

    common = build_common_timestamp_set(
        _adequacy_forecasts(), candidates, "Germany", jobs=pd.DataFrame([job])
    )

    assert common.empty


def test_native_summary_reports_nonblank_reasons_for_every_failed_job():
    jobs = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "status": "failed",
                "error_message": "",
                "weekly_invalid": False,
            },
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "weekly_differenced",
                "specification_id": "weekly_a",
                "status": "weekly_lag_invalid",
                "error_message": "unexpected mapping failure",
                "weekly_invalid": True,
            },
        ]
    )
    manifest = jobs.copy()
    manifest["weekly_invalid"] = False
    manifest["expected_observations"] = 24

    summary = build_native_and_common_summary(jobs, pd.DataFrame(), manifest)

    ordinary_reason = summary.loc[
        summary["specification_id"].eq("ordinary_a"), "invalid_reasons"
    ].iloc[0]
    weekly_reason = summary.loc[
        summary["specification_id"].eq("weekly_a"), "invalid_reasons"
    ].iloc[0]
    assert ordinary_reason
    assert "failed" in ordinary_reason
    assert "unexpected mapping failure" in weekly_reason


def test_weekly_mapping_invalidity_is_terminal_for_the_whole_date(tmp_path, monkeypatch):
    frames = {country: _year_frame(country, invalid_date="2023-12-25") for country in COUNTRY_CONFIG}
    shortlist = _shortlist()
    real_manifest = validation_module.build_full_validation_manifest

    monkeypatch.setattr(validation_module, "_load_validation_frames", lambda _: frames)
    monkeypatch.setattr(
        validation_module,
        "build_full_validation_manifest",
        lambda candidates, source_frames: real_manifest(candidates, source_frames).loc[
            real_manifest(candidates, source_frames)["target_date"].eq("2024-01-01")
        ].copy(),
    )
    result = run_full_validation(shortlist, tmp_path, workers=1, processed_directory=tmp_path)

    jobs = pd.read_csv(tmp_path / "weekly_differenced" / "jobs.csv")
    invalid = jobs.loc[
        jobs["target_date"].eq("2024-01-01") & jobs["country"].eq("Germany")
    ]
    assert len(invalid) == 2
    assert invalid["status"].eq("weekly_lag_invalid").all()
    assert invalid["weekly_invalid"].astype(str).str.lower().eq("true").all()
    forecasts = pd.read_csv(tmp_path / "weekly_differenced" / "forecasts.csv")
    assert not forecasts["target_date"].eq("2024-01-01").any()
    assert result["terminal_weekly_invalid_jobs"] == 4


def _forecast_rows(specification_id: str, timestamps: list[str], error: float):
    target_utc = pd.to_datetime(timestamps, utc=True)
    origin = country_forecast_origin("2024-01-01", "Germany")
    weekly_path = specification_id.startswith("weekly_")
    target_start = pd.Timestamp(
        "2024-01-01 00:00:00", tz="Europe/Berlin"
    ).tz_convert("UTC")
    bridge = pd.date_range(origin, target_start, inclusive="left", freq="h")
    utc = bridge.append(target_utc)
    local = utc.tz_convert("Europe/Berlin")
    is_target = np.asarray(local.strftime("%Y-%m-%d")) == "2024-01-01"
    return pd.DataFrame(
        {
            "country": "Germany",
            "target_date": "2024-01-01",
            "branch": (
                "weekly_differenced"
                if specification_id.startswith("weekly_")
                else "ordinary"
            ),
            "specification_id": specification_id,
            "timestamp_utc": utc,
            "interval_end_utc": utc + pd.Timedelta(hours=1),
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_load_mwh": [100.0] * len(utc),
            "forecast_mwh": [100.0 + error] * len(utc),
            "is_target_day": is_target,
            "evaluated": is_target,
            "status": "completed",
            "local_date": local.strftime("%Y-%m-%d"),
            "forecast_origin_local": origin.tz_convert(
                "Europe/Berlin"
            ).isoformat(),
            "forecast_origin_utc": origin.isoformat(),
            "information_cutoff_utc": origin.isoformat(),
            "model_family": "enhanced_arima",
            "bridge_used": ~is_target,
            "source_timestamp_utc": utc - pd.Timedelta(days=7)
            if weekly_path
            else pd.NaT,
            "source_actual_mwh": 99.0 if weekly_path else np.nan,
            "source_kind": "observed" if weekly_path else "arima_forecast",
            "mapping_issue_count": 0,
        }
    )


def test_common_set_and_native_metrics_use_exact_observation_intersection():
    candidate_definitions = [
        ("ordinary_a", "ordinary", ["00:00", "01:00", "02:00"], 1.0),
        ("ordinary_b", "ordinary", ["01:00", "02:00", "03:00"], 2.0),
        ("ordinary_c", "ordinary", ["01:00", "02:00", "04:00"], 4.0),
        ("weekly_b", "weekly_differenced", ["01:00", "02:00", "05:00"], 3.0),
        ("weekly_c", "weekly_differenced", ["01:00", "02:00", "06:00"], 5.0),
    ]
    forecasts = pd.concat(
        [
            _forecast_rows(
                specification_id,
                [f"2024-01-01 {hour}" for hour in hours],
                error,
            )
            for specification_id, _, hours, error in candidate_definitions
        ],
        ignore_index=True,
    )
    candidates = pd.DataFrame(
        {
            "country": ["Germany"] * 5,
            "branch": [branch for _, branch, _, _ in candidate_definitions],
            "specification_id": [item[0] for item in candidate_definitions],
        }
    )
    jobs = pd.DataFrame(
        {
            "country": ["Germany"] * 5,
            "target_date": ["2024-01-01"] * 5,
            "branch": [branch for _, branch, _, _ in candidate_definitions],
            "specification_id": [item[0] for item in candidate_definitions],
            "status": ["completed"] * 5,
            "error_message": [""] * 5,
            "fit_status": ["success"] * 5,
            "convergence_status": ["converged"] * 5,
            "weekly_invalid": [False] * 5,
            "expected_observations": [3] * 5,
            "converged": [True] * 5,
            "parameters_finite": [True] * 5,
            "standard_errors_finite": [True] * 5,
            "stationarity_ok": [True] * 5,
            "invertibility_ok": [True] * 5,
            "log_likelihood": [-1.0] * 5,
            "aic": [1.0] * 5,
            "aicc": [1.0] * 5,
            "bic": [1.0] * 5,
            "residual_n_effective": [100] * 5,
            "residual_rejected": [False] * 5,
            "residual_rejection_reason": [""] * 5,
            "residual_acf_values": ['{"1": 0.1}'] * 5,
            "residual_acf_flagged_lags": ["[]"] * 5,
            "residual_ljung_box_statistics": ['{"24": 1.0, "48": 2.0}'] * 5,
            "residual_ljung_box_pvalues": ['{"24": 0.5, "48": 0.4}'] * 5,
        }
    )
    manifest = jobs.copy()
    manifest["expected_observations"] = 3

    common = build_common_timestamp_set(forecasts, candidates, "Germany")
    assert len(candidates) == 5
    assert common["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
        pd.Timestamp("2024-01-01 02:00", tz="UTC"),
    ]

    full_index = pd.date_range(
        "2024-01-01",
        "2024-01-02",
        inclusive="left",
        freq="h",
        tz="Europe/Berlin",
    ).tz_convert("UTC")
    summary_forecasts = pd.concat(
        [
            _forecast_rows(
                specification_id,
                [timestamp.isoformat() for timestamp in full_index],
                error,
            )
            for specification_id, _, _, error in candidate_definitions
        ],
        ignore_index=True,
    )
    summary_jobs = jobs.copy()
    summary_jobs["expected_observations"] = 24
    summary_manifest = summary_jobs.copy()
    summary = build_native_and_common_summary(
        summary_jobs, summary_forecasts, summary_manifest
    )
    ordinary = summary.loc[summary["specification_id"].eq("ordinary_a")].iloc[0]
    weekly = summary.loc[summary["specification_id"].eq("weekly_b")].iloc[0]
    assert len(summary) == 5
    assert ordinary["native_observations"] == 24
    assert ordinary["common_observations"] == 24
    assert ordinary["common_mae"] == pytest.approx(1.0)
    assert weekly["common_mae"] == pytest.approx(3.0)
    assert ordinary["native_coverage"] == pytest.approx(1.0)


def _adequacy_jobs(*, warning: bool = False, complete: bool = True):
    return pd.DataFrame(
        [
            {
                "country": "Germany",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "target_date": "2024-01-01",
                "status": "completed",
                "error_message": "" if complete else "fit failed",
                "fit_status": "success" if complete else "failed",
                "convergence_status": "converged" if complete else "failed",
                "converged": complete,
                "parameters_finite": complete,
                "standard_errors_finite": complete,
                "stationarity_ok": complete,
                "invertibility_ok": complete,
                "log_likelihood": -1.0 if complete else np.nan,
                "aic": 1.0 if complete else np.nan,
                "aicc": 1.0 if complete else np.nan,
                "bic": 1.0 if complete else np.nan,
                "residual_n_effective": 100 if complete else 0,
                "residual_rejected": False if complete else True,
                "residual_rejection_reason": "" if complete else "missing",
                "residual_acf_values": '{"1": 0.1}' if complete else "{}",
                "residual_acf_flagged_lags": "[1]" if warning else "[]",
                "residual_ljung_box_statistics": '{"24": 1.0, "48": 2.0}' if complete else "{}",
                "residual_ljung_box_pvalues": '{"24": 0.5, "48": 0.4}' if complete else "{}",
                "mae": 1.0 if complete else np.nan,
                "rmse": 1.0 if complete else np.nan,
                "mape": 1.0 if complete else np.nan,
                "evaluated_observations": 24 if complete else 0,
                "coverage": 1.0 if complete else 0.0,
                "weekly_invalid": False,
                "expected_observations": 24,
            }
        ]
    )


def _diagnostics_for_adequacy_job():
    return pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "diagnostic": "residual_acf",
                "lag": 1,
                "value": 0.1,
                "threshold": 0.2,
                "flagged": True,
            },
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "diagnostic": "ljung_box_statistic",
                "lag": 24,
                "value": 1.0,
                "threshold": 0.01,
                "flagged": False,
            },
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "diagnostic": "ljung_box_statistic",
                "lag": 48,
                "value": 2.0,
                "threshold": 0.01,
                "flagged": False,
            },
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "diagnostic": "ljung_box_pvalue",
                "lag": 24,
                "value": 0.5,
                "threshold": 0.01,
                "flagged": False,
            },
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "diagnostic": "ljung_box_pvalue",
                "lag": 48,
                "value": 0.4,
                "threshold": 0.01,
                "flagged": False,
            },
        ]
    )


def _adequacy_forecasts(target_date: str = "2024-01-01"):
    origin = country_forecast_origin(target_date, "Germany")
    end_date = pd.Timestamp(target_date).date() + pd.Timedelta(days=1)
    end = pd.Timestamp(f"{end_date} 00:00:00", tz="Europe/Berlin").tz_convert(
        "UTC"
    )
    utc_index = pd.date_range(origin, end, inclusive="left", freq="h")
    local_index = utc_index.tz_convert("Europe/Berlin")
    is_target = np.asarray(local_index.strftime("%Y-%m-%d")) == target_date
    return pd.DataFrame(
        {
            "country": ["Germany"] * len(utc_index),
            "branch": ["ordinary"] * len(utc_index),
            "specification_id": ["ordinary_a"] * len(utc_index),
            "target_date": [target_date] * len(utc_index),
            "timestamp_utc": utc_index,
            "interval_end_utc": utc_index + pd.Timedelta(hours=1),
            "timestamp_local": local_index.map(pd.Timestamp.isoformat),
            "local_date": local_index.strftime("%Y-%m-%d"),
            "forecast_origin_local": [
                origin.tz_convert("Europe/Berlin").isoformat()
            ] * len(utc_index),
            "forecast_origin_utc": [origin.isoformat()] * len(utc_index),
            "information_cutoff_utc": [origin.isoformat()] * len(utc_index),
            "actual_load_mwh": [100.0] * len(utc_index),
            "forecast_mwh": [101.0] * len(utc_index),
            "is_target_day": is_target,
            "evaluated": is_target,
            "status": ["completed"] * len(utc_index),
            "model_family": ["enhanced_arima"] * len(utc_index),
            "bridge_used": ~is_target,
            "source_kind": ["test"] * len(utc_index),
            "mapping_issue_count": [0] * len(utc_index),
        }
    )


def test_adequacy_requires_complete_finite_fit_and_reviewed_diagnostics():
    jobs = _adequacy_jobs(warning=True)
    forecasts = _adequacy_forecasts()
    summary = pd.DataFrame(
        {
            "country": ["Germany"],
            "branch": ["ordinary"],
            "specification_id": ["ordinary_a"],
            "native_mae": [1.0],
            "common_mae": [1.0],
        }
    )
    diagnostics = _diagnostics_for_adequacy_job()
    review = build_adequacy_review(jobs, forecasts, summary, diagnostics)
    row = review.iloc[0]
    assert bool(row["adequate"])
    assert bool(row["diagnostics_reviewed"])
    assert row["warning_count"] == 1

    mismatched = diagnostics.copy()
    mismatched.loc[0, "value"] = 999.0
    mismatched_review = build_adequacy_review(
        jobs, _adequacy_forecasts(), summary, mismatched
    )
    assert not bool(mismatched_review.iloc[0]["diagnostics_reviewed"])

    truncated = build_adequacy_review(
        jobs, forecasts, summary, diagnostics.iloc[:-1].copy()
    )
    assert not bool(truncated.iloc[0]["diagnostics_reviewed"])
    assert not bool(truncated.iloc[0]["adequate"])

    incomplete = build_adequacy_review(
        _adequacy_jobs(complete=False), forecasts, summary, diagnostics
    )
    assert not bool(incomplete.iloc[0]["adequate"])
    assert "fit" in str(incomplete.iloc[0]["inadequacy_reasons"])


def test_adequacy_accepts_ljung_box_statistic_flagged_by_its_pvalue():
    jobs = _adequacy_jobs(warning=True)
    jobs.loc[0, "residual_ljung_box_pvalues"] = '{"24": 0.001, "48": 0.4}'
    forecasts = _adequacy_forecasts()
    summary = pd.DataFrame(
        {
            "country": ["Germany"],
            "branch": ["ordinary"],
            "specification_id": ["ordinary_a"],
            "native_mae": [1.0],
            "common_mae": [1.0],
        }
    )
    diagnostics = _diagnostics_for_adequacy_job()
    diagnostics.loc[
        (diagnostics["diagnostic"] == "ljung_box_statistic")
        & diagnostics["lag"].eq(24),
        "flagged",
    ] = True
    diagnostics.loc[
        (diagnostics["diagnostic"] == "ljung_box_pvalue")
        & diagnostics["lag"].eq(24),
        ["value", "flagged"],
    ] = [0.001, True]

    review = build_adequacy_review(jobs, forecasts, summary, diagnostics)

    assert bool(review.iloc[0]["adequate"])
    assert bool(review.iloc[0]["diagnostics_reviewed"])


def test_freeze_selects_common_mae_then_rmse_mape_then_order_only_when_adequate():
    summary = pd.DataFrame(
        [
            {
                "country": "Germany",
                "branch": "weekly_differenced",
                "specification_id": "weekly_b",
                "specification_order": 2,
                "native_mae": 1.0,
                "native_rmse": 1.0,
                "native_mape": 1.0,
                "common_mae": 1.0,
                "common_rmse": 2.0,
                "common_mape": 3.0,
                "valid_dates": 365,
                "invalid_dates": 1,
                "native_coverage": 0.99,
            },
            {
                "country": "Germany",
                "branch": "ordinary",
                "specification_id": "ordinary_a",
                "specification_order": 5,
                "native_mae": 1.0,
                "native_rmse": 1.0,
                "native_mape": 1.0,
                "common_mae": 1.0,
                "common_rmse": 2.0,
                "common_mape": 3.0,
                "valid_dates": 366,
                "invalid_dates": 0,
                "native_coverage": 1.0,
            },
        ]
    )
    adequacy = pd.DataFrame(
        {
            "country": ["Germany", "Germany"],
            "branch": ["weekly_differenced", "ordinary"],
            "specification_id": ["weekly_b", "ordinary_a"],
            "adequate": [True, True],
            "diagnostics_reviewed": [True, True],
            "warning_count": [0, 0],
            "residual_warning_count": [0, 0],
            "valid_jobs": [365, 366],
            "invalid_jobs": [1, 0],
        }
    )

    frozen = freeze_specifications(summary, adequacy)

    assert len(frozen) == 2
    germany = frozen.loc[frozen["country"].eq("Germany")].iloc[0]
    assert germany["specification_id"] == "weekly_b"
    assert germany["selection_rule"] == (
        "common_mae, common_rmse, common_mape, specification_order"
    )
    assert germany["branch"] == "weekly_differenced"


def test_partial_manifest_candidate_cannot_pass_adequacy():
    jobs = _adequacy_jobs()
    forecasts = _adequacy_forecasts()
    summary = pd.DataFrame(
        {
            "country": ["Germany"],
            "branch": ["ordinary"],
            "specification_id": ["ordinary_a"],
            "expected_jobs": [2],
            "expected_dates": [2],
        }
    )
    review = build_adequacy_review(
        jobs,
        forecasts,
        summary,
        _diagnostics_for_adequacy_job(),
    )
    assert not bool(review.iloc[0]["adequate"])
    assert review.iloc[0]["invalid_jobs"] >= 1


def test_adequacy_rejects_substituted_manifest_job_key():
    jobs = _adequacy_jobs().copy()
    jobs["target_date"] = "2024-01-02"
    forecasts = _adequacy_forecasts("2024-01-02")
    diagnostics = _diagnostics_for_adequacy_job().copy()
    diagnostics["target_date"] = "2024-01-02"
    manifest = pd.DataFrame(
        {
            "country": ["Germany"],
            "target_date": ["2024-01-01"],
            "branch": ["ordinary"],
            "specification_id": ["ordinary_a"],
            "expected_observations": [24],
        }
    )
    summary = pd.DataFrame(
        {
            "country": ["Germany"],
            "branch": ["ordinary"],
            "specification_id": ["ordinary_a"],
            "expected_jobs": [1],
            "expected_dates": [1],
            "expected_job_keys": [
                json.dumps([["Germany", "2024-01-01", "ordinary", "ordinary_a"]])
            ],
        }
    )
    review = build_adequacy_review(
        jobs, forecasts, summary, diagnostics, manifest=manifest
    )
    assert not bool(review.iloc[0]["adequate"])
    assert "job key" in str(review.iloc[0]["inadequacy_reasons"])


def test_unexpected_weekly_failure_is_inadequate_even_when_flagged_invalid():
    jobs = _adequacy_jobs().copy()
    jobs.loc[0, "branch"] = "weekly_differenced"
    jobs.loc[0, "status"] = "weekly_lag_invalid"
    jobs.loc[0, "weekly_invalid"] = True
    jobs.loc[0, "weekly_invalid_reason"] = "missing_source"
    jobs.loc[0, "error_message"] = "unexpected weekly source mapping issue"
    summary = pd.DataFrame(
        {
            "country": ["Germany"],
            "branch": ["weekly_differenced"],
            "specification_id": ["ordinary_a"],
            "expected_jobs": [1],
            "expected_dates": [1],
        }
    )
    review = build_adequacy_review(
        jobs,
        pd.DataFrame(),
        summary,
        pd.DataFrame(),
    )
    assert not bool(review.iloc[0]["adequate"])
    assert review.iloc[0]["invalid_jobs"] == 1


def test_terminal_weekly_invalid_requires_derived_key_and_reason():
    jobs = pd.DataFrame(
        {
            "country": ["Germany"],
            "target_date": ["2024-01-01"],
            "branch": ["weekly_differenced"],
            "specification_id": ["weekly_b"],
            "status": ["weekly_lag_invalid"],
            "weekly_invalid": [True],
            "weekly_invalid_reason": ["missing_source"],
            "error_message": ["unrelated failure"],
        }
    )
    key = ("Germany", "2024-01-01", "weekly_differenced", "weekly_b")
    assert key not in validation_module._terminal_job_keys(
        jobs, pd.DataFrame(), {key: "missing_source"}
    )


def test_common_intersection_excludes_non_completed_jobs():
    candidates = pd.DataFrame(
        {
            "country": ["Germany"] * 5,
            "branch": ["ordinary"] * 3 + ["weekly_differenced"] * 2,
            "specification_id": [
                "ordinary_a",
                "ordinary_b",
                "ordinary_c",
                "weekly_b",
                "weekly_c",
            ],
        }
    )
    forecasts = pd.concat(
        [
            _forecast_rows(
                row.specification_id,
                ["2024-01-01 00:00", "2024-01-01 01:00"],
                1.0,
            )
            for row in candidates.itertuples(index=False)
        ],
        ignore_index=True,
    )
    jobs = candidates.assign(
        target_date="2024-01-01",
        status="completed",
        weekly_invalid=False,
    )
    jobs.loc[jobs["specification_id"].eq("ordinary_a"), "status"] = "failed"
    common = build_common_timestamp_set(
        forecasts, candidates, "Germany", jobs=jobs
    )
    assert common.empty


def test_native_summary_output_retains_invalid_reasons(tmp_path):
    jobs = pd.DataFrame(
        {
            "country": ["Germany"],
            "target_date": ["2024-01-01"],
            "branch": ["weekly_differenced"],
            "specification_id": ["weekly_b"],
            "status": ["weekly_lag_invalid"],
            "weekly_invalid": [True],
            "error_message": ["missing_source"],
        }
    )
    manifest = jobs.copy()
    manifest["expected_observations"] = 24
    summary, _ = validation_module._write_common_outputs(
        tmp_path, jobs, pd.DataFrame(), manifest
    )
    native = pd.read_csv(tmp_path / "native_summary.csv")
    assert "invalid_reasons" in summary.columns
    assert "invalid_reasons" in native.columns


def test_native_summary_retains_all_failed_and_unexpected_reasons():
    jobs = pd.DataFrame(
        {
            "country": ["Germany", "Germany"],
            "target_date": ["2024-01-01", "2024-01-01"],
            "branch": ["ordinary", "weekly_differenced"],
            "specification_id": ["ordinary_a", "weekly_b"],
            "status": ["failed", "weekly_lag_invalid"],
            "weekly_invalid": [False, True],
            "weekly_invalid_reason": ["", "missing_source"],
            "error_message": ["ordinary fit failed", "unexpected failure"],
        }
    )
    manifest = jobs.copy()
    manifest["expected_observations"] = 24
    summary = build_native_and_common_summary(jobs, pd.DataFrame(), manifest)
    ordinary = summary.loc[summary["branch"].eq("ordinary")].iloc[0]
    weekly = summary.loc[summary["branch"].eq("weekly_differenced")].iloc[0]
    assert ordinary["failed_jobs"] == 1
    assert "ordinary fit failed" in ordinary["invalid_reasons"]
    assert weekly["failed_jobs"] == 1
    assert weekly["invalid_dates"] == 0
    assert "unexpected failure" in weekly["invalid_reasons"]


def test_reuse_adapter_rejects_non_completed_or_incomplete_authoritative_jobs():
    job = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "branch": "ordinary",
        "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",
    }
    reusable = {"job": {"status": "failed"}, "forecasts": pd.DataFrame()}
    with pytest.raises(ValueError, match="completed"):
        screening_module._adapt_reused_result(job, reusable)

    reusable["job"] = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "specification_id": "arima_p1_d1_q1",
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",
        "status": "completed",
    }
    with pytest.raises(ValueError, match="forecast"):
        screening_module._adapt_reused_result(job, reusable)


def test_residual_metadata_rejects_flagged_unknown_lags_and_rejection_inconsistency():
    job = _adequacy_jobs().iloc[0].to_dict()
    job["residual_acf_flagged_lags"] = "[2]"
    assert not validation_module._residual_metadata_complete(job)[0]

    job["residual_acf_flagged_lags"] = "[]"
    job["residual_rejected"] = False
    job["residual_rejection_reason"] = "unexpected rejection reason"
    assert not validation_module._residual_metadata_complete(job)[0]


def test_empty_freeze_preserves_required_columns():
    frozen = freeze_specifications(pd.DataFrame(), pd.DataFrame())
    assert set(
        [
            "country",
            "branch",
            "specification_id",
            "common_mae",
            "common_rmse",
            "common_mape",
            "native_mae",
            "native_rmse",
            "native_mape",
            "valid_dates",
            "invalid_dates",
            "coverage",
            "convergence_adequate",
            "diagnostics_reviewed",
            "selection_rule",
        ]
    ).issubset(frozen.columns)


def test_empty_untyped_adequacy_frame_does_not_crash_freeze_schema():
    summary = pd.DataFrame(
        {
            "country": ["Germany"],
            "branch": ["ordinary"],
            "specification_id": ["ordinary_a"],
            "specification_order": [1],
            "common_mae": [1.0],
            "common_rmse": [1.0],
            "common_mape": [1.0],
        }
    )
    frozen = freeze_specifications(summary, pd.DataFrame())
    assert len(frozen) == 2
    assert frozen.loc[frozen["country"].eq("Germany"), "selection_status"].item() == (
        "no_eligible_candidate"
    )


def test_run_full_validation_suppresses_freeze_artifact_for_incomplete_validation(
    tmp_path, monkeypatch
):
    frames = {country: _year_frame(country) for country in COUNTRY_CONFIG}
    real_manifest = validation_module.build_full_validation_manifest
    manifest = real_manifest(_shortlist(), frames).iloc[[0]].copy()
    freeze_output = tmp_path.parent / "freeze_2024"
    freeze_output.mkdir()
    frozen_path = freeze_output / "frozen_specifications_2024.csv"
    adequacy_path = freeze_output / "adequacy_review_2024.csv"
    frozen_path.write_text("stale\n", encoding="utf-8")
    adequacy_path.write_text("stale\n", encoding="utf-8")
    writes: list[Path] = []
    real_write = screening_module._atomic_write_csv

    monkeypatch.setattr(validation_module, "_load_validation_frames", lambda _: frames)
    monkeypatch.setattr(validation_module, "_load_reusable_candidates", lambda: {})
    monkeypatch.setattr(
        validation_module,
        "build_full_validation_manifest",
        lambda candidates, source_frames: manifest.copy(),
    )
    monkeypatch.setattr(
        validation_module,
        "_execute_validation_job",
        lambda job, frame: {
            "job": {
                **job,
                "specification_id": "foreign_specification",
                "status": "failed",
                "error_message": "foreign result",
            },
            "forecasts": [],
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(
        screening_module,
        "_atomic_write_csv",
        lambda frame, path: (writes.append(Path(path)), real_write(frame, path))[1],
    )

    run_full_validation(_shortlist(), tmp_path, workers=1, processed_directory=tmp_path)

    assert not any(path.name == "frozen_specifications_2024.csv" for path in writes)
    assert not frozen_path.exists()
    assert not adequacy_path.exists()


def test_freeze_publishable_requires_frozen_adequate_row_per_country():
    manifest = pd.DataFrame(
        [
            {
                "country": country,
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": f"{country.lower()}_ordinary",
            }
            for country in COUNTRY_CONFIG
        ]
    )
    jobs = manifest.copy()
    adequacy = manifest.assign(adequate=True)
    freeze = pd.DataFrame(
        {
            "country": list(COUNTRY_CONFIG),
            "selection_status": ["frozen"] * len(COUNTRY_CONFIG),
            "adequate": [True] * len(COUNTRY_CONFIG),
        }
    )

    assert validation_module._freeze_publishable(
        manifest, jobs, adequacy, freeze
    )

    not_frozen = freeze.copy()
    not_frozen.loc[0, "selection_status"] = "no_eligible_candidate"
    assert not validation_module._freeze_publishable(
        manifest, jobs, adequacy, not_frozen
    )

    not_adequate = freeze.copy()
    not_adequate.loc[0, "adequate"] = False
    assert not validation_module._freeze_publishable(
        manifest, jobs, adequacy, not_adequate
    )
