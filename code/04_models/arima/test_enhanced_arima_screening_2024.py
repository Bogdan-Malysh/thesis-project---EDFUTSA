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
from enhanced_arima_models import (
    derive_weekly_invalid_target_dates,
    enhanced_specifications,
)
from enhanced_arima_screening_2024 import (
    _atomic_write_csv,
    _execute_job,
    _load_csv,
    _load_setup_country_data,
    _read_reusable_original_artifacts,
    _summary_frame,
    JOB_COLUMNS,
    _year_frame,
    audit_existing_arima,
    build_screening_calendar,
    build_screening_manifest,
    build_shortlist,
    run_screening,
    write_initial_artifacts,
)
import enhanced_arima_screening_2024 as screening_module


EXPECTED_DATES = (
    "2024-01-01",
    "2024-01-08",
    "2024-01-27",
    "2024-02-15",
    "2024-03-31",
    "2024-04-01",
    "2024-05-09",
    "2024-06-17",
    "2024-07-25",
    "2024-08-18",
    "2024-09-12",
    "2024-10-03",
    "2024-10-26",
    "2024-10-27",
    "2024-11-11",
    "2024-12-25",
)
EXPECTED_CATEGORIES = (
    "New Year holiday / cold season",
    "Monday / cold season",
    "Saturday / cold season",
    "ordinary weekday",
    "spring-DST Sunday",
    "Easter Monday holiday",
    "Ascension holiday",
    "Monday / warm season",
    "warm-season weekday",
    "warm-season Sunday",
    "ordinary weekday",
    "Germany-specific public holiday",
    "Austria-specific public holiday / Saturday",
    "autumn-DST Sunday",
    "Monday",
    "Christmas holiday / cold season",
)


def _calendar_frame(country: str) -> pd.DataFrame:
    rows = []
    config = COUNTRY_CONFIG[country]
    for target_date in EXPECTED_DATES:
        local_index = pd.date_range(
            f"{target_date} 00:00:00",
            f"{target_date} 23:00:00",
            freq="h",
            tz=config["timezone"],
        )
        utc_index = local_index.tz_convert("UTC")
        rows.extend(
            {
                "timestamp_utc": timestamp,
                "timestamp_local": timestamp.tz_convert(config["timezone"]).isoformat(),
                "actual_grid_load_mwh": 1000.0,
                "hour": timestamp.tz_convert(config["timezone"]).hour,
                "day_of_week": timestamp.tz_convert(config["timezone"]).dayofweek,
            }
            for timestamp in utc_index
        )
    return pd.DataFrame(rows)


def test_screening_calendar_has_exact_dates_and_persisted_categories_for_each_country():
    calendar = build_screening_calendar()

    assert list(calendar.columns) == ["target_date", "category"]
    assert list(calendar["target_date"]) == list(EXPECTED_DATES)
    assert list(calendar["category"]) == list(EXPECTED_CATEGORIES)
    assert len(calendar) == 16
    assert pd.to_datetime(calendar["target_date"]).dt.year.eq(2024).all()


def test_screening_manifest_is_unique_and_contains_country_origins_observations_and_derived_invalidity():
    calendar = build_screening_calendar()
    specifications = enhanced_specifications()
    frames = {country: _calendar_frame(country) for country in COUNTRY_CONFIG}
    invalid_dates = pd.DataFrame(
        {
            "country": ["Germany"],
            "target_date": ["2024-03-31"],
            "reason": ["missing_source"],
        }
    )

    manifest = build_screening_manifest(
        calendar,
        specifications,
        frames,
        invalid_dates,
    )

    key = ["country", "target_date", "branch", "specification_id"]
    assert len(manifest) == 32 * 16
    assert not manifest.duplicated(key).any()
    assert set(manifest["target_date"]) == set(EXPECTED_DATES)
    assert manifest["target_date"].map(lambda value: str(value)[:4]).eq("2024").all()
    assert set(manifest["country"]) == set(COUNTRY_CONFIG)
    assert manifest["expected_observations"].between(23, 25).all()

    germany_origin = manifest.loc[
        manifest["country"].eq("Germany") & manifest["target_date"].eq("2024-01-01")
    ].iloc[0]
    austria_origin = manifest.loc[
        manifest["country"].eq("Austria") & manifest["target_date"].eq("2024-01-01")
    ].iloc[0]
    assert pd.Timestamp(germany_origin["forecast_origin_utc"]) == country_forecast_origin(
        "2024-01-01", "Germany"
    )
    assert pd.Timestamp(austria_origin["forecast_origin_utc"]) == country_forecast_origin(
        "2024-01-01", "Austria"
    )
    assert germany_origin["forecast_origin_local"].endswith("+01:00")
    assert austria_origin["forecast_origin_local"].endswith("+01:00")

    germany_weekly = manifest.loc[
        manifest["country"].eq("Germany")
        & manifest["target_date"].eq("2024-03-31")
        & manifest["branch"].eq("weekly_differenced")
    ]
    austria_weekly = manifest.loc[
        manifest["country"].eq("Austria")
        & manifest["target_date"].eq("2024-03-31")
        & manifest["branch"].eq("weekly_differenced")
    ]
    assert germany_weekly["weekly_invalid"].eq(True).all()
    assert germany_weekly["weekly_invalid_reason"].eq("missing_source").all()
    assert austria_weekly["weekly_invalid"].eq(False).all()
    assert manifest.loc[
        manifest["branch"].eq("ordinary") & manifest["target_date"].eq("2024-03-31")
    ]["weekly_invalid"].eq(False).all()


def test_audit_records_original_grid_selection_overlap_evidence_and_reuse_eligibility():
    audit = audit_existing_arima()

    required = {
        "audit_type",
        "country",
        "branch",
        "specification_id",
        "order",
        "overlap_status",
        "evidence_completeness",
        "source_paths",
        "reuse_eligible",
    }
    assert required.issubset(audit.columns)
    assert audit["source_paths"].astype(str).str.len().gt(0).all()

    grid = audit.loc[audit["audit_type"].eq("original_candidate_grid")]
    assert set(grid["order"]) == {
        "(0, 1, 0)",
        "(1, 1, 0)",
        "(0, 1, 1)",
        "(1, 1, 1)",
        "(2, 1, 0)",
        "(0, 1, 2)",
        "(2, 1, 1)",
        "(1, 1, 2)",
        "(2, 1, 2)",
        "(3, 1, 0)",
        "(0, 1, 3)",
    }
    selected = audit.loc[audit["audit_type"].eq("selected_model")]
    assert set(selected["order"]) == {"(2, 1, 2)"}
    assert set(selected["country"]) == set(COUNTRY_CONFIG)

    ordinary = audit.loc[audit["audit_type"].eq("ordinary_enhanced_candidate")]
    assert set(ordinary["country"]) == set(COUNTRY_CONFIG)
    assert set(ordinary["specification_id"]) == {
        specification.specification_id
        for specification in specifications_for_branch("ordinary")
    }
    overlap = ordinary.loc[ordinary["overlap_status"].eq("overlap")]
    unexplored = ordinary.loc[ordinary["overlap_status"].eq("unexplored")]
    assert set(overlap["order"]) == {"(1, 1, 1)", "(1, 1, 2)", "(2, 1, 1)", "(2, 1, 2)"}
    assert len(unexplored) == 7 * len(COUNTRY_CONFIG)
    assert ordinary["evidence_completeness"].isin({"complete", "missing"}).all()
    assert ordinary.loc[ordinary["overlap_status"].eq("unexplored"), "reuse_eligible"].eq(
        False
    ).all()


def specifications_for_branch(branch: str):
    return [specification for specification in enhanced_specifications() if specification.branch == branch]


def test_write_initial_artifacts_creates_four_atomic_csv_artifacts(tmp_path):
    paths = write_initial_artifacts(tmp_path)

    expected = {
        "audit": tmp_path / "audit" / "original_arima_audit.csv",
        "run_manifest": tmp_path / "audit" / "enhanced_arima_run_manifest.csv",
        "candidate_definitions": tmp_path
        / "specifications"
        / "enhanced_arima_candidate_definitions.csv",
        "calendar": tmp_path / "specifications" / "screening_calendar_2024.csv",
    }
    assert paths == expected
    assert all(path.exists() and path.suffix == ".csv" for path in paths.values())
    assert not list(tmp_path.rglob("*.tmp"))

    candidates = pd.read_csv(paths["candidate_definitions"])
    assert len(candidates) == 16
    assert candidates[
        ["branch", "specification_id", "p", "d", "q", "trend", "transform"]
    ].to_dict("records") == [
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
            "p": 1,
            "d": 1,
            "q": 1,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p1_d1_q2",
            "p": 1,
            "d": 1,
            "q": 2,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p1_d1_q3",
            "p": 1,
            "d": 1,
            "q": 3,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p2_d1_q1",
            "p": 2,
            "d": 1,
            "q": 1,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p2_d1_q2",
            "p": 2,
            "d": 1,
            "q": 2,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p2_d1_q3",
            "p": 2,
            "d": 1,
            "q": 3,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p3_d1_q1",
            "p": 3,
            "d": 1,
            "q": 1,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p3_d1_q2",
            "p": 3,
            "d": 1,
            "q": 2,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p3_d1_q3",
            "p": 3,
            "d": 1,
            "q": 3,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p4_d1_q1",
            "p": 4,
            "d": 1,
            "q": 1,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "ordinary",
            "specification_id": "enhanced_ordinary_arima_p4_d1_q2",
            "p": 4,
            "d": 1,
            "q": 2,
            "trend": "n",
            "transform": "identity",
        },
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_differenced_arima_p1_d0_q1",
            "p": 1,
            "d": 0,
            "q": 1,
            "trend": "c",
            "transform": "weekly_difference",
        },
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_differenced_arima_p2_d0_q0",
            "p": 2,
            "d": 0,
            "q": 0,
            "trend": "c",
            "transform": "weekly_difference",
        },
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_differenced_arima_p2_d0_q1",
            "p": 2,
            "d": 0,
            "q": 1,
            "trend": "c",
            "transform": "weekly_difference",
        },
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_differenced_arima_p2_d0_q2",
            "p": 2,
            "d": 0,
            "q": 2,
            "trend": "c",
            "transform": "weekly_difference",
        },
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_differenced_arima_p3_d0_q1",
            "p": 3,
            "d": 0,
            "q": 1,
            "trend": "c",
            "transform": "weekly_difference",
        },
    ]

    calendar = pd.read_csv(paths["calendar"])
    assert len(calendar) == 16
    assert set(calendar["target_date"]) == set(EXPECTED_DATES)

    run_manifest = pd.read_csv(paths["run_manifest"])
    assert len(run_manifest) == 1
    row = run_manifest.iloc[0]
    for column in (
        "python_version",
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "statsmodels_version",
        "platform",
        "logical_cpu_count",
        "physical_ram_bytes",
        "available_ram_bytes",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "selected_worker",
        "run_timestamp_utc",
        "enhanced_source_files",
    ):
        assert column in run_manifest
    assert pd.isna(row["selected_worker"])
    assert pd.Timestamp(row["run_timestamp_utc"]).tzinfo is not None


def _full_year_frame(
    country: str,
    future_value: float = 0.0,
    future_start: str = "2024-01-01",
    start_utc: str = "2023-12-25 00:00:00+00:00",
) -> pd.DataFrame:
    timezone = COUNTRY_CONFIG[country]["timezone"]
    utc_index = pd.date_range(
        start_utc,
        "2024-12-31 23:00:00+00:00",
        freq="h",
    )
    local_index = utc_index.tz_convert(timezone)
    actual = np.arange(len(utc_index), dtype=float) + 1000.0
    actual[utc_index >= pd.Timestamp(future_start, tz="UTC")] += future_value
    return pd.DataFrame(
        {
            "timestamp_utc": utc_index,
            "timestamp_local": local_index.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": actual,
            "hour": local_index.hour,
            "day_of_week": local_index.dayofweek,
        }
    )


def _fake_target_forecasts(job, frame):
    path = screening_module._expected_forecast_path_index(
        str(job["country"]), str(job["target_date"])
    )
    timezone = COUNTRY_CONFIG[str(job["country"])]["timezone"]
    local_index = path.tz_convert(timezone)
    frame_values = pd.Series(
        frame["actual_grid_load_mwh"].to_numpy(dtype=float),
        index=pd.to_datetime(frame["timestamp_utc"], utc=True),
    )
    actual = frame_values.reindex(path).to_numpy(dtype=float)
    target = np.asarray(local_index.strftime("%Y-%m-%d")) == str(job["target_date"])
    return [
        {
            "country": job["country"],
            "model_family": "enhanced_arima",
            "target_date": job["target_date"],
            "branch": job["branch"],
            "specification_id": job["specification_id"],
            "forecast_origin_local": job["forecast_origin_local"],
            "forecast_origin_utc": job["forecast_origin_utc"],
            "information_cutoff_utc": job["information_cutoff_utc"],
            "timestamp_utc": timestamp,
            "interval_end_utc": pd.Timestamp(timestamp) + pd.Timedelta(hours=1),
            "timestamp_local": local,
            "local_date": local[:10],
            "actual_load_mwh": actual,
            "forecast_mwh": actual,
            "bridge_used": not is_target,
            "source_timestamp_utc": pd.NaT,
            "source_actual_mwh": np.nan,
            "source_kind": "arima_forecast",
            "is_target_day": is_target,
            "evaluated": is_target,
            "status": "completed",
            "mapping_issue_count": 0,
        }
        for timestamp, local, actual, is_target in zip(
            path,
            local_index.map(pd.Timestamp.isoformat),
            actual,
            target,
        )
    ]


def _screening_job_rows(
    *,
    weekly_invalid_dates: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    weekly_invalid_dates = weekly_invalid_dates or {}
    rows = []
    specifications = enhanced_specifications()
    for country in COUNTRY_CONFIG:
        for specification_order, specification in enumerate(specifications, 1):
            invalid_dates = weekly_invalid_dates.get(
                specification.specification_id,
                set(),
            )
            for target_date in EXPECTED_DATES:
                invalid = (
                    specification.branch == "weekly_differenced"
                    and target_date in invalid_dates
                )
                score = float(
                    [
                        item.specification_id
                        for item in specifications
                        if item.branch == specification.branch
                    ].index(specification.specification_id)
                    + 1
                )
                rows.append(
                    {
                        "country": country,
                        "target_date": target_date,
                        "category": "test",
                        "branch": specification.branch,
                        "specification_id": specification.specification_id,
                        "specification_order": specification_order,
                        "p": specification.order[0],
                        "d": specification.order[1],
                        "q": specification.order[2],
                        "trend": specification.trend,
                        "transform": specification.transform,
                        "weekly_invalid": invalid,
                        "status": "weekly_lag_invalid" if invalid else "completed",
                        "fit_status": "not_run" if invalid else "success",
                        "converged": not invalid,
                        "parameters_finite": not invalid,
                        "standard_errors_finite": not invalid,
                        "stationarity_ok": not invalid,
                        "invertibility_ok": not invalid,
                        "mae": score if not invalid else 999999.0,
                        "rmse": score if not invalid else 999999.0,
                        "mape": score if not invalid else 999999.0,
                        "evaluated_observations": 24 if not invalid else 0,
                        "expected_observations": 24,
                        "coverage": 1.0 if not invalid else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_build_shortlist_uses_only_valid_weekly_dates_and_enforces_15_of_16_gate():
    weekly_invalid_dates = {
        "weekly_differenced_arima_p1_d0_q1": {"2024-03-31"},
        "weekly_differenced_arima_p2_d0_q0": {
            "2024-03-31",
            "2024-10-27",
        },
    }
    jobs = _screening_job_rows(weekly_invalid_dates=weekly_invalid_dates)

    shortlist = build_shortlist(jobs)

    assert len(shortlist) == 10
    assert shortlist.groupby(["country", "branch"]).size().to_dict() == {
        ("Germany", "ordinary"): 3,
        ("Germany", "weekly_differenced"): 2,
        ("Austria", "ordinary"): 3,
        ("Austria", "weekly_differenced"): 2,
    }
    accepted_weekly = shortlist.loc[
        shortlist["branch"].eq("weekly_differenced")
    ]
    assert "weekly_differenced_arima_p2_d0_q0" not in set(
        accepted_weekly["specification_id"]
    )
    first_weekly = accepted_weekly.loc[
        accepted_weekly["specification_id"].eq(
            "weekly_differenced_arima_p1_d0_q1"
        )
    ].iloc[0]
    assert first_weekly["valid_dates"] == 15
    assert first_weekly["invalid_dates"] == 1
    assert first_weekly["mae"] == 1.0
    assert first_weekly["coverage"] == 15 / 16


def test_shortlist_rejects_unexpected_weekly_mapping_statuses():
    jobs = _screening_job_rows()
    unexpected = (
        jobs["branch"].eq("weekly_differenced")
        & jobs["specification_id"].eq("weekly_differenced_arima_p1_d0_q1")
        & jobs["target_date"].eq("2024-03-31")
    )
    jobs.loc[unexpected, "status"] = "weekly_lag_invalid"
    jobs.loc[unexpected, "weekly_invalid"] = False

    shortlist = build_shortlist(jobs)

    assert "weekly_differenced_arima_p1_d0_q1" not in set(
        shortlist.loc[
            shortlist["branch"].eq("weekly_differenced"), "specification_id"
        ]
    )


def test_shortlist_ranking_preserves_observation_level_23_24_25_weighting():
    specification = enhanced_specifications()[0]
    expected_counts = [23, 25, *([24] * 14)]
    manifest = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": target_date,
                "branch": "ordinary",
                "specification_id": specification.specification_id,
                "p": specification.order[0],
                "d": specification.order[1],
                "q": specification.order[2],
                "trend": specification.trend,
                "transform": specification.transform,
                "expected_observations": expected,
            }
            for target_date, expected in zip(EXPECTED_DATES, expected_counts)
        ]
    )
    jobs = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": target_date,
                "branch": "ordinary",
                "specification_id": specification.specification_id,
                "status": "completed",
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "evaluated_observations": expected,
                "expected_observations": expected,
            }
            for target_date, expected in zip(EXPECTED_DATES, expected_counts)
        ]
    )
    forecast_rows = []
    for date_index, (target_date, count) in enumerate(
        zip(EXPECTED_DATES, expected_counts)
    ):
        error = 1.0 if date_index == 0 else 100.0 if date_index == 1 else 0.0
        forecast_rows.extend(
            {
                "country": "Germany",
                "target_date": target_date,
                "branch": "ordinary",
                "specification_id": specification.specification_id,
                "actual_load_mwh": 100.0,
                "forecast_mwh": 100.0 + error,
                "is_target_day": True,
                "evaluated": True,
                "status": "completed",
            }
            for _ in range(count)
        )

    summary = _summary_frame(jobs, manifest, pd.DataFrame(forecast_rows))
    shortlist = build_shortlist(summary)
    expected_mae = (23 * 1.0 + 25 * 100.0) / sum(expected_counts)

    assert summary.iloc[0]["mae"] == expected_mae
    assert shortlist.iloc[0]["mae"] == expected_mae


@pytest.mark.parametrize(
    "target_count, diagnostics_present, expected_reuse",
    [(23, True, False), (24, True, True), (24, False, False)],
)
def test_reuse_requires_complete_authoritative_artifacts(
    tmp_path,
    monkeypatch,
    target_count,
    diagnostics_present,
    expected_reuse,
):
    original_id = "arima_p1_d1_q1"
    enhanced_id = "enhanced_ordinary_arima_p1_d1_q1"
    monkeypatch.setattr(screening_module, "ORIGINAL_VALIDATION_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        screening_module,
        "audit_existing_arima",
        lambda: pd.DataFrame(
            {
                "audit_type": ["ordinary_enhanced_candidate"],
                "country": ["Germany"],
                "specification_id": [enhanced_id],
                "reuse_eligible": [True],
            }
        ),
    )
    expected_origin = country_forecast_origin("2024-01-01", "Germany")
    source_job = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "specification_id": original_id,
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",
        "status": "completed",
        "fit_status": "success",
        "convergence_status": "converged",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "nobs": 100,
        "n_params": 2,
        "log_likelihood": -1.0,
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "ar_root_minimum": 2.0,
        "ma_root_minimum": 2.0,
        "warning_messages": "[]",
        "optimizer_retry_count": 0,
        "mae": 1.0,
        "rmse": 1.0,
        "mape": 1.0,
        "expected_observations": 24,
        "evaluated_observations": 24,
        "coverage": 1.0,
        "residual_n_effective": 100,
        "residual_rejected": False,
        "residual_acf_values": json.dumps({"1": 0.1}),
        "residual_acf_flagged_lags": "[]",
        "residual_ljung_box_statistics": json.dumps({"24": 1.0, "48": 1.0}),
        "residual_ljung_box_pvalues": json.dumps({"24": 0.5, "48": 0.5}),
    }
    if not diagnostics_present:
        source_job["residual_ljung_box_pvalues"] = ""
    pd.DataFrame([source_job]).to_csv(
        tmp_path / "arima_validation_2024_jobs.csv",
        index=False,
    )
    target_index = pd.date_range(
        "2024-01-01 00:00:00",
        "2024-01-01 23:00:00",
        freq="h",
        tz=COUNTRY_CONFIG["Germany"]["timezone"],
    )
    bridge_index = pd.date_range(
        expected_origin,
        target_index[0].tz_convert("UTC"),
        inclusive="left",
        freq="h",
    ).tz_convert(COUNTRY_CONFIG["Germany"]["timezone"])
    path_index = bridge_index.append(target_index[:target_count])
    timestamps = path_index.tz_convert("UTC")
    is_target = np.asarray(path_index.strftime("%Y-%m-%d")) == "2024-01-01"
    pd.DataFrame(
        {
            "country": "Germany",
            "target_date": "2024-01-01",
            "specification_id": original_id,
            "model_family": "ARIMA",
            "forecast_origin_local": expected_origin.tz_convert(
                COUNTRY_CONFIG["Germany"]["timezone"]
            ).isoformat(),
            "forecast_origin_utc": expected_origin.isoformat(),
            "information_cutoff_utc": expected_origin.isoformat(),
            "timestamp_utc": timestamps,
            "interval_end_utc": timestamps + pd.Timedelta(hours=1),
            "timestamp_local": path_index.map(
                pd.Timestamp.isoformat
            ),
            "local_date": path_index.strftime("%Y-%m-%d"),
            "actual_load_mwh": 100.0,
            "forecast_mwh": 101.0,
            "bridge_used": ~is_target,
            "source_kind": "arima_forecast",
            "is_target_day": is_target,
            "evaluated": is_target,
        }
    ).to_csv(
        tmp_path / "arima_validation_2024_forecasts.csv",
        index=False,
    )

    reusable = _read_reusable_original_artifacts()

    assert bool(reusable) is expected_reuse
    if expected_reuse:
        adapted = screening_module._adapt_reused_result(
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": enhanced_id,
                "p": 1,
                "d": 1,
                "q": 1,
                "trend": "n",
            },
            reusable[("Germany", "2024-01-01", original_id)],
        )
        assert adapted["job"]["artifact_source"] == "original_arima_validation_2024"
        assert adapted["job"]["source_job_path"].endswith(
            "arima_validation_2024_jobs.csv"
        )
        assert adapted["job"]["source_forecasts_path"].endswith(
            "arima_validation_2024_forecasts.csv"
        )
        assert "arima_validation_2024_jobs.csv" in adapted["job"]["source_artifact_path"]


def test_reuse_rejects_source_setting_metric_and_forecast_path_mismatches():
    original_id = "arima_p1_d1_q1"
    expected_origin = country_forecast_origin("2024-01-01", "Germany")
    target_index = pd.date_range(
        "2024-01-01 00:00:00",
        "2024-01-02 00:00:00",
        freq="h",
        inclusive="left",
        tz=COUNTRY_CONFIG["Germany"]["timezone"],
    )
    bridge_index = pd.date_range(
        expected_origin,
        target_index[0].tz_convert("UTC"),
        inclusive="left",
        freq="h",
    ).tz_convert(COUNTRY_CONFIG["Germany"]["timezone"])
    path_index = bridge_index.append(target_index)
    utc_index = path_index.tz_convert("UTC")
    is_target = np.asarray(path_index.strftime("%Y-%m-%d")) == "2024-01-01"
    source_job = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "specification_id": original_id,
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",
        "status": "completed",
        "error_message": "",
        "fit_status": "success",
        "convergence_status": "converged",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "nobs": 100,
        "n_params": 2,
        "log_likelihood": -1.0,
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "ar_root_minimum": 2.0,
        "ma_root_minimum": 2.0,
        "warning_messages": "[]",
        "optimizer_retry_count": 0,
        "residual_n_effective": 100,
        "residual_rejected": False,
        "residual_rejection_reason": "",
        "residual_acf_values": json.dumps({"1": 0.1}),
        "residual_acf_flagged_lags": "[]",
        "residual_ljung_box_statistics": json.dumps({"24": 1.0, "48": 1.0}),
        "residual_ljung_box_pvalues": json.dumps({"24": 0.5, "48": 0.5}),
        "expected_observations": 24,
        "evaluated_observations": 24,
        "coverage": 1.0,
        "mae": 1.0,
        "rmse": 1.0,
        "mape": 1.0,
    }
    forecasts = pd.DataFrame(
        {
            "country": "Germany",
            "target_date": "2024-01-01",
            "specification_id": original_id,
            "model_family": "ARIMA",
            "forecast_origin_local": expected_origin.tz_convert(
                COUNTRY_CONFIG["Germany"]["timezone"]
            ).isoformat(),
            "forecast_origin_utc": expected_origin.isoformat(),
            "information_cutoff_utc": expected_origin.isoformat(),
            "timestamp_utc": utc_index,
            "interval_end_utc": utc_index + pd.Timedelta(hours=1),
            "timestamp_local": path_index.map(pd.Timestamp.isoformat),
            "local_date": path_index.strftime("%Y-%m-%d"),
            "actual_load_mwh": 100.0,
            "forecast_mwh": 101.0,
            "bridge_used": ~is_target,
            "source_kind": "arima_forecast",
            "is_target_day": is_target,
            "evaluated": is_target,
        }
    )
    target_job = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "branch": "ordinary",
        "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "c",
    }
    reusable = {"job": source_job, "forecasts": forecasts}

    with pytest.raises(ValueError, match="settings"):
        screening_module._adapt_reused_result(target_job, reusable)

    target_job["trend"] = "n"
    source_job["mae"] = 2.0
    with pytest.raises(ValueError, match="metrics"):
        screening_module._adapt_reused_result(target_job, reusable)

    source_job["mae"] = 1.0
    with pytest.raises(ValueError, match="forecast"):
        screening_module._adapt_reused_result(
            target_job,
            {"job": source_job, "forecasts": forecasts.loc[~forecasts["bridge_used"]]},
        )

    forecasts.loc[0, "timestamp_local"] = "invalid"
    with pytest.raises(ValueError, match="forecast"):
        screening_module._adapt_reused_result(target_job, reusable)


def test_reuse_rejects_fit_settings_that_violate_fit_arima_conventions():
    original_id = "arima_p1_d1_q1"
    expected_origin = country_forecast_origin("2024-01-01", "Germany")
    source_job = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "specification_id": original_id,
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "c",
        "status": "completed",
        "error_message": "",
    }
    target_job = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "branch": "ordinary",
        "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "c",
    }

    assert not screening_module._source_fit_settings_match(source_job, target_job)
    with pytest.raises(ValueError, match="settings"):
        screening_module._adapt_reused_result(
            target_job,
            {"job": source_job, "forecasts": pd.DataFrame()},
        )


def test_reuse_rejects_source_and_requested_job_key_mismatch():
    source_job = {
        "country": "Austria",
        "target_date": "2024-01-01",
        "specification_id": "arima_p1_d1_q1",
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",
        "status": "completed",
        "error_message": "",
    }
    target_job = {
        "country": "Germany",
        "target_date": "2024-01-01",
        "branch": "ordinary",
        "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",
    }

    with pytest.raises(ValueError, match="key"):
        screening_module._adapt_reused_result(
            target_job,
            {"job": source_job, "forecasts": pd.DataFrame()},
        )


def test_run_screening_resumes_completed_jobs_retries_failures_and_upserts_without_duplicates(
    tmp_path,
    monkeypatch,
):
    frames = {country: _full_year_frame(country) for country in COUNTRY_CONFIG}
    monkeypatch.setattr(
        screening_module,
        "_load_setup_country_data",
        lambda country, processed_directory: frames[country],
    )
    one_specification = enhanced_specifications()[0]
    monkeypatch.setattr(
        screening_module,
        "enhanced_specifications",
        lambda: (one_specification,),
    )
    calls: list[tuple[str, str, str, str]] = []

    def fake_execute(job, frame):
        key = (
            str(job["country"]),
            str(job["target_date"]),
            str(job["branch"]),
            str(job["specification_id"]),
        )
        calls.append(key)
        return {
            "job": {
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
                    "mae": 1.0,
                "rmse": 1.0,
                "mape": 1.0,
                "evaluated_observations": int(job["expected_observations"]),
                "coverage": 1.0,
            },
                "forecasts": _fake_target_forecasts(job, frame),
            "diagnostics": [
                {
                    "country": job["country"],
                    "target_date": job["target_date"],
                    "branch": job["branch"],
                    "specification_id": job["specification_id"],
                    "diagnostic": "residual_acf",
                    "lag": 1,
                    "value": 0.1,
                    "threshold": 0.2,
                    "flagged": False,
                },
                {
                    "country": job["country"],
                    "target_date": job["target_date"],
                    "branch": job["branch"],
                    "specification_id": job["specification_id"],
                    "diagnostic": "ljung_box_pvalue",
                    "lag": 24,
                    "value": 0.5,
                    "threshold": 0.01,
                    "flagged": False,
                },
            ],
        }

    monkeypatch.setattr(screening_module, "_execute_job", fake_execute, raising=False)
    monkeypatch.setattr(
        screening_module,
        "_read_reusable_original_artifacts",
        lambda: {},
    )

    first = run_screening(tmp_path, processed_directory=tmp_path)
    assert first["total_jobs"] == 2 * 16
    assert len(calls) == first["total_jobs"]

    jobs_path = tmp_path / "screening_2024" / "ordinary" / "jobs.csv"
    jobs = pd.read_csv(jobs_path)
    assert not jobs.duplicated(
        ["country", "target_date", "branch", "specification_id"]
    ).any()
    diagnostics_path = tmp_path / "screening_2024" / "ordinary" / "diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    assert len(diagnostics) == 2 * len(jobs)
    assert not diagnostics.duplicated(
        [
            "country",
            "target_date",
            "branch",
            "specification_id",
            "diagnostic",
            "lag",
        ]
    ).any()

    forecasts_path = tmp_path / "screening_2024" / "ordinary" / "forecasts.csv"
    forecasts = pd.read_csv(forecasts_path)
    assert not forecasts.duplicated(
        [
            "country",
            "target_date",
            "branch",
            "specification_id",
            "timestamp_utc",
        ]
    ).any()
    first_key = tuple(
        str(jobs.iloc[0][column])
        for column in ("country", "target_date", "branch", "specification_id")
    )
    forecasts = forecasts.loc[
        ~(
            forecasts["country"].eq(first_key[0])
            & forecasts["target_date"].eq(first_key[1])
            & forecasts["branch"].eq(first_key[2])
            & forecasts["specification_id"].eq(first_key[3])
        )
    ]
    forecasts.to_csv(forecasts_path, index=False)

    calls.clear()
    second = run_screening(tmp_path, processed_directory=tmp_path)
    assert second["total_jobs"] == first["total_jobs"]
    assert calls == [first_key]

    jobs.loc[0, "status"] = "failed"
    jobs.loc[0, "error_message"] = np.nan
    jobs.to_csv(jobs_path, index=False)
    failed_key = tuple(
        str(jobs.iloc[0][column])
        for column in ("country", "target_date", "branch", "specification_id")
    )

    calls.clear()
    third = run_screening(tmp_path, processed_directory=tmp_path)

    assert third["total_jobs"] == first["total_jobs"]
    assert calls == [failed_key]
    final_jobs = pd.read_csv(jobs_path)
    assert len(final_jobs) == 2 * 16
    assert not final_jobs.duplicated(
        ["country", "target_date", "branch", "specification_id"]
    ).any()


def test_run_screening_fits_only_origin_bounded_values_not_future_actuals(
    tmp_path,
    monkeypatch,
):
    frames = {
        country: _full_year_frame(
            country,
            future_value=1_000_000.0,
            future_start="2024-12-26",
        )
        for country in COUNTRY_CONFIG
    }
    captured: list[tuple[str, int, float]] = []

    def fake_loader(country, processed_directory):
        return frames[country]

    def fake_fit(specification, values):
        values = np.asarray(values, dtype=float)
        captured.append((specification.branch, len(values), float(values[-1])))
        return SimpleNamespace(
            fit_status="success",
            convergence_status="converged",
            converged=True,
            parameters_finite=True,
            standard_errors_finite=True,
            stationarity_ok=True,
            invertibility_ok=True,
            nobs=len(values),
            n_params=1,
            log_likelihood=-1.0,
            aic=1.0,
            aicc=1.0,
            bic=1.0,
            ar_root_minimum=2.0,
            ma_root_minimum=2.0,
            warning_messages=(),
            optimizer_retry_count=0,
            residual_diagnostics=None,
            error_message=None,
            eligible=True,
            fitted_result=SimpleNamespace(
                get_forecast=lambda steps: SimpleNamespace(
                    predicted_mean=np.ones(steps, dtype=float)
                )
            ),
        )

    monkeypatch.setattr(screening_module, "_load_setup_country_data", fake_loader)
    monkeypatch.setattr(screening_module, "fit_enhanced", fake_fit, raising=False)
    monkeypatch.setattr(
        screening_module,
        "_read_reusable_original_artifacts",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        screening_module,
        "enhanced_specifications",
        lambda: (enhanced_specifications()[0],),
    )

    run_screening(tmp_path, processed_directory=tmp_path)

    assert captured
    assert all(length > 50 for _, length, _ in captured)
    assert all(value < 100000.0 for _, _, value in captured)


def test_valid_weekly_execution_persists_the_weekly_branch_on_forecasts(monkeypatch):
    frame = _full_year_frame("Germany")
    specification = next(
        item
        for item in enhanced_specifications()
        if item.branch == "weekly_differenced"
    )
    target_date = "2024-02-15"
    origin = country_forecast_origin(target_date, "Germany")
    expected_observations = len(
        frame["timestamp_local"].str[:10].eq(target_date)
    )

    class FakeFitted:
        def get_forecast(self, steps):
            return SimpleNamespace(predicted_mean=np.zeros(steps, dtype=float))

    monkeypatch.setattr(
        screening_module,
        "fit_enhanced",
        lambda spec, values: SimpleNamespace(
            fit_status="success",
            convergence_status="converged",
            converged=True,
            parameters_finite=True,
            standard_errors_finite=True,
            stationarity_ok=True,
            invertibility_ok=True,
            nobs=len(values),
            n_params=1,
            log_likelihood=-1.0,
            aic=1.0,
            aicc=1.0,
            bic=1.0,
            ar_root_minimum=2.0,
            ma_root_minimum=2.0,
            warning_messages=(),
            hard_warning_messages=(),
            optimizer_retry_count=0,
            residual_diagnostics=None,
            error_message=None,
            eligible=True,
            fitted_result=FakeFitted(),
        ),
    )

    result = _execute_job(
        {
            "country": "Germany",
            "target_date": target_date,
            "branch": specification.branch,
            "specification_id": specification.specification_id,
            "p": specification.order[0],
            "d": specification.order[1],
            "q": specification.order[2],
            "trend": specification.trend,
            "transform": specification.transform,
            "expected_observations": expected_observations,
            "weekly_invalid": False,
            "weekly_invalid_reason": "",
            "forecast_origin_utc": origin.isoformat(),
        },
        frame,
    )

    assert result["job"]["status"] == "completed"
    assert result["forecasts"]
    assert {row["branch"] for row in result["forecasts"]} == {
        "weekly_differenced"
    }


def test_unexpected_runtime_weekly_mapping_failure_is_not_a_derived_invalid_date(
    monkeypatch,
):
    frame = _full_year_frame("Germany")
    specification = next(
        item
        for item in enhanced_specifications()
        if item.branch == "weekly_differenced"
    )
    target_date = "2024-02-15"
    expected_observations = int(frame["timestamp_local"].str[:10].eq(target_date).sum())

    monkeypatch.setattr(
        screening_module,
        "fit_enhanced",
        lambda spec, values: SimpleNamespace(
            fit_status="success",
            convergence_status="converged",
            converged=True,
            parameters_finite=True,
            standard_errors_finite=True,
            stationarity_ok=True,
            invertibility_ok=True,
            nobs=len(values),
            n_params=1,
            log_likelihood=-1.0,
            aic=1.0,
            aicc=1.0,
            bic=1.0,
            ar_root_minimum=2.0,
            ma_root_minimum=2.0,
            warning_messages=(),
            hard_warning_messages=(),
            optimizer_retry_count=0,
            residual_diagnostics=None,
            error_message=None,
            eligible=True,
            fitted_result=SimpleNamespace(
                get_forecast=lambda steps: SimpleNamespace(
                    predicted_mean=np.zeros(steps, dtype=float)
                )
            ),
        ),
    )
    monkeypatch.setattr(
        screening_module,
        "build_weekly_forecast_path",
        lambda *args: pd.DataFrame({"status": ["weekly_lag_invalid"]}),
    )

    result = _execute_job(
        {
            "country": "Germany",
            "target_date": target_date,
            "branch": specification.branch,
            "specification_id": specification.specification_id,
            "p": specification.order[0],
            "d": specification.order[1],
            "q": specification.order[2],
            "trend": specification.trend,
            "transform": specification.transform,
            "expected_observations": expected_observations,
            "weekly_invalid": False,
            "weekly_invalid_reason": "",
        },
        frame,
    )

    assert result["job"]["status"] == "failed"
    assert result["job"]["error_message"] == "unexpected weekly source mapping issue"


@pytest.mark.parametrize("country", ["Germany", "Austria"])
def test_january_one_weekly_bridge_retains_december_24_source(
    tmp_path,
    monkeypatch,
    country,
):
    raw = _full_year_frame(
        country,
        start_utc="2023-12-24 00:00:00+00:00",
    )
    raw.to_csv(tmp_path / COUNTRY_CONFIG[country]["filename"], index=False)
    loaded = _load_setup_country_data(country, tmp_path)
    specification = next(
        item
        for item in enhanced_specifications()
        if item.branch == "weekly_differenced"
    )
    target_date = "2024-01-01"
    expected_observations = int(loaded["local_date"].eq(target_date).sum())

    class FakeFitted:
        def get_forecast(self, steps):
            return SimpleNamespace(predicted_mean=np.zeros(steps, dtype=float))

    monkeypatch.setattr(
        screening_module,
        "fit_enhanced",
        lambda spec, values: SimpleNamespace(
            fit_status="success",
            convergence_status="converged",
            converged=True,
            parameters_finite=True,
            standard_errors_finite=True,
            stationarity_ok=True,
            invertibility_ok=True,
            nobs=len(values),
            n_params=1,
            log_likelihood=-1.0,
            aic=1.0,
            aicc=1.0,
            bic=1.0,
            ar_root_minimum=2.0,
            ma_root_minimum=2.0,
            warning_messages=(),
            hard_warning_messages=(),
            optimizer_retry_count=0,
            residual_diagnostics=None,
            error_message=None,
            eligible=True,
            fitted_result=FakeFitted(),
        ),
    )

    result = _execute_job(
        {
            "country": country,
            "target_date": target_date,
            "branch": specification.branch,
            "specification_id": specification.specification_id,
            "p": specification.order[0],
            "d": specification.order[1],
            "q": specification.order[2],
            "trend": specification.trend,
            "transform": specification.transform,
            "expected_observations": expected_observations,
            "weekly_invalid": False,
            "weekly_invalid_reason": "",
        },
        loaded,
    )

    assert loaded["local_date"].eq("2023-12-24").any()
    assert result["job"]["status"] == "completed"
    assert result["forecasts"]
    assert {row["status"] for row in result["forecasts"]} == {"completed"}
    bridge = [row for row in result["forecasts"] if row["bridge_used"]]
    assert bridge
    assert pd.Timestamp(bridge[0]["source_timestamp_utc"]).date().isoformat() == (
        "2023-12-24"
    )


def test_setup_loader_stops_before_2025_and_preserves_weekly_bridge(tmp_path):
    path = tmp_path / COUNTRY_CONFIG["Germany"]["filename"]
    utc_index = pd.date_range(
        "2023-12-24 22:00:00+00:00",
        "2024-12-31 22:00:00+00:00",
        freq="h",
    )
    local_index = utc_index.tz_convert(COUNTRY_CONFIG["Germany"]["timezone"])
    pd.DataFrame(
        {
            "timestamp_utc": utc_index,
            "timestamp_local": local_index.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": np.arange(len(utc_index), dtype=float),
            "hour": local_index.hour,
            "day_of_week": local_index.dayofweek,
        }
    ).to_csv(path, index=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("2025-01-01T00:00:00+00:00,malformed-2025-row\n")

    loaded = _load_setup_country_data("Germany", tmp_path)

    assert loaded["local_date"].min() == "2023-12-24"
    assert loaded["local_date"].max() == "2024-12-31"
    assert loaded["local_date"].eq("2023-12-24").any()
    assert not loaded["local_date"].str.startswith("2025-").any()
    invalid = derive_weekly_invalid_target_dates(loaded, 2024)
    assert not invalid["target_date"].eq("2024-01-01").any()


def test_year_frame_keeps_the_pre_2024_bridge_and_excludes_2025():
    frame = pd.DataFrame(
        {
            "timestamp_local": [
                "2023-12-25T00:00:00+01:00",
                "2024-01-01T00:00:00+01:00",
                "2025-01-01T00:00:00+01:00",
            ],
        }
    )

    bounded = _year_frame(frame)

    assert bounded["timestamp_local"].tolist() == [
        "2023-12-25T00:00:00+01:00",
        "2024-01-01T00:00:00+01:00",
    ]


def test_atomic_write_preserves_existing_target_when_write_fails(tmp_path):
    path = tmp_path / "artifact.csv"
    path.write_text("old\n", encoding="utf-8")

    class FailingFrame:
        def to_csv(self, handle, index):
            handle.write("partial\n")
            raise RuntimeError("simulated write failure")

    with pytest.raises(RuntimeError, match="simulated write failure"):
        _atomic_write_csv(FailingFrame(), path)

    assert path.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_loader_normalizes_dates_and_keeps_last_duplicate_logical_row(tmp_path):
    path = tmp_path / "jobs.csv"
    pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-01T00:00:00+01:00",
                "branch": "ordinary",
                "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
                "status": "failed",
            },
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "branch": "ordinary",
                "specification_id": "enhanced_ordinary_arima_p1_d1_q1",
                "status": "completed",
            },
        ]
    ).to_csv(path, index=False)

    loaded = _load_csv(path, JOB_COLUMNS)

    assert len(loaded) == 1
    assert loaded.iloc[0]["target_date"] == "2024-01-01"
    assert loaded.iloc[0]["status"] == "completed"


def _complete_screening_checkpoint():
    target_date = "2024-01-01"
    origin = country_forecast_origin(target_date, "Germany")
    job = {
        "country": "Germany",
        "target_date": target_date,
        "category": "test",
        "branch": "ordinary",
        "specification_id": "ordinary_a",
        "p": 1,
        "d": 1,
        "q": 1,
        "trend": "n",
        "transform": "identity",
        "forecast_origin_local": origin.tz_convert("Europe/Berlin").isoformat(),
        "forecast_origin_utc": origin.isoformat(),
        "information_cutoff_utc": origin.isoformat(),
        "expected_observations": 24,
        "weekly_invalid": False,
        "weekly_invalid_reason": "",
        "status": "completed",
        "error_message": "",
        "fit_status": "success",
        "convergence_status": "converged",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "nobs": 100,
        "n_params": 2,
        "log_likelihood": -1.0,
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "ar_root_minimum": 2.0,
        "ma_root_minimum": 2.0,
        "warning_messages": "[]",
        "hard_warning_messages": "[]",
        "optimizer_retry_count": 0,
        "residual_n_effective": 100,
        "residual_rejected": False,
        "residual_rejection_reason": "",
        "residual_acf_values": '{"1": 0.1}',
        "residual_acf_flagged_lags": "[]",
        "residual_ljung_box_statistics": '{"24": 1.0, "48": 2.0}',
        "residual_ljung_box_pvalues": '{"24": 0.5, "48": 0.4}',
    }
    path = screening_module._expected_forecast_path_index("Germany", target_date)
    local = path.tz_convert("Europe/Berlin")
    target = np.asarray(local.strftime("%Y-%m-%d")) == target_date
    forecasts = pd.DataFrame(
        {
            "country": "Germany",
            "model_family": "enhanced_arima",
            "target_date": target_date,
            "branch": "ordinary",
            "specification_id": "ordinary_a",
            "forecast_origin_local": job["forecast_origin_local"],
            "forecast_origin_utc": job["forecast_origin_utc"],
            "information_cutoff_utc": job["information_cutoff_utc"],
            "timestamp_utc": path,
            "interval_end_utc": path + pd.Timedelta(hours=1),
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "local_date": local.strftime("%Y-%m-%d"),
            "actual_load_mwh": 100.0,
            "forecast_mwh": 101.0,
            "bridge_used": ~target,
            "source_timestamp_utc": path - pd.Timedelta(days=7),
            "source_actual_mwh": 99.0,
            "source_kind": "observed",
            "is_target_day": target,
            "evaluated": target,
            "status": "completed",
            "mapping_issue_count": 0,
        }
    )
    return job, forecasts


def test_screening_checkpoint_requires_fit_metadata_and_complete_dst_path():
    job, forecasts = _complete_screening_checkpoint()
    manifest = pd.DataFrame([job])
    key = ("Germany", "2024-01-01", "ordinary", "ordinary_a")

    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), forecasts, manifest
    ) == {key}

    incomplete_fit = job.copy()
    incomplete_fit["fit_status"] = np.nan
    assert screening_module._terminal_job_keys(
        pd.DataFrame([incomplete_fit]), forecasts, manifest
    ) == set()

    incomplete_path = forecasts.iloc[1:].copy()
    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), incomplete_path, manifest
    ) == set()

    malformed_dst = forecasts.copy()
    malformed_dst.loc[0, "timestamp_local"] = "2024-01-01T00:00:00+00:00"
    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), malformed_dst, manifest
    ) == set()


def test_screening_checkpoint_accepts_only_derived_weekly_invalid_status():
    job, _ = _complete_screening_checkpoint()
    job.update(
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_a",
            "transform": "weekly_difference",
            "weekly_invalid": True,
            "weekly_invalid_reason": "missing_source",
            "status": "weekly_lag_invalid",
            "error_message": "missing_source",
        }
    )
    manifest = pd.DataFrame([job])
    key = ("Germany", "2024-01-01", "weekly_differenced", "weekly_a")

    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), pd.DataFrame(), manifest
    ) == {key}

    unexpected = job.copy()
    unexpected["error_message"] = "runtime mapping failure"
    assert screening_module._terminal_job_keys(
        pd.DataFrame([unexpected]), pd.DataFrame(), manifest
    ) == set()

    not_derived = job.copy()
    not_derived["weekly_invalid"] = False
    not_derived["weekly_invalid_reason"] = ""
    not_derived_manifest = manifest.copy()
    not_derived_manifest["weekly_invalid"] = False
    not_derived_manifest["weekly_invalid_reason"] = ""
    assert screening_module._terminal_job_keys(
        pd.DataFrame([not_derived]), pd.DataFrame(), not_derived_manifest
    ) == set()


def test_screening_checkpoint_reconciles_same_key_manifest_metadata_before_skip():
    job, forecasts = _complete_screening_checkpoint()
    manifest = pd.DataFrame([job])
    key = ("Germany", "2024-01-01", "ordinary", "ordinary_a")

    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), forecasts, manifest
    ) == {key}

    for column, stale_value in (
        ("category", "stale category"),
        ("forecast_origin_local", "2024-01-01T00:00:00+01:00"),
        ("p", 9),
        ("transform", "weekly_difference"),
        ("weekly_invalid", True),
    ):
        stale = job.copy()
        stale[column] = stale_value
        assert screening_module._terminal_job_keys(
            pd.DataFrame([stale]), forecasts, manifest
        ) == set(), column


def test_screening_weekly_checkpoint_requires_observed_strict_seven_day_sources():
    job, forecasts = _complete_screening_checkpoint()
    job.update(
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_a",
            "transform": "weekly_difference",
            "weekly_invalid": False,
        }
    )
    forecasts = forecasts.copy()
    forecasts["branch"] = "weekly_differenced"
    forecasts["specification_id"] = "weekly_a"
    key = ("Germany", "2024-01-01", "weekly_differenced", "weekly_a")
    manifest = pd.DataFrame([job])

    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), forecasts, manifest
    ) == {key}

    non_observed = forecasts.copy()
    non_observed.loc[non_observed.index[0], "source_kind"] = "forecast"
    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), non_observed, manifest
    ) == set()

    misaligned = forecasts.copy()
    misaligned.loc[misaligned.index[0], "source_timestamp_utc"] += pd.Timedelta(hours=1)
    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]), misaligned, manifest
    ) == set()


def test_screening_weekly_checkpoint_rejects_source_actual_mismatch_to_prepared_frame():
    job, forecasts = _complete_screening_checkpoint()
    job.update(
        {
            "branch": "weekly_differenced",
            "specification_id": "weekly_a",
            "transform": "weekly_difference",
            "weekly_invalid": False,
        }
    )
    forecasts = forecasts.copy()
    forecasts["branch"] = "weekly_differenced"
    forecasts["specification_id"] = "weekly_a"
    source_frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                forecasts["source_timestamp_utc"], utc=True
            ),
            "actual_load_mwh": 99.0,
        }
    )
    manifest = pd.DataFrame([job])
    key = ("Germany", "2024-01-01", "weekly_differenced", "weekly_a")

    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]),
        forecasts,
        manifest,
        source_frames={"Germany": source_frame},
    ) == {key}

    mismatch = source_frame.copy()
    mismatch.loc[0, "actual_load_mwh"] = 98.0
    assert screening_module._terminal_job_keys(
        pd.DataFrame([job]),
        forecasts,
        manifest,
        source_frames={"Germany": mismatch},
    ) == set()
