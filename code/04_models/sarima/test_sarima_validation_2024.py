from datetime import date

import numpy as np
import pandas as pd
import pytest

import sarima_validation_2024 as validation
from sarima_models import (
    ResidualDiagnostics,
    RootDiagnostics,
    SarimaFitRecord,
    SarimaOrder,
    fit_sarima as real_fit_sarima,
)
from common.forecasting_framework import (
    country_forecast_origin,
    evaluate_target_day,
    extract_target_day,
)


MINI_DATES = [date(2024, 2, 15), date(2024, 3, 31), date(2024, 10, 27)]


def _shortlists() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": "Germany",
                "specification_id": "sarima_p1_d1_q1_P0_D0_Q1_s24",
                "specification_order": 1,
                "p": 1,
                "d": 1,
                "q": 1,
                "P": 0,
                "D": 0,
                "Q": 1,
                "seasonal_period": 24,
                "trend": "n",
            },
            {
                "country": "Austria",
                "specification_id": "sarima_p1_d0_q1_P1_D1_Q0_s24",
                "specification_order": 1,
                "p": 1,
                "d": 0,
                "q": 1,
                "P": 1,
                "D": 1,
                "Q": 0,
                "seasonal_period": 24,
                "trend": "n",
            },
        ]
    )


def _frame(start: str, end: str) -> pd.DataFrame:
    utc = pd.date_range(start, end, freq="h", tz="UTC")
    local = utc.tz_convert("Europe/Berlin")
    values = np.arange(len(utc), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": values,
            "hour": local.hour,
            "day_of_week": local.dayofweek,
        }
    )


def _fit_record(order: SarimaOrder, result: object) -> SarimaFitRecord:
    residual = ResidualDiagnostics(
        burn_in=48,
        state_space_loglikelihood_burn=2,
        conservative_burn_in=48,
        n_effective=100,
        acf_values={24: 0.2},
        acf_threshold=0.1,
        flagged_acf_lags=(24,),
        ljung_box_statistics={24: 1.0, 48: 2.0},
        ljung_box_pvalues={24: 0.2, 48: 0.3},
        training_adequacy_rejected=True,
        training_adequacy_reason="diagnostic flag",
        warnings=("UserWarning: diagnostic warning",),
    )
    return SarimaFitRecord(
        order=order,
        trend="n",
        fitted_result=result,
        fit_status="success",
        convergence_status="converged",
        optimizer_success=True,
        optimizer_status=0,
        optimizer_warnflag=0,
        warning_messages=(),
        hard_warning_messages=(),
        optimizer_retry_count=0,
        error_message=None,
        parameters_finite=True,
        standard_errors_finite=True,
        converged=True,
        forecast_valid=True,
        forecast_error_message=None,
        nobs=100,
        effective_nobs=100,
        n_params=4,
        log_likelihood=-10.0,
        aic=20.0,
        aicc=20.1,
        bic=21.0,
        root_diagnostics=RootDiagnostics.empty(),
        ar_root_minimum=1.5,
        ma_root_minimum=1.5,
        stationarity_ok=True,
        invertibility_ok=True,
        residual_diagnostics=residual,
        training_adequacy_rejected=True,
    )


def _forecast_rows(job, frame):
    prepared = validation._as_prepared(frame)
    target_date = job["target_date"]
    target = prepared.loc[prepared["local_date"].eq(target_date)].set_index(
        "timestamp_utc"
    )
    return [
        {
            "country": job["country"],
            "model_family": "SARIMA",
            "target_date": job["target_date"],
            "specification_id": job["specification_id"],
            "timestamp_utc": timestamp,
            "interval_end_utc": pd.Timestamp(timestamp) + pd.Timedelta(hours=1),
            "timestamp_local": pd.Timestamp(timestamp)
            .tz_convert("Europe/Berlin")
            .isoformat(),
            "local_date": target_date,
            "actual_load_mwh": float(
                target.loc[pd.Timestamp(timestamp), "actual_load_mwh"]
            )
            if pd.Timestamp(timestamp) in target.index
            else 1.0,
            "forecast_mwh": 1.0,
            "bridge_used": False,
            "source_kind": "test",
            "is_target_day": True,
            "evaluated": True,
        }
        for timestamp in sorted(
            validation.expected_forecast_timestamps(
                frame, job["country"], job["target_date"]
            )
        )
    ]


def test_manifest_has_unique_sorted_country_date_specification_keys():
    manifest = validation.build_manifest(_shortlists(), MINI_DATES)

    assert len(manifest) == 6
    assert list(manifest.columns[:3]) == ["country", "target_date", "specification_id"]
    assert not manifest.duplicated(list(validation.JOB_KEY_COLUMNS)).any()
    assert manifest.iloc[0]["country"] == "Germany"
    assert manifest.iloc[0]["target_date"] == "2024-02-15"
    assert manifest.iloc[-1]["country"] == "Austria"


def test_full_year_manifest_has_1464_jobs_and_366_dates_per_country():
    full_dates = list(pd.date_range("2024-01-01", "2024-12-31", freq="D").date)
    shortlists = pd.concat(
        [
            _shortlists(),
            pd.DataFrame(
                [
                    {
                        "country": "Germany",
                        "specification_id": "sarima_p2_d0_q0_P0_D1_Q0_s24",
                        "specification_order": 5,
                        "p": 2,
                        "d": 0,
                        "q": 0,
                        "P": 0,
                        "D": 1,
                        "Q": 0,
                        "seasonal_period": 24,
                        "trend": "n",
                    },
                    {
                        "country": "Austria",
                        "specification_id": "sarima_p1_d1_q1_P0_D0_Q1_s24",
                        "specification_order": 11,
                        "p": 1,
                        "d": 1,
                        "q": 1,
                        "P": 0,
                        "D": 0,
                        "Q": 1,
                        "seasonal_period": 24,
                        "trend": "n",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    manifest = validation.build_manifest(
        shortlists,
        {"Germany": full_dates, "Austria": full_dates},
        allow_full_year=True,
    )

    assert len(manifest) == 1464
    assert manifest.groupby("country").size().to_dict() == {
        "Germany": 732,
        "Austria": 732,
    }
    assert manifest.groupby(["country", "target_date"]).size().eq(2).all()


def test_full_year_manifest_rejects_non_2024_or_incomplete_dates():
    full_dates = list(pd.date_range("2024-01-01", "2024-12-31", freq="D").date)

    with pytest.raises(ValueError, match="full 2024"):
        validation.build_manifest(
            _shortlists(),
            {"Germany": full_dates[:-1], "Austria": full_dates},
            allow_full_year=True,
        )


def test_country_origins_are_local_preceding_day_cutoffs():
    germany = country_forecast_origin(date(2024, 2, 15), "Germany")
    austria = country_forecast_origin(date(2024, 2, 15), "Austria")

    assert germany.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M") == "2024-02-14 18:00"
    assert austria.tz_convert("Europe/Vienna").strftime("%Y-%m-%d %H:%M") == "2024-02-14 08:00"


def test_frozen_shortlist_rejects_specification_id_or_country_errors():
    invalid = _shortlists().copy()
    invalid.loc[0, "specification_id"] = "not-the-order"

    with pytest.raises(ValueError, match="specification_id"):
        validation.load_shortlists(invalid)


@pytest.mark.parametrize(
    ("target_dates", "message"),
    [
        ([date(2024, 2, 14)], "approved mini-validation dates"),
        ([date(2024, 2, 15), date(2024, 2, 15)], "duplicate target dates"),
        ([date(2024, 2, 15), date(2024, 3, 31)], "approved mini-validation dates"),
        (
            list(pd.date_range("2024-01-01", "2024-12-31", freq="D")),
            "approved mini-validation dates",
        ),
    ],
)
def test_validation_accepts_only_the_exact_approved_target_date_set(
    target_dates, message
):
    with pytest.raises(ValueError, match=message):
        validation._target_date_strings(target_dates)


def test_forecast_path_uses_only_information_available_at_origin(monkeypatch):
    frame = validation._as_prepared(_frame("2024-02-13 00:00", "2024-02-15 23:00"))
    order = SarimaOrder(1, 1, 1, 0, 0, 1)

    class FakeResult:
        def get_forecast(self, steps):
            return type("Forecast", (), {"predicted_mean": np.arange(steps, dtype=float)})()

    original = validation.build_sarima_forecast_path(
        frame, "Germany", date(2024, 2, 15), order.specification_id, FakeResult()
    )
    altered = frame.copy()
    cutoff = pd.Timestamp(original["information_cutoff_utc"].iloc[0])
    altered.loc[altered["interval_end_utc"] > cutoff, "actual_grid_load_mwh"] = -999999.0
    changed = validation.build_sarima_forecast_path(
        altered, "Germany", date(2024, 2, 15), order.specification_id, FakeResult()
    )

    assert original["forecast_mwh"].tolist() == changed["forecast_mwh"].tolist()


@pytest.mark.parametrize(
    ("target_date", "expected_observations"),
    [
        (date(2024, 2, 15), 24),
        (date(2024, 3, 31), 23),
        (date(2024, 10, 27), 25),
    ],
)
def test_forecast_path_preserves_ordinary_and_dst_lengths(target_date, expected_observations):
    frame = _frame("2024-02-13 00:00", "2024-10-28 23:00")

    class FakeResult:
        def get_forecast(self, steps):
            return type("Forecast", (), {"predicted_mean": np.ones(steps)})()

    path = validation.build_sarima_forecast_path(
        frame,
        "Germany",
        target_date,
        "sarima_p1_d1_q1_P0_D0_Q1_s24",
        FakeResult(),
    )

    assert len(extract_target_day(frame, target_date)) == expected_observations
    assert int(path["is_target_day"].sum()) == expected_observations
    assert path["forecast_mwh"].notna().all()
    assert path.loc[~path["is_target_day"], "bridge_used"].all()


def test_missing_bridge_interval_fails_before_forecast_completion():
    frame = _frame("2024-02-13 00:00", "2024-02-15 23:00")
    origin = country_forecast_origin(date(2024, 2, 15), "Germany")
    frame = frame.loc[frame["timestamp_utc"].ne(origin + pd.Timedelta(hours=1))]

    class FakeResult:
        def get_forecast(self, steps):
            return type("Forecast", (), {"predicted_mean": np.ones(steps)})()

    with pytest.raises(ValueError, match="hourly continuity"):
        validation.build_sarima_forecast_path(
            frame,
            "Germany",
            date(2024, 2, 15),
            "sarima_p1_d1_q1_P0_D0_Q1_s24",
            FakeResult(),
        )


def test_target_day_metrics_ignore_bridge_rows():
    path = pd.DataFrame(
        {
            "actual_load_mwh": [999.0, 10.0],
            "forecast_mwh": [0.0, 10.0],
            "is_target_day": [False, True],
        }
    )

    metrics = evaluate_target_day(path, expected_observations=1)

    assert metrics["evaluated_observations"] == 1
    assert metrics["coverage"] == 1.0
    assert metrics["mae"] == 0.0


def test_execute_job_retains_residual_flags_as_diagnostics(monkeypatch):
    frame = _frame("2024-02-13 00:00", "2024-02-15 23:00")
    order = SarimaOrder(1, 1, 1, 0, 0, 1)
    fake_result = object()
    monkeypatch.setattr(
        validation,
        "fit_sarima",
        lambda values, requested_order: _fit_record(requested_order, fake_result),
    )
    monkeypatch.setattr(
        validation,
        "forecast_values",
        lambda result, steps: np.ones(steps),
    )

    result = validation.execute_job(
        {
            "country": "Germany",
            "target_date": "2024-02-15",
            "specification_id": order.specification_id,
            "p": 1,
            "d": 1,
            "q": 1,
            "P": 0,
            "D": 0,
            "Q": 1,
            "seasonal_period": 24,
            "trend": "n",
        },
        frame,
    )

    assert result["job"]["status"] == "completed"
    assert result["job"]["residual_rejected"] is True
    assert "diagnostic warning" in result["job"]["residual_diagnostic_warnings"]
    assert result["job"]["evaluated_observations"] == 24
    assert result["diagnostics"]["residual_adequacy_rejected"] is True
    assert "diagnostic warning" in result["diagnostics"]["residual_diagnostic_warnings"]


def test_failed_fit_records_zero_coverage_and_error_diagnostics(monkeypatch):
    frame = _frame("2024-02-13 00:00", "2024-02-15 23:00")
    order = SarimaOrder(1, 1, 1, 0, 0, 1)
    monkeypatch.setattr(
        validation,
        "fit_sarima",
        lambda values, requested_order: real_fit_sarima([], requested_order),
    )

    result = validation.execute_job(
        {
            "country": "Germany",
            "target_date": "2024-02-15",
            "specification_id": order.specification_id,
            "p": 1,
            "d": 1,
            "q": 1,
            "P": 0,
            "D": 0,
            "Q": 1,
            "seasonal_period": 24,
            "trend": "n",
        },
        frame,
    )

    assert result["job"]["status"] == "failed"
    assert result["job"]["coverage"] == 0.0
    assert result["job"]["error_message"]
    assert result["diagnostics"]["status"] == "failed"


@pytest.mark.parametrize("target_date", MINI_DATES)
def test_truncated_forecast_artifacts_are_not_marked_complete(target_date):
    frame = _frame("2024-02-13 00:00", "2024-10-28 23:00")
    job_key = ("Germany", target_date.isoformat(), "specification")
    jobs = pd.DataFrame(
        [{"country": job_key[0], "target_date": job_key[1], "specification_id": job_key[2], "status": "completed"}]
    )
    expected = {
        job_key: validation.expected_forecast_timestamps(frame, job_key[0], target_date)
    }
    forecasts = pd.DataFrame(
        [
            {
                "country": job_key[0],
                "target_date": job_key[1],
                "specification_id": job_key[2],
                "timestamp_utc": sorted(expected[job_key])[0],
            }
        ]
    )
    diagnostics = pd.DataFrame(
        [{"country": job_key[0], "target_date": job_key[1], "specification_id": job_key[2]}]
    )

    assert validation.load_completed_job_keys(
        jobs, forecasts, diagnostics, expected
    ) == set()


def test_key_only_forecast_artifacts_are_not_marked_complete():
    frame = _frame("2024-02-13 00:00", "2024-02-15 23:00")
    job_key = ("Germany", "2024-02-15", "specification")
    expected = {
        job_key: validation.expected_forecast_timestamps(
            frame, job_key[0], job_key[1]
        )
    }
    forecasts = pd.DataFrame(
        {
            "country": [job_key[0]] * len(expected[job_key]),
            "target_date": [job_key[1]] * len(expected[job_key]),
            "specification_id": [job_key[2]] * len(expected[job_key]),
            "timestamp_utc": sorted(expected[job_key]),
        }
    )
    jobs = pd.DataFrame(
        [{"country": job_key[0], "target_date": job_key[1], "specification_id": job_key[2], "status": "completed"}]
    )
    diagnostics = pd.DataFrame(
        [{"country": job_key[0], "target_date": job_key[1], "specification_id": job_key[2], "diagnostic_type": "sarima_fit"}]
    )

    assert validation.load_completed_job_keys(jobs, forecasts, diagnostics, expected) == set()


def test_summary_uses_only_completed_jobs_and_counts_actual_failures():
    frame = _frame("2024-02-13 00:00", "2024-02-15 23:00")
    job = {
        "country": "Germany",
        "target_date": "2024-02-15",
        "specification_id": "specification",
        "specification_order": 1,
        "expected_observations": 24,
    }
    manifest = pd.DataFrame([job])
    expected = {
        (job["country"], job["target_date"], job["specification_id"]): validation.expected_forecast_timestamps(
            frame, job["country"], job["target_date"]
        )
    }
    jobs = pd.DataFrame([{**job, "status": "failed"}])
    forecasts = pd.DataFrame(_forecast_rows(job, frame))
    diagnostics = pd.DataFrame([{**job, "diagnostic_type": "sarima_fit"}])

    summary = validation._summary_frame(
        jobs,
        manifest,
        forecasts,
        diagnostics=diagnostics,
        expected_forecast_timestamps=expected,
    )

    row = summary.loc[summary["country"].eq("Germany")].iloc[0]
    assert row["completed_jobs"] == 0
    assert row["failed_jobs"] == 1
    assert row["evaluated_observations"] == 0


def test_checkpoint_upsert_and_failed_retry(tmp_path, monkeypatch):
    frame = _frame("2024-02-13 00:00", "2024-10-28 23:00")
    shortlist = _shortlists()
    monkeypatch.setattr(
        validation, "load_validation_country_data", lambda country, directory: frame
    )
    calls = []

    def fake_execute(job, worker_frame):
        calls.append(job["specification_id"])
        return {
            "job": {
                **job,
                "status": "completed",
                "error_message": None,
                "evaluated_observations": 24,
                "coverage": 1.0,
            },
            "forecasts": _forecast_rows(job, worker_frame),
            "diagnostics": {**job, "diagnostic_type": "sarima_fit"},
        }

    monkeypatch.setattr(validation, "execute_job", fake_execute)
    first = validation.run_validation(
        shortlist,
        MINI_DATES,
        tmp_path,
        processed_directory=tmp_path,
    )
    second = validation.run_validation(
        shortlist,
        MINI_DATES,
        tmp_path,
        processed_directory=tmp_path,
    )

    assert first["completed_jobs"] == second["completed_jobs"] == 6
    assert len(calls) == 6
    jobs = pd.read_csv(tmp_path / validation.JOB_FILENAME)
    forecasts = pd.read_csv(tmp_path / validation.FORECAST_FILENAME)
    assert len(jobs) == 6
    assert len(forecasts) > len(jobs)
    assert not jobs.duplicated(list(validation.JOB_KEY_COLUMNS)).any()
    assert not forecasts.duplicated(list(validation.FORECAST_KEY_COLUMNS)).any()


def test_failed_job_is_retried_on_next_run(tmp_path, monkeypatch):
    shortlist = _shortlists()
    frame = _frame("2024-02-13 00:00", "2024-10-28 23:00")
    monkeypatch.setattr(
        validation, "load_validation_country_data", lambda country, directory: frame
    )
    attempts = iter(["failed"] * 6 + ["completed"] * 6)

    def fake_execute(job, worker_frame):
        status = next(attempts)
        return {
            "job": {
                **job,
                "status": status,
                "error_message": None if status == "completed" else "retry",
                "evaluated_observations": 24 if status == "completed" else 0,
                "coverage": 1.0 if status == "completed" else 0.0,
            },
            "forecasts": [] if status == "failed" else _forecast_rows(job, worker_frame),
            "diagnostics": {**job, "diagnostic_type": "sarima_fit"},
        }

    monkeypatch.setattr(validation, "execute_job", fake_execute)
    validation.run_validation(shortlist, MINI_DATES, tmp_path, processed_directory=tmp_path)
    result = validation.run_validation(
        shortlist, MINI_DATES, tmp_path, processed_directory=tmp_path
    )

    assert result["completed_jobs"] == 6
    assert set(pd.read_csv(tmp_path / validation.JOB_FILENAME)["status"]) == {"completed"}


def test_sequential_and_parallel_workers_use_deterministic_job_order(monkeypatch, tmp_path):
    shortlist = _shortlists()
    frame = _frame("2024-02-13 00:00", "2024-10-28 23:00")
    monkeypatch.setattr(
        validation, "load_validation_country_data", lambda country, directory: frame
    )
    monkeypatch.setattr(
        validation,
        "execute_job",
        lambda job, worker_frame: {
            "job": {**job, "status": "completed", "error_message": None},
            "forecasts": _forecast_rows(job, worker_frame),
            "diagnostics": {**job, "diagnostic_type": "sarima_fit"},
            },
        )

    class FakeExecutor:
        def __init__(self, max_workers, mp_context, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, jobs):
            return [function(job) for job in jobs]

    monkeypatch.setattr(validation, "ProcessPoolExecutor", FakeExecutor)

    sequential = validation.run_validation(
        shortlist, MINI_DATES, tmp_path / "sequential", processed_directory=tmp_path
    )
    parallel = validation.run_validation(
        shortlist, MINI_DATES, tmp_path / "parallel", processed_directory=tmp_path, workers=2
    )

    seq_jobs = pd.read_csv(tmp_path / "sequential" / validation.JOB_FILENAME)
    par_jobs = pd.read_csv(tmp_path / "parallel" / validation.JOB_FILENAME)
    assert seq_jobs[list(validation.JOB_KEY_COLUMNS)].to_dict("records") == par_jobs[
        list(validation.JOB_KEY_COLUMNS)
    ].to_dict("records")
    assert sequential["total_jobs"] == parallel["total_jobs"] == 6


def test_interruption_before_job_checkpoint_forces_artifact_retry(
    tmp_path, monkeypatch
):
    shortlist = _shortlists()
    frame = _frame("2024-02-13 00:00", "2024-10-28 23:00")
    monkeypatch.setattr(
        validation, "load_validation_country_data", lambda country, directory: frame
    )
    calls = []

    def fake_execute(job, worker_frame):
        calls.append(job["specification_id"] + ":" + job["target_date"])
        return {
            "job": {**job, "status": "completed", "error_message": None},
            "forecasts": _forecast_rows(job, worker_frame),
            "diagnostics": {
                **job,
                "diagnostic_type": "sarima_fit",
                "residual_diagnostic_warnings": "[]",
            },
        }

    monkeypatch.setattr(validation, "execute_job", fake_execute)
    original_atomic_write = validation._atomic_write
    interrupted = {"value": True}

    def interrupt_before_job(frame_to_write, path, columns):
        if interrupted["value"] and path.name == validation.DIAGNOSTIC_FILENAME:
            interrupted["value"] = False
            raise RuntimeError("simulated persistence interruption")
        return original_atomic_write(frame_to_write, path, columns)

    monkeypatch.setattr(validation, "_atomic_write", interrupt_before_job)
    with pytest.raises(RuntimeError, match="simulated persistence interruption"):
        validation.run_validation(
            shortlist, MINI_DATES, tmp_path, processed_directory=tmp_path
        )

    result = validation.run_validation(
        shortlist, MINI_DATES, tmp_path, processed_directory=tmp_path
    )

    assert result["completed_jobs"] == 6
    assert len(calls) == 7
    assert calls.count(calls[0]) == 2
    jobs = pd.read_csv(tmp_path / validation.JOB_FILENAME)
    forecasts = pd.read_csv(tmp_path / validation.FORECAST_FILENAME)
    diagnostics = pd.read_csv(tmp_path / validation.DIAGNOSTIC_FILENAME)
    expected_forecast_rows = sum(
        len(validation.expected_forecast_timestamps(frame, country, target_date))
        for country in validation.COUNTRY_CONFIG
        for target_date in MINI_DATES
    )
    assert len(jobs) == len(diagnostics) == 6
    assert len(forecasts) == expected_forecast_rows


def test_validation_loader_filters_2025_rows_before_preparation(tmp_path):
    raw = _frame("2024-02-13 00:00", "2024-02-15 23:00")
    sentinel = raw.iloc[[0]].copy()
    sentinel["timestamp_utc"] = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    sentinel["timestamp_local"] = "2025-01-01T01:00:00+01:00"
    sentinel["actual_grid_load_mwh"] = 999999999.0
    raw = pd.concat([raw, sentinel], ignore_index=True)
    path = tmp_path / validation.COUNTRY_CONFIG["Germany"]["filename"]
    raw.to_csv(path, index=False)

    loaded = validation.load_validation_country_data("Germany", tmp_path)

    assert not loaded["local_date"].astype(str).str.startswith("2025-").any()
    assert 999999999.0 not in loaded["actual_load_mwh"].tolist()
