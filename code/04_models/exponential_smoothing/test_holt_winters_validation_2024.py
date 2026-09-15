from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from holt_winters_models import build_specification_grid
from holt_winters_validation_2024 import (
    JOB_KEY_COLUMNS,
    ValidationResult,
    ValidationJob,
    ValidationStore,
    build_summary,
    build_validation_jobs,
    execute_jobs,
    run_validation,
    select_final_specifications,
)


def _shortlists(countries=("Germany", "Austria"), specifications=None):
    specifications = specifications or build_specification_grid()[:2]
    rows = []
    for country in countries:
        for order, specification in enumerate(specifications):
            rows.append(
                {
                    "country": country,
                    "specification_id": specification.specification_id,
                    "seasonal_form": specification.seasonal_form,
                    "seasonal_periods": specification.seasonal_periods,
                    "damped_trend": specification.damped_trend,
                    "specification_order": order,
                    "optimizer_used": "L-BFGS-B",
                    "optimizer_attempts_json": json.dumps(
                        [
                            {
                                "optimizer": "L-BFGS-B",
                                "minimize_kwargs": {},
                                "success": True,
                                "parameters_finite": True,
                            }
                        ]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _job_row(job, status="completed", **overrides):
    row = {
        "country": job.country,
        "target_date": job.target_date,
        "specification_id": job.specification_id,
        "specification_order": job.specification_order,
        "status": status,
        "mae": 1.0,
        "rmse": 1.0,
        "mape": 1.0,
        "evaluated_observations": 24,
        "expected_observations": 24,
        "coverage": 1.0,
        "error_message": None,
    }
    row.update(overrides)
    return row


def _forecast_rows(job, values=(1.0, 3.0), target_flags=(True, True)):
    return pd.DataFrame(
        {
            "country": job.country,
            "model_family": "Holt-Winters",
            "specification_id": job.specification_id,
            "target_date": job.target_date,
            "timestamp_utc": [
                f"{job.target_date}T00:00:00+00:00",
                f"{job.target_date}T01:00:00+00:00",
            ],
            "actual_load_mwh": [1.0, 1.0],
            "forecast_mwh": list(values),
            "is_target_day": list(target_flags),
        }
    )


def _deterministic_worker(job):
    return (job.job_key, job.specification_order + len(job.target_date))


def _prepared_day_frame():
    utc = pd.date_range("2024-01-14 23:00", periods=24, freq="h", tz="UTC")
    local = utc.tz_convert("Europe/Berlin")
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_load_mwh": np.arange(24, dtype=float) + 100,
            "actual_grid_load_mwh": np.arange(24, dtype=float) + 100,
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "interval_end_utc": utc + pd.Timedelta(hours=1),
            "local_date": local.strftime("%Y-%m-%d"),
            "local_occurrence": 0,
        }
    )


def test_build_validation_jobs_is_stable_and_rejects_duplicate_keys():
    shortlist = _shortlists()

    jobs = build_validation_jobs(shortlist, [date(2024, 3, 31), date(2024, 1, 15)])

    assert len(jobs) == 8
    assert [job.country for job in jobs[:4]] == ["Germany"] * 4
    assert [job.target_date for job in jobs[:2]] == ["2024-01-15"] * 2
    assert len({job.job_key for job in jobs}) == len(jobs)

    duplicate = pd.concat([shortlist, shortlist.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate validation job"):
        build_validation_jobs(duplicate, [date(2024, 1, 15)])


def test_checkpoint_store_skips_only_completed_jobs_and_upserts_without_duplicates(tmp_path):
    jobs = build_validation_jobs(_shortlists(countries=("Germany",)), [date(2024, 1, 15)])
    completed, failed = jobs
    store = ValidationStore(tmp_path)

    store.persist_result(_job_row(completed), _forecast_rows(completed))
    store.persist_result(
        _job_row(failed, status="failed", error_message="temporary failure"),
        pd.DataFrame(),
    )
    store.persist_result(
        _job_row(completed, mae=2.0, rmse=2.0),
        _forecast_rows(completed, values=(2.0, 4.0)),
    )

    assert store.completed_keys() == {completed.job_key}
    saved_jobs = store.read_jobs()
    saved_forecasts = store.read_forecasts()
    assert len(saved_jobs) == 2
    assert len(saved_forecasts) == 2
    assert saved_jobs.loc[
        saved_jobs["specification_id"].eq(completed.specification_id), "mae"
    ].item() == 2.0


def test_checkpoint_store_recovers_a_completed_job_with_missing_forecast_rows(tmp_path):
    job = build_validation_jobs(_shortlists(countries=("Germany",)), [date(2024, 1, 15)])[0]
    store = ValidationStore(tmp_path)
    pd.DataFrame([_job_row(job)]).to_csv(store.jobs_path, index=False)

    assert store.completed_keys() == set()


def test_aggregate_uses_only_target_rows_and_selection_tie_breakers():
    jobs = build_validation_jobs(_shortlists(countries=("Germany",)), [date(2024, 1, 15)])
    rows = []
    forecasts = []
    for job in jobs:
        rows.append(_job_row(job, expected_observations=4))
        forecasts.append(_forecast_rows(job, values=(1.0, 3.0,))[0:2])
    job_frame = pd.DataFrame(rows)
    forecast_frame = pd.concat(forecasts, ignore_index=True)
    forecast_frame.loc[0, "is_target_day"] = False
    forecast_frame.loc[0, "forecast_mwh"] = 1000.0

    summary = build_summary(
        jobs,
        job_frame,
        forecast_frame,
        expected_observations_by_country={"Germany": 4},
    )

    assert summary["evaluated_observations"].tolist() == [1, 2]
    assert summary["coverage"].tolist() == [0.25, 0.5]
    assert summary["mae"].notna().all()

    selection_input = pd.DataFrame(
        [
            {
                "country": "Germany",
                "specification_id": "rmse_favored",
                "specification_order": 1,
                "coverage": 1.0,
                "rmse": 0.1,
                "mae": 0.6,
                "mape": 1.0,
            },
            {
                "country": "Germany",
                "specification_id": "mae_favored",
                "specification_order": 2,
                "coverage": 1.0,
                "mae": 0.5,
                "rmse": 2.0,
                "mape": 5.0,
            },
            {
                "country": "Austria",
                "specification_id": "incomplete",
                "specification_order": 0,
                "coverage": 0.5,
                "rmse": 0.1,
                "mae": 0.1,
                "mape": 0.1,
            },
        ]
    )

    selected = select_final_specifications(selection_input, minimum_coverage=1.0)

    assert (
        selected.set_index("country").loc["Germany", "specification_id"]
        == "mae_favored"
    )
    assert "Austria" not in set(selected["country"])

    tie_input = pd.DataFrame(
        [
            {
                "country": "Germany",
                "specification_id": "later",
                "specification_order": 1,
                "coverage": 1.0,
                "rmse": 1.0,
                "mae": 0.5,
                "mape": 2.0,
            },
            {
                "country": "Germany",
                "specification_id": "earlier",
                "specification_order": 0,
                "coverage": 1.0,
                "rmse": 1.0,
                "mae": 0.5,
                "mape": 2.0,
            },
        ]
    )
    tie_selected = select_final_specifications(tie_input)
    assert tie_selected.iloc[0]["specification_id"] == "earlier"


def test_sequential_and_parallel_execution_have_identical_sorted_results():
    jobs = build_validation_jobs(_shortlists(countries=("Germany",)), [date(2024, 1, 15), date(2024, 1, 16)])

    sequential = execute_jobs(jobs, workers=1, worker_function=_deterministic_worker)
    parallel = execute_jobs(jobs, workers=2, worker_function=_deterministic_worker)

    assert sorted(sequential, key=lambda item: item[0]) == sorted(
        parallel, key=lambda item: item[0]
    )


def test_run_validation_retries_failed_and_missing_jobs_without_repeating_completed(
    tmp_path, monkeypatch
):
    shortlist = _shortlists()
    frame = _prepared_day_frame()
    monkeypatch.setattr(
        "holt_winters_validation_2024.load_country_data",
        lambda country, processed_directory: frame,
    )
    monkeypatch.setattr(
        "holt_winters_validation_2024.load_frozen_shortlists",
        lambda path: shortlist,
    )
    manifest = build_validation_jobs(shortlist, [date(2024, 1, 15)])
    store = ValidationStore(tmp_path)
    store.persist_result(_job_row(manifest[0]), _forecast_rows(manifest[0]))
    store.persist_result(
        _job_row(manifest[1], status="failed", error_message="retry me"),
        pd.DataFrame(),
    )
    pending_keys = []

    def fake_iter_job_results(jobs, **kwargs):
        pending_keys.extend(job.job_key for job in jobs)
        for job in jobs:
            yield ValidationResult(_job_row(job), _forecast_rows(job))

    monkeypatch.setattr(
        "holt_winters_validation_2024.iter_job_results", fake_iter_job_results
    )
    monkeypatch.setattr(
        "holt_winters_validation_2024.select_final_specifications",
        lambda summary, minimum_coverage: pd.DataFrame(
            {
                "country": ["Germany", "Austria"],
                "specification_id": ["germany_final", "austria_final"],
            }
        ),
    )

    result = run_validation(
        processed_directory=tmp_path,
        shortlist_path=tmp_path / "shortlists.csv",
        output_directory=tmp_path,
        workers=2,
        target_dates=[date(2024, 1, 15)],
    )

    assert len(pending_keys) == len(manifest) - 1
    assert manifest[0].job_key not in pending_keys
    assert result["completed_jobs"] == len(manifest)
    saved_jobs = pd.read_csv(store.jobs_path)
    saved_forecasts = pd.read_csv(store.forecasts_path)
    assert len(saved_jobs) == len(manifest)
    assert len(saved_forecasts) == len(manifest) * 2
    assert not saved_jobs.duplicated(list(JOB_KEY_COLUMNS)).any()
    assert not store.selection_path.exists()
