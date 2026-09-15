from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import COUNTRY_CONFIG
from enhanced_arima_models import enhanced_specifications
from enhanced_arima_benchmark import (
    build_benchmark_workload,
    run_worker_benchmark,
    select_fastest_safe_worker,
)
import enhanced_arima_benchmark as benchmark_module
import enhanced_arima_screening_2024 as screening_module


def _shortlist() -> pd.DataFrame:
    rows = []
    specifications = enhanced_specifications()
    for country in COUNTRY_CONFIG:
        for branch, count in (("ordinary", 3), ("weekly_differenced", 2)):
            candidates = [
                specification
                for specification in specifications
                if specification.branch == branch
            ][:count]
            for specification_order, specification in enumerate(candidates, 1):
                rows.append(
                    {
                        "country": country,
                        "branch": specification.branch,
                        "specification_id": specification.specification_id,
                        "specification_order": specification_order,
                        "shortlist_rank": specification_order,
                        "p": specification.order[0],
                        "d": specification.order[1],
                        "q": specification.order[2],
                        "trend": specification.trend,
                        "transform": specification.transform,
                    }
                )
    return pd.DataFrame(rows)


def _forecast_rows(workload: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for job in workload.to_dict("records"):
        timezone = COUNTRY_CONFIG[job["country"]]["timezone"]
        path = screening_module._expected_forecast_path_index(
            job["country"], job["target_date"]
        )
        local_path = path.tz_convert(timezone)
        target = np.asarray(local_path.strftime("%Y-%m-%d")) == job["target_date"]
        occurrences = pd.DataFrame(
            {"local_date": local_path.date, "hour": local_path.hour}
        ).groupby(["local_date", "hour"], sort=False).cumcount()
        source_utc = [
            pd.Timestamp(
                f"{timestamp.date() - pd.Timedelta(days=7)} {timestamp.hour:02d}:00:00"
            )
            .tz_localize(timezone, ambiguous=int(occurrence) == 0)
            .tz_convert("UTC")
            for timestamp, occurrence in zip(local_path, occurrences)
        ]
        weekly = job["branch"] == "weekly_differenced"
        for timestamp, local_timestamp, source_timestamp, is_target in zip(
            path, local_path, source_utc, target
        ):
            rows.append(
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
                    "interval_end_utc": timestamp + pd.Timedelta(hours=1),
                    "timestamp_local": local_timestamp.isoformat(),
                    "local_date": local_timestamp.date().isoformat(),
                    "forecast_mwh": 100.0,
                    "actual_load_mwh": 100.0,
                    "bridge_used": not is_target,
                    "is_target_day": is_target,
                    "evaluated": is_target,
                    "status": "completed",
                    "source_timestamp_utc": source_timestamp if weekly else pd.NaT,
                    "source_actual_mwh": 99.0 if weekly else np.nan,
                    "source_kind": "observed" if weekly else "arima_forecast",
                    "mapping_issue_count": 0,
                }
            )
    return pd.DataFrame(rows)


def _source_frames(forecasts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        country: pd.DataFrame(
            {
                "timestamp_utc": group["source_timestamp_utc"].dropna().unique(),
                "actual_load_mwh": 99.0,
            }
        )
        for country in COUNTRY_CONFIG
        for group in [
            forecasts.loc[
                forecasts["country"].eq(country)
                & forecasts["branch"].eq("weekly_differenced")
            ]
        ]
    }


def _run_benchmark_with_defect(tmp_path, monkeypatch, defect: str) -> pd.DataFrame:
    shortlist = _shortlist()
    workload = build_benchmark_workload(
        shortlist, pd.DataFrame({"country": [], "target_date": [], "reason": []})
    )
    valid_forecasts = _forecast_rows(workload)
    frames = _source_frames(valid_forecasts)
    monkeypatch.setattr(benchmark_module, "_load_benchmark_frames", lambda _: frames)
    monkeypatch.setattr(
        benchmark_module,
        "_derive_invalid_dates",
        lambda _: pd.DataFrame(columns=["country", "target_date", "reason"]),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_worker_count_support",
        lambda workers: (workers == 1, "only one worker in test"),
    )

    def fake_run(workload, workers, source_frames, processed_directory):
        jobs = workload.assign(status="completed", error_message="")
        forecasts = valid_forecasts.copy()
        if defect == "extra_bridge":
            extra = forecasts.loc[
                forecasts["branch"].eq("ordinary")
                & ~forecasts["is_target_day"]
            ].iloc[[0]].copy()
            extra["timestamp_utc"] = pd.Timestamp(
                extra.iloc[0]["timestamp_utc"]
            ) + pd.Timedelta(minutes=30)
            forecasts = pd.concat([forecasts, extra], ignore_index=True)
        elif defect == "source_actual":
            index = forecasts.index[forecasts["branch"].eq("weekly_differenced")][0]
            forecasts.loc[index, "source_actual_mwh"] = 98.0
        elif defect == "missing_status":
            jobs = workload.copy()
        return {
            "jobs": jobs,
            "forecasts": forecasts,
            "active_workers": 1,
            "peak_rss_bytes": 1000,
            "peak_swap_bytes": 0,
        }

    monkeypatch.setattr(benchmark_module, "_run_worker_count", fake_run)
    return run_worker_benchmark(shortlist, tmp_path, processed_directory=tmp_path)


def _single_benchmark_job(branch: str = "ordinary") -> pd.DataFrame:
    invalid_dates = pd.DataFrame(
        {"country": [], "target_date": [], "reason": []}
    )
    workload = build_benchmark_workload(_shortlist(), invalid_dates)
    return workload.loc[workload["branch"].eq(branch)].iloc[[0]].reset_index(drop=True)


def _completed_benchmark_run(branch: str = "ordinary") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workload = _single_benchmark_job(branch)
    jobs = workload.assign(status="completed", error_message="")
    return workload, jobs, _forecast_rows(workload)


def test_benchmark_workload_contains_two_candidates_per_branch_and_four_valid_dates():
    invalid_dates = pd.DataFrame(
        {
            "country": ["Germany", "Austria"],
            "target_date": ["2024-11-03", "2024-11-03"],
            "reason": ["ambiguous_source", "ambiguous_source"],
        }
    )

    workload = build_benchmark_workload(_shortlist(), invalid_dates)

    assert len(workload) == 32
    assert set(workload["country"]) == set(COUNTRY_CONFIG)
    assert workload["target_date"].nunique() == 4
    assert "2024-11-03" not in set(workload["target_date"])
    assert workload.groupby(["country", "branch"]).size().to_dict() == {
        (country, branch): 8
        for country in COUNTRY_CONFIG
        for branch in ("ordinary", "weekly_differenced")
    }
    assert not workload.duplicated(
        ["country", "target_date", "branch", "specification_id"]
    ).any()
    assert not workload["weekly_invalid"].astype(bool).any()
    assert workload["target_date"].astype(str).str.startswith("2024-").all()


def test_runner_reuses_persisted_workload_and_records_telemetry_and_support_status(
    tmp_path, monkeypatch
):
    shortlist = _shortlist()
    fixture_workload = build_benchmark_workload(
        shortlist,
        pd.DataFrame({"country": [], "target_date": [], "reason": []}),
    )
    frames = _source_frames(_forecast_rows(fixture_workload))
    captured_workloads = []

    monkeypatch.setattr(benchmark_module, "_load_benchmark_frames", lambda _: frames)
    monkeypatch.setattr(
        benchmark_module,
        "_derive_invalid_dates",
        lambda _: pd.DataFrame(columns=["country", "target_date", "reason"]),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_worker_count_support",
        lambda workers: (workers <= 2, "" if workers <= 2 else "logical CPU capacity"),
    )

    def fake_run(workload, workers, source_frames, processed_directory):
        captured_workloads.append(workload.copy())
        jobs = workload.copy()
        jobs["status"] = "completed"
        jobs["error_message"] = ""
        forecasts = _forecast_rows(workload)
        return {
            "jobs": jobs,
            "forecasts": forecasts,
            "peak_rss_bytes": 1000 + workers,
            "peak_swap_bytes": 0,
        }

    monkeypatch.setattr(benchmark_module, "_run_worker_count", fake_run)

    summary = run_worker_benchmark(shortlist, tmp_path, processed_directory=tmp_path)

    assert len(summary) == 8
    assert set(summary["requested_workers"]) == set(range(1, 9))
    assert summary.loc[summary["requested_workers"].le(2), "supported"].all()
    assert not summary.loc[summary["requested_workers"].gt(2), "supported"].any()
    assert summary.loc[summary["requested_workers"].le(2), "safe"].all()
    assert summary.loc[summary["requested_workers"].gt(2), "status"].eq(
        "unsupported"
    ).all()
    for column in (
        "active_workers",
        "elapsed_seconds",
        "jobs_per_minute",
        "failures",
        "peak_rss_bytes",
        "peak_swap_bytes",
        "max_absolute_forecast_difference",
        "max_relative_forecast_difference",
        "equivalence",
        "safe",
    ):
        assert column in summary
    assert summary.loc[summary["requested_workers"].eq(2), "peak_rss_bytes"].item() == 1002

    assert len(captured_workloads) == 2
    pd.testing.assert_frame_equal(captured_workloads[0], captured_workloads[1])
    persisted = pd.read_csv(tmp_path / "benchmark" / "workload.csv")
    pd.testing.assert_frame_equal(
        captured_workloads[0].reset_index(drop=True), persisted
    )
    assert (tmp_path / "benchmark" / "workers_1" / "result.csv").exists()
    assert (tmp_path / "benchmark" / "workers_2" / "result.csv").exists()
    assert (tmp_path / "benchmark" / "worker_benchmark_summary.csv").exists()
    assert (tmp_path / "benchmark" / "worker_selection.csv").exists()
    audit = pd.read_csv(tmp_path / "audit" / "enhanced_arima_run_manifest.csv")
    assert int(audit.loc[0, "selected_worker"]) in {1, 2}


def test_update_run_manifest_handles_existing_blank_selected_worker(tmp_path):
    audit_path = tmp_path / "audit" / "enhanced_arima_run_manifest.csv"
    audit_path.parent.mkdir(parents=True)
    pd.DataFrame({"selected_worker": [""]}).to_csv(audit_path, index=False)

    benchmark_module._update_run_manifest(tmp_path, 4)

    audit = pd.read_csv(audit_path)
    assert int(audit.loc[0, "selected_worker"]) == 4


def test_proc_status_keeps_parent_pid_in_process_id_units():
    status = benchmark_module._read_proc_status(os.getpid())

    assert status is not None
    assert status[0] == os.getppid()


def test_process_tree_metrics_counts_spawn_workers_not_resource_tracker(monkeypatch):
    statuses = {
        100: (0, 10, 0),
        101: (100, 20, 0),
        102: (100, 30, 0),
    }
    commands = {
        100: "main",
        101: "from multiprocessing.resource_tracker import main",
        102: "from multiprocessing.spawn import spawn_main",
    }
    monkeypatch.setattr(
        benchmark_module.os,
        "listdir",
        lambda path: [str(pid) for pid in statuses] if path == "/proc" else [],
    )
    monkeypatch.setattr(
        benchmark_module,
        "_read_proc_status",
        lambda pid: statuses[pid],
    )
    monkeypatch.setattr(
        benchmark_module,
        "_process_command",
        lambda pid: commands[pid],
    )

    _, _, active_workers = benchmark_module._process_tree_metrics(100)

    assert active_workers == 1


def test_fastest_safe_worker_ignores_unsafe_and_unsupported_rows():
    summary = pd.DataFrame(
        {
            "requested_workers": [1, 2, 3, 4],
            "supported": [True, True, False, True],
            "safe": [True, False, False, True],
            "elapsed_seconds": [10.0, 1.0, 0.1, 20.0],
        }
    )

    assert select_fastest_safe_worker(summary) == 1


def test_fastest_safe_worker_rejects_missing_safe_supported_count():
    summary = pd.DataFrame(
        {
            "requested_workers": [1],
            "supported": [True],
            "safe": [False],
            "elapsed_seconds": [10.0],
        }
    )

    with np.testing.assert_raises(ValueError):
        select_fastest_safe_worker(summary)


def test_validate_run_rejects_forecast_rows_for_foreign_jobs():
    workload, jobs, forecasts = _completed_benchmark_run()
    foreign = forecasts.iloc[[0]].copy()
    foreign["country"] = "Austria"

    validation = benchmark_module._validate_run(
        workload, jobs, pd.concat([forecasts, foreign], ignore_index=True)
    )

    assert not validation["valid"]


def test_validate_run_rejects_incorrect_target_timestamps():
    workload, jobs, forecasts = _completed_benchmark_run()
    malformed = forecasts.copy()
    malformed.loc[0, "timestamp_utc"] = (
        pd.Timestamp(malformed.loc[0, "timestamp_utc"]) + pd.Timedelta(hours=1)
    )

    validation = benchmark_module._validate_run(workload, jobs, malformed)

    assert not validation["valid"]


def test_validate_run_rejects_completed_jobs_with_nonempty_errors():
    workload, jobs, forecasts = _completed_benchmark_run()
    jobs.loc[0, "error_message"] = "stale error"

    validation = benchmark_module._validate_run(workload, jobs, forecasts)

    assert not validation["valid"]


def test_validate_run_rejects_weekly_source_actuals_that_do_not_match_prepared_rows():
    workload, jobs, forecasts = _completed_benchmark_run("weekly_differenced")
    source_timestamps = pd.to_datetime(forecasts["source_timestamp_utc"], utc=True)
    source_frames = {
        country: pd.DataFrame(columns=["timestamp_utc", "actual_load_mwh"])
        for country in COUNTRY_CONFIG
    }
    source_frames["Germany"] = pd.DataFrame(
        {
            "timestamp_utc": source_timestamps,
            "actual_load_mwh": 98.0,
        }
    )

    validation = benchmark_module._validate_run(
        workload, jobs, forecasts, source_frames=source_frames
    )

    assert not validation["valid"]


def test_worker_count_one_uses_spawn_and_reports_measured_active_workers(
    tmp_path, monkeypatch
):
    workload = _single_benchmark_job()
    calls = []

    class FakeMonitor:
        peak_active_workers = 1

        def __init__(self, root_pid):
            pass

        def start(self):
            pass

        def stop(self):
            return 1000, 0

    class FakeExecutor:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            calls.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            return False

        def map(self, function, jobs):
            return [function(job) for job in jobs]

    monkeypatch.setattr(benchmark_module, "_ResourceMonitor", FakeMonitor)
    monkeypatch.setattr(benchmark_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        benchmark_module,
        "_worker_execute",
        lambda job: {"job": job, "forecasts": [], "diagnostics": []},
    )

    result = benchmark_module._run_worker_count(
        workload, 1, {country: pd.DataFrame() for country in COUNTRY_CONFIG}, tmp_path
    )

    assert calls == [1]
    assert result["active_workers"] == 1


def test_summary_uses_measured_active_workers_instead_of_requested_count():
    workload, jobs, forecasts = _completed_benchmark_run()

    row = benchmark_module._summary_row(
        2,
        {
            "jobs": jobs,
            "forecasts": forecasts,
            "active_workers": 1,
            "peak_rss_bytes": 1000,
            "peak_swap_bytes": 0,
        },
        workload,
        None,
        1.0,
        True,
        "",
    )

    assert row["active_workers"] == 1


def test_summary_marks_swap_telemetry_as_resource_exhaustion():
    workload, jobs, forecasts = _completed_benchmark_run()

    row = benchmark_module._summary_row(
        1,
        {
            "jobs": jobs,
            "forecasts": forecasts,
            "active_workers": 1,
            "peak_rss_bytes": 1000,
            "peak_swap_bytes": 1,
        },
        workload,
        None,
        1.0,
        True,
        "",
    )

    assert not row["safe"]
    assert "resource exhaustion" in row["reason"]


def test_shortlist_rejects_supplied_specification_order_mismatch():
    shortlist = _shortlist()
    shortlist.loc[0, "specification_order"] = 999

    with pytest.raises(ValueError, match="specification order"):
        build_benchmark_workload(
            shortlist,
            pd.DataFrame({"country": [], "target_date": [], "reason": []}),
        )


def test_benchmark_accepts_screening_produced_branch_local_specification_orders():
    workload = build_benchmark_workload(
        _shortlist(),
        pd.DataFrame({"country": [], "target_date": [], "reason": []}),
    )

    assert set(
        workload.loc[workload["branch"].eq("ordinary"), "specification_order"]
    ) == {1, 2}
    assert set(
        workload.loc[workload["branch"].eq("weekly_differenced"), "specification_order"]
    ) == {1, 2}


def test_benchmark_runner_rejects_extra_bridge_rows_for_existing_job_keys(
    tmp_path, monkeypatch
):
    summary = _run_benchmark_with_defect(tmp_path, monkeypatch, "extra_bridge")

    row = summary.loc[summary["requested_workers"].eq(1)].iloc[0]
    assert not row["safe"]


def test_benchmark_runner_propagates_source_frames_into_weekly_validation(
    tmp_path, monkeypatch
):
    summary = _run_benchmark_with_defect(tmp_path, monkeypatch, "source_actual")

    row = summary.loc[summary["requested_workers"].eq(1)].iloc[0]
    assert not row["safe"]


def test_benchmark_runner_rejects_jobs_without_status(
    tmp_path, monkeypatch
):
    summary = _run_benchmark_with_defect(tmp_path, monkeypatch, "missing_status")

    row = summary.loc[summary["requested_workers"].eq(1)].iloc[0]
    assert not row["safe"]
