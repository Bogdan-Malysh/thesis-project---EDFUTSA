from __future__ import annotations

import json

import pandas as pd

from arima_intervention_report_2024 import (
    build_metric_comparison,
    summarize_coefficients,
    summarize_jobs,
)


def test_metric_comparison_reports_absolute_and_relative_changes() -> None:
    base = pd.DataFrame(
        [
            {
                "country": "Germany",
                "specification_id": "arima_p2_d1_q2",
                "mae": 100.0,
                "rmse": 200.0,
                "mape": 10.0,
                "coverage": 1.0,
                "evaluated_observations": 8784,
                "completed_jobs": 366,
                "failed_jobs": 0,
            }
        ]
    )
    crisis = pd.DataFrame(
        [
            {
                "country": "Germany",
                "specification_id": "arima_p2_d1_q2_crisis",
                "mae": 90.0,
                "rmse": 220.0,
                "mape": 8.5,
                "coverage": 1.0,
                "completed_jobs": 366,
                "failed_jobs": 0,
                "evaluated_observations": 8784,
            }
        ]
    )

    result = build_metric_comparison(base, crisis).iloc[0]

    assert result["mae_change_mwh"] == -10.0
    assert result["mae_change_pct"] == -10.0
    assert result["rmse_change_mwh"] == 20.0
    assert result["rmse_change_pct"] == 10.0
    assert result["mape_change_pp"] == -1.5


def test_job_summary_counts_convergence_retries_and_failures() -> None:
    jobs = pd.DataFrame(
        [
            {
                "country": "Austria",
                "status": "completed",
                "converged": True,
                "optimizer_retry_count": 1,
                "warning_messages": json.dumps(["warning"]),
                "expected_observations": 24,
                "evaluated_observations": 24,
                "coverage": 1.0,
            },
            {
                "country": "Austria",
                "status": "failed",
                "converged": False,
                "optimizer_retry_count": 0,
                "warning_messages": "[]",
                "expected_observations": 24,
                "evaluated_observations": 0,
                "coverage": 0.0,
            },
        ]
    )

    result = summarize_jobs(jobs).loc[lambda frame: frame["country"].eq("Austria")].iloc[0]

    assert result["total_jobs"] == 2
    assert result["completed_jobs"] == 1
    assert result["failed_jobs"] == 1
    assert result["converged_jobs"] == 1
    assert result["optimizer_retry_jobs"] == 1
    assert result["warning_jobs"] == 1


def test_coefficient_summary_reads_crisis_parameter_names() -> None:
    jobs = pd.DataFrame(
        [
            {
                "country": "Germany",
                "parameter_estimates": json.dumps({"is_covid_period": -4.0, "is_post_invasion": -2.0}),
                "standard_errors": json.dumps({"is_covid_period": 2.0, "is_post_invasion": 1.0}),
            }
        ]
    )

    result = summarize_coefficients(jobs)

    assert set(result["coefficient"]) == {"is_covid_period", "is_post_invasion"}
    assert result.loc[result["coefficient"].eq("is_covid_period"), "mean_estimate"].iloc[0] == -4.0
    assert result.loc[result["coefficient"].eq("is_post_invasion"), "absolute_t_ge_1_96_jobs"].iloc[0] == 1
