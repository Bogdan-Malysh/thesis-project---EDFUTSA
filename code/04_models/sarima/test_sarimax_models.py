from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

import benchmark_sarimax_fit_2024 as benchmark
import sarimax_models
from benchmark_sarimax_fit_2024 import benchmark_row
from sarimax_models import (
    COUNTRY_ORDERS,
    SARIMA_INTERVENTION_SPECIFICATION_IDS,
    SARIMAX_SPECIFICATION_IDS,
    build_exog,
    fit_sarimax,
    order_for_country,
    specification_columns,
)


def test_sarimax_specifications_reuse_calendar_and_crisis_columns() -> None:
    calendar = specification_columns("sarimax_calendar")
    crisis = specification_columns("sarimax_calendar_crisis")

    assert SARIMAX_SPECIFICATION_IDS == (
        "sarimax_calendar",
        "sarimax_calendar_crisis",
    )
    assert crisis[: len(calendar)] == calendar
    assert crisis[-2:] == ("is_covid_period", "is_post_invasion")
    assert "is_covid_period" not in calendar
    assert "is_post_invasion" not in calendar


def test_sarima_intervention_specifications_use_crisis_columns_only() -> None:
    assert SARIMA_INTERVENTION_SPECIFICATION_IDS == {
        "Germany": "sarima_p2_d0_q0_P0_D1_Q0_s24_crisis",
        "Austria": "sarima_p1_d0_q1_P1_D1_Q0_s24_crisis",
    }
    for specification_id in SARIMA_INTERVENTION_SPECIFICATION_IDS.values():
        assert specification_columns(specification_id) == (
            "is_covid_period",
            "is_post_invasion",
        )


def test_sarimax_orders_are_the_selected_country_specific_sarima_orders() -> None:
    assert COUNTRY_ORDERS["Germany"].nonseasonal_order == (2, 0, 0)
    assert COUNTRY_ORDERS["Germany"].seasonal_order == (0, 1, 0, 24)
    assert COUNTRY_ORDERS["Austria"].nonseasonal_order == (1, 0, 1)
    assert COUNTRY_ORDERS["Austria"].seasonal_order == (1, 1, 0, 24)
    assert order_for_country("Germany") == COUNTRY_ORDERS["Germany"]


def test_sarimax_exog_is_finite_unique_and_full_rank() -> None:
    timestamps = pd.date_range(
        "2020-01-01 00:00:00+00:00",
        "2024-12-31 23:00:00+00:00",
        freq="h",
    )

    for specification_id in SARIMAX_SPECIFICATION_IDS:
        exog = build_exog(timestamps, "Germany", specification_id)
        assert not exog.columns.duplicated().any()
        assert np.isfinite(exog.to_numpy()).all()
        assert np.linalg.matrix_rank(exog.to_numpy()) == len(exog.columns)


def test_forecast_fit_uses_bounded_likelihood_without_inference(monkeypatch) -> None:
    timestamps = pd.date_range(
        "2020-01-01 00:00:00+00:00",
        periods=200,
        freq="h",
    )
    exog = build_exog(timestamps, "Germany", "sarimax_calendar")
    calls = []

    def fake_fit(values, order, exog=None, **kwargs):
        calls.append((values, order, exog, kwargs))
        return object()

    monkeypatch.setattr(sarimax_models, "fit_sarima", fake_fit)

    record = fit_sarimax(
        np.arange(len(exog), dtype=float),
        exog,
        COUNTRY_ORDERS["Germany"],
    )

    assert record.sarima_fit is not None
    assert sarimax_models.SARIMAX_FORECAST_FIT_KWARGS["maxfun"] == 1000
    assert sarimax_models.SARIMAX_BENCHMARK_FIT_KWARGS["maxfun"] == 3000
    assert calls[0][3] == {
        "fit_kwargs": sarimax_models.SARIMAX_FORECAST_FIT_KWARGS,
        "require_standard_errors": False,
        "retry_maxiter": None,
        "validate_exog": False,
    }


def test_reporting_fit_keeps_full_inference_defaults(monkeypatch) -> None:
    timestamps = pd.date_range(
        "2020-01-01 00:00:00+00:00",
        periods=200,
        freq="h",
    )
    exog = build_exog(timestamps, "Germany", "sarimax_calendar")
    calls = []

    def fake_fit(values, order, exog=None, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(sarimax_models, "fit_sarima", fake_fit)

    fit_sarimax(
        np.arange(len(exog), dtype=float),
        exog,
        COUNTRY_ORDERS["Germany"],
        forecast_only=False,
    )

    assert calls == [
        {
            "fit_kwargs": None,
            "require_standard_errors": True,
            "retry_maxiter": 1000,
            "validate_exog": False,
        }
    ]


def test_benchmark_row_contains_required_fit_metrics() -> None:
    fit = SimpleNamespace(
        sarima_fit=SimpleNamespace(
            n_params=44,
            converged=True,
            log_likelihood=-100.0,
            standard_errors_finite=False,
            parameters_finite=True,
            stationarity_ok=True,
            invertibility_ok=True,
            optimizer_status=None,
            optimizer_warnflag=None,
            optimizer_message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
            optimizer_gradient_norm=0.0001,
            optimizer_iterations=4,
            optimizer_function_calls=125,
        ),
        fitted_result=SimpleNamespace(
            cov_type="none",
            model=SimpleNamespace(ssm=SimpleNamespace(k_states=26)),
        ),
    )

    row = benchmark_row(
        fit,
        observations=35058,
        fit_seconds=12.5,
        rss_start_bytes=100,
        rss_finish_bytes=200,
    )

    assert row == {
        "observations": 35058,
        "parameters": 44,
        "state_dimension": 26,
        "optimizer_method": "lbfgs",
        "function_evaluations": 125,
        "iterations": 4,
        "converged": True,
        "termination_message": "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
        "parameters_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "gradient_norm": 0.0001,
        "maxiter_limit_reached": False,
        "maxfun_limit_reached": False,
        "optimizer_status": None,
        "optimizer_warnflag": None,
        "log_likelihood": -100.0,
        "fit_seconds": 12.5,
        "rss_start_bytes": 100,
        "rss_finish_bytes": 200,
        "covariance_skipped": True,
    }


def test_working_set_bytes_supports_linux_without_windows_api(monkeypatch) -> None:
    monkeypatch.delattr(benchmark.ctypes, "windll", raising=False)

    working_set = benchmark.working_set_bytes()

    assert working_set > 0
