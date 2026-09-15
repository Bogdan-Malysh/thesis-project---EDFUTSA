import numpy as np
import pytest

import arima_screening
from arima_screening import (
    build_candidate_orders,
    narrow_shortlist,
    screening_row_is_model_valid,
    select_d,
    stationarity_diagnostics,
)


def test_stationarity_tests_use_frozen_adf_and_kpss_configuration(monkeypatch):
    adf_calls = []
    kpss_calls = []

    def fake_adfuller(values, **kwargs):
        adf_calls.append(kwargs)
        return (-5.0, 0.01, 3, len(values), {}, 0.0)

    def fake_kpss(values, **kwargs):
        kpss_calls.append(kwargs)
        return (0.1, 0.1, 5, {"1%": 0.7, "5%": 0.46, "10%": 0.35})

    monkeypatch.setattr(arima_screening, "adfuller", fake_adfuller)
    monkeypatch.setattr(arima_screening, "kpss", fake_kpss)

    diagnostics = stationarity_diagnostics(np.arange(120.0))

    assert len(diagnostics) == 3
    assert adf_calls[0] == {"regression": "ct", "maxlag": 24, "autolag": "AIC"}
    assert kpss_calls[0] == {"regression": "ct", "nlags": "auto"}
    assert all(call == {"regression": "c", "maxlag": 24, "autolag": "AIC"} for call in adf_calls[1:])
    assert all(call == {"regression": "c", "nlags": "auto"} for call in kpss_calls[1:])


def test_select_d_chooses_smallest_d_supported_by_both_tests():
    diagnostics = [
        {"d": 0, "adf_pvalue": 0.20, "kpss_pvalue": 0.01},
        {"d": 1, "adf_pvalue": 0.01, "kpss_pvalue": 0.10},
        {"d": 2, "adf_pvalue": 0.01, "kpss_pvalue": 0.10},
    ]

    assert select_d(diagnostics) == 1


def test_select_d_returns_no_order_when_tests_have_no_common_decision():
    diagnostics = [
        {"d": 0, "adf_pvalue": 0.20, "kpss_pvalue": 0.01},
        {"d": 1, "adf_pvalue": 0.20, "kpss_pvalue": 0.10},
        {"d": 2, "adf_pvalue": 0.20, "kpss_pvalue": 0.10},
    ]

    assert select_d(diagnostics) is None


def test_fixed_candidate_set_has_no_high_orders():
    assert build_candidate_orders(1) == (
        (0, 1, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 0),
        (0, 1, 2),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 0),
        (0, 1, 3),
    )
    assert all(order[0] <= 3 and order[2] <= 3 for order in build_candidate_orders(1))


def test_shortlist_is_union_of_top_two_each_information_criterion():
    rows = [
        {"specification_id": "a", "eligible": True, "aic": 1, "aicc": 4, "bic": 6, "specification_order": 1},
        {"specification_id": "b", "eligible": True, "aic": 2, "aicc": 1, "bic": 5, "specification_order": 2},
        {"specification_id": "c", "eligible": True, "aic": 3, "aicc": 2, "bic": 1, "specification_order": 3},
        {"specification_id": "d", "eligible": True, "aic": 4, "aicc": 3, "bic": 2, "specification_order": 4},
        {"specification_id": "e", "eligible": False, "aic": 0, "aicc": 0, "bic": 0, "specification_order": 5},
    ]

    selected = narrow_shortlist(rows)

    assert [row["specification_id"] for row in selected] == ["a", "b", "c", "d"]


def test_shortlist_is_empty_when_all_candidates_are_rejected():
    rows = [
        {
            "specification_id": "a",
            "eligible": False,
            "aic": 1,
            "aicc": 1,
            "bic": 1,
            "specification_order": 1,
        }
    ]

    assert narrow_shortlist(rows) == []


def test_residual_adequacy_flag_does_not_change_model_validity():
    row = {
        "fit_status": "success",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "residual_rejected": True,
        "error_message": "joint Ljung-Box and residual ACF rule",
    }

    assert screening_row_is_model_valid(row)
