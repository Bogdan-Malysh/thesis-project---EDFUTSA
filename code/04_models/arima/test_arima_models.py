import numpy as np
import pytest

import arima_models
from arima_models import (
    ResidualDiagnostics,
    classify_warning,
    fit_arima,
    order_id,
    residual_acf_threshold,
    residual_rejection,
    root_is_stable,
    trend_for_d,
)


@pytest.mark.parametrize(
    ("d", "expected"),
    [(0, "c"), (1, "n"), (2, "n")],
)
def test_deterministic_term_is_frozen_by_d(d, expected):
    assert trend_for_d(d) == expected


def test_order_id_preserves_nonseasonal_order():
    assert order_id((2, 1, 0)) == "arima_p2_d1_q0"


def test_root_stability_uses_strict_one_percent_unit_circle_margin():
    assert root_is_stable(np.array([1.010001 + 0j]))
    assert not root_is_stable(np.array([1.01 + 0j]))
    assert root_is_stable(np.array([], dtype=complex))


def test_residual_threshold_has_practical_magnitude_floor():
    assert residual_acf_threshold(10_000) == pytest.approx(0.05)
    assert residual_acf_threshold(10) == pytest.approx(2.576 / np.sqrt(10))


def test_one_significant_ljung_box_result_does_not_reject():
    rejected, threshold, flagged = residual_rejection(
        {24: 0.001, 48: 0.2},
        {1: 0.10, 2: 0.08},
        10_000,
    )

    assert not rejected
    assert threshold == pytest.approx(0.05)
    assert flagged == [1, 2]


def test_joint_ljung_box_and_acf_rule_rejects_residual_structure():
    rejected, _, flagged = residual_rejection(
        {24: 0.001, 48: 0.009},
        {1: 0.06, 2: -0.07, 3: 0.01},
        10_000,
    )

    assert rejected
    assert flagged == [1, 2]


def test_residual_diagnostic_flag_does_not_make_fit_ineligible(monkeypatch):
    diagnostic = ResidualDiagnostics(
        n_effective=100,
        acf_values={1: 0.1, 2: 0.1},
        acf_threshold=0.05,
        flagged_acf_lags=(1, 2),
        ljung_box_statistics={24: 10.0, 48: 20.0},
        ljung_box_pvalues={24: 0.001, 48: 0.001},
        rejected=True,
        rejection_reason="diagnostic adequacy flag",
    )
    monkeypatch.setattr(arima_models, "_residual_diagnostics", lambda result, order: diagnostic)

    record = fit_arima(np.random.default_rng(0).normal(size=200), (0, 0, 0))

    assert record.residual_diagnostics is diagnostic
    assert record.error_message is None
    assert record.eligible


def test_nonconverged_fit_uses_one_recorded_maxiter_retry(monkeypatch):
    class FakeResult:
        def __init__(self, converged):
            self.mle_retvals = {
                "converged": converged,
                "success": converged,
                "warnflag": 0 if converged else 1,
            }
            self.params = np.array([1.0])
            self.bse = np.array([0.1])
            self.nobs = 200
            self.llf = -100.0
            self.aic = 202.0
            self.bic = 205.0
            self.arroots = np.array([], dtype=complex)
            self.maroots = np.array([], dtype=complex)
            self.resid = np.zeros(200)

    class FakeModel:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, method_kwargs=None):
            self.calls.append(method_kwargs or {})
            return FakeResult(converged=len(self.calls) > 1)

    diagnostic = ResidualDiagnostics(
        n_effective=100,
        acf_values={},
        acf_threshold=0.05,
        flagged_acf_lags=(),
        ljung_box_statistics={24: 0.0, 48: 0.0},
        ljung_box_pvalues={24: 1.0, 48: 1.0},
        rejected=False,
        rejection_reason=None,
    )
    monkeypatch.setattr(arima_models, "ARIMA", FakeModel)
    monkeypatch.setattr(arima_models, "_residual_diagnostics", lambda result, order: diagnostic)

    record = fit_arima(np.arange(200.0), (0, 0, 0))

    assert FakeModel.calls == [{}, {"maxiter": 1000}]
    assert record.optimizer_retry_count == 1
    assert record.converged
    assert record.eligible


@pytest.mark.parametrize(
    ("category", "message", "hard_rejection"),
    [
        ("ConvergenceWarning", "failed to converge", True),
        ("RuntimeWarning", "invalid value encountered", True),
        ("RuntimeWarning", "Hessian inversion failed", False),
        ("UserWarning", "singular covariance matrix", False),
    ],
)
def test_hessian_and_covariance_warnings_are_diagnostic_only(
    category, message, hard_rejection
):
    actual, _ = classify_warning(category, message)
    assert actual is hard_rejection
