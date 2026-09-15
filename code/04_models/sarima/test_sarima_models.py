import warnings

import numpy as np
import pandas as pd
import pytest

import sarima_models
from sarima_models import (
    ResidualDiagnostics,
    SarimaOrder,
    candidate_pool,
    fit_sarima,
    forecast_values,
    root_is_stable,
    specification_id,
    trend_for_d_D,
)


def test_sarima_order_has_machine_readable_specification_id():
    order = SarimaOrder(2, 1, 0, 1, 0, 1)

    assert specification_id(order) == "sarima_p2_d1_q0_P1_D0_Q1_s24"
    assert order.specification_id == "sarima_p2_d1_q0_P1_D0_Q1_s24"


@pytest.mark.parametrize(
    ("d", "D", "expected"),
    [(0, 0, "c"), (1, 0, "n"), (2, 0, "n"), (0, 1, "n"), (2, 1, "n")],
)
def test_deterministic_term_is_frozen_by_d_and_D(d, D, expected):
    assert trend_for_d_D(d, D) == expected


@pytest.mark.parametrize(
    "values",
    [
        (3, 0, 0, 0, 0, 0),
        (0, 3, 0, 0, 0, 0),
        (0, 0, 2, 2, 0, 0),
        (0, 0, 0, 0, 2, 0),
        (0, 0, 0, 0, 0, 2),
    ],
)
def test_sarima_order_rejects_values_outside_fixed_bounds(values):
    with pytest.raises(ValueError):
        SarimaOrder(*values)


def test_candidate_pool_is_exactly_the_eight_core_and_six_interactions():
    actual = [(order.p, order.q, order.P, order.Q) for order in candidate_pool(1, 0)]

    assert actual == [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 1, 0, 0),
        (2, 0, 0, 0),
        (0, 2, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 1, 0),
        (0, 1, 0, 1),
        (1, 1, 1, 0),
        (1, 1, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
    ]
    assert len(actual) == len(set(actual)) == 14
    assert all(order.d == 1 and order.D == 0 for order in candidate_pool(1, 0))


def test_root_stability_requires_modulus_strictly_above_threshold():
    assert root_is_stable([1.010001 + 0j])
    assert not root_is_stable([1.01 + 0j])
    assert not root_is_stable([np.nan + 0j])
    assert root_is_stable([np.inf + 0j])


class _FakeForecast:
    def __init__(self, values):
        self.predicted_mean = np.asarray(values, dtype=float)


class _FakeResult:
    def __init__(self, *, converged=True, roots=(1.2, 1.3), forecast=(2.0,)):
        self.mle_retvals = {
            "converged": converged,
            "success": converged,
            "status": 0 if converged else 1,
            "warnflag": 0 if converged else 1,
            "iterations": 3,
            "fcalls": 17,
            "gopt": np.array([3.0, 4.0]),
        }
        self.params = np.array([1.0])
        self.bse = np.array([0.1])
        self.param_names = ["sigma2"]
        self.nobs = 200
        self.llf = -100.0
        self.aic = 202.0
        self.bic = 205.0
        self.arroots = np.asarray(roots, dtype=complex)
        self.maroots = np.array([], dtype=complex)
        self.resid = np.zeros(200)
        self.loglikelihood_burn = 7
        self._forecast = forecast

    def get_forecast(self, steps):
        return _FakeForecast(self._forecast)


class _RetryModel:
    calls = []
    result_factory = staticmethod(lambda: _FakeResult())

    def __init__(self, values, **kwargs):
        self.values = values
        self.kwargs = kwargs

    def fit(self, maxiter=50, method_kwargs=None):
        self.__class__.calls.append(
            {} if maxiter == 50 else {"maxiter": maxiter}
        )
        return self.result_factory()


def test_fit_uses_sarimax_constraints_and_one_maxiter_retry(monkeypatch):
    class RetryModel(_RetryModel):
        result_factory = staticmethod(
            lambda: _FakeResult(converged=len(RetryModel.calls) > 1)
        )

    RetryModel.calls = []
    monkeypatch.setattr(sarima_models, "SARIMAX", RetryModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(1, 0, 0, 0, 0, 0))

    assert RetryModel.calls == [{}, {"maxiter": 1000}]
    assert record.optimizer_retry_count == 1
    assert record.converged
    assert record.eligible
    assert record.specification_id == "sarima_p1_d0_q0_P0_D0_Q0_s24"
    assert record.fitted_result is not None


def test_fit_accepts_bounded_forecast_options_without_standard_errors(monkeypatch):
    class ForecastModel:
        calls = []

        def __init__(self, values, **kwargs):
            self.values = values
            self.kwargs = kwargs

        def fit(self, **kwargs):
            self.calls.append(kwargs)
            result = _FakeResult()
            result.bse = np.array([np.nan])
            return result

    monkeypatch.setattr(sarima_models, "SARIMAX", ForecastModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(
        np.arange(200.0),
        SarimaOrder(1, 0, 0, 0, 0, 0),
        fit_kwargs={
            "method": "lbfgs",
            "maxiter": 50,
            "maxfun": 1000,
            "cov_type": "none",
            "low_memory": True,
        },
        require_standard_errors=False,
        retry_maxiter=None,
    )

    assert ForecastModel.calls == [
        {
            "method": "lbfgs",
            "maxiter": 50,
            "maxfun": 1000,
            "cov_type": "none",
            "low_memory": True,
        }
    ]
    assert record.optimizer_iterations == 3
    assert record.optimizer_function_calls == 17
    assert record.optimizer_gradient_norm == 5.0
    assert not record.standard_errors_finite
    assert record.eligible


def test_fit_records_bounded_nonconvergence_without_retry(monkeypatch):
    class BoundedModel:
        calls = []

        def __init__(self, values, **kwargs):
            self.values = values
            self.kwargs = kwargs

        def fit(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeResult(converged=False)

    monkeypatch.setattr(sarima_models, "SARIMAX", BoundedModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(
        np.arange(200.0),
        SarimaOrder(1, 0, 0, 0, 0, 0),
        fit_kwargs={"method": "lbfgs", "maxiter": 50, "maxfun": 1000},
        require_standard_errors=False,
        retry_maxiter=None,
    )

    assert len(BoundedModel.calls) == 1
    assert record.optimizer_retry_count == 0
    assert not record.converged
    assert not record.eligible


def test_fit_forwards_optimizer_progress_without_changing_fit_options(monkeypatch):
    events = []

    class ProgressModel:
        def __init__(self, values, **kwargs):
            self.values = values
            self.kwargs = kwargs

        def fit(self, **kwargs):
            kwargs["callback"](np.array([2.0]))
            return _FakeResult()

    monkeypatch.setattr(sarima_models, "SARIMAX", ProgressModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(
        np.arange(200.0),
        SarimaOrder(0, 0, 0, 0, 0, 0),
        fit_kwargs={"method": "lbfgs", "maxiter": 10, "maxfun": 20},
        progress_callback=lambda model, params: events.append((model, params.copy())),
        retry_maxiter=None,
    )

    assert len(events) == 1
    assert isinstance(events[0][0], ProgressModel)
    assert np.array_equal(events[0][1], [2.0])
    assert record.eligible


def test_fit_filters_persisted_parameters_without_reoptimizing(monkeypatch):
    class FixedModel:
        filter_calls = []
        fit_calls = []

        def __init__(self, values, **kwargs):
            self.values = values
            self.kwargs = kwargs

        def filter(self, params, **kwargs):
            self.filter_calls.append((params, kwargs))
            return _FakeResult()

        def fit(self, **kwargs):
            self.fit_calls.append(kwargs)
            return _FakeResult(converged=False)

    monkeypatch.setattr(sarima_models, "SARIMAX", FixedModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(
        np.arange(200.0),
        SarimaOrder(0, 0, 0, 0, 0, 0),
        fixed_params=np.array([1.0]),
        require_standard_errors=False,
        retry_maxiter=None,
    )

    assert len(FixedModel.filter_calls) == 1
    assert FixedModel.fit_calls == []
    assert record.converged
    assert record.optimizer_iterations == 0
    assert record.optimizer_function_calls == 0
    assert record.eligible


def test_retry_passes_maxiter_to_installed_sarimax_api(monkeypatch):
    class DirectRetryModel:
        calls = []

        def __init__(self, values, **kwargs):
            self.values = values
            self.kwargs = kwargs

        def fit(self, maxiter=50, **kwargs):
            self.calls.append((maxiter, kwargs))
            return _FakeResult(converged=maxiter == 1000)

    monkeypatch.setattr(sarima_models, "SARIMAX", DirectRetryModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(0, 0, 0, 0, 0, 0))

    assert DirectRetryModel.calls == [(50, {}), (1000, {})]
    assert record.optimizer_retry_count == 1
    assert record.eligible


def test_fit_normalizes_missing_optimizer_success_and_status(monkeypatch):
    result = _FakeResult()
    result.mle_retvals = {"converged": True, "warnflag": 0}

    class IncompleteRetvalsModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", IncompleteRetvalsModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(0, 0, 0, 0, 0, 0))

    assert record.optimizer_success is True
    assert record.optimizer_status == 0
    assert record.optimizer_warnflag == 0


def test_fit_records_warnings_and_hard_final_status(monkeypatch):
    class WarningModel(_RetryModel):
        def fit(self, method_kwargs=None):
            with warnings.catch_warnings():
                warnings.warn("covariance matrix is singular", UserWarning)
            result = _FakeResult()
            result.mle_retvals["warnflag"] = 2
            return result

    monkeypatch.setattr(sarima_models, "SARIMAX", WarningModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(0, 0, 0, 0, 0, 0))

    assert record.warning_messages == (
        "UserWarning: covariance matrix is singular",
    )
    assert not record.eligible
    assert "warnflag" in record.error_message


def test_component_roots_are_reconstructed_and_not_relabelled_as_combined(monkeypatch):
    result = _FakeResult(roots=(1.5,))
    result.params = np.array([0.5, 0.2, 0.1, 1.0])
    result.bse = np.array([0.1, 0.1, 0.1, 0.1])
    result.param_names = ["ar.L1", "ma.L1", "ar.S.L24", "sigma2"]
    result.maroots = np.array([1.6 + 0j])

    class ComponentModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", ComponentModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(1, 0, 1, 1, 0, 0))

    assert record.ar_root_minimum == pytest.approx(1.5)
    assert record.ma_root_minimum == pytest.approx(1.6)
    assert record.root_diagnostics.combined_ar_roots == (1.5 + 0j,)
    assert record.root_diagnostics.reconstructed_nonseasonal_ar_roots[0] == pytest.approx(2.0)
    assert record.root_diagnostics.reconstructed_seasonal_ar_roots[0] == pytest.approx(10.0)
    assert record.root_diagnostics.reconstructed_component_method.startswith("numpy.roots")
    assert record.eligible


def test_invalid_reconstructed_component_root_rejects_fit(monkeypatch):
    result = _FakeResult(roots=(1.5,))
    result.params = np.array([1.2, 1.0])
    result.bse = np.array([0.1, 0.1])
    result.param_names = ["ar.L1", "sigma2"]

    class ComponentModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", ComponentModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(1, 0, 0, 0, 0, 0))

    assert record.ar_root_minimum == pytest.approx(1.5)
    assert not record.stationarity_ok
    assert not record.eligible
    assert "reconstructed" in record.error_message


def test_missing_combined_roots_reject_positive_order_fit(monkeypatch):
    result = _FakeResult(roots=())
    result.param_names = ["ar.L1", "sigma2"]
    result.params = np.array([0.2, 1.0])
    result.bse = np.array([0.1, 0.1])

    class MissingRootModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", MissingRootModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(1, 0, 0, 0, 0, 0))

    assert not record.stationarity_ok
    assert not record.eligible
    assert "combined AR roots" in record.error_message


def test_nonfinite_parameters_produce_failed_record_not_root_exception(monkeypatch):
    result = _FakeResult()
    result.params = np.array([np.nan])
    result.bse = np.array([0.1])
    result.param_names = ["sigma2"]

    class NonfiniteModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", NonfiniteModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(0, 0, 0, 0, 0, 0))

    assert not record.parameters_finite
    assert not record.eligible
    assert "non-finite parameter estimates" in record.error_message


def test_nonfinite_parameters_skip_root_property_access(monkeypatch):
    class BrokenRootResult(_FakeResult):
        def __init__(self):
            super().__init__()
            self.params = np.array([np.nan])
            self.bse = np.array([0.1])
            self.param_names = ["ar.L1"]

        @property
        def arroots(self):
            raise np.linalg.LinAlgError("invalid root polynomial")

        @arroots.setter
        def arroots(self, value):
            self._arroots = value

    result = BrokenRootResult()

    class BrokenRootModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", BrokenRootModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(1, 0, 0, 0, 0, 0))

    assert not record.parameters_finite
    assert not record.eligible
    assert "non-finite parameter estimates" in record.error_message


def test_residual_burn_in_uses_larger_state_space_and_conservative_value(monkeypatch):
    captured = {}
    original_diagnostics = sarima_models._residual_diagnostics

    def diagnostics(result, order):
        captured["result"] = result
        captured["order"] = order
        return original_diagnostics(result, order)

    result = _FakeResult()
    result.resid = np.random.default_rng(1).normal(size=250)
    result.loglikelihood_burn = 100

    class BurnModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", BurnModel)
    monkeypatch.setattr(sarima_models, "_residual_diagnostics", diagnostics)

    record = fit_sarima(np.arange(250.0), SarimaOrder(2, 1, 2, 1, 1, 1))

    assert captured["result"] is result
    assert record.residual_diagnostics.burn_in == 100
    assert record.residual_diagnostics.conservative_burn_in == 77
    assert record.residual_diagnostics.n_effective == 150
    assert set(record.residual_diagnostics.acf_values) == set(range(1, 49))
    assert set(record.residual_diagnostics.ljung_box_pvalues) == {24, 48}


def test_residual_autocorrelation_is_diagnostic_only_but_training_flag_is_retained(
    monkeypatch,
):
    diagnostic = ResidualDiagnostics(
        burn_in=48,
        state_space_loglikelihood_burn=0,
        conservative_burn_in=48,
        n_effective=100,
        acf_values={1: 0.1, 2: 0.1},
        acf_threshold=0.05,
        flagged_acf_lags=(1, 2),
        ljung_box_statistics={24: 10.0, 48: 20.0},
        ljung_box_pvalues={24: 0.001, 48: 0.001},
        training_adequacy_rejected=True,
        training_adequacy_reason="joint Ljung-Box and residual ACF rule",
    )
    monkeypatch.setattr(sarima_models, "_residual_diagnostics", lambda result, order: diagnostic)

    class DiagnosticModel(_RetryModel):
        result_factory = staticmethod(lambda: _FakeResult())

    monkeypatch.setattr(sarima_models, "SARIMAX", DiagnosticModel)

    record = fit_sarima(np.arange(200.0), SarimaOrder(0, 0, 0, 0, 0, 0))

    assert record.residual_diagnostics is diagnostic
    assert record.training_adequacy_rejected
    assert record.error_message is None
    assert record.eligible


def test_aicc_is_finite_and_uses_fitted_parameter_count(monkeypatch):
    result = _FakeResult()
    result.params = np.array([1.0, 2.0, 3.0])
    result.bse = np.array([0.1, 0.1, 0.1])
    result.nobs = 100
    result.nobs_effective = 90
    result.aic = 50.0

    class AicModel(_RetryModel):
        result_factory = staticmethod(lambda: result)

    monkeypatch.setattr(sarima_models, "SARIMAX", AicModel)
    monkeypatch.setattr(
        sarima_models,
        "_residual_diagnostics",
        lambda result, order: ResidualDiagnostics.empty(100, 48),
    )

    record = fit_sarima(np.arange(200.0), SarimaOrder(0, 0, 0, 0, 0, 0))

    assert record.n_params == 3
    assert record.aicc == pytest.approx(50.0 + (2 * 3 * 4) / 86)
    assert np.isfinite(record.aicc)


def test_nonfinite_residual_diagnostics_are_recorded_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        sarima_models,
        "acf",
        lambda *args, **kwargs: np.full(49, np.nan),
    )
    monkeypatch.setattr(
        sarima_models,
        "acorr_ljungbox",
        lambda *args, **kwargs: pd.DataFrame(
            {"lb_stat": [np.nan, np.nan], "lb_pvalue": [np.nan, np.nan]},
            index=[24, 48],
        ),
    )
    result = _FakeResult()
    result.resid = np.random.default_rng(3).normal(size=200)

    diagnostics = sarima_models._residual_diagnostics(
        result,
        SarimaOrder(0, 0, 0, 0, 0, 0),
    )

    assert diagnostics.training_adequacy_rejected
    assert "non-finite" in diagnostics.training_adequacy_reason


def test_residual_diagnostics_retain_diagnostic_warnings(monkeypatch):
    original_acf = sarima_models.acf

    def warned_acf(*args, **kwargs):
        warnings.warn("diagnostic warning", UserWarning)
        return original_acf(*args, **kwargs)

    monkeypatch.setattr(sarima_models, "acf", warned_acf)
    result = _FakeResult()
    result.resid = np.random.default_rng(2).normal(size=200)

    diagnostics = sarima_models._residual_diagnostics(
        result,
        SarimaOrder(0, 0, 0, 0, 0, 0),
    )

    assert diagnostics.warnings == ("UserWarning: diagnostic warning",)


def test_forecast_values_rejects_exception_shape_and_nonfinite_output():
    class BrokenResult:
        def __init__(self, values):
            self.values = values

        def get_forecast(self, steps):
            if isinstance(self.values, Exception):
                raise self.values
            return _FakeForecast(self.values)

    with pytest.raises(ValueError, match="unusable"):
        forecast_values(BrokenResult([1.0, 2.0]), 1)
    with pytest.raises(ValueError, match="unusable"):
        forecast_values(BrokenResult([np.nan]), 1)
    with pytest.raises(ValueError, match="unusable"):
        forecast_values(BrokenResult([[1.0]]), 1)
    with pytest.raises(ValueError, match="failed"):
        forecast_values(BrokenResult(RuntimeError("boom")), 1)
