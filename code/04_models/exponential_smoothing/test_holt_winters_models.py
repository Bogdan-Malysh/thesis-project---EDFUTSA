from datetime import date
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import holt_winters_models as hwm
from holt_winters_models import (
    HoltWintersSpecification,
    FitRecord,
    HoltWintersForecastError,
    ShortlistingError,
    build_specification_grid,
    forecast_holt_winters,
    fit_specification,
    select_shortlists,
)


def _hourly_frame(
    start: str,
    end: str,
    timezone_name: str = "Europe/Berlin",
    positive: bool = True,
) -> pd.DataFrame:
    utc = pd.date_range(start, end, freq="h", tz="UTC")
    local = utc.tz_convert(timezone_name)
    values = np.arange(len(utc), dtype=float) + (100 if positive else -100)
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "actual_grid_load_mwh": values,
            "temperature_c": 10.0,
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_public_holiday": 0,
        }
    )


def _screening_rows(
    *,
    aicc_by_spec: dict[str, float],
    aic_by_spec: dict[str, float] | None = None,
    eligible_by_spec: dict[str, bool] | None = None,
) -> pd.DataFrame:
    rows = []
    for order, specification in enumerate(build_specification_grid()):
        aicc = aicc_by_spec.get(specification.specification_id, np.nan)
        aic = (aic_by_spec or {}).get(specification.specification_id, 100.0)
        eligible = (eligible_by_spec or {}).get(specification.specification_id, True)
        if eligible and np.isfinite(aicc):
            criterion, value = "AICc", aicc
        elif eligible and np.isfinite(aic):
            criterion, value = "AIC", aic
        else:
            criterion, value = None, np.nan
        rows.append(
            {
                "country": "Germany",
                "specification_id": specification.specification_id,
                "seasonal_form": specification.seasonal_form,
                "seasonal_periods": specification.seasonal_periods,
                "damped_trend": specification.damped_trend,
                "specification_order": order,
                "aic": aic,
                "aicc": aicc,
                "bic": 100.0,
                "convergence_status": "converged",
                "fit_status": "success" if eligible else "failed",
                "selection_criterion": criterion,
                "selection_value": value,
                "eligible": eligible and criterion is not None,
                "error_message": None if eligible else "forced failure",
            }
        )
    return pd.DataFrame(rows)


def test_build_specification_grid_contains_all_eight_specifications_in_stable_order():
    specifications = build_specification_grid()

    assert len(specifications) == 8
    assert [spec.specification_id for spec in specifications] == [
        "hw_add_s24_undamped",
        "hw_add_s24_damped",
        "hw_add_s168_undamped",
        "hw_add_s168_damped",
        "hw_mul_s24_undamped",
        "hw_mul_s24_damped",
        "hw_mul_s168_undamped",
        "hw_mul_s168_damped",
    ]
    assert {
        (spec.seasonal_form, spec.seasonal_periods, spec.damped_trend)
        for spec in specifications
    } == {
        ("add", 24, False),
        ("add", 24, True),
        ("add", 168, False),
        ("add", 168, True),
        ("mul", 24, False),
        ("mul", 24, True),
        ("mul", 168, False),
        ("mul", 168, True),
    }


def test_shortlisting_selects_lowest_criterion_and_sole_eligible_candidate():
    specifications = build_specification_grid()
    aicc = {spec.specification_id: 100.0 + order for order, spec in enumerate(specifications)}
    aicc[specifications[1].specification_id] = 1.0
    aicc[specifications[3].specification_id] = np.nan
    eligible = {spec.specification_id: True for spec in specifications}
    eligible[specifications[2].specification_id] = False
    rows = _screening_rows(aicc_by_spec=aicc, eligible_by_spec=eligible)

    shortlist = select_shortlists(rows)

    assert len(shortlist) == 4
    assert shortlist.loc[
        (shortlist["seasonal_form"] == "add")
        & (shortlist["seasonal_periods"] == 24),
        "specification_id",
    ].item() == specifications[1].specification_id
    assert shortlist.loc[
        (shortlist["seasonal_form"] == "add")
        & (shortlist["seasonal_periods"] == 168),
        "specification_id",
    ].item() == specifications[3].specification_id
    assert shortlist.loc[
        shortlist["specification_id"] == specifications[3].specification_id,
        "selection_criterion",
    ].item() == "AIC"


def test_shortlisting_tie_breaks_by_specification_order():
    specifications = build_specification_grid()
    aicc = {spec.specification_id: 10.0 for spec in specifications}
    rows = _screening_rows(aicc_by_spec=aicc)

    shortlist = select_shortlists(rows)

    assert shortlist["specification_id"].tolist() == [
        specifications[0].specification_id,
        specifications[2].specification_id,
        specifications[4].specification_id,
        specifications[6].specification_id,
    ]


def test_shortlisting_sorts_numeric_selection_values_not_numeric_strings():
    specifications = build_specification_grid()
    rows = _screening_rows(
        aicc_by_spec={spec.specification_id: 10.0 for spec in specifications}
    )
    rows["selection_value"] = rows["selection_value"].astype(object)
    rows.loc[
        rows["specification_id"] == specifications[0].specification_id,
        "selection_value",
    ] = "10"
    rows.loc[
        rows["specification_id"] == specifications[1].specification_id,
        "selection_value",
    ] = "2"

    shortlist = select_shortlists(rows)

    assert shortlist.loc[
        (shortlist["seasonal_form"] == "add")
        & (shortlist["seasonal_periods"] == 24),
        "specification_id",
    ].item() == specifications[1].specification_id


def test_shortlisting_does_not_treat_string_false_as_eligible():
    specifications = build_specification_grid()
    rows = _screening_rows(
        aicc_by_spec={spec.specification_id: 10.0 for spec in specifications}
    )
    rows["eligible"] = rows["eligible"].astype(object)
    group = rows["specification_id"].isin(
        [specifications[0].specification_id, specifications[1].specification_id]
    )
    rows.loc[group, "eligible"] = ["False", False]

    with pytest.raises(ShortlistingError):
        select_shortlists(rows)


def test_shortlisting_reports_both_failures_when_a_group_has_no_eligible_candidate():
    specifications = build_specification_grid()
    aicc = {spec.specification_id: 10.0 for spec in specifications}
    eligible = {spec.specification_id: True for spec in specifications}
    eligible[specifications[0].specification_id] = False
    eligible[specifications[1].specification_id] = False
    rows = _screening_rows(aicc_by_spec=aicc, eligible_by_spec=eligible)

    with pytest.raises(ShortlistingError) as error:
        select_shortlists(rows)

    message = str(error.value)
    assert specifications[0].specification_id in message
    assert specifications[1].specification_id in message
    assert "forced failure" in message


class _FakeFit:
    aic = 10.0
    aicc = 11.0
    bic = 12.0
    params_formatted = pd.DataFrame({"param": [1.0, 2.0]})

    def __init__(self, mle_retvals):
        self.mle_retvals = mle_retvals

    def forecast(self, count):
        return np.full(count, 123.0)


class _FakeModel:
    def __init__(self, fit_result):
        self.fit_result = fit_result

    def fit(self, **kwargs):
        return self.fit_result


@pytest.mark.parametrize(
    ("mle_retvals", "expected_status", "expected_eligible"),
    [
        ({}, "unknown", True),
        ({"success": False}, "failed", False),
        ({"converged": False}, "failed", False),
        ({"converged": True}, "converged", True),
    ],
)
def test_fit_metadata_handles_unknown_and_explicit_optimizer_failure(
    monkeypatch, mle_retvals, expected_status, expected_eligible
):
    fake_fit = _FakeFit(mle_retvals)
    monkeypatch.setattr(
        hwm,
        "ExponentialSmoothing",
        lambda *args, **kwargs: _FakeModel(fake_fit),
    )

    record = fit_specification(
        np.arange(100, dtype=float) + 100,
        build_specification_grid()[0],
    )

    assert record.convergence_status == expected_status
    assert record.fit_status == ("success" if expected_eligible else "failed")
    assert record.eligible is expected_eligible
    assert record.optimizer_attempts[-1].convergence_status == expected_status
    assert record.optimizer_attempts[-1].success is (
        None if expected_status == "unknown" else expected_eligible
    )


def test_iteration_limit_uses_fixed_optimizer_fallback_order_and_records_attempts(
    monkeypatch,
):
    fits = [
        _FakeFit({
            "message": "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
            "status": 1,
            "success": False,
            "nit": 69,
            "nfev": 15138,
        }),
        _FakeFit({
            "message": "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
            "status": 1,
            "success": False,
            "nit": 146,
            "nfev": 30102,
        }),
        _FakeFit({
            "message": "`xtol` termination condition is satisfied.",
            "status": 3,
            "success": True,
            "nfev": 14,
            "njev": 9,
        }),
    ]
    fit_calls = []

    class _SequenceModel:
        def fit(self, **kwargs):
            fit_calls.append(kwargs)
            return fits.pop(0)

    monkeypatch.setattr(hwm, "ExponentialSmoothing", lambda *args, **kwargs: _SequenceModel())

    record = fit_specification(
        np.arange(100, dtype=float) + 100,
        next(spec for spec in build_specification_grid() if spec.seasonal_form == "mul"),
    )

    assert [attempt.optimizer for attempt in record.optimizer_attempts] == [
        "L-BFGS-B",
        "L-BFGS-B",
        "least_squares",
    ]
    assert fit_calls[0]["method"] == "L-BFGS-B"
    assert fit_calls[1]["minimize_kwargs"] == {
        "options": {"maxfun": 30000, "maxiter": 2000}
    }
    assert fit_calls[2]["method"] == "least_squares"
    assert fit_calls[2]["minimize_kwargs"] == {"max_nfev": 2000}
    assert record.optimizer_used == "least_squares"
    assert record.eligible is True
    assert all(attempt.parameters_finite for attempt in record.optimizer_attempts)
    assert all(attempt.criteria_finite for attempt in record.optimizer_attempts)
    serialized = hwm.fit_record_to_row(
        record, "Germany", pd.Timestamp("2024-01-01 00:00", tz="UTC")
    )
    assert len(json.loads(serialized["optimizer_attempts_json"])) == 3


def test_non_iteration_optimizer_failure_does_not_trigger_fallback(monkeypatch):
    fit_calls = []

    class _FailedModel:
        def fit(self, **kwargs):
            fit_calls.append(kwargs)
            return _FakeFit(
                {
                    "message": "ABNORMAL_TERMINATION_IN_LNSRCH",
                    "status": 2,
                    "success": False,
                }
            )

    monkeypatch.setattr(hwm, "ExponentialSmoothing", lambda *args, **kwargs: _FailedModel())

    record = fit_specification(np.arange(100, dtype=float) + 100, build_specification_grid()[0])

    assert len(record.optimizer_attempts) == 1
    assert fit_calls[0]["method"] == "L-BFGS-B"
    assert record.eligible is False


def test_frozen_optimizer_runs_first_and_skips_duplicate_fallback_configuration(
    monkeypatch,
):
    fits = [
        _FakeFit({"message": "frozen failure", "status": 2, "success": False}),
        _FakeFit({"message": "retry failure", "status": 1, "success": False}),
        _FakeFit({"message": "ok", "status": 0, "success": True}),
    ]
    fit_calls = []

    class _SequenceModel:
        def fit(self, **kwargs):
            fit_calls.append(kwargs)
            return fits.pop(0)

    monkeypatch.setattr(hwm, "ExponentialSmoothing", lambda *args, **kwargs: _SequenceModel())

    record = fit_specification(
        np.arange(100, dtype=float) + 100,
        build_specification_grid()[-1],
        optimizer_configuration=hwm.OptimizerConfiguration(
            "least_squares", {"max_nfev": 2000}
        ),
    )

    assert [call["method"] for call in fit_calls] == [
        "least_squares",
        "L-BFGS-B",
        "L-BFGS-B",
    ]
    assert fit_calls[0]["minimize_kwargs"] == {"max_nfev": 2000}
    assert fit_calls[1].get("minimize_kwargs") is None
    assert fit_calls[2]["minimize_kwargs"] == {
        "options": {"maxfun": 30000, "maxiter": 2000}
    }
    assert [attempt.optimizer for attempt in record.optimizer_attempts].count(
        "least_squares"
    ) == 1
    assert record.optimizer_used == "L-BFGS-B"


def test_multiplicative_fit_fails_clearly_for_non_positive_training_values():
    record = fit_specification(
        np.array([10.0, 0.0, 11.0, 12.0]),
        next(spec for spec in build_specification_grid() if spec.seasonal_form == "mul"),
    )

    assert record.fit_status == "failed"
    assert record.eligible is False
    assert "strictly positive" in record.error_message


def test_forecast_uses_only_approved_information_set_and_returns_continuous_path():
    frame = _hourly_frame("2023-10-01 00:00", "2024-01-02 23:00")
    specification = build_specification_grid()[0]
    altered = frame.copy()
    origin = hwm.country_forecast_origin(date(2024, 1, 2), "Germany")
    altered.loc[
        pd.to_datetime(altered["timestamp_utc"], utc=True) + pd.Timedelta(hours=1)
        > origin,
        "actual_grid_load_mwh",
    ] = 999999.0

    original = forecast_holt_winters(frame, "Germany", date(2024, 1, 2), specification)
    changed = forecast_holt_winters(altered, "Germany", date(2024, 1, 2), specification)

    assert len(original.forecast) == 30
    assert original.forecast.iloc[0]["timestamp_utc"] == origin
    assert original.forecast["timestamp_utc"].is_monotonic_increasing
    assert original.forecast["forecast_mwh"].notna().all()
    assert original.forecast["forecast_mwh"].tolist() == changed.forecast[
        "forecast_mwh"
    ].tolist()
    assert original.forecast["optimizer_used"].eq("L-BFGS-B").all()
    assert original.forecast["optimizer_attempt_count"].eq(1).all()


def test_forecast_rejects_ineligible_fit_even_when_result_is_present(monkeypatch):
    frame = _hourly_frame("2023-10-01 00:00", "2024-01-02 23:00")
    specification = build_specification_grid()[0]
    eligible_record = fit_specification(np.arange(100, dtype=float) + 100, specification)
    ineligible_record = replace(
        eligible_record,
        eligible=False,
        error_message="forced ineligible fit",
    )
    monkeypatch.setattr(hwm, "fit_specification", lambda *args, **kwargs: ineligible_record)

    with pytest.raises(HoltWintersForecastError, match="forced ineligible fit"):
        forecast_holt_winters(frame, "Germany", date(2024, 1, 2), specification)


@pytest.mark.parametrize(
    ("target_date", "expected_path", "expected_target"),
    [(date(2024, 3, 31), 29, 23), (date(2024, 10, 27), 31, 25)],
)
def test_german_dst_paths_preserve_bridge_and_target_counts(
    monkeypatch, target_date, expected_path, expected_target
):
    frame = _hourly_frame("2023-10-01 00:00", "2024-10-28 23:00")
    specification = build_specification_grid()[0]
    fitted = _FakeFit({"success": True, "converged": True})
    record = FitRecord(
        specification=specification,
        specification_order=0,
        fitted_result=fitted,
        aic=10.0,
        aicc=11.0,
        bic=12.0,
        convergence_status="converged",
        fit_status="success",
        fitting_time_seconds=0.1,
        selection_criterion="AICc",
        selection_value=11.0,
        training_observations=100,
        eligible=True,
        error_message=None,
        optimizer_used="L-BFGS-B",
        optimizer_attempts=(),
    )
    monkeypatch.setattr(hwm, "fit_specification", lambda *args, **kwargs: record)

    result = forecast_holt_winters(frame, "Germany", target_date, specification)

    assert len(result.forecast) == expected_path
    assert int(result.forecast["is_target_day"].sum()) == expected_target
    assert int((~result.forecast["is_target_day"]).sum()) == 6


def test_austria_ordinary_path_contains_sixteen_bridge_intervals():
    frame = _hourly_frame("2023-10-01 00:00", "2024-01-02 23:00", "Europe/Vienna")

    result = forecast_holt_winters(
        frame, "Austria", date(2024, 1, 2), build_specification_grid()[0]
    )

    assert len(result.forecast) == 40
    assert int((~result.forecast["is_target_day"]).sum()) == 16
    assert int(result.forecast["is_target_day"].sum()) == 24
