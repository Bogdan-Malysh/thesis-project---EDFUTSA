from datetime import date
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import holt_winters_phase2a as phase2a
from holt_winters_models import ShortlistingError, build_specification_grid


def _screening_frame(country: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    specifications = build_specification_grid()
    for order, specification in enumerate(specifications):
        rows.append(
            {
                "country": country,
                "specification_id": specification.specification_id,
                "seasonal_form": specification.seasonal_form,
                "seasonal_periods": specification.seasonal_periods,
                "trend": "add",
                "damped_trend": specification.damped_trend,
                "specification_order": order,
                "selection_criterion": "AICc",
                "selection_value": float(order),
                "eligible": True,
                "fit_status": "success",
                "convergence_status": "converged",
                "error_message": None,
                "optimizer_used": "L-BFGS-B",
                "optimizer_attempt_count": 1,
                "fitting_time_seconds": 0.1,
                "optimizer_attempts_json": json.dumps(
                    [
                        {
                            "attempt_number": 1,
                            "optimizer": "L-BFGS-B",
                            "minimize_kwargs": {},
                            "message": "ok",
                            "status": 0,
                            "success": True,
                            "nit": 1,
                            "nfev": 2,
                            "njev": 1,
                            "warning_messages": [],
                            "parameters_finite": True,
                            "aic": 1.0,
                            "aic_finite": True,
                            "aicc": 1.0,
                            "aicc_finite": True,
                            "bic": 1.0,
                            "bic_finite": True,
                            "fitting_time_seconds": 0.1,
                        }
                    ]
                ),
            }
        )
    screening = pd.DataFrame(rows)
    shortlist = screening.iloc[[0, 2, 4, 6]].copy()
    return screening, shortlist


def test_smoke_cases_use_july_ordinary_dates_and_german_dst_dates():
    assert phase2a.SMOKE_CASES == (
        ("Germany", date(2024, 7, 15)),
        ("Austria", date(2024, 7, 15)),
        ("Germany", date(2024, 3, 31)),
        ("Germany", date(2024, 10, 27)),
    )


def test_screening_writes_sixteen_candidates_and_eight_shortlist_rows(
    tmp_path, monkeypatch
):
    germany_screening, germany_shortlist = _screening_frame("Germany")
    austria_screening, austria_shortlist = _screening_frame("Austria")
    responses = iter(
        [
            (germany_screening, germany_shortlist),
            (austria_screening, austria_shortlist),
        ]
    )
    monkeypatch.setattr(phase2a, "screen_country", lambda *args: next(responses))
    monkeypatch.setattr(phase2a, "load_country_data", lambda *args: object())
    monkeypatch.setattr(phase2a, "TABLE_DIRECTORY", tmp_path)

    screening, shortlist = phase2a.run_screening()

    assert len(screening) == 16
    assert len(shortlist) == 8
    assert len(pd.read_csv(tmp_path / "holt_winters_screening_2024.csv")) == 16
    assert len(pd.read_csv(tmp_path / "holt_winters_shortlists_2024.csv")) == 8
    assert len(pd.read_csv(tmp_path / "holt_winters_optimizer_attempts_2024.csv")) == 16


def test_unselectable_group_stops_before_smoke_testing(monkeypatch):
    specifications = build_specification_grid()
    failed_rows = pd.DataFrame(
        [
            {
                "specification_id": specifications[0].specification_id,
                "fit_status": "failed",
                "error_message": "first failure",
            },
            {
                "specification_id": specifications[1].specification_id,
                "fit_status": "failed",
                "error_message": "second failure",
            },
        ]
    )
    error = ShortlistingError([(("add", 24), failed_rows)])
    smoke_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal smoke_called
        smoke_called = True

    monkeypatch.setattr(
        phase2a,
        "run_screening",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(phase2a, "run_smoke_tests", fail_if_called)

    with pytest.raises(ShortlistingError, match="first failure"):
        phase2a.run_phase2a()

    assert smoke_called is False


def test_failed_screening_persists_diagnostics_before_stopping(tmp_path, monkeypatch):
    screening, _ = _screening_frame("Germany")
    specifications = build_specification_grid()
    failed_rows = screening.iloc[[0, 1]].copy()
    failed_rows["fit_status"] = "failed"
    failed_rows["eligible"] = False
    failed_rows["error_message"] = ["first failure", "second failure"]
    error = ShortlistingError([(("add", 24), failed_rows)])

    monkeypatch.setattr(phase2a, "load_country_data", lambda *args: screening)

    def fail_screening(*args):
        error.screening = screening
        raise error

    monkeypatch.setattr(phase2a, "screen_country", fail_screening)
    monkeypatch.setattr(phase2a, "TABLE_DIRECTORY", tmp_path)

    with pytest.raises(ShortlistingError, match=specifications[0].specification_id):
        phase2a.run_screening()

    saved_screening = pd.read_csv(tmp_path / "holt_winters_screening_2024.csv")
    assert len(saved_screening) == 8
    saved_shortlists = pd.read_csv(tmp_path / "holt_winters_shortlists_2024.csv")
    assert saved_shortlists.empty
    assert saved_shortlists.columns.tolist() == saved_screening.columns.tolist()


def test_phase2a_result_reports_each_smoke_case(monkeypatch):
    screening, shortlist = _screening_frame("Germany")
    smoke = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-07-15",
                "specification_id": "hw_add_s24_undamped",
                "target_intervals": 24,
                "path_intervals": 30,
                "bridge_intervals": 6,
                "missing_predictions": 0,
                "failure": False,
                "error_message": None,
                "convergence_status": "converged",
                "fitting_time_seconds": 1.0,
            }
        ]
    )
    monkeypatch.setattr(phase2a, "run_screening", lambda *args: (screening, shortlist))
    monkeypatch.setattr(phase2a, "load_frozen_shortlists", lambda: shortlist)
    monkeypatch.setattr(phase2a, "run_smoke_tests", lambda *args: smoke)

    result = phase2a.run_phase2a()

    assert result["smoke_results"] == smoke.to_dict(orient="records")


def _valid_frozen_shortlist() -> pd.DataFrame:
    germany_screening, _ = _screening_frame("Germany")
    austria_screening, _ = _screening_frame("Austria")
    germany_shortlist = germany_screening.iloc[[1, 3, 5, 7]].copy()
    austria_shortlist = austria_screening.iloc[[1, 3, 5, 7]].copy()
    shortlists = pd.concat([germany_shortlist, austria_shortlist], ignore_index=True)
    weekly = shortlists["specification_id"].eq("hw_mul_s168_damped")
    shortlists.loc[weekly, "optimizer_used"] = "least_squares"
    shortlists.loc[weekly, "optimizer_attempts_json"] = json.dumps(
        [
            {
                "attempt_number": 1,
                "optimizer": "least_squares",
                "minimize_kwargs": {"max_nfev": 2000},
                "message": "ok",
                "status": 2,
                "success": True,
                "parameters_finite": True,
                "aic": 1.0,
                "aic_finite": True,
                "aicc": 1.0,
                "aicc_finite": True,
                "bic": 1.0,
                "bic_finite": True,
                "warning_messages": [],
                "fitting_time_seconds": 0.1,
            }
        ]
    )
    return shortlists


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.assign(damped_trend="not-a-boolean"),
        lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
        lambda frame: frame.iloc[:-1].copy(),
        lambda frame: frame.assign(selection_value=float("inf")),
        lambda frame: frame.assign(eligible=False),
        lambda frame: frame.assign(specification_id="unknown_specification"),
        lambda frame: frame.assign(seasonal_periods=24),
    ],
    ids=[
        "malformed-damping",
        "duplicate-group",
        "incomplete-country",
        "infinite-selection",
        "ineligible",
        "unknown-specification",
        "inconsistent-identity",
    ],
)
def test_load_frozen_shortlists_rejects_invalid_files(tmp_path, mutate):
    path = tmp_path / "shortlists.csv"
    mutate(_valid_frozen_shortlist()).to_csv(path, index=False)

    with pytest.raises(ValueError):
        phase2a.load_frozen_shortlists(path)


def test_smoke_reuses_frozen_optimizer_settings_from_shortlist_metadata(monkeypatch):
    shortlists = _valid_frozen_shortlist()
    captured = []

    def fake_forecast(frame, country, target_date, specification, **kwargs):
        captured.append((country, specification.specification_id, kwargs))
        configuration = kwargs["optimizer_configuration"]
        return SimpleNamespace(
            forecast=pd.DataFrame(
                {
                    "is_target_day": [True] * 24,
                    "forecast_mwh": [1.0] * 24,
                }
            ),
            fit_record=SimpleNamespace(
                convergence_status="converged",
                fitting_time_seconds=1.0,
                optimizer_used=configuration.optimizer,
                optimizer_attempts=(object(),),
            ),
        )

    monkeypatch.setattr(phase2a, "load_country_data", lambda *args: object())
    monkeypatch.setattr(phase2a, "forecast_holt_winters", fake_forecast)
    monkeypatch.setattr(
        phase2a,
        "evaluate_target_day",
        lambda forecast: {"evaluated_observations": len(forecast)},
    )

    results = phase2a.run_smoke_tests(shortlists)

    weekly_multiplicative = [
        item
        for item in captured
        if item[1] == "hw_mul_s168_damped"
    ]
    assert weekly_multiplicative
    assert all(
        item[2]["optimizer_configuration"].optimizer == "least_squares"
        and item[2]["optimizer_configuration"].minimize_kwargs == {"max_nfev": 2000}
        for item in weekly_multiplicative
    )
    assert (
        results["optimizer_used"].eq("least_squares")
        == results["specification_id"].eq("hw_mul_s168_damped")
    ).all()
    assert results["optimizer_attempt_count"].eq(1).all()


def test_runtime_estimation_stops_when_any_smoke_fit_fails():
    smoke = pd.DataFrame(
        [
            {
                "country": "Germany",
                "specification_id": "hw_add_s24_damped",
                "failure": False,
                "fitting_time_seconds": 1.0,
            },
            {
                "country": "Germany",
                "specification_id": "hw_add_s168_damped",
                "failure": True,
                "fitting_time_seconds": None,
            },
        ]
    )

    with pytest.raises(RuntimeError, match="smoke fit failed"):
        phase2a.estimate_full_validation_runtime(smoke)
