from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sarima_screening
from sarima_models import ResidualDiagnostics, SarimaOrder
from sarima_screening import (
    assess_differencing_paths,
    build_candidate_orders,
    narrow_shortlist,
    recompute_shortlist_from_persisted_screening,
    screening_row_is_eligible,
    screening_row_is_model_valid,
    select_d,
    stationarity_diagnostics,
    training_values,
)


def _hourly_frame(start="2020-01-01", end="2025-01-03"):
    utc = pd.date_range(start, end, freq="h", tz="UTC")
    local = utc.tz_convert("Europe/Berlin")
    values = np.arange(len(utc), dtype=float)
    local_dates = local.strftime("%Y-%m-%d")
    values[local_dates >= "2024-01-01"] = 2024_000_000.0
    values[local_dates >= "2025-01-01"] = 2025_000_000.0
    return pd.DataFrame(
        {
            "timestamp_utc": utc,
            "timestamp_local": local.map(pd.Timestamp.isoformat),
            "local_date": local_dates,
            "actual_load_mwh": values,
        }
    )


def test_training_values_selects_only_contiguous_2020_2023_observations():
    frame = _hourly_frame()

    values = training_values(frame)

    training_mask = pd.to_datetime(frame["local_date"]).dt.date.between(
        pd.Timestamp("2020-01-01").date(), pd.Timestamp("2023-12-31").date()
    )
    dates = pd.to_datetime(frame.loc[training_mask, "timestamp_utc"], utc=True)
    expected = frame.loc[training_mask, "actual_load_mwh"].to_numpy()

    assert len(values) == len(expected)
    assert np.array_equal(values, expected)
    assert not np.isin(values, [2024_000_000.0, 2025_000_000.0]).any()
    assert dates.is_monotonic_increasing


def test_training_values_rejects_missing_hour():
    frame = _hourly_frame("2020-01-01", "2020-01-03")
    frame = frame.drop(index=frame.index[10]).reset_index(drop=True)

    with pytest.raises(ValueError, match="contiguous"):
        training_values(frame)


def test_training_values_ignores_invalid_dates_outside_training_window():
    frame = _hourly_frame("2020-01-01", "2020-01-03")
    future = frame.iloc[[0]].copy()
    future["local_date"] = "not-a-date"
    future["timestamp_local"] = "not-a-date"
    future["timestamp_utc"] = pd.Timestamp("2025-01-01", tz="UTC")
    frame = pd.concat([frame, future], ignore_index=True)

    values = training_values(frame)

    assert len(values) == 49


def test_stationarity_tests_use_exact_adf_and_kpss_configuration(monkeypatch):
    adf_calls = []
    kpss_calls = []

    def fake_adfuller(values, **kwargs):
        adf_calls.append(kwargs)
        return (-5.0, 0.01, 3, len(values), {}, 0.0)

    def fake_kpss(values, **kwargs):
        kpss_calls.append(kwargs)
        return (0.1, 0.1, 5, {"1%": 0.7, "5%": 0.46, "10%": 0.35})

    monkeypatch.setattr(sarima_screening, "adfuller", fake_adfuller)
    monkeypatch.setattr(sarima_screening, "kpss", fake_kpss)

    diagnostics = stationarity_diagnostics(np.arange(120.0), D=0)

    assert len(diagnostics) == 3
    assert adf_calls[0] == {"regression": "ct", "maxlag": 24, "autolag": "AIC"}
    assert kpss_calls[0] == {"regression": "ct", "nlags": "auto"}
    assert all(
        call == {"regression": "c", "maxlag": 24, "autolag": "AIC"}
        for call in adf_calls[1:]
    )
    assert all(
        call == {"regression": "c", "nlags": "auto"}
        for call in kpss_calls[1:]
    )


def test_select_d_chooses_smallest_supported_order():
    diagnostics = [
        {"d": 0, "stationary_supported": False},
        {"d": 1, "stationary_supported": True},
        {"d": 2, "stationary_supported": True},
    ]

    assert select_d(diagnostics) == 1
    assert select_d([{"d": 0, "stationary_supported": False}]) is None


def test_differencing_assessment_reports_seasonal_evidence_and_viability(monkeypatch):
    def fake_seasonal(values, D):
        return {
            "D": D,
            "lags": (24, 48, 72),
            "original_acf": {24: 0.8, 48: 0.6, 72: 0.4},
            "transformed_acf": {
                24: 0.8 if D == 0 else 0.1,
                48: 0.6 if D == 0 else 0.1,
                72: 0.4 if D == 0 else 0.1,
            },
            "persistence_change": {
                lag: 0.0 if D == 0 else -0.5 for lag in (24, 48, 72)
            },
            "seasonal_evidence_supported": True,
        }

    def fake_stationarity(values, D):
        supported = D == 0
        return [
            {"D": D, "d": 0, "stationary_supported": supported},
            {"D": D, "d": 1, "stationary_supported": False},
            {"D": D, "d": 2, "stationary_supported": False},
        ]

    monkeypatch.setattr(
        sarima_screening, "seasonal_dependence_diagnostics", fake_seasonal
    )
    monkeypatch.setattr(
        sarima_screening, "stationarity_diagnostics", fake_stationarity
    )

    paths = assess_differencing_paths(np.arange(200.0))

    assert set(paths) == {0, 1}
    assert paths[0]["selected_d"] == 0
    assert paths[0]["viable"]
    assert paths[1]["selected_d"] is None
    assert not paths[1]["viable"]
    assert paths[1]["seasonal"]["lags"] == (24, 48, 72)
    assert "seasonal_unit_root" not in str(paths).lower()


def test_candidate_orders_are_core_by_default_and_exactly_bounded_with_evidence():
    core = build_candidate_orders(1, 0)
    all_evidence = {
        "ordinary_ar": True,
        "ordinary_ma": True,
        "seasonal_ar": True,
        "seasonal_ma": True,
    }
    expanded = build_candidate_orders(1, 0, all_evidence)
    signatures = [(order.p, order.q, order.P, order.Q) for order in expanded]

    assert len(core) == 8
    assert len(expanded) == 14
    assert len(signatures) == len(set(signatures))
    assert len({order.specification_id for order in expanded}) == 14
    assert signatures == [
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
    assert all(order.p <= 2 and order.q <= 2 for order in expanded)
    assert all(order.P <= 1 and order.Q <= 1 for order in expanded)


def test_narrow_shortlist_unions_best_criterion_within_each_d_D_path():
    rows = [
        {
            "specification_id": "a",
            "d": 1,
            "D": 0,
            "eligible": True,
            "aic": 1,
            "aicc": 5,
            "bic": 9,
            "specification_order": 1,
        },
        {
            "specification_id": "b",
            "d": 1,
            "D": 0,
            "eligible": True,
            "aic": 2,
            "aicc": 1,
            "bic": 8,
            "specification_order": 2,
        },
        {
            "specification_id": "c",
            "d": 1,
            "D": 0,
            "eligible": True,
            "aic": 3,
            "aicc": 2,
            "bic": 1,
            "specification_order": 3,
        },
        {
            "specification_id": "d",
            "d": 0,
            "D": 1,
            "eligible": True,
            "aic": 1,
            "aicc": 1,
            "bic": 1,
            "specification_order": 1,
        },
        {
            "specification_id": "e",
            "d": 0,
            "D": 1,
            "eligible": True,
            "aic": 2,
            "aicc": 2,
            "bic": 2,
            "specification_order": 2,
        },
    ]

    selected = narrow_shortlist(rows)

    assert [row["specification_id"] for row in selected] == ["a", "b", "c", "d"]
    assert all(
        (row["d"], row["D"]) in {(1, 0), (0, 1)} for row in selected
    )
    assert len(selected) <= 6


def test_residual_adequacy_is_screening_rejection_but_not_hard_model_invalidity():
    row = {
        "fit_status": "success",
        "convergence_status": "converged",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "forecast_valid": True,
        "optimizer_success": True,
        "optimizer_status": 0,
        "optimizer_warnflag": 0,
        "log_likelihood": -1.0,
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "residual_adequacy_rejected": True,
        "error_message": None,
    }

    assert screening_row_is_model_valid(row)
    assert screening_row_is_eligible(row)


def test_model_validity_requires_explicit_optimizer_forecast_and_finite_likelihood():
    row = {
        "fit_status": "success",
        "convergence_status": "converged",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "forecast_valid": False,
        "optimizer_success": False,
        "optimizer_status": 0,
        "optimizer_warnflag": 0,
        "log_likelihood": float("nan"),
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "residual_adequacy_rejected": False,
        "error_message": None,
    }

    assert not screening_row_is_model_valid(row)


def test_residual_error_message_alone_prevents_screening_eligibility():
    row = {
        "fit_status": "success",
        "convergence_status": "converged",
        "converged": True,
        "parameters_finite": True,
        "standard_errors_finite": True,
        "stationarity_ok": True,
        "invertibility_ok": True,
        "forecast_valid": True,
        "optimizer_success": True,
        "optimizer_status": 0,
        "optimizer_warnflag": 0,
        "log_likelihood": -1.0,
        "aic": 1.0,
        "aicc": 1.0,
        "bic": 1.0,
        "residual_adequacy_rejected": False,
        "error_message": "joint Ljung-Box and residual ACF rule",
    }

    assert screening_row_is_model_valid(row)
    assert screening_row_is_eligible(row)


def test_recompute_shortlist_uses_persisted_rows_without_refitting(tmp_path, monkeypatch):
    rows = [
        {
            "country": "Germany",
            "specification_id": "a",
            "specification_order": 1,
            "p": 0,
            "d": 1,
            "q": 0,
            "P": 0,
            "D": 0,
            "Q": 0,
            "fit_status": "success",
            "convergence_status": "converged",
            "optimizer_success": np.nan,
            "optimizer_status": np.nan,
            "optimizer_warnflag": 0,
            "converged": True,
            "parameters_finite": True,
            "standard_errors_finite": True,
            "forecast_valid": True,
            "stationarity_ok": True,
            "invertibility_ok": True,
            "log_likelihood": -1.0,
            "aic": 1.0,
            "aicc": 1.0,
            "bic": 1.0,
            "residual_adequacy_rejected": True,
            "residual_adequacy_reason": "joint Ljung-Box and residual ACF rule",
            "error_message": np.nan,
        },
        {
            "country": "Germany",
            "specification_id": "b",
            "specification_order": 2,
            "p": 1,
            "d": 1,
            "q": 0,
            "P": 0,
            "D": 0,
            "Q": 0,
            "fit_status": "failed",
            "convergence_status": "failed",
            "optimizer_success": False,
            "optimizer_status": 1,
            "optimizer_warnflag": 1,
            "converged": False,
            "parameters_finite": True,
            "standard_errors_finite": True,
            "forecast_valid": True,
            "stationarity_ok": True,
            "invertibility_ok": True,
            "log_likelihood": -1.0,
            "aic": 0.0,
            "aicc": 0.0,
            "bic": 0.0,
            "residual_adequacy_rejected": False,
            "residual_adequacy_reason": np.nan,
            "error_message": "optimizer did not converge",
        },
    ]
    screening_path = tmp_path / "screening.csv"
    shortlist_path = tmp_path / "shortlist.csv"
    pd.DataFrame(rows).to_csv(screening_path, index=False)
    monkeypatch.setattr(sarima_screening, "fit_sarima", lambda *args: pytest.fail("refit"))

    summary = recompute_shortlist_from_persisted_screening(
        screening_path, shortlist_path
    )

    revised = pd.read_csv(summary["revised_screening_path"])
    shortlist = pd.read_csv(shortlist_path)
    assert summary["eligible_candidate_count"] == 1
    assert shortlist["specification_id"].tolist() == ["a"]
    assert bool(revised.loc[0, "residual_adequacy_rejected"])
    assert bool(revised.loc[0, "eligible"])
    assert screening_path.read_bytes() == pd.DataFrame(rows).to_csv(index=False).encode()


def test_screening_keeps_germany_and_austria_rows_separate_and_writes_only_four_tables(
    monkeypatch, tmp_path
):
    frame = _hourly_frame("2020-01-01", "2020-01-03")
    residual = ResidualDiagnostics(
        burn_in=48,
        state_space_loglikelihood_burn=0,
        conservative_burn_in=48,
        n_effective=100,
        acf_values={1: 0.0},
        acf_threshold=0.05,
        flagged_acf_lags=(),
        ljung_box_statistics={24: 1.0, 48: 1.0},
        ljung_box_pvalues={24: 0.5, 48: 0.5},
        training_adequacy_rejected=False,
        training_adequacy_reason=None,
    )

    def fake_assessment(values):
        return {
            0: {
                "D": 0,
                "selected_d": 0,
                "viable": True,
                "seasonal_evidence_supported": True,
                "seasonal": {
                    "lags": (24, 48, 72),
                    "original_acf": {24: 0.1, 48: 0.1, 72: 0.1},
                    "transformed_acf": {24: 0.1, 48: 0.1, 72: 0.1},
                    "persistence_change": {24: 0.0, 48: 0.0, 72: 0.0},
                },
                "stationarity": [
                    {"d": 0, "stationary_supported": True, "observations": len(values)}
                ],
            },
            1: {
                "D": 1,
                "selected_d": None,
                "viable": False,
                "seasonal_evidence_supported": False,
                "seasonal": {"lags": (24, 48, 72)},
                "stationarity": [],
            },
        }

    def fake_fit(values, order):
        return SimpleNamespace(
            order=order,
            trend="c",
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
            nobs=len(values),
            effective_nobs=len(values),
            n_params=1,
            log_likelihood=-1.0,
            aic=1.0 + order.p,
            aicc=1.0 + order.p,
            bic=1.0 + order.p,
            ar_root_minimum=2.0,
            ma_root_minimum=2.0,
            stationarity_ok=True,
            invertibility_ok=True,
            residual_diagnostics=residual,
            training_adequacy_rejected=False,
        )

    monkeypatch.setattr(sarima_screening, "assess_differencing_paths", fake_assessment)
    monkeypatch.setattr(sarima_screening, "acf_pacf_evidence", lambda *args: {})
    monkeypatch.setattr(sarima_screening, "fit_sarima", fake_fit)
    monkeypatch.setattr(sarima_screening, "load_country_data", lambda country, directory: frame)
    monkeypatch.setattr(sarima_screening, "TABLE_DIRECTORY", tmp_path)

    result = sarima_screening.run_screening(tmp_path)

    assert result["countries"] == ["Germany", "Austria"]
    assert set(result["countries_with_empty_shortlists"]) == set()
    expected_files = {
        "sarima_differencing_diagnostics_2024.csv",
        "sarima_candidate_screening_training_2024.csv",
        "sarima_residual_diagnostics_training_2024.csv",
        "sarima_shortlists_2024.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    screening = pd.read_csv(tmp_path / "sarima_candidate_screening_training_2024.csv")
    assert set(screening["country"]) == {"Germany", "Austria"}
    assert set(screening["training_start"]) == {"2020-01-01"}
    assert set(screening["training_end"]) == {"2023-12-31"}
    assert not screening.astype(str).apply(lambda column: column.str.contains("2025").any()).any()
