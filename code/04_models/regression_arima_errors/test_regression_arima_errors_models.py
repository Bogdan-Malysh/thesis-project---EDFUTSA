from __future__ import annotations

import numpy as np
import pandas as pd

from regression_arima_errors_models import (
    CRISIS_SPECIFICATION_ID,
    ERROR_ORDER,
    INTERVENTION_SPECIFICATION_ID,
    SPECIFICATION_IDS,
    build_exog,
    fit_regression_arima,
    fit_with_retry,
    specification_columns,
)


def test_frozen_specifications_have_reference_coded_columns() -> None:
    timestamps = pd.date_range(
        "2024-01-01 00:00:00+00:00", periods=48, freq="h"
    )

    expected = {
        "reg_arima_h": [f"hour_{hour:02d}" for hour in range(1, 24)],
        "reg_arima_h_d": [
            *[f"hour_{hour:02d}" for hour in range(1, 24)],
            *[f"weekday_{day}" for day in range(1, 7)],
        ],
        "reg_arima_h_d_m": [
            *[f"hour_{hour:02d}" for hour in range(1, 24)],
            *[f"weekday_{day}" for day in range(1, 7)],
            *[f"month_{month:02d}" for month in range(2, 13)],
        ],
        "reg_arima_h_d_m_hol": [
            *[f"hour_{hour:02d}" for hour in range(1, 24)],
            *[f"weekday_{day}" for day in range(1, 7)],
            *[f"month_{month:02d}" for month in range(2, 13)],
            "public_holiday",
        ],
    }

    assert SPECIFICATION_IDS == tuple(expected)
    for specification_id, columns in expected.items():
        actual = build_exog(timestamps, "Germany", specification_id)
        assert list(actual.columns) == columns
        assert actual.shape == (48, len(columns))
        assert actual.index.equals(pd.RangeIndex(48))
        assert not set(actual.columns) & {
            "temperature_c",
            "forecasted_grid_load_mwh",
            "is_covid_period",
            "is_post_invasion",
        }


def test_country_holiday_logic_excludes_regional_german_holidays() -> None:
    national = build_exog(
        pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]),
        "Germany",
        "reg_arima_h_d_m_hol",
    )
    regional = build_exog(
        pd.DatetimeIndex([pd.Timestamp("2024-06-20", tz="UTC")]),
        "Germany",
        "reg_arima_h_d_m_hol",
    )

    assert national["public_holiday"].tolist() == [1]
    assert regional["public_holiday"].tolist() == [0]


def test_actual_exog_rank_is_hard_validity_but_differenced_rank_is_diagnostic() -> None:
    values = np.linspace(100.0, 200.0, 80)
    exog = pd.DataFrame(
        np.ones((80, len(specification_columns("reg_arima_h")))),
        columns=specification_columns("reg_arima_h"),
    )

    fit = fit_regression_arima(values, exog, "reg_arima_h")

    assert fit.exog_rank == 1
    assert fit.exog_condition_number > 1e12
    assert fit.transformed_exog_rank is not None
    assert not fit.eligible
    assert "rank-deficient actual exogenous design" in (fit.error_message or "")


def test_retry_uses_only_the_approved_second_attempt() -> None:
    calls: list[dict[str, int]] = []

    class Result:
        def __init__(self, converged: bool) -> None:
            self.mle_retvals = {
                "converged": converged,
                "success": converged,
                "warnflag": 0 if converged else 1,
            }

    def fit_once(method_kwargs: dict[str, int] | None) -> tuple[Result, tuple[str, ...], str | None]:
        calls.append(method_kwargs or {})
        return Result(len(calls) == 2), (), None

    result, attempts, retry_count = fit_with_retry(fit_once)

    assert result is not None
    assert len(attempts) == 2
    assert attempts[0]["method_kwargs"] == {}
    assert attempts[1]["method_kwargs"] == {"maxiter": 1000}
    assert retry_count == 1
    assert calls == [{}, {"maxiter": 1000}]


def test_frozen_error_order_is_nonseasonal_arima_212() -> None:
    assert ERROR_ORDER == (2, 1, 2)


def test_crisis_specification_adds_only_the_two_interventions() -> None:
    base = specification_columns("reg_arima_h_d_m_hol")
    crisis = specification_columns(INTERVENTION_SPECIFICATION_ID)

    assert crisis[: len(base)] == base
    assert crisis[-2:] == ("is_covid_period", "is_post_invasion")
    assert specification_columns("reg_arima_h_d_m_hol") == base


def test_crisis_exog_is_timestamp_only_finite_and_full_rank() -> None:
    timestamps = pd.date_range(
        "2020-01-01 00:00:00+00:00",
        "2024-12-31 23:00:00+00:00",
        freq="h",
    )

    exog = build_exog(timestamps, "Germany", INTERVENTION_SPECIFICATION_ID)

    assert list(exog.columns) == list(specification_columns(INTERVENTION_SPECIFICATION_ID))
    assert not exog.columns.duplicated().any()
    assert np.isfinite(exog.to_numpy()).all()
    assert np.linalg.matrix_rank(exog.to_numpy()) == len(exog.columns)
    assert exog.loc[0, "is_covid_period"] == 0
    assert exog.loc[24 * 70, "is_covid_period"] == 1
    assert exog.loc[24 * 800, "is_post_invasion"] == 1


def test_crisis_only_specification_uses_only_the_two_interventions() -> None:
    timestamps = pd.date_range(
        "2020-03-10 00:00:00+00:00",
        "2022-02-24 23:00:00+00:00",
        freq="h",
    )

    columns = specification_columns(CRISIS_SPECIFICATION_ID)
    exog = build_exog(timestamps, "Germany", CRISIS_SPECIFICATION_ID)

    assert columns == ("is_covid_period", "is_post_invasion")
    assert list(exog.columns) == list(columns)
    assert np.isfinite(exog.to_numpy()).all()
    assert set(np.unique(exog.to_numpy())) <= {0.0, 1.0}
    assert np.linalg.matrix_rank(exog.to_numpy()) == len(columns)
    assert exog.iloc[0].tolist() == [0.0, 0.0]
    assert exog.iloc[-1].tolist() == [1.0, 1.0]
