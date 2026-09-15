from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

import sarimax_state_validation_2024 as state_validation
import sarimax_validation_2024 as sarimax_validation
from sarimax_models import order_for_country, specification_columns
from sarimax_state_update import (
    FixedParameterState,
    build_initial_training_set,
    observations_between_cutoffs,
)


class _FakeStateResult:
    def __init__(self, params: np.ndarray, *, mutate: bool = False) -> None:
        self.params = params.copy()
        self.mutate = mutate
        self.extend_calls: list[tuple[object, object]] = []

    def extend(self, endog, exog=None):
        self.extend_calls.append((endog, exog))
        if self.mutate:
            self.params = self.params + 1.0
        return self


def test_fixed_state_updates_with_extend_and_preserves_parameters() -> None:
    result = _FakeStateResult(np.array([1.0, 2.0]))
    state = FixedParameterState(result=result, parameter_vector=result.params.copy())
    endog = pd.Series([10.0], index=pd.to_datetime(["2024-01-01T00:00Z"]))
    exog = pd.DataFrame({"hour_01": [1.0]}, index=endog.index)

    state.update(endog, exog)

    assert len(result.extend_calls) == 1
    assert state.result is result
    assert np.array_equal(state.parameter_vector, [1.0, 2.0])


def test_fixed_state_normalizes_inferable_datetime_frequency_for_extend() -> None:
    result = _FakeStateResult(np.array([1.0, 2.0]))
    state = FixedParameterState(result=result, parameter_vector=result.params.copy())
    endog = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.to_datetime(
            ["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-01T02:00Z"]
        ),
    )
    exog = pd.DataFrame({"hour_01": [1.0, 1.0, 1.0]}, index=endog.index)

    state.update(endog, exog)

    updated_endog = result.extend_calls[0][0]
    assert updated_endog.index.freq == pd.tseries.frequencies.to_offset("h")


def test_state_initial_fit_retains_filter_state_for_extend() -> None:
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    endog = pd.Series(np.arange(len(index), dtype=float), index=index)
    exog = pd.DataFrame({"x": np.ones(len(index))}, index=index)
    model = SARIMAX(endog, exog=exog, order=(1, 0, 0), trend="n")
    fit = model.fit(
        method="lbfgs",
        maxiter=20,
        maxfun=100,
        disp=0,
        cov_type="none",
        low_memory=state_validation.STATE_INITIAL_FIT_KWARGS["low_memory"],
    )
    state = FixedParameterState(result=fit, parameter_vector=fit.params.copy())
    update_index = pd.date_range("2024-01-03", periods=3, freq="h", tz="UTC")

    state.update(
        pd.Series([48.0, 49.0, 50.0], index=update_index),
        pd.DataFrame({"x": np.ones(3)}, index=update_index),
    )

    assert state.result.model._index[-1] == update_index[-1]


def test_fixed_state_rejects_parameter_changes() -> None:
    result = _FakeStateResult(np.array([1.0, 2.0]), mutate=True)
    state = FixedParameterState(result=result, parameter_vector=result.params.copy())

    with pytest.raises(ValueError, match="parameters changed"):
        state.update(
            pd.Series([10.0]),
            pd.DataFrame({"hour_01": [1.0]}),
        )


def test_initial_training_set_excludes_2024_observations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2022-12-31T16:00Z",
                    "2023-12-31T16:00Z",
                    "2023-12-31T17:00Z",
                    "2024-01-01T00:00Z",
                ]
            ),
            "timestamp_local": [
                "2022-12-31T17:00:00+01:00",
                "2023-12-31T17:00:00+01:00",
                "2023-12-31T18:00:00+01:00",
                "2024-01-01T01:00:00+01:00",
            ],
            "interval_end_utc": pd.to_datetime(
                [
                    "2022-12-31T17:00Z",
                    "2023-12-31T17:00Z",
                    "2023-12-31T18:00Z",
                    "2024-01-01T01:00Z",
                ]
            ),
            "local_date": ["2022-12-31", "2023-12-31", "2023-12-31", "2024-01-01"],
            "actual_load_mwh": [0.0, 1.0, 2.0, 3.0],
            "actual_grid_load_mwh": [0.0, 1.0, 2.0, 3.0],
            "hour": [17, 17, 18, 1],
            "day_of_week": [5, 6, 6, 0],
        }
    )

    training, cutoff = build_initial_training_set(frame, "Germany")

    assert cutoff.tz is not None
    assert training["local_date"].min() == "2022-12-31"
    assert training["local_date"].max() == "2023-12-31"
    assert training["timestamp_utc"].max() == pd.Timestamp("2023-12-31T16:00:00Z")
    assert not training["timestamp_utc"].astype(str).str.startswith("2024").any()


@pytest.mark.parametrize(
    ("country", "expected_last_timestamp"),
    [
        ("Germany", pd.Timestamp("2023-12-31T16:00:00Z")),
        ("Austria", pd.Timestamp("2023-12-31T06:00:00Z")),
    ],
)
def test_initial_training_set_respects_country_origin_boundary(
    country: str,
    expected_last_timestamp: pd.Timestamp,
) -> None:
    timestamps = pd.date_range("2023-12-31T06:00:00Z", periods=12, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "timestamp_local": timestamps.tz_convert("Europe/Berlin").astype(str),
            "interval_end_utc": timestamps + pd.Timedelta(hours=1),
            "local_date": ["2023-12-31"] * len(timestamps),
            "actual_load_mwh": np.arange(len(timestamps), dtype=float),
            "actual_grid_load_mwh": np.arange(len(timestamps), dtype=float),
            "hour": list(range(len(timestamps))),
            "day_of_week": [6] * len(timestamps),
        }
    )

    training, cutoff = build_initial_training_set(frame, country)

    assert training["timestamp_utc"].max() == expected_last_timestamp
    assert training["interval_end_utc"].max() == cutoff


def test_observations_between_cutoffs_are_chronological_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "interval_end_utc": pd.date_range("2024-01-01T01:00Z", periods=4, freq="h"),
            "actual_load_mwh": [1.0, 2.0, 3.0, 4.0],
        }
    )

    updated = observations_between_cutoffs(
        frame,
        pd.Timestamp("2024-01-01T01:00Z"),
        pd.Timestamp("2024-01-01T03:00Z"),
    )

    assert updated["actual_load_mwh"].tolist() == [2.0, 3.0]
    assert updated["timestamp_utc"].is_monotonic_increasing


def test_fixed_state_forecast_path_uses_shared_prepared_frame_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = pd.date_range(
        "2024-02-14T16:00:00Z", "2024-02-15T23:00:00Z", freq="h"
    )
    local = timestamps.tz_convert("Europe/Berlin")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "timestamp_local": local.astype(str),
            "interval_end_utc": timestamps + pd.Timedelta(hours=1),
            "local_date": local.strftime("%Y-%m-%d"),
            "local_occurrence": 0,
            "actual_load_mwh": np.arange(len(timestamps), dtype=float),
            "actual_grid_load_mwh": np.arange(len(timestamps), dtype=float),
            "hour": local.hour,
            "day_of_week": local.dayofweek,
        }
    )
    exog_columns = specification_columns("sarimax_calendar")
    monkeypatch.setattr(
        state_validation,
        "build_exog",
        lambda values, country, specification_id: pd.DataFrame(
            0.0, index=range(len(values)), columns=exog_columns
        ),
    )
    monkeypatch.setattr(
        state_validation,
        "forecast_values",
        lambda result, steps, exog=None: np.zeros(steps),
    )
    context = SimpleNamespace(
        cutoff_utc=pd.Timestamp("2024-02-14T17:00:00Z"),
        origin_local="2024-02-14T18:00:00+01:00",
    )

    path = state_validation._forecast_path(
        frame,
        "Germany",
        "2024-02-15",
        "sarimax_calendar",
        object(),
        context,
        order_for_country("Germany"),
    )

    assert len(path) == 30
    assert path["forecast_mwh"].eq(0.0).all()


def test_fixed_state_output_columns_are_unique() -> None:
    assert len(state_validation.JOB_COLUMNS) == len(set(state_validation.JOB_COLUMNS))
    assert len(state_validation.DIAGNOSTIC_COLUMNS) == len(
        set(state_validation.DIAGNOSTIC_COLUMNS)
    )


def test_sarimax_validation_exports_forecast_timestamp_grouping_helper() -> None:
    assert callable(sarimax_validation._forecast_timestamps_by_job)
