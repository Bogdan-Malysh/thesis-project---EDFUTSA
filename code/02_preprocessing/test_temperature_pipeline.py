import numpy as np
import pandas as pd
import pytest
import xarray as xr

from temperature_pipeline import (
    _weighted_temperature,
    build_weights,
    required_key_groups,
    validate_temperature_output,
)


def test_required_key_groups_preserves_boundary_year_and_month():
    keys = pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(
                ["2019-12-31 23:00", "2020-01-01 00:00"], utc=True
            ),
            "interval_end_utc": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00"], utc=True
            ),
        }
    )

    groups = required_key_groups(keys)

    assert list(groups) == [(2019, 12), (2020, 1)]
    assert groups[(2019, 12)][0] == pd.Timestamp("2019-12-31 23:00", tz="UTC")


def test_weighted_temperature_returns_kelvin_weighted_values():
    dataset = xr.Dataset(
        {
            "t2m": (
                ("valid_time", "latitude", "longitude"),
                np.array([[[280.0, 282.0], [284.0, 286.0]]]),
            )
        },
        coords={
            "valid_time": pd.to_datetime(["2020-01-01 00:00"], utc=True),
            "latitude": [1.0, 0.0],
            "longitude": [0.0, 1.0],
        },
    )
    weights = pd.DataFrame(
        {
            "latitude_index": [0, 1],
            "longitude_index": [0, 1],
            "weight": [0.25, 0.75],
        }
    )

    result = _weighted_temperature(dataset, "valid_time", "t2m", weights)

    assert result.loc[0, "temperature_k"] == pytest.approx(284.5)


def test_build_weights_redirects_population_from_nonfinite_grid(monkeypatch):
    population = pd.DataFrame(
        {
            "SPATIAL": ["a", "b"],
            "population": [2.0, 3.0],
            "longitude": [0.0, 1.0],
            "latitude": [1.0, 0.0],
        }
    )
    monkeypatch.setattr(
        "temperature_pipeline._read_population_country", lambda country: population
    )

    weights, total = build_weights(
        "Germany",
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        valid_grid_mask=np.array([[True, False], [False, False]]),
    )

    assert total == 5.0
    assert weights["weight"].sum() == pytest.approx(1.0)
    assert weights[["latitude_index", "longitude_index"]].to_numpy().tolist() == [
        [0, 0]
    ]


def test_validate_temperature_output_requires_exact_smard_keys():
    keys = pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00"], utc=True
            ),
            "interval_end_utc": pd.to_datetime(
                ["2020-01-01 01:00", "2020-01-01 02:00"], utc=True
            ),
        }
    )
    output = keys.assign(
        interval_start_local=keys["interval_start_utc"],
        interval_end_local=keys["interval_end_utc"],
        temperature_c=[0.0, 1.0],
        temperature_valid=True,
    )

    validation = validate_temperature_output(output, keys, "Germany")

    assert validation["row_count"] == 2
    assert validation["missing_count"] == 0
