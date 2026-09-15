import numpy as np
import pandas as pd
import pytest

from dwd_station_validation import (
    find_dwd_archive,
    nearest_finite_grid_cell,
    select_exact_dwd_observation,
    source_filename_for_url,
)


def test_nearest_finite_grid_cell_records_a_coastal_missing_cell_fallback():
    latitudes = np.array([54.0, 53.9, 53.8])
    longitudes = np.array([8.6, 8.7, 8.8])
    values = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [-5.5, -5.4, -5.7],
        ]
    )

    result = nearest_finite_grid_cell(53.861, 8.694, latitudes, longitudes, values)

    assert result["requested_nearest_value_valid"] is False
    assert result["latitude"] == pytest.approx(53.8)
    assert result["longitude"] == pytest.approx(8.7)
    assert result["temperature_c"] == pytest.approx(-5.4)


def test_select_exact_dwd_observation_requires_the_requested_utc_hour():
    observations = pd.DataFrame(
        {
            "MESS_DATUM": ["2021021305", "2021021306", "2021021307"],
            "QN_9": [3, 3, 3],
            "TT_TU": [-4.0, -5.9, -6.2],
        }
    )

    result = select_exact_dwd_observation(
        observations, pd.Timestamp("2021-02-13 06:00", tz="UTC")
    )

    assert result["temperature_c"] == pytest.approx(-5.9)
    assert result["quality_flag"] == 3


def test_select_exact_dwd_observation_rejects_missing_or_invalid_rows():
    observations = pd.DataFrame(
        {
            "MESS_DATUM": ["2021021305", "2021021307"],
            "QN_9": [3, 3],
            "TT_TU": [-4.0, -6.2],
        }
    )

    with pytest.raises(ValueError, match="exact UTC observation"):
        select_exact_dwd_observation(
            observations, pd.Timestamp("2021-02-13 06:00", tz="UTC")
        )


def test_find_dwd_archive_accepts_local_source_filename_without_hist_suffix(tmp_path):
    archive = tmp_path / "stundenwerte_TU_00891_19510101_20251231.zip"
    archive.write_bytes(b"test")

    assert find_dwd_archive(tmp_path, "00891") == archive


def test_source_filename_for_url_restores_dwd_historical_suffix():
    assert (
        source_filename_for_url("stundenwerte_TU_00891_19510101_20251231.zip")
        == "stundenwerte_TU_00891_19510101_20251231_hist.zip"
    )
