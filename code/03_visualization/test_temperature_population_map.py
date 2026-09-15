import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from temperature_population_map import (
    FIGURE_TITLE,
    MAP_ASPECT,
    OUTPUT_DPI,
    OUTPUT_PATH,
    _geometry_path,
    _local_time_summary,
    bubble_area,
    bubble_marker_size,
    mapped_grid_values,
    mask_grid_to_geometry,
    population_legend_values,
    select_cold_timestamp,
    validate_color_scale,
    validate_plot_data,
)


def _temperature_frame(values: list[float], timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(timestamps, utc=True),
            "temperature_c": values,
            "temperature_valid": True,
        }
    )


def test_select_cold_timestamp_uses_joint_mean_over_the_requested_period():
    timestamps = [
        "2019-12-31 23:00+00:00",
        "2020-01-01 00:00+00:00",
        "2021-02-03 04:00+00:00",
        "2025-12-31 23:00+00:00",
        "2026-01-01 00:00+00:00",
    ]
    germany = _temperature_frame([100.0, -4.0, -12.0, -3.0, -50.0], timestamps)
    austria = _temperature_frame([100.0, -6.0, -8.0, -5.0, -50.0], timestamps)

    selected, national = select_cold_timestamp(germany, austria)

    assert selected == pd.Timestamp("2021-02-03 04:00", tz="UTC")
    assert national == {"Germany": pytest.approx(-12.0), "Austria": pytest.approx(-8.0)}


def test_select_cold_timestamp_accepts_a_utc_override():
    timestamps = ["2020-01-01 00:00+00:00", "2021-02-03 04:00+00:00"]
    germany = _temperature_frame([-4.0, -12.0], timestamps)
    austria = _temperature_frame([-6.0, -8.0], timestamps)

    selected, national = select_cold_timestamp(
        germany, austria, override="2020-01-01T00:00:00Z"
    )

    assert selected == pd.Timestamp("2020-01-01 00:00", tz="UTC")
    assert national == {"Germany": pytest.approx(-4.0), "Austria": pytest.approx(-6.0)}


def test_mask_grid_to_geometry_masks_grid_centres_outside_country():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    latitudes = np.array([0.5, 1.5])
    longitudes = np.array([0.5, 1.5])

    masked = mask_grid_to_geometry(values, latitudes, longitudes, box(0.0, 0.0, 1.0, 1.0))

    assert masked[0, 0] == pytest.approx(1.0)
    assert np.isnan(masked[0, 1])
    assert np.isnan(masked[1, 0])
    assert np.isnan(masked[1, 1])


def test_validate_plot_data_rejects_nonfinite_values_and_bad_weight_sums():
    with pytest.raises(ValueError, match="finite"):
        validate_plot_data(
            {"Germany": np.array([1.0, np.nan])},
            {"Germany": pd.DataFrame({"weight": [1.0]})},
        )

    with pytest.raises(ValueError, match="weight sum"):
        validate_plot_data(
            {"Germany": np.array([1.0, 2.0])},
            {"Germany": pd.DataFrame({"weight": [0.4, 0.5]})},
        )


def test_mapped_bubbles_use_unclipped_grid_values():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = pd.DataFrame(
        {
            "latitude_index": [0, 1],
            "longitude_index": [0, 1],
            "weight": [0.5, 0.5],
        }
    )

    result = mapped_grid_values(values, weights)

    assert result.tolist() == [1.0, 4.0]


def test_geometry_path_represents_the_country_boundary():
    path = _geometry_path(box(0.0, 0.0, 1.0, 1.0))

    assert path.contains_point((0.5, 0.5))
    assert not path.contains_point((1.5, 0.5))


def test_output_contract_is_a_300_dpi_png():
    assert OUTPUT_DPI == 300
    assert OUTPUT_PATH.suffix == ".png"


def test_final_figure_title_is_exact():
    assert FIGURE_TITLE == "Temperature Field and Population Weighting in Germany and Austria"


def test_selected_timestamp_has_the_required_shared_local_time_text():
    timestamp = pd.Timestamp("2021-02-13 06:00", tz="UTC")

    assert (
        _local_time_summary(timestamp)
        == "Local time in Germany and Austria: 13 February 2021, 07:00 CET"
    )


def test_map_uses_equal_geographic_aspect():
    assert MAP_ASPECT == "equal"


def test_population_legend_values_are_supported_by_both_weight_distributions():
    weights = {
        "Germany": pd.DataFrame({"weight": [0.001, 0.005, 0.009]}),
        "Austria": pd.DataFrame({"weight": [0.002, 0.006, 0.08]}),
    }

    values = population_legend_values(weights)

    assert len(values) == 3
    assert values == tuple(sorted(values))
    assert all(value <= 0.009 for value in values)


def test_map_and_legend_use_the_same_area_to_marker_size_rule():
    share = 0.005

    assert bubble_marker_size(share) ** 2 == pytest.approx(bubble_area(share))


def test_validate_color_scale_reports_unclipped_values():
    result = validate_color_scale(
        {"Germany": np.array([-2.0, 0.0]), "Austria": np.array([-1.0, 1.0])},
        vmin=-2.0,
        vmax=1.0,
    )

    assert result == {"below_min": 0, "above_max": 0, "total": 4}
