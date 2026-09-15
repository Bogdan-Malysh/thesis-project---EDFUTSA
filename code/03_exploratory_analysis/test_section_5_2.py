from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from section_5_2 import (
    EXPECTED_ROWS,
    FIGURE_DPI,
    HOUR_TICKS,
    HOURS,
    build_hourly_profile,
    build_profile_table,
    load_country_data,
    run_section_5_2,
    validate_country_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
COUNTRIES = ["Germany", "Austria"]


def _small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual_grid_load_mwh": [10.0, 20.0, 30.0, 40.0],
            "hour": [0, 0, 1, 1],
        }
    )


def test_figure_contract_uses_local_hours_and_300_dpi():
    assert list(HOURS) == list(range(24))
    assert list(HOUR_TICKS) == list(range(0, 24, 2))
    assert FIGURE_DPI == 300


@pytest.mark.parametrize("country", COUNTRIES)
def test_real_inputs_validate_with_complete_local_hour_coverage(country):
    path = PROCESSED / f"modelling_{country.lower()}_hourly.csv"
    before = path.read_bytes()

    data = load_country_data(country)
    validation = validate_country_data(data, country)

    assert list(data.columns) == ["actual_grid_load_mwh", "hour"]
    assert validation["row_count"] == EXPECTED_ROWS
    assert validation["hours"] == list(range(24))
    assert validation["hour_counts"] == [2_192] * 24
    assert validation["finite_actual_load"] is True
    assert path.read_bytes() == before


def test_validation_rejects_invalid_hour_and_nonfinite_load():
    data = _small_frame()
    data.loc[0, "actual_grid_load_mwh"] = np.nan
    data.loc[1, "hour"] = 24

    with pytest.raises(ValueError, match="finite"):
        validate_country_data(data, "Germany")


def test_hourly_profile_calculates_requested_statistics():
    profile = build_hourly_profile(_small_frame(), "Germany")

    hour_zero = profile.loc[profile["hour"].eq(0)].iloc[0]
    assert hour_zero["number_of_observations"] == 2
    assert hour_zero["mean_hourly_grid_load_mwh"] == pytest.approx(15.0)
    assert hour_zero["median_hourly_grid_load_mwh"] == pytest.approx(15.0)
    assert hour_zero["standard_deviation_mwh"] == pytest.approx(np.std([10, 20], ddof=1))
    assert hour_zero["first_quartile_mwh"] == pytest.approx(12.5)
    assert hour_zero["third_quartile_mwh"] == pytest.approx(17.5)


@pytest.mark.parametrize("country", COUNTRIES)
def test_real_profiles_have_24_hours_and_independent_extrema(country):
    data = load_country_data(country)
    profile = build_hourly_profile(data, country)
    independent = data.groupby("hour")["actual_grid_load_mwh"].mean()

    assert len(profile) == 24
    assert profile["hour"].tolist() == list(range(24))
    assert profile["number_of_observations"].sum() == EXPECTED_ROWS
    assert profile["number_of_observations"].tolist() == [2_192] * 24
    assert not profile.isna().any().any()
    assert np.isfinite(profile.select_dtypes(include=np.number).to_numpy()).all()
    assert profile.loc[profile["mean_hourly_grid_load_mwh"].idxmin(), "hour"] == independent.idxmin()
    assert profile.loc[profile["mean_hourly_grid_load_mwh"].idxmax(), "hour"] == independent.idxmax()


def test_profile_table_has_exactly_48_rows():
    table = build_profile_table(
        {country: load_country_data(country) for country in COUNTRIES}
    )

    assert len(table) == 48
    assert table["country"].tolist().count("Germany") == 24
    assert table["country"].tolist().count("Austria") == 24
    assert table.groupby("country")["hour"].apply(list).to_dict() == {
        "Germany": list(range(24)),
        "Austria": list(range(24)),
    }


def test_section_outputs_are_generated_in_an_isolated_directory(tmp_path):
    input_paths = [
        PROCESSED / "modelling_germany_hourly.csv",
        PROCESSED / "modelling_austria_hourly.csv",
    ]
    before = {path: path.read_bytes() for path in input_paths}

    result = run_section_5_2(tmp_path)

    expected_names = {
        "hourly_demand_profile.csv",
        "figure_5_2_hourly_demand_patterns.png",
        "figure_5_2_hourly_demand_patterns.pdf",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    assert len(result["profile"]) == 48
    assert all(path.read_bytes() == content for path, content in before.items())

    with Image.open(tmp_path / "figure_5_2_hourly_demand_patterns.png") as image:
        assert image.info["dpi"][0] >= 299
        assert image.width > 700
        assert image.height > 500
    assert (tmp_path / "figure_5_2_hourly_demand_patterns.pdf").stat().st_size > 0
