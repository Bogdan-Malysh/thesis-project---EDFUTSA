from pathlib import Path

import pandas as pd
import pytest

from compare_apg_smard import compare_processed_files


def write_processed(
    path: Path,
    actual: list[float],
    forecast: list[float],
    forecast_valid: list[bool],
    smard: bool = False,
) -> None:
    starts = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-06-01 00:00",
            "2021-01-01 00:00",
            "2021-06-01 00:00",
            "2022-01-01 00:00",
            "2022-06-01 00:00",
        ],
        utc=True,
    )
    local_starts = starts.tz_convert("Europe/Vienna")
    data = pd.DataFrame(
        {
            "interval_start_utc": starts,
            "interval_start_local": local_starts,
            ("actual_grid_load_mwh" if smard else "actual_load_mwh"): actual,
            ("forecasted_grid_load_mwh" if smard else "forecasted_load_mwh"): forecast,
            ("actual_grid_load_valid" if smard else "actual_load_valid"): True,
            ("forecasted_grid_load_valid" if smard else "forecasted_load_valid"): forecast_valid,
        }
    )
    data.to_csv(path, index=False)


def test_compare_calculates_overall_and_annual_metrics_on_valid_utc_hours(tmp_path):
    write_processed(
        tmp_path / "apg.csv",
        [110, 90, 120, 80, 100, 100],
        [105, 95, 110, 90, 102, 98],
        [True, True, True, False, True, True],
        smard=False,
    )
    write_processed(
        tmp_path / "smard.csv",
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [True, True, True, True, True, True],
        smard=True,
    )

    result = compare_processed_files(tmp_path / "apg.csv", tmp_path / "smard.csv")

    assert len(result) == 8
    actual_overall = result.query("period == 'overall' and series == 'actual'").iloc[0]
    assert actual_overall["matched_row_count"] == 6
    assert actual_overall["apg_mean_mwh"] == 100.0
    assert actual_overall["smard_mean_mwh"] == 100.0
    assert actual_overall["mean_difference_mwh"] == 0.0
    assert actual_overall["mae_mwh"] == 10.0
    assert actual_overall["mape_percent"] == pytest.approx(10.0)
    assert pd.isna(actual_overall["correlation"])

    forecast_overall = result.query("period == 'overall' and series == 'forecast'").iloc[0]
    assert forecast_overall["matched_row_count"] == 5
    assert forecast_overall["apg_mean_mwh"] == 102.0
    assert forecast_overall["smard_mean_mwh"] == 100.0
    assert forecast_overall["mean_difference_mwh"] == 2.0
    assert forecast_overall["mae_mwh"] == pytest.approx(4.8)
    assert forecast_overall["mape_percent"] == pytest.approx(4.8)

    annual_2020 = result.query("period == '2020' and series == 'actual'").iloc[0]
    assert annual_2020["matched_row_count"] == 2
    assert annual_2020["mae_mwh"] == 10.0
    assert annual_2020["rmse_mwh"] == 10.0


def test_compare_uses_apg_local_year_for_the_comparison_window(tmp_path):
    write_processed(
        tmp_path / "apg.csv",
        [110, 90, 120, 80, 100, 100],
        [105, 95, 110, 90, 102, 98],
        [True, True, True, True, True, True],
        smard=False,
    )
    write_processed(
        tmp_path / "smard.csv",
        [100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100],
        [True, True, True, True, True, True],
        smard=True,
    )
    apg = pd.read_csv(tmp_path / "apg.csv")
    smard = pd.read_csv(tmp_path / "smard.csv")
    extra = apg.iloc[[0]].copy()
    extra["interval_start_utc"] = "2019-12-31 23:00:00+00:00"
    extra["interval_start_local"] = "2020-01-01 00:00:00+01:00"
    apg = pd.concat([apg, extra], ignore_index=True)
    extra = smard.iloc[[0]].copy()
    extra["interval_start_utc"] = "2019-12-31 23:00:00+00:00"
    extra["interval_start_local"] = "2020-01-01 00:00:00+01:00"
    smard = pd.concat([smard, extra], ignore_index=True)
    apg.to_csv(tmp_path / "apg.csv", index=False)
    smard.to_csv(tmp_path / "smard.csv", index=False)

    result = compare_processed_files(tmp_path / "apg.csv", tmp_path / "smard.csv")

    assert result.query("period == 'overall' and series == 'actual'").iloc[0][
        "matched_row_count"
    ] == 7
    assert result.query("period == '2020' and series == 'actual'").iloc[0][
        "matched_row_count"
    ] == 3
