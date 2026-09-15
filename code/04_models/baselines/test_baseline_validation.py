import numpy as np
import pandas as pd

import baseline_validation_2024 as validation


def _write_minimal_inputs(directory):
    utc = pd.date_range("2023-12-31 00:00", "2024-01-02 22:00", freq="h", tz="UTC")
    for country, timezone_name, filename in (
        ("Germany", "Europe/Berlin", "modelling_germany_hourly.csv"),
        ("Austria", "Europe/Vienna", "modelling_austria_hourly.csv"),
    ):
        local = utc.tz_convert(timezone_name)
        pd.DataFrame(
            {
                "timestamp_utc": utc,
                "timestamp_local": local.map(pd.Timestamp.isoformat),
                "actual_grid_load_mwh": np.arange(len(utc), dtype=float) + 100,
                "hour": local.hour,
                "day_of_week": local.dayofweek,
            }
        ).to_csv(directory / filename, index=False)


def test_smoke_test_covers_ordinary_and_dst_days():
    result = validation.run_smoke_test()

    assert result["ordinary_target_intervals"] == 24
    assert result["spring_dst_target_intervals"] == 23
    assert result["autumn_dst_target_intervals"] == 25
    assert result["countries"] == ["Germany", "Austria"]


def test_complete_model_failure_writes_zero_coverage_without_name_error(tmp_path, monkeypatch):
    _write_minimal_inputs(tmp_path)
    output_tables = tmp_path / "tables"
    output_forecasts = tmp_path / "forecasts"

    def fail_forecast(*args, **kwargs):
        raise RuntimeError("forced forecast failure")

    monkeypatch.setattr(validation, "forecast_path", fail_forecast)
    monkeypatch.setattr(validation, "TABLE_DIRECTORY", output_tables)
    monkeypatch.setattr(validation, "FORECAST_DIRECTORY", output_forecasts)

    result = validation.run_validation(processed_directory=tmp_path)

    summary = pd.read_csv(output_tables / "baseline_validation_2024.csv")
    assert result["failure_rows"] == 16
    assert len(summary) == 8
    assert summary["evaluated_observations"].eq(0).all()
    assert summary["coverage"].eq(0.0).all()
