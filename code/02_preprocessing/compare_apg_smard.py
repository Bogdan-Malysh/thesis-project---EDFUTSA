from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "period",
    "series",
    "matched_row_count",
    "apg_mean_mwh",
    "smard_mean_mwh",
    "mean_difference_mwh",
    "mae_mwh",
    "rmse_mwh",
    "mape_percent",
    "correlation",
]


def _metrics(apg: pd.Series, smard: pd.Series) -> dict[str, float | int]:
    apg = apg.astype(float)
    smard = smard.astype(float)
    difference = apg - smard
    nonzero_reference = smard != 0
    if len(apg) == 0:
        return {
            "matched_row_count": 0,
            "apg_mean_mwh": np.nan,
            "smard_mean_mwh": np.nan,
            "mean_difference_mwh": np.nan,
            "mae_mwh": np.nan,
            "rmse_mwh": np.nan,
            "mape_percent": np.nan,
            "correlation": np.nan,
        }
    return {
        "matched_row_count": int(len(apg)),
        "apg_mean_mwh": float(apg.mean()),
        "smard_mean_mwh": float(smard.mean()),
        "mean_difference_mwh": float(difference.mean()),
        "mae_mwh": float(difference.abs().mean()),
        "rmse_mwh": float(np.sqrt((difference**2).mean())),
        "mape_percent": float(
            (difference[nonzero_reference].abs() / smard[nonzero_reference].abs()).mean()
            * 100
        )
        if nonzero_reference.any()
        else np.nan,
        "correlation": (
            float(apg.corr(smard))
            if len(apg) > 1 and apg.nunique() > 1 and smard.nunique() > 1
            else np.nan
        ),
    }


def _valid_mask(data: pd.DataFrame, value_column: str, flag_column: str) -> pd.Series:
    mask = data[value_column].notna() & np.isfinite(data[value_column])
    if flag_column in data.columns:
        mask &= data[flag_column].astype(bool)
    return mask


def compare_processed_files(
    apg_path: str | Path, smard_path: str | Path
) -> pd.DataFrame:
    apg = pd.read_csv(apg_path)
    smard = pd.read_csv(smard_path)
    apg["interval_start_utc"] = pd.to_datetime(apg["interval_start_utc"], utc=True)
    smard["interval_start_utc"] = pd.to_datetime(
        smard["interval_start_utc"], utc=True
    )
    apg = apg.rename(
        columns={
            "interval_start_local": "apg_interval_start_local",
            "actual_load_mwh": "apg_actual_load_mwh",
            "forecasted_load_mwh": "apg_forecasted_load_mwh",
            "actual_load_valid": "apg_actual_load_valid",
            "forecasted_load_valid": "apg_forecasted_load_valid",
        }
    )
    smard = smard.rename(
        columns={
            "actual_grid_load_mwh": "smard_actual_grid_load_mwh",
            "forecasted_grid_load_mwh": "smard_forecasted_grid_load_mwh",
            "actual_grid_load_valid": "smard_actual_grid_load_valid",
            "forecasted_grid_load_valid": "smard_forecasted_grid_load_valid",
        }
    )
    merged = apg.merge(
        smard,
        on="interval_start_utc",
        how="inner",
        validate="one_to_one",
        suffixes=("_apg", "_smard"),
    )
    merged["apg_local_year"] = (
        merged["apg_interval_start_local"].astype(str).str.slice(0, 4).astype(int)
    )
    window = merged[merged["apg_local_year"].between(2020, 2022)].copy()
    series = {
        "actual": (
            "apg_actual_load_mwh",
            "smard_actual_grid_load_mwh",
            "apg_actual_load_valid",
            "smard_actual_grid_load_valid",
        ),
        "forecast": (
            "apg_forecasted_load_mwh",
            "smard_forecasted_grid_load_mwh",
            "apg_forecasted_load_valid",
            "smard_forecasted_grid_load_valid",
        ),
    }
    rows = []
    periods = [("overall", window)] + [
        (str(year), window[window["apg_local_year"] == year])
        for year in (2020, 2021, 2022)
    ]
    for period, period_data in periods:
        for name, (apg_value, smard_value, apg_flag, smard_flag) in series.items():
            valid = _valid_mask(period_data, apg_value, apg_flag) & _valid_mask(
                period_data, smard_value, smard_flag
            )
            metrics = _metrics(
                period_data.loc[valid, apg_value], period_data.loc[valid, smard_value]
            )
            rows.append({"period": period, "series": name, **metrics})
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def write_comparison(
    apg_path: str | Path,
    smard_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    comparison = compare_processed_files(apg_path, smard_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return comparison


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    comparison = write_comparison(
        project_root / "data" / "processed" / "apg_austria_hourly.csv",
        project_root / "data" / "processed" / "smard_austria_hourly.csv",
        project_root / "results" / "tables" / "apg_smard_austria_comparison.csv",
    )
    print(
        f"APG/SMARD comparison: rows={len(comparison)} | "
        "APG local years=2020-2022"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
