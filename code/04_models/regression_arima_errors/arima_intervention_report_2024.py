from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION = PROJECT_ROOT / "results" / "arima_intervention" / "validation_2024" / "full_validation"
TABLES = PROJECT_ROOT / "results" / "arima_intervention" / "tables"
BASE_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "arima"
    / "validation_2024"
    / "full_validation"
    / "arima_validation_2024_summary.csv"
)
CRISIS_SPECIFICATION_ID = "arima_p2_d1_q2_crisis"
BASE_SPECIFICATION_ID = "arima_p2_d1_q2"
COUNTRIES = ("Germany", "Austria")
COEFFICIENTS = ("is_covid_period", "is_post_invasion")


def _json_value(value: object, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def build_metric_comparison(base: pd.DataFrame, crisis: pd.DataFrame) -> pd.DataFrame:
    base = base.loc[base["specification_id"].eq(BASE_SPECIFICATION_ID)].copy()
    crisis = crisis.loc[crisis["specification_id"].eq(CRISIS_SPECIFICATION_ID)].copy()
    merged = base.merge(crisis, on="country", suffixes=("_base", "_crisis"), validate="one_to_one")
    rows: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        base_mae = float(row["mae_base"])
        crisis_mae = float(row["mae_crisis"])
        base_rmse = float(row["rmse_base"])
        crisis_rmse = float(row["rmse_crisis"])
        rows.append(
            {
                "country": row["country"],
                "base_model_family": "arima",
                "base_specification_id": BASE_SPECIFICATION_ID,
                "crisis_model_family": "arima_intervention",
                "crisis_specification_id": CRISIS_SPECIFICATION_ID,
                "base_mae": base_mae,
                "crisis_mae": crisis_mae,
                "mae_change_mwh": crisis_mae - base_mae,
                "mae_change_pct": (crisis_mae - base_mae) / base_mae * 100.0,
                "base_rmse": base_rmse,
                "crisis_rmse": crisis_rmse,
                "rmse_change_mwh": crisis_rmse - base_rmse,
                "rmse_change_pct": (crisis_rmse - base_rmse) / base_rmse * 100.0,
                "base_mape": float(row["mape_base"]),
                "crisis_mape": float(row["mape_crisis"]),
                "mape_change_pp": float(row["mape_crisis"]) - float(row["mape_base"]),
                "base_coverage": float(row["coverage_base"]),
                "crisis_coverage": float(row["coverage_crisis"]),
                "crisis_completed_jobs": int(row.get("completed_jobs_crisis", row.get("completed_jobs", 0))),
                "crisis_failed_jobs": int(row.get("failed_jobs_crisis", row.get("failed_jobs", 0))),
                "crisis_evaluated_observations": int(
                    row.get("evaluated_observations_crisis", row.get("evaluated_observations", 0))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("country", key=lambda values: values.map({country: index for index, country in enumerate(COUNTRIES)}), ignore_index=True)


def summarize_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        frame = jobs.loc[jobs["country"].eq(country)].copy()
        retries = pd.to_numeric(frame["optimizer_retry_count"], errors="coerce").fillna(0)
        converged = frame["converged"].map(_as_bool)
        warnings = frame["warning_messages"].map(lambda value: bool(_json_value(value, [])))
        expected = pd.to_numeric(frame["expected_observations"], errors="coerce").sum()
        evaluated = pd.to_numeric(frame["evaluated_observations"], errors="coerce").sum()
        rows.append(
            {
                "country": country,
                "total_jobs": len(frame),
                "completed_jobs": int(frame["status"].eq("completed").sum()),
                "failed_jobs": int(frame["status"].eq("failed").sum()),
                "converged_jobs": int(converged.sum()),
                "nonconverged_jobs": int((~converged).sum()),
                "optimizer_retry_total": int(retries.sum()),
                "optimizer_retry_jobs": int(retries.gt(0).sum()),
                "warning_jobs": int(sum(bool(value) for value in warnings.tolist())),
                "expected_observations": int(expected),
                "evaluated_observations": int(evaluated),
                "coverage": float(evaluated / expected) if expected else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_coefficients(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        frame = jobs.loc[jobs["country"].eq(country)]
        for coefficient in COEFFICIENTS:
            estimates: list[float] = []
            standard_errors: list[float] = []
            for row in frame.itertuples(index=False):
                parameters = _json_value(row.parameter_estimates, {})
                errors = _json_value(row.standard_errors, {})
                estimate = parameters.get(coefficient)
                standard_error = errors.get(coefficient)
                if estimate is not None and np.isfinite(float(estimate)):
                    estimates.append(float(estimate))
                if standard_error is not None and np.isfinite(float(standard_error)):
                    standard_errors.append(float(standard_error))
            t_values = [
                abs(estimate / standard_error)
                for estimate, standard_error in zip(estimates, standard_errors)
                if standard_error > 0
            ]
            rows.append(
                {
                    "country": country,
                    "coefficient": coefficient,
                    "jobs_with_finite_estimate": len(estimates),
                    "mean_estimate": float(np.mean(estimates)) if estimates else float("nan"),
                    "median_estimate": float(np.median(estimates)) if estimates else float("nan"),
                    "std_estimate": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
                    "minimum_estimate": min(estimates) if estimates else float("nan"),
                    "maximum_estimate": max(estimates) if estimates else float("nan"),
                    "jobs_with_finite_standard_error": len(standard_errors),
                    "mean_standard_error": float(np.mean(standard_errors)) if standard_errors else float("nan"),
                    "absolute_t_ge_1_96_jobs": int(sum(value >= 1.96 for value in t_values)),
                }
            )
    return pd.DataFrame(rows)


def summarize_residuals(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        frame = jobs.loc[jobs["country"].eq(country)]
        acf_flagged = []
        significant = []
        warning = []
        effective = pd.to_numeric(frame["residual_n_effective"], errors="coerce")
        pvalues_by_lag: dict[int, list[float]] = {24: [], 48: []}
        for row in frame.itertuples(index=False):
            flagged = _json_value(row.residual_acf_flagged_lags, [])
            pvalues = _json_value(row.residual_ljung_box_pvalues, {})
            flagged = list(flagged) if isinstance(flagged, list) else []
            pvalues = pvalues if isinstance(pvalues, dict) else {}
            pvalue_values = [float(value) for value in pvalues.values() if np.isfinite(float(value))]
            has_significant = any(value < 0.01 for value in pvalue_values)
            acf_flagged.append(bool(flagged))
            significant.append(has_significant)
            warning.append(bool(flagged) or has_significant)
            for lag in pvalues_by_lag:
                value = pvalues.get(str(lag), pvalues.get(lag))
                if value is not None and np.isfinite(float(value)):
                    pvalues_by_lag[lag].append(float(value))
        rows.append(
            {
                "country": country,
                "jobs": len(frame),
                "mean_residual_n_effective": float(effective.mean()),
                "acf_flagged_jobs": int(sum(acf_flagged)),
                "ljung_box_significant_jobs_alpha_0_01": int(sum(significant)),
                "residual_warning_jobs": int(sum(warning)),
                "mean_ljung_box_pvalue_lag_24": float(np.mean(pvalues_by_lag[24])) if pvalues_by_lag[24] else float("nan"),
                "mean_ljung_box_pvalue_lag_48": float(np.mean(pvalues_by_lag[48])) if pvalues_by_lag[48] else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, name: str) -> Path:
    path = TABLES / name
    frame.to_csv(path, index=False)
    return path


def _fmt(value: object) -> str:
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    crisis_summary = pd.read_csv(VALIDATION / "arima_intervention_validation_2024_summary.csv")
    base_summary = pd.read_csv(BASE_SUMMARY)
    jobs = pd.read_csv(VALIDATION / "arima_intervention_validation_2024_jobs.csv")
    comparison = build_metric_comparison(base_summary, crisis_summary)
    run_summary = summarize_jobs(jobs)
    coefficients = summarize_coefficients(jobs)
    residuals = summarize_residuals(jobs)
    paths = {
        "comparison": _write(comparison, "arima_vs_crisis_comparison_2024.csv"),
        "coefficients": _write(coefficients, "arima_intervention_coefficients_2024.csv"),
        "residuals": _write(residuals, "arima_intervention_residual_diagnostics_2024.csv"),
        "run_summary": _write(run_summary, "arima_intervention_run_summary_2024.csv"),
    }

    report = [
        "# 2024 ARIMA Crisis Experiment",
        "",
        "Model family: `arima_intervention`.",
        "Specification: `arima_p2_d1_q2_crisis`.",
        "Base error model: ARIMA(2,1,2), trend=`n`, jointly estimated regression coefficients and ARIMA errors, with stationarity and invertibility enforcement enabled.",
        "",
        "## Exogenous Variables",
        "",
        "Only `is_covid_period` and `is_post_invasion` were used. COVID is inclusive from 2020-03-11 through 2023-05-05; post-invasion is inclusive from 2022-02-24 onward. No calendar variables, temperature, SMARD/APG forecasts, future load, future observed temperature, or 2025 data were used.",
        "",
        "## Validation Design",
        "",
        "Daily refitting used the established rolling information set, Germany's 18:00 previous-day origin, Austria's 08:00 previous-day origin, bridge forecasts through the target day, target-day-only scoring, and the existing DST-aware target extraction. Full validation completed 366 days per country and 8,784 target observations per country.",
        "",
        "## Accuracy Comparison",
        "",
    ]
    for row in comparison.to_dict(orient="records"):
        report.extend(
            [
                f"### {row['country']}",
                "",
                f"- Plain ARIMA(2,1,2): MAE {_fmt(row['base_mae'])} MWh; RMSE {_fmt(row['base_rmse'])} MWh; MAPE {_fmt(row['base_mape'])}%; coverage {_fmt(row['base_coverage'])}.",
                f"- Crisis ARIMA: MAE {_fmt(row['crisis_mae'])} MWh; RMSE {_fmt(row['crisis_rmse'])} MWh; MAPE {_fmt(row['crisis_mape'])}%; coverage {_fmt(row['crisis_coverage'])}.",
                f"- Change: MAE {_fmt(row['mae_change_mwh'])} MWh ({_fmt(row['mae_change_pct'])}%); RMSE {_fmt(row['rmse_change_mwh'])} MWh ({_fmt(row['rmse_change_pct'])}%); MAPE {_fmt(row['mape_change_pp'])} percentage points.",
                "",
            ]
        )
    report.extend(
        [
            "## Fit and Residual Diagnostics",
            "",
            "Detailed convergence/retry counts, crisis coefficient summaries, and residual diagnostics are in the companion CSV tables. A coefficient's `absolute_t_ge_1_96_jobs` count uses the absolute estimate divided by its finite standard error.",
            "",
            "## Central Summary Status",
            "",
            "The experiment is retained as an unselected comparator. It does not replace the already selected plain ARIMA specification and is not added to the frozen model registry.",
        ]
    )
    report_path = TABLES / "arima_intervention_report_2024.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    paths["report"] = report_path
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
