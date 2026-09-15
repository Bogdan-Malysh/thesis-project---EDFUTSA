from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from build_model_validation_summaries import (
    ALL_MODELS_COLUMNS,
    FamilySource,
    build_model_validation_summaries,
)


BASELINE_MODELS = (
    "Naive",
    "Daily seasonal naive",
    "Weekly seasonal naive",
    "Average hour-of-week profile",
)


def _write_sources(
    directory: Path,
    *,
    shuffled: bool = False,
    duplicate_baseline: bool = False,
) -> dict[str, FamilySource]:
    directory.mkdir(parents=True, exist_ok=True)
    baseline_rows = []
    for country, offset in (("Germany", 0), ("Austria", 100)):
        for index, specification_id in enumerate(BASELINE_MODELS):
            baseline_rows.append(
                {
                    "country": country,
                    "model_family": specification_id,
                    "mae": float(offset + index + 1),
                    "rmse": float(offset + index + 11),
                    "mape": float(offset + index + 21),
                    "evaluated_observations": 10,
                    "coverage": 0.5,
                    "diagnostic_note": "source baseline note",
                }
            )
    if duplicate_baseline:
        baseline_rows.append(dict(baseline_rows[0]))
    if shuffled:
        baseline_rows = list(reversed(baseline_rows))

    statistical_rows = []
    for country, offset in (("Germany", 0), ("Austria", 100)):
        for index, specification_id in enumerate(("spec_b", "spec_a")):
            statistical_rows.append(
                {
                    "country": country,
                    "specification_id": specification_id,
                    "specification_order": 2 - index,
                    "completed_jobs": 8,
                    "failed_jobs": 0,
                    "evaluated_observations": 10,
                    "coverage": 1.0,
                    "rmse": float(offset + index + 31),
                    "mae": float(offset + index + 41),
                    "mape": float(offset + index + 51),
                }
            )
    if shuffled:
        statistical_rows = list(reversed(statistical_rows))

    arima_rows = [
        {
            "country": country,
            "specification_id": specification_id,
            "completed_jobs": 8,
            "failed_jobs": 0,
            "evaluated_observations": 10,
            "coverage": 1.0,
            "mae": float(offset + index + 61),
            "rmse": float(offset + index + 71),
            "mape": float(offset + index + 81),
        }
        for country, offset in (("Germany", 0), ("Austria", 100))
        for index, specification_id in enumerate(("arima_b", "arima_a"))
    ]
    if shuffled:
        arima_rows = list(reversed(arima_rows))

    baseline_summary = directory / "baseline.csv"
    baseline_forecasts = directory / "baseline_forecasts.csv"
    baseline_failures = directory / "baseline_failures.csv"
    holt_summary = directory / "holt_summary.csv"
    holt_selection = directory / "holt_selection.csv"
    arima_summary = directory / "arima_summary.csv"
    arima_selection = directory / "arima_selection.csv"

    pd.DataFrame(baseline_rows).to_csv(baseline_summary, index=False)
    forecast_rows = [
        {"country": "Germany", "model_family": "Naive", "target_date": "2024-01-01", "evaluated": True},
        {"country": "Germany", "model_family": "Naive", "target_date": "2024-01-02", "evaluated": True},
        {"country": "Germany", "model_family": "Daily seasonal naive", "target_date": "2024-01-01", "evaluated": True},
        {"country": "Germany", "model_family": "Weekly seasonal naive", "target_date": "2024-01-01", "evaluated": True},
    ] + [
        {"country": "Austria", "model_family": model, "target_date": "2024-01-01", "evaluated": True}
        for model in BASELINE_MODELS
    ]
    pd.DataFrame(forecast_rows).to_csv(baseline_forecasts, index=False)
    pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-03",
                "model_family": "Weekly seasonal naive",
                "stage": "forecast",
                "error_type": "RuntimeError",
                "message": "fixture failure",
            }
        ]
    ).to_csv(baseline_failures, index=False)

    pd.DataFrame(statistical_rows).to_csv(holt_summary, index=False)
    pd.DataFrame(
        [
            {"country": "Germany", "specification_id": "spec_a"},
            {"country": "Austria", "specification_id": "spec_a"},
        ]
    ).to_csv(holt_selection, index=False)
    pd.DataFrame(arima_rows).to_csv(arima_summary, index=False)
    pd.DataFrame(
        [
            {"country": "Germany", "selected_specification_id": "arima_a"},
            {"country": "Austria", "selected_specification_id": "arima_a"},
        ]
    ).to_csv(arima_selection, index=False)

    return {
        "baselines": FamilySource(
            summary_path=baseline_summary,
            forecasts_path=baseline_forecasts,
            failures_path=baseline_failures,
        ),
        "holt_winters": FamilySource(
            summary_path=holt_summary,
            selection_path=holt_selection,
        ),
        "arima": FamilySource(
            summary_path=arima_summary,
            selection_path=arima_selection,
        ),
    }


def test_normalization_preserves_source_values_and_uses_machine_family_labels(tmp_path):
    artifacts = build_model_validation_summaries(_write_sources(tmp_path))

    all_models = artifacts["all_models"]

    assert list(all_models.columns) == ALL_MODELS_COLUMNS
    assert set(all_models["model_family"]) == {"baselines", "holt_winters", "arima"}
    hw_row = all_models.loc[
        (all_models["country"] == "Germany")
        & (all_models["model_family"] == "holt_winters")
        & (all_models["specification_id"] == "spec_a")
    ].iloc[0]
    assert hw_row["mae"] == 42.0
    assert hw_row["rmse"] == 32.0
    assert hw_row["mape"] == 52.0
    assert hw_row["completed_jobs"] == 8
    assert hw_row["validation_year"] == 2024


def test_diagnostic_notes_propagate_to_selected_summary_and_registry(tmp_path):
    sources = _write_sources(tmp_path)
    summary_path = sources["arima"].summary_path
    summary = pd.read_csv(summary_path)
    summary["diagnostic_note"] = "Residual adequacy warning: daily autocorrelation remains."
    summary.to_csv(summary_path, index=False)

    artifacts = build_model_validation_summaries(sources)

    selected = artifacts["selected_overview"].loc[
        artifacts["selected_overview"]["model_family"].eq("arima")
    ]
    assert selected["notes"].str.contains("daily autocorrelation remains").all()
    registry = artifacts["registry"].loc[
        artifacts["registry"]["model_family"].eq("arima")
    ]
    assert registry["notes"].str.contains("daily autocorrelation remains").all()


def test_baseline_job_counts_use_distinct_forecast_and_failure_keys(tmp_path):
    all_models = build_model_validation_summaries(_write_sources(tmp_path))["all_models"]

    naive = all_models.loc[
        (all_models["country"] == "Germany")
        & (all_models["specification_id"] == "Naive")
    ].iloc[0]
    weekly = all_models.loc[
        (all_models["country"] == "Germany")
        & (all_models["specification_id"] == "Weekly seasonal naive")
    ].iloc[0]

    assert naive["completed_jobs"] == 2
    assert naive["failed_jobs"] == 0
    assert weekly["completed_jobs"] == 1
    assert weekly["failed_jobs"] == 1
    assert naive["mae"] == 1.0


def test_selection_flags_overview_and_registry_statuses(tmp_path):
    artifacts = build_model_validation_summaries(_write_sources(tmp_path))
    all_models = artifacts["all_models"]
    overview = artifacts["selected_overview"]
    registry = artifacts["registry"]

    germany_baseline = all_models.loc[
        (all_models["country"] == "Germany")
        & (all_models["model_family"] == "baselines")
    ]
    assert germany_baseline.loc[
        germany_baseline["specification_id"] == "Naive", "selected_within_family"
    ].item()
    assert not germany_baseline.loc[
        germany_baseline["specification_id"] == "Daily seasonal naive", "selected_within_family"
    ].item()
    assert len(overview) == 6
    assert set(overview["model_family"]) == {"baselines", "holt_winters", "arima"}

    assert set(registry["status"]) == {"benchmark", "frozen"}
    assert (registry.loc[registry["model_family"] == "baselines", "status"] == "benchmark").all()
    assert (registry.loc[registry["model_family"] != "baselines", "status"] == "frozen").all()
    baseline_note = registry.loc[registry["model_family"] == "baselines", "notes"].iloc[0]
    assert "not frozen" in baseline_note
    assert "future testing" in baseline_note


def test_ordering_is_deterministic_and_duplicate_keys_are_rejected(tmp_path):
    artifacts = build_model_validation_summaries(
        _write_sources(tmp_path / "shuffled", shuffled=True)
    )
    all_models = artifacts["all_models"]
    expected_order = [
        ("Germany", "baselines", model) for model in BASELINE_MODELS
    ] + [
        ("Germany", "holt_winters", "spec_a"),
        ("Germany", "holt_winters", "spec_b"),
        ("Germany", "arima", "arima_a"),
        ("Germany", "arima", "arima_b"),
    ] + [
        ("Austria", "baselines", model) for model in BASELINE_MODELS
    ] + [
        ("Austria", "holt_winters", "spec_a"),
        ("Austria", "holt_winters", "spec_b"),
        ("Austria", "arima", "arima_a"),
        ("Austria", "arima", "arima_b"),
    ]
    assert list(
        all_models[["country", "model_family", "specification_id"]]
        .itertuples(index=False, name=None)
    ) == expected_order
    assert not all_models.duplicated(
        ["country", "model_family", "specification_id"]
    ).any()

    duplicate_sources = _write_sources(tmp_path / "duplicate", duplicate_baseline=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_model_validation_summaries(duplicate_sources)


def test_baseline_ties_use_fixed_specification_order_not_source_row_order(tmp_path):
    sources = _write_sources(tmp_path, shuffled=True)
    summary_path = sources["baselines"].summary_path
    summary = pd.read_csv(summary_path)
    tied = summary.loc[
        summary["country"].eq("Germany")
        & summary["model_family"].isin(["Naive", "Daily seasonal naive"])
    ].index
    summary.loc[tied, ["mae", "rmse", "mape"]] = 1.0
    summary.to_csv(summary_path, index=False)

    all_models = build_model_validation_summaries(sources)["all_models"]

    selected = all_models.loc[
        all_models["country"].eq("Germany")
        & all_models["model_family"].eq("baselines")
        & all_models["selected_within_family"]
    ]
    assert selected["specification_id"].tolist() == ["Naive"]


def test_incomplete_authoritative_family_rows_are_rejected(tmp_path):
    sources = _write_sources(tmp_path)
    summary_path = sources["baselines"].summary_path
    summary = pd.read_csv(summary_path)
    summary = summary.loc[summary["country"].ne("Austria")]
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="must contain exactly Germany and Austria"):
        build_model_validation_summaries(sources)


def test_selection_file_must_contain_both_countries(tmp_path):
    sources = _write_sources(tmp_path)
    selection_path = sources["holt_winters"].selection_path
    selection = pd.read_csv(selection_path)
    selection = selection.loc[selection["country"].eq("Germany")]
    selection.to_csv(selection_path, index=False)

    with pytest.raises(ValueError, match="must contain exactly Germany and Austria"):
        build_model_validation_summaries(sources)


def test_configured_future_family_is_added_without_schema_changes(tmp_path):
    sources = _write_sources(tmp_path)
    sources["sarima"] = sources["arima"]

    artifacts = build_model_validation_summaries(sources)

    all_models = artifacts["all_models"]
    assert set(all_models["model_family"]) == {
        "baselines",
        "holt_winters",
        "arima",
        "sarima",
    }
    assert len(all_models.loc[all_models["model_family"].eq("sarima")]) == 4
    assert (artifacts["registry"].loc[
        artifacts["registry"]["model_family"].eq("sarima"), "status"
    ] == "frozen").all()


def test_unselected_experiment_family_is_kept_out_of_registry(tmp_path):
    sources = _write_sources(tmp_path)
    crisis_summary = tmp_path / "arima_intervention_summary.csv"
    pd.DataFrame(
        [
            {
                "country": country,
                "specification_id": "arima_p2_d1_q2_crisis",
                "completed_jobs": 366,
                "failed_jobs": 0,
                "evaluated_observations": 8784,
                "coverage": 1.0,
                "mae": 10.0,
                "rmse": 20.0,
                "mape": 3.0,
            }
            for country in ("Germany", "Austria")
        ]
    ).to_csv(crisis_summary, index=False)
    sources["arima_intervention"] = FamilySource(summary_path=crisis_summary)

    artifacts = build_model_validation_summaries(sources)

    rows = artifacts["all_models"].loc[
        artifacts["all_models"]["model_family"].eq("arima_intervention")
    ]
    assert len(rows) == 2
    assert not rows["selected_within_family"].any()
    assert artifacts["registry"].loc[
        artifacts["registry"]["model_family"].eq("arima_intervention")
    ].empty


def test_sarima_intervention_is_reported_but_not_selected(tmp_path):
    sources = _write_sources(tmp_path)
    intervention_summary = tmp_path / "sarima_intervention_summary.csv"
    pd.DataFrame(
        [
            {
                "country": country,
                "specification_id": specification_id,
                "completed_jobs": 366,
                "failed_jobs": 0,
                "evaluated_observations": 8784,
                "coverage": 1.0,
                "mae": 10.0,
                "rmse": 20.0,
                "mape": 1.0,
            }
            for country, specification_id in (
                ("Germany", "sarima_p2_d0_q0_P0_D1_Q0_s24_crisis"),
                ("Austria", "sarima_p1_d0_q1_P1_D1_Q0_s24_crisis"),
            )
        ]
    ).to_csv(intervention_summary, index=False)
    sources["sarima_intervention"] = FamilySource(summary_path=intervention_summary)

    artifacts = build_model_validation_summaries(sources)

    rows = artifacts["all_models"].loc[
        artifacts["all_models"]["model_family"].eq("sarima_intervention")
    ]
    assert len(rows) == 2
    assert not rows["selected_within_family"].any()
    assert artifacts["registry"].loc[
        artifacts["registry"]["model_family"].eq("sarima_intervention")
    ].empty


def test_baseline_selection_rejects_country_without_finite_metrics(tmp_path):
    sources = _write_sources(tmp_path)
    summary_path = sources["baselines"].summary_path
    summary = pd.read_csv(summary_path)
    germany = summary["country"].eq("Germany")
    summary.loc[germany, ["mae", "rmse", "mape"]] = float("nan")
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="no finite baseline metrics"):
        build_model_validation_summaries(sources)


def test_baseline_selection_rejects_infinite_metrics(tmp_path):
    sources = _write_sources(tmp_path)
    summary_path = sources["baselines"].summary_path
    summary = pd.read_csv(summary_path)
    germany = summary["country"].eq("Germany")
    summary.loc[germany, ["mae", "rmse", "mape", "coverage"]] = float("inf")
    summary.to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match="no finite baseline metrics"):
        build_model_validation_summaries(sources)
