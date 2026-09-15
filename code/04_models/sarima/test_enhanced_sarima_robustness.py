from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from enhanced_sarima_robustness import (
    build_final_comparison,
    build_screening_calendar,
    candidate_definitions,
    select_shortlist,
)
import enhanced_sarima_robustness as robustness


def test_candidate_definitions_are_exact_targeted_set():
    candidates = candidate_definitions()

    assert len(candidates) == 14
    assert list(candidates.loc[candidates["seasonal_period"].eq(24), "order"]) == [
        (1, 0, 0, 0, 1, 0),
        (2, 0, 0, 0, 1, 0),
        (3, 0, 0, 0, 1, 0),
        (1, 0, 1, 0, 1, 0),
        (2, 0, 1, 0, 1, 0),
        (1, 0, 2, 0, 1, 0),
        (2, 0, 2, 0, 1, 0),
        (1, 0, 0, 1, 1, 0),
        (1, 0, 1, 1, 1, 0),
        (2, 0, 0, 1, 1, 0),
        (2, 0, 1, 1, 1, 0),
        (1, 0, 1, 0, 1, 1),
    ]
    assert list(candidates.loc[candidates["seasonal_period"].eq(168), "order"]) == [
        (2, 0, 0, 0, 1, 0),
        (1, 0, 1, 0, 1, 0),
    ]


def test_screening_calendar_is_fixed_and_covers_dst_seasons_and_weekends():
    calendar = build_screening_calendar()

    assert len(calendar) == 16
    assert set(calendar["target_date"]) >= {"2024-03-31", "2024-10-27"}
    assert calendar["target_date"].is_unique
    assert set(calendar["season"]) == {"winter", "spring", "summer", "autumn"}
    assert calendar["is_weekend"].any()
    assert (~calendar["is_weekend"]).any()


def test_shortlist_keeps_at_most_three_eligible_candidates_per_country():
    rows = []
    for country in ("Germany", "Austria"):
        for index in range(5):
            rows.append(
                {
                    "country": country,
                    "branch": "s24",
                    "specification_id": f"candidate_{index}",
                    "mae": float(index + (0 if country == "Germany" else 10)),
                    "rmse": float(index),
                    "mape": float(index),
                    "eligible": index != 4,
                }
            )

    shortlist = select_shortlist(pd.DataFrame(rows))

    assert shortlist.groupby("country").size().to_dict() == {
        "Germany": 3,
        "Austria": 3,
    }
    assert shortlist["eligible"].all()


def test_final_comparison_has_seven_models_per_country_and_labels_robustness():
    summary = pd.DataFrame(
        [
            {
                "country": country,
                "specification_id": "enhanced_sarima_test",
                "mae": 2.0,
                "rmse": 3.0,
                "mape": 4.0,
                "evaluated_observations": 2,
                "expected_observations": 2,
                "coverage": 1.0,
            }
            for country in ("Germany", "Austria")
        ]
    )
    forecasts = pd.DataFrame()
    comparator_metrics = pd.DataFrame(
        [
            {
                "country": country,
                "model": model,
                "mae": 1.0,
                "rmse": 2.0,
                "mape": 3.0,
                "evaluated_observations": 2,
                "expected_observations": 2,
                "coverage": 1.0,
            }
            for country in ("Germany", "Austria")
            for model in (
                "Original SARIMA",
                "Original ARIMA",
                "Enhanced ARIMA",
                "Weekly seasonal naive",
                "Holt-Winters",
            )
        ]
    )
    smard = pd.DataFrame(
        {
            "country": ["Germany", "Austria"],
            "n_observations": [2, 2],
            "coverage": [1.0, 1.0],
            "mae": [5.0, 6.0],
            "rmse": [6.0, 7.0],
            "mape": [7.0, 8.0],
        }
    )

    comparison = build_final_comparison(
        summary, forecasts, comparator_metrics, smard
    )

    assert comparison.groupby("country").size().to_dict() == {
        "Germany": 7,
        "Austria": 7,
    }
    enhanced = comparison.loc[comparison["model"] == "Enhanced SARIMA"]
    assert len(enhanced) == 2
    assert enhanced["analysis_label"].eq(
        "enhanced/post-hoc SARIMA robustness analysis"
    ).all()
    assert enhanced["selection_role"].eq("not_used_for_selection").all()
    assert comparison.loc[comparison["model"] == "SMARD", "mae"].notna().all()


def test_parallel_checkpoint_persists_before_executor_batch_returns(tmp_path, monkeypatch):
    manifest = pd.DataFrame(
        [
            {
                "country": "Germany",
                "target_date": "2024-01-01",
                "specification_id": "candidate_1",
            },
            {
                "country": "Germany",
                "target_date": "2024-01-02",
                "specification_id": "candidate_1",
            },
        ]
    )
    output = tmp_path / "validation"
    jobs, forecasts, diagnostics = robustness._load_checkpoint(output)

    def fake_worker(job):
        return {
            "job": {**job, "status": "completed", "reused_authoritative": False},
            "forecasts": [],
            "diagnostics": [],
        }

    class FakeExecutor:
        def __init__(self, max_workers, mp_context, initializer, initargs):
            self.initializer = initializer
            self.initargs = initargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, records, chunksize=1):
            records = list(records)
            yield function(records[0])
            if len(records) > 1:
                assert {
                    tuple(row)
                    for row in pd.read_csv(output / "jobs.csv")[
                        ["country", "target_date", "specification_id"]
                    ]
                    .astype(str)
                    .itertuples(index=False, name=None)
                } == {("Germany", "2024-01-01", "candidate_1")}
                yield function(records[1])

    monkeypatch.setattr(robustness, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(robustness, "_worker_execute", fake_worker)
    interrupted = {"value": True}

    def persist(result):
        nonlocal jobs, forecasts, diagnostics
        jobs, forecasts, diagnostics = robustness._persist_result(
            jobs, forecasts, diagnostics, result, output
        )
        if interrupted["value"]:
            interrupted["value"] = False
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        robustness._run_parallel(
            manifest, workers=2, processed_directory=tmp_path, on_result=persist
        )

    jobs, forecasts, diagnostics = robustness._load_checkpoint(output)
    terminal = robustness._terminal_keys(jobs)
    remaining = manifest.loc[
        ~manifest.apply(lambda row: robustness._job_key(row) in terminal, axis=1)
    ]
    assert terminal == {("Germany", "2024-01-01", "candidate_1")}
    robustness._run_parallel(
        remaining,
        workers=2,
        processed_directory=tmp_path,
        on_result=persist,
    )
    jobs, _, _ = robustness._load_checkpoint(output)
    assert set(robustness._job_key(row) for row in jobs.to_dict("records")) == {
        ("Germany", "2024-01-01", "candidate_1"),
        ("Germany", "2024-01-02", "candidate_1"),
    }
