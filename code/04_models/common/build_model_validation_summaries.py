from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_YEAR = 2024
COUNTRIES = ("Germany", "Austria")
MODEL_FAMILY_ORDER = (
    "baselines",
    "holt_winters",
    "arima",
    "sarima",
    "sarima_intervention",
    "regression_arima_errors",
    "arima_intervention",
    "regression_arima_errors_intervention",
    "sarimax",
)
FUTURE_MODEL_FAMILIES = MODEL_FAMILY_ORDER[3:]
BASELINE_SPECIFICATION_ORDER = (
    "Naive",
    "Daily seasonal naive",
    "Weekly seasonal naive",
    "Average hour-of-week profile",
)

ALL_MODELS_COLUMNS = [
    "country",
    "model_family",
    "specification_id",
    "selected_within_family",
    "mae",
    "rmse",
    "mape",
    "coverage",
    "completed_jobs",
    "failed_jobs",
    "evaluated_observations",
    "validation_year",
    "notes",
    "source_result_file",
]
REGISTRY_COLUMNS = [
    "country",
    "model_family",
    "selected_specification_id",
    "selection_year",
    "primary_selection_metric",
    "selection_value",
    "selection_source",
    "status",
    "notes",
]


@dataclass(frozen=True)
class FamilySource:
    summary_path: Path
    selection_path: Path | None = None
    forecasts_path: Path | None = None
    failures_path: Path | None = None


SOURCE_CONFIG: dict[str, FamilySource | None] = {
    "baselines": FamilySource(
        summary_path=Path("results/baselines/tables/baseline_validation_2024.csv"),
        forecasts_path=Path("results/baselines/forecasts/baseline_forecasts_2024.csv"),
        failures_path=Path("results/baselines/tables/validation_failures.csv"),
    ),
    "holt_winters": FamilySource(
        summary_path=Path(
            "results/exponential_smoothing/holt_winters/validation_2024/"
            "holt_winters_validation_2024_summary.csv"
        ),
        selection_path=Path(
            "results/exponential_smoothing/holt_winters/validation_2024/"
            "holt_winters_selected_specifications_2024.csv"
        ),
    ),
    "arima": FamilySource(
        summary_path=Path(
            "results/arima/validation_2024/full_validation/"
            "arima_validation_2024_summary.csv"
        ),
        selection_path=Path("results/arima/tables/arima_selected_2024.csv"),
    ),
    "sarima": FamilySource(
        summary_path=Path(
            "results/arima_sarima/validation_2024/full_validation/"
            "sarima_validation_2024_summary.csv"
        ),
        selection_path=Path("results/arima_sarima/tables/sarima_selected_2024.csv"),
    ),
    "regression_arima_errors": FamilySource(
        summary_path=Path(
            "results/regression_arima_errors/validation_2024/full_validation/"
            "regression_arima_errors_validation_2024_summary.csv"
        ),
        selection_path=Path(
            "results/regression_arima_errors/tables/"
            "regression_arima_errors_selected_2024.csv"
        ),
    ),
    "arima_intervention": FamilySource(
        summary_path=Path(
            "results/arima_intervention/validation_2024/full_validation/"
            "arima_intervention_validation_2024_summary.csv"
        ),
    ),
    "sarima_intervention": FamilySource(
        summary_path=Path(
            "results/arima_sarima/validation_2024/full_validation_crisis/"
            "sarima_intervention_validation_2024_summary.csv"
        ),
    ),
    "regression_arima_errors_intervention": FamilySource(
        summary_path=Path(
            "results/regression_arima_errors_intervention/validation_2024/"
            "full_validation/"
            "regression_arima_errors_intervention_validation_2024_summary.csv"
        ),
        selection_path=Path(
            "results/regression_arima_errors_intervention/tables/"
            "regression_arima_errors_intervention_selected_2024.csv"
        ),
    ),
    "sarimax": None,
}


def _resolve(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _baseline_job_counts(
    source: FamilySource,
    project_root: Path,
) -> dict[tuple[str, str], tuple[int, int]]:
    if source.forecasts_path is None or source.failures_path is None:
        raise ValueError("baseline source requires forecasts and failures paths")
    forecasts = _read_csv(
        _resolve(source.forecasts_path, project_root),
        {"country", "model_family", "target_date", "evaluated"},
    )
    failures = _read_csv(
        _resolve(source.failures_path, project_root),
        {"country", "model_family", "target_date"},
    )
    successful = forecasts.loc[forecasts["evaluated"].map(_as_bool)].copy()
    successful_keys = successful.loc[
        :, ["country", "model_family", "target_date"]
    ].drop_duplicates()
    failure_keys = failures.loc[
        :, ["country", "model_family", "target_date"]
    ].drop_duplicates()
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    keys = pd.concat(
        [
            successful_keys.loc[:, ["country", "model_family"]],
            failure_keys.loc[:, ["country", "model_family"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    for row in keys.itertuples(index=False):
        key = (str(row.country), str(row.model_family))
        completed = int(
            (
                successful_keys["country"].eq(key[0])
                & successful_keys["model_family"].eq(key[1])
            ).sum()
        )
        failed = int(
            (
                failure_keys["country"].eq(key[0])
                & failure_keys["model_family"].eq(key[1])
            ).sum()
        )
        counts[key] = (completed, failed)
    return counts


def _required_summary_columns(model_family: str) -> set[str]:
    common = {
        "country",
        "mae",
        "rmse",
        "mape",
        "coverage",
        "evaluated_observations",
    }
    if model_family == "baselines":
        return common | {"model_family"}
    return common | {"specification_id", "completed_jobs", "failed_jobs"}


def _validate_summary_coverage(
    model_family: str,
    frame: pd.DataFrame,
    summary_path: Path,
) -> None:
    if set(frame["country"].astype(str)) != set(COUNTRIES):
        raise ValueError(f"{summary_path} must contain exactly Germany and Austria")
    key_column = "model_family" if model_family == "baselines" else "specification_id"
    if frame.duplicated(["country", key_column]).any():
        raise ValueError(f"{summary_path} contains duplicate country/specification rows")
    if model_family == "baselines":
        specifications = set(frame[key_column].astype(str))
        if specifications != set(BASELINE_SPECIFICATION_ORDER):
            raise ValueError(
                f"{summary_path} must contain exactly the four configured baseline specifications"
            )
        expected_keys = {
            (country, specification)
            for country in COUNTRIES
            for specification in BASELINE_SPECIFICATION_ORDER
        }
        actual_keys = set(
            zip(frame["country"].astype(str), frame[key_column].astype(str))
        )
        if actual_keys != expected_keys:
            raise ValueError(
                f"{summary_path} must contain every country/specification baseline row"
            )


def _normalize_summary(
    model_family: str,
    source: FamilySource,
    project_root: Path,
) -> pd.DataFrame:
    summary_path = _resolve(source.summary_path, project_root)
    frame = _read_csv(summary_path, _required_summary_columns(model_family))
    _validate_summary_coverage(model_family, frame, summary_path)
    rows: list[dict[str, object]] = []
    baseline_counts = (
        _baseline_job_counts(source, project_root)
        if model_family == "baselines"
        else {}
    )
    for source_order, row in enumerate(frame.to_dict(orient="records")):
        country = str(row["country"])
        specification_id = (
            str(row["model_family"])
            if model_family == "baselines"
            else str(row["specification_id"])
        )
        if model_family == "baselines":
            completed_jobs, failed_jobs = baseline_counts.get(
                (country, specification_id), (0, 0)
            )
            notes = row.get("diagnostic_note", "Deterministic baseline validation summary")
        else:
            completed_jobs = row["completed_jobs"]
            failed_jobs = row["failed_jobs"]
            diagnostic_note = row.get("diagnostic_note")
            notes = f"Authoritative {model_family} validation summary."
            if pd.notna(diagnostic_note) and str(diagnostic_note).strip():
                notes = f"{notes} {str(diagnostic_note).strip()}"
        normalized: dict[str, object] = {
            "country": country,
            "model_family": model_family,
            "specification_id": specification_id,
            "selected_within_family": False,
            "mae": row["mae"],
            "rmse": row["rmse"],
            "mape": row["mape"],
            "coverage": row["coverage"],
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "evaluated_observations": row["evaluated_observations"],
            "validation_year": VALIDATION_YEAR,
            "notes": notes,
            "diagnostic_note": row.get("diagnostic_note"),
            "source_result_file": _display_path(summary_path, project_root),
            "_source_order": source_order,
        }
        if model_family == "holt_winters" and "specification_order" in row:
            normalized["_specification_order"] = row["specification_order"]
        rows.append(normalized)
    return pd.DataFrame(rows)


def _selection_ids(
    model_family: str,
    source: FamilySource,
    project_root: Path,
) -> tuple[dict[str, str], str]:
    if source.selection_path is None:
        raise ValueError(f"{model_family} source requires a selection path")
    selection_path = _resolve(source.selection_path, project_root)
    selection = _read_csv(selection_path, {"country"})
    if "selected_specification_id" in selection.columns:
        id_column = "selected_specification_id"
    elif "specification_id" in selection.columns:
        id_column = "specification_id"
    else:
        raise ValueError(
            f"{selection_path} must contain selected_specification_id or specification_id"
        )
    if set(selection["country"].astype(str)) != set(COUNTRIES):
        raise ValueError(f"{selection_path} must contain exactly Germany and Austria")
    if selection.duplicated(["country"]).any():
        raise ValueError(f"{selection_path} contains duplicate selected countries")
    selected = {
        str(row.country): str(getattr(row, id_column))
        for row in selection.itertuples(index=False)
    }
    return selected, _display_path(selection_path, project_root)


def _metric_value(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    return number if math.isfinite(number) else math.inf


def _select_baselines(frame: pd.DataFrame) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    specification_order = {
        specification_id: index
        for index, specification_id in enumerate(BASELINE_SPECIFICATION_ORDER)
    }
    for country in COUNTRIES:
        candidates = frame.loc[frame["country"].eq(country)]
        if candidates.empty:
            continue
        finite = candidates.loc[
            candidates[["mae", "rmse", "mape", "coverage"]]
            .apply(pd.to_numeric, errors="coerce")
            .map(math.isfinite)
            .all(axis=1)
        ]
        if finite.empty:
            raise ValueError(f"no finite baseline metrics for {country}")
        winner = min(
            finite.to_dict(orient="records"),
            key=lambda row: (
                _metric_value(row["mae"]),
                _metric_value(row["rmse"]),
                _metric_value(row["mape"]),
                specification_order.get(
                    str(row["specification_id"]), len(specification_order)
                ),
                str(row["specification_id"]),
            ),
        )
        selected.add((country, str(winner["specification_id"])))
    return selected


def _specification_sort_key(row: pd.Series) -> str:
    family = str(row["model_family"])
    specification_id = str(row["specification_id"])
    if family == "baselines":
        rank = {
            value: index for index, value in enumerate(BASELINE_SPECIFICATION_ORDER)
        }.get(specification_id, len(BASELINE_SPECIFICATION_ORDER))
        return f"{rank:08d}:{specification_id}"
    if family == "holt_winters":
        order = row.get("_specification_order")
        try:
            if pd.notna(order):
                return f"{int(order):08d}:{specification_id}"
        except (TypeError, ValueError):
            pass
    return f"99999999:{specification_id}"


def _sort_all_models(frame: pd.DataFrame) -> pd.DataFrame:
    country_order = {country: index for index, country in enumerate(COUNTRIES)}
    family_order = {
        family: index for index, family in enumerate(MODEL_FAMILY_ORDER)
    }
    ordered = frame.copy()
    ordered["_country_order"] = ordered["country"].map(country_order).fillna(len(COUNTRIES))
    ordered["_family_order"] = ordered["model_family"].map(family_order).fillna(
        len(MODEL_FAMILY_ORDER)
    )
    ordered["_specification_sort"] = ordered.apply(_specification_sort_key, axis=1)
    return (
        ordered.sort_values(
            ["_country_order", "_family_order", "_specification_sort", "_source_order"],
            kind="mergesort",
        )
        .drop(columns=["_country_order", "_family_order", "_specification_sort"])
        .reset_index(drop=True)
    )


def _validate_unique_keys(frame: pd.DataFrame) -> None:
    key_columns = ["country", "model_family", "specification_id"]
    if frame.duplicated(key_columns).any():
        duplicates = frame.loc[frame.duplicated(key_columns, keep=False), key_columns]
        raise ValueError(f"duplicate normalized keys: {duplicates.to_dict(orient='records')}")


def build_model_validation_summaries(
    source_config: Mapping[str, FamilySource | None] = SOURCE_CONFIG,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, pd.DataFrame]:
    root = Path(project_root)
    required_families = ("baselines", "holt_winters", "arima")
    missing = [
        family
        for family in required_families
        if source_config.get(family) is None
    ]
    if missing:
        raise ValueError(f"missing completed family sources: {missing}")

    completed_families = tuple(
        family
        for family in MODEL_FAMILY_ORDER
        if source_config.get(family) is not None
    )
    normalized_by_family = {
        family: _normalize_summary(family, source_config[family], root)
        for family in completed_families
    }
    all_models = pd.concat(
        [normalized_by_family[family] for family in completed_families],
        ignore_index=True,
    )
    _validate_unique_keys(all_models)

    baseline_winners = _select_baselines(normalized_by_family["baselines"])
    selected_sources: dict[tuple[str, str, str], str] = {}
    for family in completed_families:
        if family == "baselines" or source_config[family].selection_path is None:
            continue
        normalized = normalized_by_family[family]
        source = source_config[family]
        selected, selection_source = _selection_ids(family, source, root)
        for country, specification_id in selected.items():
            key = (country, family, specification_id)
            if not ((normalized["country"] == country) & (normalized["specification_id"] == specification_id)).any():
                raise ValueError(
                    f"{family} selection does not identify a summary row: {country}, {specification_id}"
                )
            selected_row = normalized.loc[
                normalized["country"].eq(country)
                & normalized["specification_id"].eq(specification_id)
            ].iloc[0]
            if not all(
                math.isfinite(float(selected_row[column]))
                for column in ("mae", "rmse", "mape", "coverage")
            ):
                raise ValueError(
                    f"{family} selection has non-finite validation metrics: "
                    f"{country}, {specification_id}"
                )
            selected_sources[key] = selection_source

    for index, row in all_models.iterrows():
        key = (row["country"], row["model_family"], row["specification_id"])
        is_selected = (
            (row["country"], row["specification_id"]) in baseline_winners
            if row["model_family"] == "baselines"
            else key in selected_sources
        )
        all_models.at[index, "selected_within_family"] = bool(is_selected)
        if row["model_family"] != "baselines" and is_selected:
            all_models.at[index, "_selection_source"] = selected_sources[key]
        elif row["model_family"] == "baselines" and is_selected:
            all_models.at[index, "_selection_source"] = row["source_result_file"]

    all_models = _sort_all_models(all_models)
    output_all_models = all_models.loc[:, ALL_MODELS_COLUMNS].copy()
    selected_overview = _sort_all_models(
        all_models.loc[all_models["selected_within_family"]]
    ).loc[:, ALL_MODELS_COLUMNS]

    registry_rows: list[dict[str, object]] = []
    for row in all_models.loc[all_models["selected_within_family"]].to_dict(
        orient="records"
    ):
        family = str(row["model_family"])
        if family == "baselines":
            status = "benchmark"
            selection_source = row["source_result_file"]
            notes = (
                "Benchmark representative; not frozen and retained as a candidate "
                "for future testing."
            )
        else:
            status = "frozen"
            selection_source = row.get("_selection_source")
            notes = f"Selected {VALIDATION_YEAR} specification; frozen for registry use."
            diagnostic_note = row.get("diagnostic_note")
            if pd.notna(diagnostic_note) and str(diagnostic_note).strip():
                notes = f"{notes} {str(diagnostic_note).strip()}"
        registry_rows.append(
            {
                "country": row["country"],
                "model_family": family,
                "selected_specification_id": row["specification_id"],
                "selection_year": VALIDATION_YEAR,
                "primary_selection_metric": "mae",
                "selection_value": row["mae"],
                "selection_source": selection_source,
                "status": status,
                "notes": notes,
            }
        )
    registry = pd.DataFrame(registry_rows, columns=REGISTRY_COLUMNS)
    if not registry.empty:
        registry["_country_order"] = registry["country"].map(
            {country: index for index, country in enumerate(COUNTRIES)}
        )
        registry["_family_order"] = registry["model_family"].map(
            {family: index for index, family in enumerate(MODEL_FAMILY_ORDER)}
        )
        registry = (
            registry.sort_values(["_country_order", "_family_order"], kind="mergesort")
            .drop(columns=["_country_order", "_family_order"])
            .reset_index(drop=True)
        )
    return {
        "all_models": output_all_models,
        "selected_overview": selected_overview.reset_index(drop=True),
        "registry": registry.loc[:, REGISTRY_COLUMNS],
    }


def update_model_validation_summaries(
    source_config: Mapping[str, FamilySource | None] = SOURCE_CONFIG,
    output_directory: str | Path = PROJECT_ROOT / "results",
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Path]:
    artifacts = build_model_validation_summaries(source_config, project_root)
    output = _resolve(Path(output_directory), Path(project_root))
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_models": output / "model_validation_2024_all_models.csv",
        "selected_overview": output / "model_validation_2024_selected_overview.csv",
        "registry": output / "selected_models_registry.csv",
    }
    for name, path in paths.items():
        artifacts[name].to_csv(path, index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build central 2024 model-validation summary tables"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    args = parser.parse_args()
    paths = update_model_validation_summaries(output_directory=args.output_directory)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
