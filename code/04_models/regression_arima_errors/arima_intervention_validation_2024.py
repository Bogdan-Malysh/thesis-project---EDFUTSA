from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import regression_arima_errors_validation_2024 as base_validation


MODEL_FAMILY = "arima_intervention"
SPECIFICATION_IDS = ("arima_p2_d1_q2_crisis",)
MINI_DATES = base_validation.MINI_DATES
PROCESSED = base_validation.PROCESSED
COUNTRY_CONFIG = base_validation.COUNTRY_CONFIG


def build_manifest(
    target_dates_by_country: dict[str, Iterable[date]],
) -> pd.DataFrame:
    return base_validation.build_manifest(
        target_dates_by_country,
        specification_ids=SPECIFICATION_IDS,
    )


def run_validation(
    target_dates_by_country: dict[str, Iterable[date]],
    output_directory: str | Path,
    workers: int = 1,
    processed_directory: str | Path = PROCESSED,
) -> dict[str, object]:
    return base_validation.run_validation(
        target_dates_by_country,
        output_directory,
        workers=workers,
        processed_directory=processed_directory,
        model_family=MODEL_FAMILY,
        specification_ids=SPECIFICATION_IDS,
    )


def _full_target_dates(processed_directory: str | Path) -> dict[str, tuple[date, ...]]:
    return {
        country: tuple(
            pd.to_datetime(
                base_validation._validation_frame(country, processed_directory)
                .loc[lambda frame: frame["local_date"].str.startswith("2024"), "local_date"]
            )
            .dt.date.unique()
            .tolist()
        )
        for country in COUNTRY_CONFIG
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 2024 ARIMA(2,1,2) crisis-only validation"
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--mini", action="store_true")
    stage.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--processed-directory", default=str(PROCESSED))
    args = parser.parse_args()
    dates = MINI_DATES if args.mini else _full_target_dates(args.processed_directory)
    result = run_validation(
        dates,
        args.output_dir,
        workers=args.workers,
        processed_directory=args.processed_directory,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
