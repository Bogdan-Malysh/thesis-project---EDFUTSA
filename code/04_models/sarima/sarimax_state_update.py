from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from common.forecasting_framework import build_information_context, information_set


@dataclass
class FixedParameterState:
    result: Any
    parameter_vector: np.ndarray

    def update(self, endog: object, exog: object) -> None:
        if len(endog) == 0:
            return
        if len(exog) != len(endog):
            raise ValueError("state-update endog and exog lengths do not match")
        for name, value in (("endog", endog), ("exog", exog)):
            index = getattr(value, "index", None)
            if (
                not isinstance(index, pd.DatetimeIndex)
                or index.freq is not None
                or len(index) < 3
            ):
                continue
            inferred = pd.infer_freq(index)
            if inferred is None:
                continue
            normalized = value.copy()
            normalized.index = pd.DatetimeIndex(index, freq=inferred)
            if name == "endog":
                endog = normalized
            else:
                exog = normalized
        # statsmodels extend() filters only the new observations and internally uses refit=False.
        updated = self.result.extend(endog, exog=exog)
        updated_parameters = np.asarray(getattr(updated, "params", []), dtype=float)
        if not np.array_equal(updated_parameters, self.parameter_vector):
            raise ValueError("fixed SARIMAX parameters changed during state update")
        self.result = updated


def build_initial_training_set(
    frame: pd.DataFrame,
    country: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    context = build_information_context(frame, country, "2024-01-01")
    available = information_set(frame, context.cutoff_utc)
    local_dates = pd.to_datetime(available["local_date"], errors="raise").dt.date
    training = available.loc[local_dates <= pd.Timestamp("2023-12-31").date()].copy()
    if training.empty or training["local_date"].astype(str).max() > "2023-12-31":
        raise ValueError("initial SARIMAX training set is not bounded at 2023-12-31")
    return training, context.cutoff_utc


def observations_between_cutoffs(
    frame: pd.DataFrame,
    previous_cutoff: pd.Timestamp,
    current_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    previous = pd.Timestamp(previous_cutoff).tz_convert("UTC")
    current = pd.Timestamp(current_cutoff).tz_convert("UTC")
    if current <= previous:
        raise ValueError("state-update cutoff must move forward chronologically")
    return frame.loc[
        frame["interval_end_utc"].gt(previous)
        & frame["interval_end_utc"].le(current)
    ].sort_values("timestamp_utc", ignore_index=True)
