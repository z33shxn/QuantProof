"""Execution-timing analysis: is the backtest assuming instantaneous execution?"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantproof._utils import timedelta_ns
from quantproof.errors import QuantProofInputError


@dataclass(frozen=True)
class ExecutionTimingReport:
    """Latency statistics between signal and execution timestamps."""

    n_trades: int
    n_instantaneous: int
    n_before_signal: int
    min_latency: str
    median_latency: str
    max_latency: str
    share_instantaneous: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_execution_timing(signal_times: Any, execution_times: Any) -> ExecutionTimingReport:
    """Compare signal and execution timestamps trade by trade.

    * ``execution == signal`` → instantaneous execution (zero latency) — usually optimistic.
    * ``execution < signal`` → impossible: the trade happens before its signal exists,
      i.e. look-ahead.
    """
    s = pd.DatetimeIndex(pd.to_datetime(pd.Series(signal_times), utc=False))
    e = pd.DatetimeIndex(pd.to_datetime(pd.Series(execution_times), utc=False))
    if len(s) != len(e):
        raise QuantProofInputError("signal_times and execution_times must have equal length.")
    if (s.tz is None) != (e.tz is None):
        raise QuantProofInputError(
            "One timestamp column is timezone-aware and the other is naive; convert both to UTC."
        )
    if len(s) == 0:
        return ExecutionTimingReport(0, 0, 0, "n/a", "n/a", "n/a", float("nan"))
    lat = e - s
    lat_ns = timedelta_ns(lat)
    n = len(lat)
    return ExecutionTimingReport(
        n_trades=n,
        n_instantaneous=int(np.sum(lat_ns == 0)),
        n_before_signal=int(np.sum(lat_ns < 0)),
        min_latency=str(pd.Timedelta(int(lat_ns.min()), unit="ns")),
        median_latency=str(pd.Timedelta(int(np.median(lat_ns)), unit="ns")),
        max_latency=str(pd.Timedelta(int(lat_ns.max()), unit="ns")),
        share_instantaneous=float(np.mean(lat_ns == 0)),
    )
