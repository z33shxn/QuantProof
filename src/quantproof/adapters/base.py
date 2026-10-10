"""Generic adapter interface.

QuantProof's core never imports a backtesting engine. Engines are integrated by
adapters that convert their native outputs into :class:`ResearchArtifacts`, a
plain container of pandas objects. A future VectorBT/Backtrader/LEAN/Nautilus/Qlib
adapter only needs to implement :class:`Adapter.load`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from quantproof.errors import QuantProofInputError


@dataclass
class ResearchArtifacts:
    """Engine-agnostic research outputs. Every field is optional.

    returns        per-period strategy returns (net or gross; say which in ``metadata``).
    trial_returns  ``T x N`` returns of every variant tried (columns = variants).
    trial_params   one row per ``trial_returns`` column with its parameter values.
    benchmark      per-period benchmark returns.
    prices         OHLCV price data with a DatetimeIndex.
    signals        target weights / signals per bar.
    positions      held positions (weights) per bar.
    trades         trade ledger; columns ``signal_time`` and ``execution_time`` enable
                   execution-timing checks.
    features       model feature matrix indexed by time.
    target         prediction target aligned with ``features``.
    metadata       free-form descriptive metadata (engine, version, notes).
    """

    returns: pd.Series | None = None
    trial_returns: pd.DataFrame | None = None
    trial_params: pd.DataFrame | None = None
    benchmark: pd.Series | None = None
    prices: pd.DataFrame | None = None
    signals: pd.Series | None = None
    positions: pd.Series | pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    target: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Basic consistency checks with actionable errors."""
        if (
            self.trial_returns is not None
            and self.trial_params is not None
            and len(self.trial_params) != self.trial_returns.shape[1]
        ):
            raise QuantProofInputError(
                "trial_params must have one row per trial_returns column "
                f"({len(self.trial_params)} rows vs {self.trial_returns.shape[1]} columns)."
            )
        if self.trades is not None and {"signal_time", "execution_time"} - set(self.trades.columns):
            missing = {"signal_time", "execution_time"} - set(self.trades.columns)
            self.metadata.setdefault("notes", []).append(
                f"trades ledger lacks {sorted(missing)}; execution-timing checks skipped."
            )
        if all(
            getattr(self, f) is None
            for f in (
                "returns",
                "trial_returns",
                "prices",
                "signals",
                "positions",
                "trades",
                "features",
            )
        ):
            raise QuantProofInputError(
                "ResearchArtifacts is empty; provide at least returns or prices."
            )


@runtime_checkable
class Adapter(Protocol):
    """Converts an engine's native output into :class:`ResearchArtifacts`."""

    name: str

    def load(self, source: Any, **kwargs: Any) -> ResearchArtifacts:
        """Build artifacts from ``source``."""
        ...
