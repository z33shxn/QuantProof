"""Engine-agnostic adapter for tabular backtest exports.

Most backtesting engines can export (a) a time series of returns or equity and (b) a
trade ledger. :class:`GenericResultsAdapter` maps such exports onto
:class:`~quantproof.adapters.ResearchArtifacts` with explicit column names, so no
engine-specific code is needed. It does not import or emulate any engine.

Conversions are explicit and recorded in ``metadata["conversions"]``:

* an equity curve becomes simple returns ``equity_t / equity_{t-1} - 1`` (first bar
  dropped);
* trade-ledger columns are renamed to ``signal_time`` / ``execution_time``;
* ``returns_are`` (``"net"`` or ``"gross"``) is recorded and must be stated.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quantproof.adapters.base import ResearchArtifacts
from quantproof.data.loaders import load_frame, parse_timestamps, resolve_timestamp_column
from quantproof.data.schemas import normalize_column_name
from quantproof.errors import QuantProofInputError


def _frame(obj: Any, name: str) -> pd.DataFrame:
    if isinstance(obj, (str, Path)):
        return load_frame(obj)
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        return obj
    raise QuantProofInputError(f"{name}: expected a DataFrame, Series or file path.")


def _time_indexed(frame: pd.DataFrame, timestamp_column: str | None, name: str) -> pd.DataFrame:
    col = resolve_timestamp_column(frame, timestamp_column)
    if col is None:
        if isinstance(frame.index, pd.DatetimeIndex):
            out = frame
        else:
            raise QuantProofInputError(
                f"{name}: no timestamp column found; pass timestamp_column=... or use a "
                "DatetimeIndex."
            )
    else:
        out = frame.drop(columns=[col])
        out.index = parse_timestamps(frame[col]).values
    if out.index.has_duplicates:
        raise QuantProofInputError(f"{name}: duplicate timestamps; aggregate them first.")
    return out.sort_index()


class GenericResultsAdapter:
    """Map a returns/equity export and an optional trade ledger onto ResearchArtifacts.

    Parameters
    ----------
    returns_column / equity_column:
        Exactly one must be given (or found: ``returns`` / ``equity`` by default).
    returns_are:
        ``"net"`` (after the engine's costs) or ``"gross"``. Required: the audit's
        interpretation of every statistic depends on it.
    timestamp_column:
        Timestamp column of the returns export (auto-detected when omitted).
    trade_columns:
        Mapping from QuantProof names (``signal_time``, ``execution_time``) to the
        ledger's column names, e.g. ``{"signal_time": "signal_ts", "execution_time":
        "fill_ts"}``.

    Example
    -------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=4, freq="D")
    >>> export = pd.DataFrame({"date": idx, "equity": [100.0, 101.0, 100.0, 102.0]})
    >>> art = GenericResultsAdapter(equity_column="equity", returns_are="net").load(export)
    >>> art.returns.round(4).tolist()
    [0.01, -0.0099, 0.02]
    """

    name = "generic"

    def __init__(
        self,
        *,
        returns_column: str | None = None,
        equity_column: str | None = None,
        returns_are: Literal["net", "gross"],
        timestamp_column: str | None = None,
        trade_columns: Mapping[str, str] | None = None,
    ) -> None:
        if returns_are not in ("net", "gross"):
            raise QuantProofInputError("returns_are must be 'net' or 'gross'.")
        if returns_column and equity_column:
            raise QuantProofInputError("Give returns_column or equity_column, not both.")
        unknown = set(trade_columns or {}) - {"signal_time", "execution_time"}
        if unknown:
            raise QuantProofInputError(
                f"trade_columns keys must be 'signal_time'/'execution_time', got {sorted(unknown)}."
            )
        self.returns_column = returns_column
        self.equity_column = equity_column
        self.returns_are = returns_are
        self.timestamp_column = timestamp_column
        self.trade_columns = dict(trade_columns or {})

    def _returns(self, frame: pd.DataFrame, conversions: list[str]) -> pd.Series:
        # File loaders normalize column names, so match names after normalization.
        cols = {normalize_column_name(c): c for c in frame.columns}

        def find(name: str) -> Any:
            col = cols.get(normalize_column_name(name))
            if col is None:
                raise QuantProofInputError(f"Column {name!r} not found in the returns export.")
            return col

        if self.returns_column is not None:
            return frame[find(self.returns_column)].astype(float).rename("returns")
        if self.equity_column is not None:
            eq_col = find(self.equity_column)
        elif "returns" in cols:
            return frame[cols["returns"]].astype(float).rename("returns")
        elif "equity" in cols:
            eq_col = cols["equity"]
        else:
            raise QuantProofInputError(
                "No returns or equity column found; pass returns_column=... or equity_column=..."
            )
        equity = frame[eq_col].astype(float)
        if (equity <= 0).any():
            raise QuantProofInputError("Equity must be strictly positive to derive returns.")
        conversions.append(f"returns derived from equity column {eq_col!r} (first bar dropped)")
        return equity.pct_change(fill_method=None).iloc[1:].rename("returns")

    def load(
        self,
        source: Any,
        *,
        trades: Any = None,
        trial_returns: Any = None,
        benchmark: pd.Series | None = None,
        prices: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchArtifacts:
        """Build artifacts from a returns/equity export (DataFrame, Series or file path)."""
        conversions: list[str] = []
        frame = _time_indexed(_frame(source, "returns export"), self.timestamp_column, "returns")
        returns = self._returns(frame, conversions)
        ledger = None
        if trades is not None:
            ledger = _frame(trades, "trades").copy()
            actual = {normalize_column_name(c): c for c in ledger.columns}
            missing = [
                v for v in self.trade_columns.values() if normalize_column_name(v) not in actual
            ]
            if missing:
                raise QuantProofInputError(f"Trade ledger lacks columns {missing}.")
            rename = {actual[normalize_column_name(v)]: k for k, v in self.trade_columns.items()}
            if rename:
                ledger = ledger.rename(columns=rename)
                conversions.append(f"trade columns renamed: {rename}")
            for col in ("signal_time", "execution_time"):
                if col in ledger.columns:
                    ledger[col] = parse_timestamps(ledger[col]).values
        trials = None
        if trial_returns is not None:
            trials = _time_indexed(_frame(trial_returns, "trial_returns"), None, "trial_returns")
            trials = trials.astype(float)
        art = ResearchArtifacts(
            returns=returns,
            trial_returns=trials,
            benchmark=benchmark,
            prices=prices,
            trades=ledger,
            metadata={
                **(metadata or {}),
                "adapter": self.name,
                "returns_are": self.returns_are,
                "conversions": conversions,
            },
        )
        art.validate()
        return art
