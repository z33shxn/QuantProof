"""Multi-asset (panel) data: long format with a symbol column, or a ``(timestamp, symbol)``
MultiIndex.

Internal representation
-----------------------
A *panel* is a DataFrame indexed by a two-level MultiIndex ``(timestamp, symbol)``,
sorted by timestamp then symbol, with one row per (timestamp, symbol) pair and the usual
price columns (``open``, ``high``, ``low``, ``close``, ``volume`` …).

Strategies receive the panel and return target weights either as a *wide* DataFrame
(index = timestamps, columns = symbols) or as a Series on the panel's MultiIndex.
Weights are fractions of total portfolio equity.

Symbols need not share the same timestamps (listings, delistings, holidays); data
validation reports unsynchronized coverage (QP-DATA-015) because cross-sectional signals
computed on unsynchronized data compare prices observed at different moments.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from quantproof.data.schemas import normalize_column_name
from quantproof.errors import QuantProofDataError

SYMBOL_CANDIDATES: tuple[str, ...] = ("symbol", "ticker", "asset", "instrument", "secid", "code")


def find_symbol_column(columns: object) -> str | None:
    """First column whose normalized name is a symbol candidate."""
    lookup = {normalize_column_name(c): c for c in columns}  # type: ignore[attr-defined]
    for cand in SYMBOL_CANDIDATES:
        if cand in lookup:
            return str(lookup[cand])
    return None


def flatten_multiindex(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a 2-level (timestamp, symbol) MultiIndex into ``timestamp``/``symbol`` columns."""
    if not isinstance(frame.index, pd.MultiIndex):
        return frame
    if frame.index.nlevels != 2:
        raise QuantProofDataError(
            f"Expected a 2-level (timestamp, symbol) MultiIndex, got {frame.index.nlevels} levels."
        )
    levels = [frame.index.get_level_values(i) for i in range(2)]
    is_time = [pd.api.types.is_datetime64_any_dtype(lv.dtype) for lv in levels]
    if is_time.count(True) != 1:
        names = [str(n) for n in frame.index.names]
        time_pos = next(
            (
                i
                for i, n in enumerate(names)
                if normalize_column_name(n) in {"timestamp", "date", "datetime", "time"}
            ),
            None,
        )
        if time_pos is None:
            raise QuantProofDataError(
                "Could not tell which MultiIndex level holds timestamps. Name the levels "
                "('timestamp', 'symbol') or use datetime values for the time level."
            )
    else:
        time_pos = is_time.index(True)
    out = frame.reset_index()
    cols = list(out.columns)
    cols[time_pos] = "timestamp"
    cols[1 - time_pos] = "symbol"
    out.columns = cols
    return out


def is_panel(prices: pd.DataFrame) -> bool:
    """True for prepared panel data (2-level MultiIndex)."""
    return isinstance(prices.index, pd.MultiIndex) and prices.index.nlevels == 2


def panel_times(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Timestamps of a single-asset frame or of a panel (with duplicates for panels)."""
    if is_panel(prices):
        return pd.DatetimeIndex(prices.index.get_level_values(0))
    return pd.DatetimeIndex(prices.index)


def unique_times(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted unique timestamps."""
    return panel_times(prices).unique().sort_values()


def symbols(prices: pd.DataFrame) -> list[str]:
    """Symbols of a panel, in sorted order."""
    if not is_panel(prices):
        return []
    return [str(s) for s in prices.index.get_level_values(1).unique().sort_values()]


def wide(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    """``column`` as a timestamps × symbols table."""
    return cast(pd.DataFrame, prices[column].unstack(level=1).sort_index())


def symbol_frame(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Single-symbol OHLCV frame indexed by timestamp."""
    return cast(pd.DataFrame, prices.xs(symbol, level=1))


def to_wide_signals(raw: pd.Series | pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Coerce panel strategy output (wide frame or MultiIndex series) to timestamps × symbols."""
    if isinstance(raw, pd.Series):
        if not isinstance(raw.index, pd.MultiIndex):
            raise QuantProofDataError(
                "Panel strategies must return a wide DataFrame (timestamps × symbols) or a Series "
                "indexed by (timestamp, symbol)."
            )
        raw = raw.unstack(level=1)
    times = unique_times(prices)
    syms = symbols(prices)
    unknown = [c for c in raw.columns if str(c) not in syms]
    if unknown:
        raise QuantProofDataError(f"Signals reference symbols not in the data: {unknown[:5]}")
    out = raw.copy()
    out.columns = [str(c) for c in out.columns]
    if not out.index.isin(times).all():
        raise QuantProofDataError("Signal timestamps are not all present in the data.")
    return out.reindex(index=times, columns=syms).astype(float)
