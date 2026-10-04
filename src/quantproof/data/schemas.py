"""Column conventions for price data."""

from __future__ import annotations

from collections.abc import Iterable

TIMESTAMP_CANDIDATES: tuple[str, ...] = ("timestamp", "datetime", "date", "time", "ts")
OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close")
VOLUME_COLUMNS: tuple[str, ...] = ("volume",)

_ALIASES = {
    "adj close": "adj_close",
    "adjclose": "adj_close",
    "adjusted_close": "adj_close",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
}


def normalize_column_name(name: object) -> str:
    """Lower-case, strip, and map common aliases (``Adj Close`` → ``adj_close``)."""
    key = str(name).strip().lower()
    return _ALIASES.get(key, key.replace(" ", "_"))


def find_timestamp_column(columns: Iterable[str]) -> str | None:
    """Return the first column whose normalized name is a timestamp candidate."""
    lookup = {normalize_column_name(c): c for c in columns}
    for cand in TIMESTAMP_CANDIDATES:
        if cand in lookup:
            return lookup[cand]
    return None


def primary_price_column(columns: Iterable[str]) -> str | None:
    """Column used to compute returns: ``close``, then ``adj_close``, then ``price``."""
    cols = list(columns)
    for cand in ("close", "adj_close", "price"):
        if cand in cols:
            return cand
    return None
