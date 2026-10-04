"""Perturbation schemes that modify data strictly *after* a decision time.

Each scheme returns a copy of the data in which rows at positions ``> k`` are
changed and rows ``<= k`` are bit-for-bit identical (this is asserted).

Price-like columns (``open``, ``high``, ``low``, ``close``, ``adj_close``, ``price``)
are perturbed jointly with the same per-row factor/offset so OHLC relationships
remain valid; other numeric columns are perturbed independently. Non-numeric
columns and the index are never changed.

Schemes
-------
``additive``        add Gaussian noise of ``magnitude × std(Δx)`` per row.
``multiplicative``  multiply by ``exp(magnitude × std(Δlog x) × Z)`` per row.
``permutation``     permute the future rows (values move, timestamps stay).
``shock``           multiply future rows by extreme factors drawn from {0.2, 5}
                    (e.g. 103 → 350 or 104 → 20).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "price")
SCHEMES = ("additive", "multiplicative", "permutation", "shock")


def _split_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        c
        for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c])
    ]
    prices = [c for c in numeric if c in PRICE_COLUMNS]
    others = [c for c in numeric if c not in PRICE_COLUMNS]
    return prices, others


def _diff_std(values: np.ndarray) -> float:
    d = np.diff(values[np.isfinite(values)])
    sd = float(np.std(d)) if d.size > 1 else 0.0
    if sd > 0:
        return sd
    scale = float(np.nanmean(np.abs(values))) if np.isfinite(values).any() else 1.0
    return max(scale * 0.01, 1e-8)


def _log_std(values: np.ndarray) -> float:
    v = values[np.isfinite(values) & (values > 0)]
    if v.size < 3:
        return 0.01
    sd = float(np.std(np.diff(np.log(v))))
    return sd if sd > 0 else 0.01


def _additive(
    data: pd.DataFrame, k: int, magnitude: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = data.copy()
    n_future = len(data) - k - 1
    prices, others = _split_columns(data)
    if prices:
        ref = prices[-1] if "close" not in prices else "close"
        sd = _diff_std(data[ref].to_numpy(dtype=float))
        offset = rng.standard_normal(n_future) * magnitude * sd * 3.0
        block = data[prices].iloc[k + 1 :].to_numpy(dtype=float)
        floor = -0.5 * np.nanmin(block, axis=1)
        offset = np.maximum(offset, floor)  # keep prices positive
        out.iloc[k + 1 :, data.columns.get_indexer(pd.Index(prices))] = block + offset[:, None]
    for c in others:
        col = data[c].to_numpy(dtype=float)
        sd = _diff_std(col)
        noise = rng.standard_normal(n_future) * magnitude * sd * 3.0
        out.iloc[k + 1 :, data.columns.get_indexer(pd.Index([c]))[0]] = col[k + 1 :] + noise
    return out


def _multiplicative(
    data: pd.DataFrame, k: int, magnitude: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = data.copy()
    n_future = len(data) - k - 1
    prices, others = _split_columns(data)
    if prices:
        ref = "close" if "close" in prices else prices[-1]
        sd = _log_std(data[ref].to_numpy(dtype=float))
        factor = np.exp(rng.standard_normal(n_future) * magnitude * sd * 3.0 + magnitude * sd)
        block = data[prices].iloc[k + 1 :].to_numpy(dtype=float)
        out.iloc[k + 1 :, data.columns.get_indexer(pd.Index(prices))] = block * factor[:, None]
    for c in others:
        col = data[c].to_numpy(dtype=float)
        factor = np.exp(rng.standard_normal(n_future) * magnitude)
        out.iloc[k + 1 :, data.columns.get_indexer(pd.Index([c]))[0]] = col[k + 1 :] * factor
    return out


def _permutation(
    data: pd.DataFrame, k: int, magnitude: float, rng: np.random.Generator
) -> pd.DataFrame:
    out = data.copy()
    n_future = len(data) - k - 1
    if n_future < 2:
        raise QuantProofInputError("Permutation needs at least two future rows.")
    perm = rng.permutation(n_future)
    attempts = 0
    while np.array_equal(perm, np.arange(n_future)) and attempts < 10:
        perm = rng.permutation(n_future)
        attempts += 1
    numeric = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    cols = data.columns.get_indexer(pd.Index(numeric))
    block = np.asarray(data.iloc[k + 1 :, cols].to_numpy())
    out.iloc[k + 1 :, cols] = block[perm]
    return out


def _shock(data: pd.DataFrame, k: int, magnitude: float, rng: np.random.Generator) -> pd.DataFrame:
    out = data.copy()
    n_future = len(data) - k - 1
    factor = np.where(rng.random(n_future) < 0.5, 0.2, 5.0)
    prices, others = _split_columns(data)
    for group in (prices, others):
        if not group:
            continue
        cols = data.columns.get_indexer(pd.Index(group))
        block = data.iloc[k + 1 :, cols].to_numpy(dtype=float)
        out.iloc[k + 1 :, cols] = block * factor[:, None]
    return out


_SCHEME_FUNCS: dict[
    str, Callable[[pd.DataFrame, int, float, np.random.Generator], pd.DataFrame]
] = {
    "additive": _additive,
    "multiplicative": _multiplicative,
    "permutation": _permutation,
    "shock": _shock,
}


def as_float_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``data`` with integer columns cast to float so perturbations are representable.

    The causality runner computes the *baseline* on this same frame, so dtype changes
    cannot be mistaken for causal violations.
    """
    base = data.copy()
    for c in base.columns:
        if pd.api.types.is_integer_dtype(base[c]) and not pd.api.types.is_bool_dtype(base[c]):
            base[c] = base[c].astype(float)
    return base


def perturb_future(
    data: pd.DataFrame,
    position: int,
    scheme: str,
    *,
    magnitude: float = 0.5,
    seed: int | np.random.Generator = 0,
) -> pd.DataFrame:
    """Return a copy of ``data`` with rows after ``position`` perturbed by ``scheme``.

    Raises if the scheme would touch any row at or before ``position`` (defensive check).
    """
    if scheme not in _SCHEME_FUNCS:
        raise QuantProofInputError(f"Unknown perturbation scheme {scheme!r}; use one of {SCHEMES}.")
    n = len(data)
    if not 0 <= position < n - 1:
        raise QuantProofInputError(f"position must be in [0, {n - 2}] to leave future rows.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    base = as_float_frame(data)
    out = _SCHEME_FUNCS[scheme](base, position, magnitude, rng)
    past_new = out.iloc[: position + 1]
    past_old = base.iloc[: position + 1]
    if not past_new.equals(past_old):
        raise AssertionError("Perturbation modified data at or before the decision time.")
    return out
