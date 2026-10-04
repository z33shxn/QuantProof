"""Perturbation schemes that modify data strictly *after* a decision time.

``perturb_after(data, cutoff, scheme)`` returns a copy of ``data`` in which every row with
timestamp ``> cutoff`` is changed and every row with timestamp ``<= cutoff`` is bit-for-bit
identical (this is asserted). It works for single-asset frames (DatetimeIndex) and for
panels (``(timestamp, symbol)`` MultiIndex); for panels each symbol is perturbed using its
own scale, and permutations/replacements never move values across symbols.

Price-like columns (``open``, ``high``, ``low``, ``close``, ``adj_close``, ``price``) are
perturbed jointly with the same per-row factor/offset so OHLC relationships remain valid;
other numeric columns are perturbed independently. Non-numeric columns and the index are
never changed.

Schemes (``m`` = ``magnitude``)
-------------------------------
``additive``        add N(0, (3m·std(Δx))²) per row (prices floored at half their value).
``multiplicative``  multiply by exp(3m·std(Δlog x)·Z + m·std(Δlog x)) per row.
``permutation``     permute the future rows (values move, timestamps stay).
``replacement``     replace each future row by a row drawn (with replacement) from the past.
``extreme``         multiply future rows by factors drawn from {0.2, 5} (e.g. 103 → 515).

All randomness comes from the supplied seed / generator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "price")
SCHEMES = ("additive", "multiplicative", "permutation", "replacement", "extreme")


def as_float_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Copy with integer (and nullable numeric) columns cast to float64.

    The causality runner computes the *baseline* on this same frame, so dtype changes cannot
    be mistaken for causal violations.
    """
    base = data.copy()
    for c in base.columns:
        dt = base[c].dtype
        if pd.api.types.is_bool_dtype(dt):
            continue
        if (pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_float_dtype(dt)) or (
            pd.api.types.is_extension_array_dtype(dt) and pd.api.types.is_numeric_dtype(dt)
        ):
            base[c] = base[c].astype("float64")
    return base


def _numeric_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        c
        for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c])
    ]
    prices = [c for c in numeric if c in PRICE_COLUMNS]
    return prices, [c for c in numeric if c not in PRICE_COLUMNS]


def _diff_std(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    d = np.diff(v)
    sd = float(np.std(d)) if d.size > 1 else 0.0
    if sd > 0:
        return sd
    scale = float(np.mean(np.abs(v))) if v.size else 1.0
    return max(scale * 0.01, 1e-8)


def _log_std(values: np.ndarray) -> float:
    v = values[np.isfinite(values) & (values > 0)]
    if v.size < 3:
        return 0.01
    sd = float(np.std(np.diff(np.log(v))))
    return sd if sd > 0 else 0.01


# Each block function receives the group's future rows (2-D, columns of one kind), the
# group's past rows, per-column scale statistics, the magnitude and the generator, and
# returns the perturbed future rows.
def _additive(
    fut: np.ndarray,
    past: np.ndarray,
    stats: dict[str, np.ndarray],
    m: float,
    rng: np.random.Generator,
    joint: bool,
) -> np.ndarray:
    n = fut.shape[0]
    if joint:
        offset = rng.standard_normal(n) * m * stats["diff_std"][0] * 3.0
        floor = -0.5 * np.nanmin(fut, axis=1)
        offset = np.maximum(offset, floor)
        return fut + offset[:, None]
    noise = rng.standard_normal(fut.shape) * m * stats["diff_std"][None, :] * 3.0
    return fut + noise


def _multiplicative(
    fut: np.ndarray,
    past: np.ndarray,
    stats: dict[str, np.ndarray],
    m: float,
    rng: np.random.Generator,
    joint: bool,
) -> np.ndarray:
    n = fut.shape[0]
    if joint:
        sd = stats["log_std"][0]
        factor = np.exp(rng.standard_normal(n) * m * sd * 3.0 + m * sd)
        return fut * factor[:, None]
    return fut * np.exp(rng.standard_normal(fut.shape) * m)


def _permutation(
    fut: np.ndarray,
    past: np.ndarray,
    stats: dict[str, np.ndarray],
    m: float,
    rng: np.random.Generator,
    joint: bool,
) -> np.ndarray:
    n = fut.shape[0]
    if n < 2:
        return fut
    perm = rng.permutation(n)
    for _ in range(10):
        if not np.array_equal(perm, np.arange(n)):
            break
        perm = rng.permutation(n)
    return fut[perm]


def _replacement(
    fut: np.ndarray,
    past: np.ndarray,
    stats: dict[str, np.ndarray],
    m: float,
    rng: np.random.Generator,
    joint: bool,
) -> np.ndarray:
    if past.shape[0] == 0:
        return fut
    idx = rng.integers(0, past.shape[0], size=fut.shape[0])
    return past[idx]


def _extreme(
    fut: np.ndarray,
    past: np.ndarray,
    stats: dict[str, np.ndarray],
    m: float,
    rng: np.random.Generator,
    joint: bool,
) -> np.ndarray:
    factor = np.where(rng.random(fut.shape[0]) < 0.5, 0.2, 5.0)
    return fut * factor[:, None]


_SCHEMES: dict[str, Callable[..., np.ndarray]] = {
    "additive": _additive,
    "multiplicative": _multiplicative,
    "permutation": _permutation,
    "replacement": _replacement,
    "extreme": _extreme,
}
# Schemes that move whole rows must treat all numeric columns as one block.
_ROW_SCHEMES = {"permutation", "replacement"}


def _groups(data: pd.DataFrame) -> np.ndarray:
    if isinstance(data.index, pd.MultiIndex):
        return data.index.get_level_values(1).astype(str).to_numpy()
    return np.zeros(len(data), dtype=int)


def perturb_after(
    data: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    scheme: str,
    *,
    magnitude: float = 0.5,
    seed: int | np.random.Generator = 0,
) -> pd.DataFrame:
    """Copy of ``data`` with every row strictly after ``cutoff`` perturbed by ``scheme``.

    Raises if the scheme would touch any row at or before ``cutoff`` (defensive check) or if
    no row lies after ``cutoff``.
    """
    if scheme not in _SCHEMES:
        raise QuantProofInputError(f"Unknown perturbation scheme {scheme!r}; use one of {SCHEMES}.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    base = as_float_frame(data)
    level0 = pd.DatetimeIndex(_index_level0(base))
    cutoff_ts = pd.Timestamp(cutoff)
    if level0.tz is not None and cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize(level0.tz)
    future = np.asarray(level0 > cutoff_ts)
    if not future.any():
        raise QuantProofInputError(f"No observations after {cutoff}; nothing to perturb.")
    if scheme == "permutation" and level0[future].nunique() < 2:
        raise QuantProofInputError(
            "The permutation scheme needs at least two timestamps after the cutoff."
        )
    if scheme == "replacement" and future.all():
        raise QuantProofInputError(
            "The replacement scheme draws from data at or before the cutoff; there is none."
        )
    groups = _groups(base)
    prices, others = _numeric_columns(base)
    out = base.copy()
    fn = _SCHEMES[scheme]
    blocks: list[tuple[list[str], bool]]
    if scheme in _ROW_SCHEMES:
        blocks = [(prices + others, True)]
    else:
        blocks = [(prices, True)] + [([c], False) for c in others]
    for g in pd.unique(groups):
        in_group = groups == g
        fut_rows = np.flatnonzero(in_group & future)
        past_rows = np.flatnonzero(in_group & ~future)
        if fut_rows.size == 0:
            continue
        for cols, joint in blocks:
            if not cols:
                continue
            pos = base.columns.get_indexer(pd.Index(cols))
            values = base.iloc[np.flatnonzero(in_group), pos].to_numpy(dtype=float)
            ref = values[:, cols.index("close")] if "close" in cols else values[:, -1]
            stats = {
                "diff_std": np.array([_diff_std(values[:, j]) for j in range(values.shape[1])])
                if not joint
                else np.array([_diff_std(ref)]),
                "log_std": np.array([_log_std(ref)]),
            }
            fut = base.iloc[fut_rows, pos].to_numpy(dtype=float)
            past = base.iloc[past_rows, pos].to_numpy(dtype=float)
            out.iloc[fut_rows, pos] = fn(fut, past, stats, magnitude, rng, joint)
    keep = ~future
    if not out.loc[keep].equals(base.loc[keep]):
        raise AssertionError("Perturbation modified data at or before the decision time.")
    return out


def _index_level0(data: pd.DataFrame) -> pd.Index:
    return data.index.get_level_values(0) if isinstance(data.index, pd.MultiIndex) else data.index


def perturb_future(
    data: pd.DataFrame,
    position: int,
    scheme: str,
    *,
    magnitude: float = 0.5,
    seed: int | np.random.Generator = 0,
) -> pd.DataFrame:
    """Positional convenience wrapper: perturb rows after the ``position``-th unique timestamp."""
    times = pd.DatetimeIndex(_index_level0(data)).unique().sort_values()
    if not 0 <= position < len(times) - 1:
        raise QuantProofInputError(
            f"position must be in [0, {len(times) - 2}] to leave future rows."
        )
    return perturb_after(data, times[position], scheme, magnitude=magnitude, seed=seed)
