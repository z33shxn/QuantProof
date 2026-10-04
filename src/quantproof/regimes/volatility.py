"""Volatility regimes.

Definition: trailing realized volatility of the *reference* return series
(benchmark if supplied, otherwise the traded asset) over ``window`` bars, lagged
one bar so the label at ``t`` uses data up to ``t-1``. Cut-offs are quantiles of
that volatility series over the full sample (by default terciles → low / normal /
high), or user-supplied absolute annualized levels.

Using full-sample quantiles for *labelling* is acceptable here because regime
analysis is descriptive (it partitions realized results); it is never fed back
into the strategy.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from quantproof.errors import QuantProofInputError


def trailing_volatility(
    returns: pd.Series, window: int = 63, periods_per_year: float = 252.0
) -> pd.Series:
    """Annualized trailing volatility, lagged one bar (label at t uses data ≤ t-1)."""
    vol = returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(periods_per_year)
    return vol.shift(1)


def volatility_regimes(
    returns: pd.Series,
    *,
    window: int = 63,
    quantiles: list[float] | None = None,
    labels: list[str] | None = None,
    thresholds: list[float] | None = None,
    periods_per_year: float = 252.0,
) -> pd.Series:
    """Label each bar with a volatility regime (NaN during warm-up)."""
    quantiles = quantiles if quantiles is not None else [1 / 3, 2 / 3]
    labels = labels if labels is not None else ["low", "normal", "high"]
    vol = trailing_volatility(returns, window, periods_per_year)
    if thresholds is None:
        valid = vol.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=returns.index, dtype=object)
        cuts = [float(valid.quantile(q)) for q in quantiles]
        if any(b <= a for a, b in pairwise(cuts)):
            # Ties between quantiles (e.g. constant prices → zero volatility everywhere):
            # regimes are not identifiable, so no bar is labelled rather than mislabelling.
            return pd.Series(np.nan, index=returns.index, dtype=object)
    else:
        cuts = list(thresholds)
        if any(b <= a for a, b in pairwise(cuts)):
            raise QuantProofInputError("Volatility thresholds must be strictly increasing.")
    if len(labels) != len(cuts) + 1:
        raise QuantProofInputError("labels must have one more entry than cut-offs.")
    bins = [-np.inf, *cuts, np.inf]
    out = pd.cut(vol, bins=bins, labels=labels, right=True)
    return out.astype(object).where(vol.notna())
