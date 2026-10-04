"""Drawdown and trend regimes of a reference (benchmark) series.

* Drawdown regime at ``t``: the benchmark's drawdown from its running peak at the
  close of ``t-1``, bucketed by ``thresholds`` (default ``[-10%, -20%]`` →
  ``normal`` / ``drawdown`` / ``deep_drawdown``).
* Trend (bull/bear) regime at ``t``: sign of the benchmark's cumulative return over the
  trailing ``window`` bars ending at ``t-1`` (``bull`` if > 0, else ``bear``).

These are simple, transparent definitions, not dating algorithms such as
Pagan-Sossounov or Lunde-Timmermann.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown of the compounded equity curve from its running peak (≤ 0)."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return equity / peak - 1.0


def drawdown_regimes(
    returns: pd.Series,
    *,
    thresholds: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.Series:
    """Label each bar by the reference drawdown at the previous close."""
    thresholds = thresholds if thresholds is not None else [-0.10, -0.20]
    labels = (
        labels
        if labels is not None
        else ["normal", "drawdown", "deep_drawdown"][: len(thresholds) + 1]
    )
    dd = drawdown_series(returns).shift(1)
    bins = [-np.inf, *sorted(thresholds), np.inf]
    ordered = list(reversed(labels))  # most negative bucket first
    out = pd.cut(dd, bins=bins, labels=ordered, right=False)
    return out.astype(object).where(dd.notna())


def trend_regimes(returns: pd.Series, *, window: int = 126) -> pd.Series:
    """``bull``/``bear`` by the sign of the trailing ``window``-bar cumulative return (lagged)."""
    growth = (1.0 + returns.fillna(0.0)).rolling(window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0
    lagged = growth.shift(1)
    out = pd.Series(np.where(lagged > 0, "bull", "bear"), index=returns.index, dtype=object)
    return out.where(lagged.notna())
