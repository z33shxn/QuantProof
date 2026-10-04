"""Turnover statistics.

Definitions (weights as fractions of equity)
--------------------------------------------
* gross turnover at bar t: ``sum_i |Δw_i,t|`` — total traded notional / equity.
* net turnover at bar t:   ``|sum_i Δw_i,t|`` — change in net exposure; equals gross
  turnover for a single instrument.
* average turnover: mean gross turnover per bar.
* annualized turnover: average × periods_per_year (e.g. 12 = portfolio traded 12×
  per year, counting both buys and sells).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def turnover_series(trades: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Gross and net turnover per bar from weight changes."""
    t = trades.to_frame() if isinstance(trades, pd.Series) else trades
    t = t.fillna(0.0)
    return pd.DataFrame({"gross": t.abs().sum(axis=1), "net": t.sum(axis=1).abs()}, index=t.index)


def turnover_stats(
    trades: pd.Series | pd.DataFrame,
    *,
    periods_per_year: float = 252.0,
    by: str | None = "YE",
) -> dict[str, Any]:
    """Summary turnover statistics; ``by`` is a pandas resample rule for the per-period table."""
    ts = turnover_series(trades)
    out: dict[str, Any] = {
        "gross_total": float(ts["gross"].sum()),
        "net_total": float(ts["net"].sum()),
        "average_per_bar": float(ts["gross"].mean()) if len(ts) else float("nan"),
        "annualized": float(ts["gross"].mean() * periods_per_year) if len(ts) else float("nan"),
        "trading_bars": int((ts["gross"] > 0).sum()),
        "bars": len(ts),
    }
    if by and isinstance(ts.index, pd.DatetimeIndex) and len(ts):
        try:
            per = ts["gross"].resample(by).sum()
        except ValueError:  # pragma: no cover - older pandas aliases
            per = ts["gross"].resample("Y").sum()
        out["by_period"] = {str(k.date()): float(v) for k, v in per.items()}
    out["holding_period_bars"] = (
        float(2.0 / out["average_per_bar"])
        if out["average_per_bar"] and np.isfinite(out["average_per_bar"])
        else float("inf")
    )
    return out
