"""Fill models and the minimal reference simulator.

QuantProof is not a backtesting engine. This module contains the smallest
simulator needed to *audit* a signal-based strategy: it converts target weights
into gross and net returns under explicit, documented fill assumptions so the
audit can compare optimistic and realistic variants.

Timing convention
-----------------
``targets[t]`` is the target weight decided with information up to and including
the close of bar ``t``.

* :class:`MarketOnClose` (``lag=L``) — the trade fills at the close of bar ``t+L``.
  ``L=0`` means filling at the very close that produced the signal (same-bar
  execution). Held weight over bar ``s`` (close[s-1] → close[s]) is
  ``targets[s-1-L]``.
* :class:`NextOpen` (``lag=L>=1``) — fills at the open of bar ``t+L``. The old weight
  earns the overnight move, the new weight the open→close move:
  ``r_s = (1 + w_old * (open_s/close_{s-1} - 1)) * (1 + w_new * (close_s/open_s - 1)) - 1``.
* :class:`LimitOrder` — a limit order at ``close_t * (1 ∓ offset)`` is live during bar
  ``t+1``; it fills if the bar's low (buy) / high (sell) touches the limit, at the
  limit price (or the open if better). Unfilled orders are cancelled. Fills on touch
  are optimistic (no queue position, no partial fills).

None of these reproduces an exchange matching engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from quantproof.data.panel import is_panel
from quantproof.errors import QuantProofDataError, QuantProofInputError
from quantproof.execution.costs import CostContext, TransactionCostModel
from quantproof.execution.turnover import turnover_stats


@dataclass
class FillOutput:
    """Weights actually held and gross returns produced by a fill model."""

    weights: pd.Series  # weight held at the end of each bar (after any fill)
    trades: pd.Series  # weight change executed during each bar
    gross_returns: pd.Series
    exec_prices: pd.Series  # price at which each bar's trade executed (NaN if none)
    fill_rate: float = 1.0


class FillModel(ABC):
    """Converts target weights into executed weights and gross returns."""

    name: str = "fill"

    @abstractmethod
    def fill(self, data: pd.DataFrame, targets: pd.Series) -> FillOutput:
        """Simulate fills for ``targets`` on OHLC ``data``."""

    def describe(self) -> dict[str, Any]:
        return {"fill_model": type(self).__name__, **vars(self)}


def _close(data: pd.DataFrame) -> pd.Series:
    for col in ("close", "adj_close", "price"):
        if col in data.columns:
            return data[col].astype(float)
    raise QuantProofDataError("Simulation needs a 'close' (or 'adj_close'/'price') column.")


class MarketOnClose(FillModel):
    """Market order filled at the close of bar ``t + lag``."""

    name = "close"

    def __init__(self, lag: int = 1) -> None:
        if lag < 0:
            raise QuantProofInputError("lag must be >= 0.")
        self.lag = int(lag)

    def fill(self, data: pd.DataFrame, targets: pd.Series) -> FillOutput:
        close = _close(data)
        r = close.pct_change(fill_method=None).fillna(0.0)
        w = targets.shift(self.lag).fillna(0.0)
        held = w.shift(1).fillna(0.0)
        trades = w.diff().fillna(w)
        gross = held * r
        prices = close.where(trades != 0)
        return FillOutput(w, trades, gross, prices)


class NextOpen(FillModel):
    """Market order filled at the open of bar ``t + lag`` (``lag >= 1``)."""

    name = "next_open"

    def __init__(self, lag: int = 1) -> None:
        if lag < 1:
            raise QuantProofInputError("NextOpen requires lag >= 1 (the next bar's open).")
        self.lag = int(lag)

    def fill(self, data: pd.DataFrame, targets: pd.Series) -> FillOutput:
        if "open" not in data.columns:
            raise QuantProofDataError("NextOpen fills need an 'open' column.")
        close = _close(data)
        open_ = data["open"].astype(float)
        prev_close = close.shift(1)
        w_new = targets.shift(self.lag).fillna(0.0)
        w_old = w_new.shift(1).fillna(0.0)
        overnight = (open_ / prev_close - 1.0).fillna(0.0)
        intraday = (close / open_ - 1.0).fillna(0.0)
        gross = (1.0 + w_old * overnight) * (1.0 + w_new * intraday) - 1.0
        trades = w_new - w_old
        prices = open_.where(trades != 0)
        return FillOutput(w_new, trades, gross, prices)


class LimitOrder(FillModel):
    """Limit order at ``close_t`` ∓ ``offset_bps`` live for one bar; cancelled if unfilled."""

    name = "limit"

    def __init__(self, offset_bps: float = 5.0) -> None:
        if offset_bps < 0:
            raise QuantProofInputError("offset_bps must be >= 0.")
        self.offset_bps = float(offset_bps)

    def fill(self, data: pd.DataFrame, targets: pd.Series) -> FillOutput:
        for col in ("open", "high", "low"):
            if col not in data.columns:
                raise QuantProofDataError(f"Limit fills need a '{col}' column.")
        close = _close(data).to_numpy()
        open_ = data["open"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        tgt = targets.fillna(0.0).to_numpy(dtype=float)
        n = len(close)
        w = np.zeros(n)
        gross = np.zeros(n)
        prices = np.full(n, np.nan)
        orders = filled = 0
        off = self.offset_bps / 1e4
        for s in range(1, n):
            w_old = w[s - 1]
            desired = tgt[s - 1]
            w_new = w_old
            fill_px = np.nan
            if desired != w_old:
                orders += 1
                if desired > w_old:  # buy
                    limit = close[s - 1] * (1 - off)
                    if low[s] <= limit:
                        fill_px = min(limit, open_[s])
                else:  # sell
                    limit = close[s - 1] * (1 + off)
                    if high[s] >= limit:
                        fill_px = max(limit, open_[s])
                if np.isfinite(fill_px):
                    filled += 1
                    w_new = desired
            if np.isfinite(fill_px):
                a = fill_px / close[s - 1] - 1.0
                b = close[s] / fill_px - 1.0
                gross[s] = (1 + w_old * a) * (1 + w_new * b) - 1.0
                prices[s] = fill_px
            else:
                gross[s] = w_old * (close[s] / close[s - 1] - 1.0)
            w[s] = w_new
        idx = targets.index
        ws = pd.Series(w, index=idx)
        return FillOutput(
            ws,
            ws.diff().fillna(ws),
            pd.Series(gross, index=idx),
            pd.Series(prices, index=idx),
            fill_rate=filled / orders if orders else 1.0,
        )


def make_fill_model(fill: str | FillModel, lag: int) -> FillModel:
    """Build a fill model from a name (``close``, ``next_open``, ``limit``) and lag."""
    if isinstance(fill, FillModel):
        return fill
    if fill == "close":
        return MarketOnClose(lag)
    if fill == "next_close":
        return MarketOnClose(max(1, lag))
    if fill == "next_open":
        return NextOpen(max(1, lag))
    if fill == "limit":
        return LimitOrder()
    raise QuantProofInputError(
        f"Unknown fill model {fill!r}; use 'close', 'next_close', 'next_open' or 'limit'. "
        "Intrabar and event-driven execution cannot be simulated from bars."
    )


@dataclass
class SimulationResult:
    """Gross/net returns, costs and turnover of a simulated strategy."""

    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    cost_breakdown: pd.DataFrame
    # Single asset: Series indexed by timestamp. Portfolio: timestamps × symbols DataFrame.
    weights: pd.Series | pd.DataFrame
    trades: pd.Series | pd.DataFrame
    fill_model: dict[str, Any]
    cost_model: list[dict[str, Any]]
    fill_rate: float = 1.0
    turnover: dict[str, Any] = field(default_factory=dict)
    gross_turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    symbol_gross: pd.DataFrame | None = None


def causal_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing return volatility known before each bar (rolling std, lagged one bar)."""
    r = close.pct_change(fill_method=None)
    vol = r.rolling(window, min_periods=max(2, window // 2)).std().shift(1)
    return vol.fillna(r.expanding(min_periods=2).std().shift(1))


def simulate(
    data: pd.DataFrame,
    targets: pd.Series | pd.DataFrame,
    *,
    fill: str | FillModel = "close",
    lag: int = 1,
    cost_model: TransactionCostModel | None = None,
    capital: float = 1_000_000.0,
    periods_per_year: float = 252.0,
    vol_window: int = 20,
) -> SimulationResult:
    """Simulate target weights on price data under a fill model and cost model.

    ``targets`` is aligned to ``data.index``; missing targets are treated as flat (0).
    Panel data (``(timestamp, symbol)`` MultiIndex) is dispatched to
    :func:`simulate_portfolio` with ``targets`` as a timestamps × symbols frame.
    """
    if is_panel(data):
        return simulate_portfolio(
            data,
            targets,
            fill=fill,
            lag=lag,
            cost_model=cost_model,
            capital=capital,
            periods_per_year=periods_per_year,
            vol_window=vol_window,
        )
    if not isinstance(targets, pd.Series):
        targets = pd.Series(np.asarray(targets, dtype=float), index=data.index)
    targets = targets.reindex(data.index).astype(float)
    if np.isinf(targets.to_numpy()).any():
        raise QuantProofInputError("Target weights contain infinite values.")
    targets = targets.fillna(0.0)
    model = make_fill_model(fill, lag)
    out = model.fill(data, targets)
    cost_model = cost_model or TransactionCostModel()
    close = _close(data)
    adv = None
    if "volume" in data.columns:
        # Causal average volume: bars before the trade only. The first bar has no history
        # (NaN), which size-dependent components treat as zero participation.
        adv = data["volume"].astype(float).rolling(20, min_periods=1).mean().shift(1).to_numpy()
    exec_px = out.exec_prices.fillna(close).to_numpy(dtype=float)
    ctx = CostContext(
        trades=out.trades.to_numpy(dtype=float),
        prices=exec_px,
        capital=capital,
        timestamps=pd.DatetimeIndex(data.index),
        volatility=causal_volatility(close, vol_window).to_numpy(dtype=float),
        adv=adv,
    )
    breakdown = cost_model.breakdown(ctx)
    costs = breakdown.sum(axis=1)
    net = out.gross_returns - costs
    return SimulationResult(
        gross_returns=out.gross_returns.rename("gross"),
        net_returns=net.rename("net"),
        costs=costs.rename("costs"),
        cost_breakdown=breakdown,
        weights=out.weights.rename("weight"),
        trades=out.trades.rename("trade"),
        fill_model=model.describe(),
        cost_model=cost_model.describe(),
        fill_rate=out.fill_rate,
        turnover=turnover_stats(out.trades, periods_per_year=periods_per_year),
        gross_turnover=out.trades.abs().rename("gross_turnover"),
    )


def simulate_portfolio(
    data: pd.DataFrame,
    targets: pd.DataFrame | pd.Series,
    *,
    fill: str | FillModel = "close",
    lag: int = 1,
    cost_model: TransactionCostModel | None = None,
    capital: float = 1_000_000.0,
    periods_per_year: float = 252.0,
    vol_window: int = 20,
) -> SimulationResult:
    """Simulate portfolio weights on panel data (one fill model per symbol, costs summed).

    Weights are fractions of total equity. With close fills the portfolio return
    ``Σ_i w_i,t-1 · r_i,t`` is exact; with next-open fills each symbol's return compounds the
    overnight and intraday legs and the portfolio return is their sum (a first-order
    approximation that ignores cross-symbol rebalancing within the bar). Symbols missing at a
    timestamp contribute nothing for that bar.
    """
    if isinstance(targets, pd.Series):
        targets = targets.unstack(level=1)
    times = pd.DatetimeIndex(data.index.get_level_values(0)).unique().sort_values()
    gross_parts: dict[str, pd.Series] = {}
    trades: dict[str, pd.Series] = {}
    weights: dict[str, pd.Series] = {}
    breakdown: pd.DataFrame | None = None
    fill_rates = []
    describe: dict[str, Any] = {}
    cost_desc: list[dict[str, Any]] = []
    for sym in data.index.get_level_values(1).unique():
        sub = cast(pd.DataFrame, data.xs(sym, level=1))
        tgt = targets[str(sym)] if str(sym) in targets.columns else pd.Series(0.0, index=sub.index)
        res = simulate(
            sub,
            tgt.reindex(sub.index),
            fill=fill,
            lag=lag,
            cost_model=cost_model,
            capital=capital,
            periods_per_year=periods_per_year,
            vol_window=vol_window,
        )
        gross_parts[str(sym)] = res.gross_returns.reindex(times, fill_value=0.0)
        trades[str(sym)] = res.trades.reindex(times, fill_value=0.0)
        weights[str(sym)] = res.weights.reindex(times).ffill().fillna(0.0)
        bd = res.cost_breakdown.reindex(times, fill_value=0.0)
        breakdown = bd if breakdown is None else breakdown.add(bd, fill_value=0.0)
        fill_rates.append(res.fill_rate)
        describe, cost_desc = res.fill_model, res.cost_model
    symbol_gross = pd.DataFrame(gross_parts, index=times)
    gross = symbol_gross.sum(axis=1).rename("gross")
    assert breakdown is not None
    costs = breakdown.sum(axis=1).rename("costs")
    trades_df = pd.DataFrame(trades, index=times)
    return SimulationResult(
        gross_returns=gross,
        net_returns=(gross - costs).rename("net"),
        costs=costs,
        cost_breakdown=breakdown,
        weights=pd.DataFrame(weights, index=times),
        trades=trades_df,
        fill_model={**describe, "portfolio": True, "symbols": len(gross_parts)},
        cost_model=cost_desc,
        fill_rate=float(np.mean(fill_rates)) if fill_rates else 1.0,
        turnover=turnover_stats(trades_df, periods_per_year=periods_per_year),
        gross_turnover=trades_df.abs().sum(axis=1).rename("gross_turnover"),
        symbol_gross=symbol_gross,
    )
