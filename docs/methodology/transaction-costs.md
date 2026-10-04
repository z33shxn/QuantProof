# Execution realism: costs, fills, turnover, timing

## Problem

Backtests commonly assume trades fill at the exact price that generated the signal and
that trading is free. Short-horizon strategies can owe most or all of their apparent edge
to those two assumptions.

## Implementation

### Timing convention

`targets[t]` is the target weight decided with information up to and including the close
of bar *t*. Fill models (`quantproof.execution.fills`):

| Model | Trade fills at | Return of bar *s* |
|---|---|---|
| `MarketOnClose(lag=L)` | close of bar *t + L* (`L = 0`: the signal bar's own close) | `targets[s−1−L] · r_s` |
| `NextOpen(lag=L≥1)` | open of bar *t + L* | `(1 + w_old·(O_s/C_{s−1} − 1)) · (1 + w_new·(C_s/O_s − 1)) − 1` |
| `LimitOrder(offset_bps)` | limit `C_t·(1 ∓ offset)` during bar *t+1* if touched, else cancelled | as next-open with the fill price |

`simulate(data, targets, fill, lag, cost_model, capital)` returns gross returns, per
component costs, net returns, held weights, trades and turnover. It is a reference
simulator for auditing, not a backtesting engine.

### Cost components

All return the cost per bar **as a fraction of equity**, so `net = gross − cost`:

| Component | Cost per bar |
|---|---|
| `BpsCommission(b)` / `PercentageCommission(p)` | `|Δw| · b/10⁴` / `|Δw| · p` |
| `PerShareCommission(c, min)` | `max(shares · c, min) / capital`, shares = `|Δw| · capital / price` |
| `FixedPerOrderCommission(f)` | `f / capital` per bar with a trade |
| `FixedSpread(s)` | `|Δw| · (s/2)/10⁴` (half the quoted spread) |
| `FixedSlippage(b)` | `|Δw| · b/10⁴` |
| `VolatilitySlippage(k)` | `|Δw| · k · σ_t` |
| `SizeSlippage(c)` | `|Δw| · c · (shares/ADV)/10⁴` |
| `SquareRootImpact(Y)` | `|Δw| · Y · σ_t · √(shares/ADV)` |

`σ_t` is a trailing return volatility lagged one bar and `ADV` a trailing average volume
lagged one bar (the first bar has no history and is treated as zero participation) — cost
estimates never use future data. `roll_spread(prices)` estimates an effective spread
(Roll 1984) from negative serial covariance of price changes.

| `TransactionTax(b, side)` | `|Δw| · b/10⁴` on buys, sells or both (stamp duty, FTT) |

`TransactionCostModel.from_bps(commission_bps, spread_bps, slippage_bps,
impact_coefficient, tax_bps)` builds the common model. Composition is explicit:

```python
from quantproof.execution import (
    BpsCommission, FixedSlippage, FixedSpread, SquareRootImpact, TransactionCostModel,
    TransactionTax,
)

model = TransactionCostModel.compose(
    commission=BpsCommission(1),
    spread=FixedSpread(4),
    slippage=FixedSlippage(2),
    impact=SquareRootImpact(0.1),   # None disables a component
    taxes=TransactionTax(5, side="buy"),
)
model.without("impact")             # drop a component
model.scaled(2.0)                   # every component x2 (sensitivity analysis)
```

### Execution semantics

| Semantics | How QuantProof treats it |
|---|---|
| close-to-close (`fill="close"`, lag L) | simulated exactly: decision at close *t*, fill at close *t+L* |
| next bar (`fill="next_close"`) | close fill with lag ≥ 1 |
| next open (`fill="next_open"`) | simulated from open and close (overnight + intraday legs) |
| delayed | any lag ≥ 1 |
| intrabar, event-driven, VWAP, TWAP | **not simulated**: declaring them gives QP-EXEC-006 (WARN, "not testable from bars") and the audit uses its own bar-level model |

Static analysis cannot always tell which semantics a vectorized backtest implies: an
un-lagged signal times same-bar returns is a FAIL only when both are close-based
(QP010); when the price sources cannot be traced QuantProof says "Execution semantics
could not be determined automatically" (WARN) instead of guessing.

### Portfolios

For panel data `simulate()` runs the fill model per symbol and sums gross returns and
costs (`Σ_i w_i,t−1 · r_i,t` for close fills, exact; next-open fills are a first-order
approximation that ignores cross-symbol rebalancing within the bar). Weights are
fractions of total equity.

### Audit scenarios

| Scenario | Assumptions |
|---|---|
| naive | lag 0, close fills, no costs (what a typical vectorized backtest shows) |
| declared | the strategy's `EXECUTION` dict |
| realistic | `ExecutionConfig`: lag 1, close fills, 1 bp commission, 2 bp spread, 2 bp slippage by default |

The report adds:

- a lag-sensitivity table (lags 0, 1, 2, 5, next open);
- **cost attribution**: gross return, each cost component (commission, spread, slippage,
  impact, taxes) and net return, as sums of per-bar returns and annualized means — the
  components add up exactly to gross − net;
- a **cost-multiplier table** (0×, 0.5×, 1×, 1.5×, 2×, 3× the whole audit cost model by
  default; `execution.cost_multipliers`). Impact is scaled as a cost *level*, not by
  changing trade size;
- the **break-even multiplier** `mean(gross) / mean(cost)` and the linear break-even
  one-way cost `mean(gross)/mean(|Δw|)·10⁴`;
- turnover.

Verdict rules:

- **QP-EXEC-001 FAIL** if the declared assumptions are optimistic (lag 0, zero costs, or
  nothing declared), the headline Sharpe is ≥ 0.5, and less than 25 % of it survives under
  audit assumptions. WARN if lag 0 is declared but the edge survives.
- **QP-EXEC-002** WARN if no non-zero cost is declared; INFO if declared costs are below
  the auditor's.
- **QP-EXEC-003** WARN if net Sharpe ≤ 0 while gross > 0, or the break-even multiplier is
  below 2 (doubling the audit cost model would erase the mean return).
- **QP-EXEC-004** WARN if costs consume ≥ 50 % of gross return.
- **QP-EXEC-005** (results mode, trade ledger with `signal_time`/`execution_time`): FAIL if
  any trade executes before its signal, WARN if more than half execute with zero latency.
- **QP-EXEC-006** WARN if the declared execution style cannot be simulated from bars.

### Turnover

Gross turnover per bar `Σ_i |Δw_i|`; net `|Σ_i Δw_i|` (equal for one instrument);
annualized = mean × periods per year; per-calendar-year totals; implied holding period
`2 / mean gross turnover` bars.

### Jurisdiction-specific providers

Statutory charges change over time, so they are **effective-dated** and kept outside the
core. A `CostProvider` holds `FeeSchedule(segment, effective_from, effective_to, buy_tax,
sell_tax, exchange_fee, regulatory_fee, stamp_duty_buy, tax_on_fees)` entries; looking up a
date outside coverage raises instead of silently applying the wrong rate. `ProviderCost`
applies a provider bar by bar, with optional broker brokerage (bps, capped per order).

`india_nse_provider()` is a reference for NSE equities and equity derivatives, coverage
from 2024-10-01:

| Segment | STT | NSE transaction | SEBI | Stamp (buy) |
|---|---|---|---|---|
| equity_delivery | 0.1 % buy & sell | 0.00297 % | ₹10/crore | 0.015 % |
| equity_intraday | 0.025 % sell | 0.00297 % | ₹10/crore | 0.003 % |
| equity_futures | 0.02 % sell → **0.05 % from 2026-04-01** | 0.00173 % | ₹10/crore | 0.002 % |
| equity_options (premium) | 0.1 % sell → **0.15 % from 2026-04-01** | 0.03503 % | ₹10/crore | 0.003 % |

GST at 18 % on brokerage + exchange + SEBI charges. Option-exercise STT is not modelled.
Rates depend on instrument, exchange, broker, transaction type, jurisdiction and date;
sources and the review date are stored in the provider. **Verify against primary sources**
(Finance Act / CBDT notifications, NSE circulars, SEBI) before use.

## Assumptions

Constant notional capital for notional-dependent components; fills at bar prices; no
partial fills or queue position; limit orders fill when touched.

## Example

`examples/unrealistic_execution` (one-day reversal, declared lag 0 and zero costs): naive
Sharpe 1.64 → gross at lag 1 0.53 → realistic net 0.10. Break-even at ≈ 1.22× the audit cost model (linear break-even ≈ 4.9 bps one-way)
against an audit assumption of 4 bps; turnover ≈ 273× equity per year. QP-EXEC-001 FAIL.

## Limitations

The square-root impact model is an order-of-magnitude approximation (no temporary vs.
permanent split, no decay, no intraday profile). None of the fill models reproduce a
matching engine.

## References

- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an
  efficient market. *Journal of Finance*, 39(4).
- Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005). Direct estimation of equity market
  impact. *Risk*, 18(7).
- Tóth, B. et al. (2011). Anomalous price impact and the critical nature of liquidity in
  financial markets. *Physical Review X*, 1(2).
- Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management*.
  Academic Press.
