# Regime analysis

## Problem

A full-sample Sharpe ratio averages over market environments. A strategy that earns all of
its return in one volatility regime may disappoint if the mix changes.

## Implementation

`quantproof.regimes.regime_analysis(strategy_returns, reference_returns, RegimeConfig)`.
The reference is the benchmark if supplied, otherwise the traded asset. All labels at time
*t* use information up to *t − 1* (tested: a shock at *t* never changes labels at ≤ *t*).

| Regime | Definition (defaults) |
|---|---|
| volatility | trailing 63-bar realized volatility of the reference, lagged one bar, cut at full-sample terciles → low / normal / high (or user thresholds) |
| drawdown | reference drawdown from its running peak at the previous close: > −10 % normal, −10 % to −20 % drawdown, ≤ −20 % deep_drawdown |
| trend (benchmark only) | bull if the benchmark's trailing 126-bar cumulative return at the previous close is positive, else bear |

Per regime: observations, share of time, annualized mean, volatility, Sharpe (only with ≥
`min_observations`, default 40), hit rate, share of P&L.

QP-REGIME-001 warns when any regime with sufficient data has a negative Sharpe while the
overall Sharpe is positive.

## Assumptions

Using full-sample quantiles for *labelling* is acceptable because regime analysis is
descriptive; the labels never feed the strategy.

## Limitations

These are simple, transparent definitions, not regime-dating algorithms (e.g.
Pagan-Sossounov, Hamilton Markov switching). Few regime switches make per-regime estimates
noisy.

## References

- Pagan, A. R. & Sossounov, K. A. (2003). A simple framework for analysing bull and bear
  markets. *Journal of Applied Econometrics*, 18(1).
- Ang, A. & Timmermann, A. (2012). Regime changes and financial markets. *Annual Review of
  Financial Economics*, 4.
