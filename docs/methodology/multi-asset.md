# Multi-asset (panel) data

## Input

Long format with one row per (timestamp, symbol):

| timestamp | symbol | open | high | low | close | volume |
|---|---|---|---|---|---|---|

The symbol column is detected by name (`symbol`, `ticker`, `asset`, `instrument`,
`secid`, `code`) or given explicitly (`audit(..., symbol_column="ric")`,
`--symbol-column`). A DataFrame with a `(timestamp, symbol)` MultiIndex also works.
Preparation sorts by (timestamp, symbol), removes duplicate (timestamp, symbol) rows
(logged as QP-DATA-000) and requires at least two distinct timestamps.

## Strategy contract

The strategy receives the panel (MultiIndex `(timestamp, symbol)`) and returns
**portfolio weights** as a wide DataFrame (timestamps × symbols) or a Series on the
panel's MultiIndex. Weights are fractions of total equity; missing weights are flat.
`data["close"].unstack("symbol")` gives the wide close matrix.
See `examples/cross_sectional_momentum/strategy.py`.

## What each check does with a panel

| Check | Panel behaviour |
|---|---|
| Data validation | every rule runs per symbol; findings are merged per rule with the worst severity and `symbols_affected` / `by_symbol` evidence. QP-DATA-015 reports symbols covering different timestamp sets |
| Static analysis | unchanged (code-level) |
| Runtime causality | perturbation by timestamp, per symbol with each symbol's own scale; values never move across symbols. Cross-sectional operations at one timestamp pass; cross-asset look-ahead fails |
| Execution | per-symbol fill model, gross returns and costs summed (exact for close fills) |
| Leakage (QP-LEAK-003) | (timestamp, symbol) pairs pooled; the binomial p-value is optimistic because pairs are cross-sectionally correlated (stated in the finding) |
| Statistics, validation, selection | on the portfolio's net returns |
| Regimes | defined on the equal-weight average of the universe's returns unless a benchmark is supplied |

## Limits

- Symbols with different calendars are flagged, not aligned; align them first if
  cross-sectional signals compare prices observed at different times.
- No corporate actions, delistings or survivorship handling beyond the prices supplied.
- Per-symbol results (a per-symbol Sharpe table) are not reported; `simulate()` returns
  `symbol_gross` for custom analysis.
