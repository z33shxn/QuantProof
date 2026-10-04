# Limitations

QuantProof looks for specific, documented failure modes. This page lists what it cannot
do, so that a PASS is not read as more than it is. Each methodology page has its own
limitations section; this is the summary.

## What a verdict means

- **PASS** means the checks that ran found no problem. It does not mean the strategy is
  profitable, robust or ready to trade. A PASS on a losing strategy is common and correct.
- **WARN** means the evidence is weak or a pattern needs a human look. Many honest
  strategies end with WARN (e.g. too few observations for PSR ≥ 0.95).
- **FAIL** means a validity violation was found (look-ahead, leakage, impossible data,
  performance that exists only under impossible execution). Fix it before reading any
  other number in the report.
- Checks that could not run are listed as INFO "not run / not computed" findings; they
  are **not** passes.

## Look-ahead and leakage

- Static analysis is heuristic (see [analysis levels](lookahead-detection.md#analysis-levels)).
  It works one file at a time, relies partly on names (`signal`, `returns`, `y`), and does
  not follow data through container mutation, dynamic attribute access, `exec`/`eval` or
  other modules.
- The future-perturbation test checks the strategy's behaviour at sampled timestamps
  under five perturbation schemes. Passing is evidence, not proof: a strategy can be
  insensitive to the particular perturbations while still reading the future.
- Neither check can see leakage that is already in the data: survivorship bias, restated
  fundamentals, timestamps recording publication instead of availability, point-in-time
  index membership. QuantProof has no way to know when a value became knowable.
- Strategy code is executed in the current Python process without sandboxing. Only audit
  code you trust (see [SECURITY.md](../../SECURITY.md)).

## Execution and costs

- The reference simulator uses bar prices. It does not model a matching engine, queue
  position, partial fills, intrabar latency, borrow costs, financing, margin, or corporate
  actions beyond what is in the prices supplied.
- Intrabar, event-driven, VWAP and TWAP execution cannot be verified from bars; declaring
  them yields QP-EXEC-006 and the audit uses its own bar-level assumptions.
- Notional-dependent costs assume constant capital. The square-root impact model is an
  order-of-magnitude approximation.
- Portfolio next-open simulation is a first-order approximation (see
  [transaction costs](transaction-costs.md#portfolios)).

## Statistics and selection bias

- PSR/DSR assume stationary returns and rely on the asymptotic normality of the Sharpe
  estimator; with fat tails or strong serial correlation they are approximations
  (the Lo-adjusted Sharpe is reported alongside).
- DSR, PBO, CPCV and the Reality Check/SPA can only account for the variants you declare
  (`PARAM_GRID`, `statistics.trials`, `trial_returns`). Undeclared exploration — other
  datasets, features, ideas dropped early — is invisible. Declaring `trials=5000` makes the
  hurdle consistent with 5000 independent tries; it does not make the count honest.
- The effective-number-of-trials estimate (Li & Ji, 2005) is a heuristic and is reported
  for context only.
- PBO is not a p-value. It measures the selection procedure on this sample.
- Bootstrap intervals are approximate; the automatic block length is itself an estimate.

## Data

- Data validation checks internal consistency (timestamps, OHLC relationships, gaps,
  stale prices, extreme moves, panel synchronisation). It cannot check that prices are
  *correct* or adjusted consistently.
- Multi-asset support covers panels of instruments sharing a calendar; symbols with
  different trading calendars are flagged (QP-DATA-015) rather than aligned
  automatically.

## Reproducibility

- Manifests record versions, hashes and seeds; bit-for-bit reproduction also depends on
  NumPy/pandas versions and platform floating-point behaviour.
