# Rule catalogue

Severity shown is the maximum the rule can emit; see the methodology pages for exact
conditions. Every rule also emits a PASS finding when it ran and found nothing.

## Data (`QP-DATA-*`) — [methodology](../methodology/data-validation.md)

| ID | Check | Max severity |
|---|---|---|
| QP-DATA-000 | preparation actions taken (sorted, de-duplicated, dropped rows) | INFO |
| QP-DATA-001 | invalid timestamps | WARN |
| QP-DATA-002 | non-monotonic timestamps | WARN |
| QP-DATA-003 | missing price column / OHLC fields | FAIL |
| QP-DATA-004 | duplicate timestamps | WARN |
| QP-DATA-005 | duplicated observations | WARN |
| QP-DATA-006 | impossible OHLC relationships | FAIL |
| QP-DATA-007 | non-positive prices | FAIL |
| QP-DATA-008 | missing values | WARN |
| QP-DATA-009 | unexplained gaps | WARN |
| QP-DATA-010 | timezone inconsistency | WARN |
| QP-DATA-011 | suspicious forward-filled (stale) prices | WARN |
| QP-DATA-012 | infinite values | FAIL |
| QP-DATA-013 | extreme single-period returns | WARN |

## Static (`QP001`–`QP015`) — [methodology](../methodology/lookahead-detection.md)

| ID | Check | Max severity |
|---|---|---|
| QP000 | source cannot be parsed | FAIL |
| QP001 | negative temporal shift | FAIL |
| QP002 | centered rolling window | FAIL |
| QP003 | suspicious future-oriented construction | FAIL |
| QP004 | random train/test split | WARN |
| QP005 | randomized / non-temporal cross-validation | WARN |
| QP006 | preprocessing fitted before the split | FAIL |
| QP007 | full-sample normalization | WARN |
| QP008 | target leakage | FAIL |
| QP009 | same-bar signal/execution | WARN |
| QP010 | missing execution lag | FAIL |
| QP011 | missing transaction costs | WARN |
| QP012 | future returns as model features | FAIL |
| QP013 | excessive hyper-parameter search | WARN |
| QP014 | no explicit out-of-sample evaluation | WARN |
| QP015 | non-causal transformation | WARN |

## Causality (`QP-CAUSAL-*`) — [methodology](../methodology/runtime-causality.md)

| ID | Check | Max severity |
|---|---|---|
| QP-CAUSAL-001 | future perturbation changed historical decisions | FAIL |
| QP-CAUSAL-002 | non-deterministic strategy (test inconclusive) | WARN |
| QP-CAUSAL-003 | strategy raised errors on perturbed data | INFO |
| QP-CAUSAL-004 | output insensitive to data (test uninformative) | INFO |

## Leakage

Value-based checks (`QP-LEAK-*`) — see [runtime causality](../methodology/runtime-causality.md) for the behavioural counterpart.

| ID | Check | Max severity |
|---|---|---|
| QP-LEAK-001 | feature replicates the target (|corr| ≥ 0.98) | FAIL |
| QP-LEAK-002 | feature replicates a future return r[t+h], h ≤ 5 | FAIL |
| QP-LEAK-003 | directional hit rate vs next return > 75 % with p < 1e-6 (≥ 50 bars) | WARN |

## Execution (`QP-EXEC-*`) — [methodology](../methodology/transaction-costs.md)

| ID | Check | Max severity |
|---|---|---|
| QP-EXEC-001 | performance depends on unrealistic execution assumptions | FAIL |
| QP-EXEC-002 | transaction-cost model missing | WARN |
| QP-EXEC-003 | edge fragile to transaction costs | WARN |
| QP-EXEC-004 | excessive turnover relative to edge | WARN |
| QP-EXEC-005 | trades executed before / instantaneously after their signal | FAIL |

## Statistics (`QP-STAT-*`) — [Sharpe/PSR](../methodology/sharpe-and-psr.md), [DSR](../methodology/deflated-sharpe.md)

| ID | Check | Max severity |
|---|---|---|
| QP-STAT-001 | small sample | WARN |
| QP-STAT-002 | Probabilistic Sharpe Ratio below confidence | WARN |
| QP-STAT-003 | Deflated Sharpe Ratio below confidence | WARN |
| QP-STAT-004 | suspiciously strong backtest statistics (any scenario) | WARN |
| QP-STAT-005 | number of trials not declared | INFO |
| QP-STAT-006 | non-normal returns | INFO |
| QP-STAT-007 | serially correlated returns | INFO |

## Validation (`QP-VAL-*`) — [walk-forward](../methodology/temporal-cross-validation.md), [PBO](../methodology/pbo.md), [CPCV](../methodology/cpcv.md), [Reality Check](../methodology/reality-check.md)

| ID | Check | Max severity |
|---|---|---|
| QP-VAL-001 | weak out-of-sample evidence (walk-forward) | WARN |
| QP-VAL-002 | Probability of Backtest Overfitting ≥ 0.5 | WARN |
| QP-VAL-003 | median CPCV path Sharpe ≤ 0 | WARN |
| QP-VAL-004 | data-snooping test not significant | WARN |

## Regimes and sensitivity

| ID | Check | Max severity |
|---|---|---|
| QP-REGIME-001 | negative Sharpe in a regime while overall Sharpe is positive | WARN |
| QP-SENS-001 | fragile parameter optimum | WARN |
| QP-SENS-002 | in-sample ranking does not persist out of sample | WARN |
