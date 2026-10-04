# Methodology

Each page states the problem, how QuantProof implements the method, the assumptions,
an example, the limitations, and references. Formulas use per-period quantities unless
stated otherwise.

| Page | Covers |
|---|---|
| [lookahead-detection.md](lookahead-detection.md) | Static AST analysis and taint tracking (QP001–QP015) |
| [runtime-causality.md](runtime-causality.md) | Future-perturbation test (QP-CAUSAL-*) |
| [temporal-cross-validation.md](temporal-cross-validation.md) | Walk-forward validation and walk-forward selection |
| [purging-and-embargo.md](purging-and-embargo.md) | Label intervals, purging, embargo, purged K-fold |
| [cpcv.md](cpcv.md) | Combinatorial Purged Cross-Validation and backtest paths |
| [sharpe-and-psr.md](sharpe-and-psr.md) | Sharpe ratio, standard error, PSR, MinTRL, Lo adjustment |
| [deflated-sharpe.md](deflated-sharpe.md) | Deflated Sharpe Ratio |
| [pbo.md](pbo.md) | Probability of Backtest Overfitting (CSCV) |
| [reality-check.md](reality-check.md) | White's Reality Check, Hansen's SPA, p-value adjustment |
| [bootstrap.md](bootstrap.md) | i.i.d., block and stationary bootstrap; block length |
| [transaction-costs.md](transaction-costs.md) | Costs, spread, slippage, impact, fills, turnover, cost providers |
| [data-validation.md](data-validation.md) | QP-DATA-* rules |
| [regimes.md](regimes.md) | Regime definitions and per-regime analysis |
| [sensitivity.md](sensitivity.md) | Parameter-surface robustness |
| [reproducibility.md](reproducibility.md) | Hashing, manifest, determinism |
| [verdict.md](verdict.md) | How findings become PASS / WARN / FAIL |
| [multi-asset.md](multi-asset.md) | Panel data: input format, strategy contract, per-check behaviour |
| [limitations.md](limitations.md) | What QuantProof cannot detect or guarantee |
