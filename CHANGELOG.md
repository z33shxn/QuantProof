# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-10-04

First public release.

### Added

- Audit engine with an explicit verdict (FAIL / WARN / PASS from findings; no score),
  strategy mode and results mode.
- Data validation rules QP-DATA-001…013 with configurable thresholds.
- AST static analyzer with taint tracking and rules QP001–QP015; inline suppression.
- Runtime future-perturbation causality test (additive, multiplicative, permutation,
  shock) with determinism and sensitivity controls.
- Value-level leakage diagnostics (QP-LEAK-001…003).
- Temporal validation: walk-forward, purging, embargo, purged K-fold, CPCV with path
  assembly.
- Statistics: Sharpe (with non-normal standard error and Lo adjustment), PSR, MinTRL, DSR,
  PBO via CSCV, White's Reality Check, Hansen's SPA, p-value adjustments, i.i.d./block/
  stationary bootstrap with automatic block length.
- Execution realism: cost components, fill models (market-on-close with lag, next open,
  limit), reference simulator, turnover, execution-timing analysis, effective-dated cost
  providers with a reference NSE (India) provider.
- Regime analysis (volatility, drawdown, trend) and parameter-surface sensitivity.
- Reproducibility manifest with deterministic content hashes and lineage.
- HTML (self-contained, offline), Markdown, JSON and text reports; `quantproof` CLI.
- Five deliberately constructed examples with synthetic data, a walkthrough notebook, and
  methodology documentation.

### Fixed during development (each covered by a regression test)

- Datetime arithmetic assumed nanosecond resolution; pandas may infer microseconds, which
  mis-scaled gap and latency measurements and misclassified daily data as intraday.
- Embargo began at the last test position instead of after the test labels' span
  (López de Prado 2018, snippet 7.3); purge and embargo are now additive.
- A declared `PARAM_GRID` was reported as an unvalidated search (QP014 false positive).
- Walk-forward validation passed when both in-sample and out-of-sample Sharpe were negative.
- Future-derived features produced one QP012 finding per downstream model call.
- Import aliases (`train_test_split as tts`) bypassed name-based static rules.
- QP-EXEC-001 ignored zero-cost declarations; QP-STAT-004 ignored the naive headline
  Sharpe.
- Size-dependent cost estimates back-filled average volume from future bars.
- TOML configuration did not work on Python 3.10 (now uses `tomli`).
- HTML reports failed to render for results-mode and static-only audits.
