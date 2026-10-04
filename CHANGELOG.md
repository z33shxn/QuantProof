# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-10-04

First public release. The API may still change before 1.0. Not published to PyPI; install
from the `v0.1.0` tag on GitHub.

### Capabilities in this release

- **Audit engine**: FAIL / WARN / PASS verdict derived from findings (no composite
  score); strategy mode, results mode and static-only mode; `quick` / `standard` /
  `strict` profiles.
- **Static research audit**: AST analysis with rules QP000–QP015 at three documented
  analysis levels; usage-aware severities; inline suppression. Heuristic.
- **Data validation**: rules QP-DATA-001…015, including per-symbol checks for panels.
- **Runtime causality testing**: future-perturbation test (additive, multiplicative,
  permutation, replacement, extreme) with determinism and sensitivity controls, plus
  value-level leakage diagnostics (QP-LEAK-001…003).
- **Temporal validation**: walk-forward, purging (bar- and event-time), embargo (bars or
  durations), purged K-fold, combinatorial purged cross-validation (CPCV) with path
  assembly.
- **Statistical diagnostics**: Sharpe with non-normal standard error, PSR, MinTRL, DSR with
  trial provenance and an effective-trials estimate, PBO via CSCV, White's Reality Check,
  Hansen's SPA, p-value adjustments, i.i.d./block/stationary bootstrap.
- **Execution realism**: fill models, signal lag, reference simulator, turnover,
  composable transaction-cost models with attribution and cost-multiplier sensitivity,
  effective-dated cost providers (reference NSE provider).
- **Multi-asset support**: long-format panels with a symbol column and portfolio
  simulation.
- **Sensitivity and regime analysis**: parameter-surface fragility; volatility, drawdown
  and trend regimes.
- **Reproducibility manifests**: content hashes and separate fingerprints for code, data,
  config, parameters and environment.
- **Reports**: self-contained offline HTML, Markdown, JSON and text; evidence-first
  narrative.
- **CLI**: `audit`, `scan`, `validate`, `statistics`, `report`, `rules`, `init-config`,
  `generate-data`, `version`; documented exit codes.
- **Adapters**: `GenericResultsAdapter` for returns/equity and trade-ledger exports, with
  export guidance for VectorBT, Backtrader and LEAN.

Limitations are documented in `docs/methodology/limitations.md` and `SECURITY.md`.

The entries below record how 0.1.0 was built: an initial implementation followed by a
second, deeper hardening review before release.

### Hardening review — added

- Multi-asset panels: long-format data with a symbol column, per-symbol data validation
  (QP-DATA-015 for unsynchronised symbols), panel-aware causality perturbation, portfolio
  simulation, a cross-sectional momentum example and `generate_universe()`.
- Rule registry (`quantproof.rules`) as the single source of truth; generated
  `docs/api/rules.md`; `quantproof rules [--category] [--json] [--markdown]` and
  `quantproof rules show ID`.
- Usage-aware findings (*FORBIDDEN IN LIVE DECISION* vs *LEGITIMATE FOR LABEL /
  ANALYSIS*); every WARN/FAIL finding explains why it matters, potential impact and how
  to investigate.
- Evidence-first narrative (primary reason, supporting evidence, recommendations, next
  steps) in every report; `AuditResult.narrative`, `.statistics`, `.validation`,
  `.execution`, `.causality`, `.metrics`, `.reproducibility`, `.to_markdown()`,
  `.to_html()`.
- Audit profiles `quick` / `standard` / `strict` (replaces `quick=True`).
- CLI exit codes 0 pass / 1 warn / 2 fail / 3 invalid input / 4 internal error; no raw
  tracebacks without `--debug`.
- Causality schemes `replacement` and `extreme` (renamed from `shock`); perturbation by
  timestamp cut-off with evidence (decision time, first modified observation, original
  vs perturbed value).
- Event-time purging API (`event_end`, `event_start`; `t1` kept as an alias) and duration
  embargoes.
- Composable cost model (`compose`, `without`, `scaled`, `TransactionTax`), cost
  attribution, cost-multiplier sensitivity, QP-EXEC-006 for untestable execution styles.
- DSR: exact expected maximum, trial provenance, Li & Ji effective-trials estimate.
- Evidence-based out-of-sample criteria (QP-VAL-001) without a composite score.
- Manifest fingerprints for code, data, config, parameters and environment.
- `GenericResultsAdapter`; adapter guidance for VectorBT, Backtrader and LEAN.
- Proof-of-value script, bad/good research-workflow tutorials, limitations, references,
  multi-asset and analysis-level documentation, runnable key-object docs (tested),
  benchmarks and work-count complexity tests, ROADMAP, banner.

### Hardening review — changed

- `quantproof.audit` is now unambiguously the function; analyzers moved to
  `quantproof.analyzers`, the result model to `quantproof.results`.
- `ExecutionConfig.cost_grid_bps` replaced by `cost_multipliers`.
- QP-LEAK-003 is two-sided; QP-EXEC-003 uses the break-even cost multiplier.

### Hardening review — fixed (each with a regression test)

- Empty or single-row data passed every data check.
- Multi-symbol long data was silently treated as one series; MultiIndex input crashed.
- Constant prices crashed volatility regimes.
- pandas 3 string columns were not coerced to numbers.
- Purging compared timezone-aware and naive label times as raw integers, accepted
  unsorted observations, silently re-labelled a `t1` with a different index, and could
  yield empty training sets.
- Hansen SPA returned a "significant" p-value for constant differentials; Li & Ji
  counted near-integer eigenvalues as two trials; the Sharpe ratio lost scale invariance
  for very small magnitudes.
- Missing report-section keys crashed HTML rendering; negative "break-even" costs were
  reported for strategies without a positive gross return.
- Static analysis missed helper-function shifts, callable aliases and module-constant
  shift periods, and reported plot-only centered windows as FAIL.


### Initial implementation — added

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

### Initial implementation — fixed during development (each covered by a regression test)

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

[Unreleased]: https://github.com/z33shxn/QuantProof/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/z33shxn/QuantProof/releases/tag/v0.1.0
