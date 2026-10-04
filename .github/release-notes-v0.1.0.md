## What is QuantProof?

QuantProof audits quantitative trading research. It is not a backtesting engine: it sits on top of your research code or your backtester's output and produces evidence about how far a result can be trusted. It looks for look-ahead bias, temporal leakage, overfitting and undeclared multiple testing, and unrealistic execution assumptions, and returns a PASS / WARN / FAIL verdict backed by findings rather than a composite score.

This is the first public release. The API may change before 1.0. QuantProof is not published to PyPI; install from this tag:

```bash
pip install "quantproof[all] @ git+https://github.com/z33shxn/QuantProof.git@v0.1.0"
```

## Highlights

- **Static research audit**: AST analysis (rules QP000–QP015) for future shifts, centered windows, full-sample normalisation, random splits on time series and similar patterns, with usage-aware severities.
- **Data validation**: QP-DATA-001…015, including per-symbol checks for multi-asset panels.
- **Runtime causality testing**: perturbs data after a cut-off and checks whether earlier signals change; value-level leakage diagnostics.
- **Temporal validation**: walk-forward, purged K-fold with bar- or event-time purging and embargo, combinatorial purged cross-validation (CPCV).
- **Statistical diagnostics**: Sharpe with non-normal standard error, PSR, MinTRL, Deflated Sharpe Ratio with trial provenance, PBO via CSCV, White's Reality Check, Hansen's SPA, bootstrap confidence intervals.
- **Execution realism**: signal lag, fill models, a reference simulator, composable transaction-cost models with cost attribution and cost-multiplier sensitivity.
- **Multi-asset support**, **parameter-sensitivity and regime analysis**, **reproducibility manifests** with code/data/config/parameter/environment fingerprints.
- **Reports**: self-contained offline HTML, Markdown, JSON and text.
- **CLI** (`audit`, `scan`, `validate`, `statistics`, `rules`, `report`, …) with documented exit codes, and a `GenericResultsAdapter` for exports from other backtesters.

Full details: [CHANGELOG.md](https://github.com/z33shxn/QuantProof/blob/v0.1.0/CHANGELOG.md).

## Validation

- CI runs the test suite on Python 3.10, 3.11, 3.12 and 3.13, plus a lowest-direct-dependency job, lint (ruff), strict type checking (mypy) and a package build with a wheel smoke test.
- Locally on this release commit (Python 3.12): 433 tests passed, 1 skipped (a doctest marked `+SKIP`); line+branch coverage 92% (CI enforces at least 85%).
- `python -m build` produces a wheel and sdist that pass `twine check`.

## Important limitations

- **Static analysis is heuristic** and intra-file. It can miss leakage and can flag legitimate code.
- **Runtime causality testing is evidence, not proof.** It covers the tested timestamps and perturbation schemes only.
- **The execution simulator is approximate.** It uses bar prices; intrabar, event-driven, VWAP/TWAP execution is reported as untestable rather than simulated.
- **Selection-bias corrections depend on declared or estimated trials.** DSR, PBO, Reality Check and SPA only account for the variants you declare.
- **Strategy code is not sandboxed.** It runs in your Python process with your privileges. Only audit code you trust (see [SECURITY.md](https://github.com/z33shxn/QuantProof/blob/v0.1.0/SECURITY.md)).
- **Market data correctness is outside QuantProof's scope.** Survivorship bias, restatements or wrong availability timestamps already in the data cannot be detected.

A PASS means the checks that ran found no problem. It does not mean a strategy is profitable or will work live. Full list: [docs/methodology/limitations.md](https://github.com/z33shxn/QuantProof/blob/v0.1.0/docs/methodology/limitations.md).
