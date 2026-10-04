# Public API

Stable, intentionally exported interfaces. Anything not listed here (modules or names
starting with `_`, analyzer internals such as `ModuleContext`) may change without notice.

## Top level — `import quantproof`

| Name | Purpose |
|---|---|
| `audit(strategy=None, data=None, *, config=None, benchmark=None, artifacts=None, returns=None, trial_returns=None, timestamp_column=None) -> AuditResult` | Run an audit (strategy mode or results mode) |
| `AuditConfig` | Configuration ([configuration.md](configuration.md)) |
| `AuditResult` | `.status`, `.summary`, `.findings`, `.issues`, `.sections`, `.manifest`, `.by_category()`, `.ids()`, `.to_dict()`, `.to_json()` |
| `Finding`, `Severity`, `Confidence`, `Category` | Result model |
| `load_strategy`, `StrategySpec` | Strategy contract ([strategy-contract.md](strategy-contract.md)) |
| `ResearchArtifacts`, `PandasAdapter` | Results-mode inputs ([adapters.md](adapters.md)) |
| `QuantProofError` and subclasses `QuantProofDataError`, `QuantProofConfigError`, `QuantProofStrategyError`, `QuantProofInputError` | Errors with actionable messages |
| `__version__` | Package version |

`quantproof.audit` is only the function; analyzers live in `quantproof.analyzers`, the
result model in `quantproof.results` and severities in `quantproof.severity`.

## `quantproof.statistics`

`sharpe_ratio`, `sharpe_summary`, `sharpe_standard_error`, `return_moments`,
`annualized_return`, `max_drawdown`, `probabilistic_sharpe_ratio`,
`minimum_track_record_length`, `deflated_sharpe_ratio`, `expected_max_sharpe`,
`probability_of_backtest_overfitting`, `reality_check`, `adjust_pvalues`,
`bootstrap_statistic`, `bootstrap_sharpe`, `bootstrap_indices`, `optimal_block_length`,
and result dataclasses `PSRResult`, `DSRResult`, `PBOResult`, `RealityCheckResult`,
`BootstrapResult` (each with `.to_dict()`).

## `quantproof.validation`

`WalkForward`, `WalkForwardWindow`, `PurgedKFold`, `CPCV`, `purge`, `apply_embargo`,
`embargo_size`, `label_intervals`, `temporal_train_test_split`, `assert_no_leakage`.
Splitters follow the scikit-learn protocol (`split(X, y=None, groups=None)`,
`get_n_splits`) and yield integer position arrays.

## `quantproof.execution`

Cost components (`BpsCommission`, `PercentageCommission`, `PerShareCommission`,
`FixedPerOrderCommission`, `FixedSpread`, `SeriesSpread`, `FixedSlippage`,
`VolatilitySlippage`, `SizeSlippage`, `SquareRootImpact`, `CostComponent`, `CostContext`),
`TransactionCostModel`, fill models (`MarketOnClose`, `NextOpen`, `LimitOrder`,
`FillModel`), `simulate`, `SimulationResult`, `turnover_series`, `turnover_stats`,
`analyze_execution_timing`, `roll_spread`, and cost providers (`CostProvider`,
`FeeSchedule`, `ProviderCost`, `india_nse_provider`).

## `quantproof.data`

`load_frame`, `prepare_prices`, `load_prices`, `load_returns`, `validate_data`,
`DATA_RULES`, `generate_prices`.

## `quantproof.analyzers.static`

`analyze_source`, `analyze_file`, `analyze_path`, `RULES`, `StaticRule`, `register`,
`list_rules`.

## `quantproof.analyzers.causal`

`run_causality_test`, `causality_findings`, `perturb_future`, `CausalityReport`,
`PerturbationTrial`, `SCHEMES`.

## `quantproof.analyzers.leakage`

`feature_leakage_findings`, `signal_foresight_finding`.

## `quantproof.regimes`, `quantproof.sensitivity`

`regime_analysis`, `performance_by_regime`, `volatility_regimes`, `drawdown_regimes`,
`trend_regimes`, `drawdown_series`, `trailing_volatility`;
`analyze_parameter_surface`, `SurfaceAnalysis`, `sensitivity_findings`, `surface_matrix`,
`render_ascii_surface`.

## `quantproof.experiments`

`hash_dataframe`, `hash_file`, `hash_config`, `canonical_json`, `schema_of`,
`build_manifest`, `manifest_to_yaml`, `git_info`, `package_versions`, `Lineage`.

## `quantproof.reports`

`render(result, fmt)`, `write_report(result, path, fmt=None)`, `render_html`,
`render_markdown`, `render_json`, `render_text`, `result_to_dict`, `load_result`.

## Command line

See [cli.md](cli.md).
