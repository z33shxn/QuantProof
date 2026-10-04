# Public API

Runnable examples for the most-used objects: [key-objects.md](key-objects.md).

Stable, intentionally exported interfaces. Anything not listed here (modules or names
starting with `_`, analyzer internals such as `ModuleContext`) may change without notice.

## Top level — `import quantproof`

| Name | Purpose |
|---|---|
| `audit(strategy=None, data=None, *, config=None, benchmark=None, artifacts=None, returns=None, trial_returns=None, timestamp_column=None, symbol_column=None) -> AuditResult` | Run an audit (strategy mode or results mode; single asset or panel) |
| `AuditConfig` | Configuration and profiles ([configuration.md](configuration.md)) |
| `AuditResult` | `.status`, `.summary`, `.narrative`, `.findings`, `.issues`, `.statistics`, `.validation`, `.execution`, `.causality`, `.metrics`, `.reproducibility`, `.sections`, `.manifest`, `.by_category()`, `.ids()`, `.to_dict()`, `.to_json()`, `.to_markdown()`, `.to_html()` |
| `Finding`, `Severity`, `Confidence`, `Category` | Result model |
| `load_strategy`, `StrategySpec` | Strategy contract ([strategy-contract.md](strategy-contract.md)) |
| `ResearchArtifacts`, `PandasAdapter`, `GenericResultsAdapter` | Results-mode inputs ([adapters.md](adapters.md)) |
| `QuantProofError` and subclasses `QuantProofDataError`, `QuantProofConfigError`, `QuantProofStrategyError`, `QuantProofInputError` | Errors with actionable messages |
| `__version__` | Package version |

`quantproof.audit` is only the function; analyzers live in `quantproof.analyzers`, the
result model in `quantproof.results` and severities in `quantproof.severity`.

## `quantproof.rules`

`list_rules(category=None)`, `get_rule(id)`, `has_rule(id)`, `REGISTRY`, `RuleSpec`
(id, name, category, analysis, max_severity, severity_policy, description, rationale,
impact, investigate, remediation, limitations, example), `rules_markdown()`.

## `quantproof.results`

`AuditResult`, `AuditSummary`, `Narrative`, `Finding`, `Location`, `Usage`, `Category`,
`build_narrative`, `determine_verdict`.

## `quantproof.statistics`

`sharpe_ratio`, `sharpe_summary`, `sharpe_standard_error`, `return_moments`,
`annualized_return`, `max_drawdown`, `probabilistic_sharpe_ratio`,
`minimum_track_record_length`, `deflated_sharpe_ratio`, `expected_max_sharpe`
(`method="approximation"|"exact"`), `expected_max_standard_normal`,
`effective_number_of_trials`,
`probability_of_backtest_overfitting`, `reality_check`, `adjust_pvalues`,
`bootstrap_statistic`, `bootstrap_sharpe`, `bootstrap_indices`, `optimal_block_length`,
and result dataclasses `PSRResult`, `DSRResult`, `PBOResult`, `RealityCheckResult`,
`BootstrapResult` (each with `.to_dict()`).

## `quantproof.validation`

`WalkForward`, `WalkForwardWindow`, `PurgedKFold` and `CPCV` (`event_end`/`event_start`
event-time labels or `label_horizon` sample-time labels; embargo as a fraction, bar count
or duration), `purge`, `apply_embargo`,
`embargo_size`, `label_intervals`, `temporal_train_test_split`, `assert_no_leakage`.
Splitters follow the scikit-learn protocol (`split(X, y=None, groups=None)`,
`get_n_splits`) and yield integer position arrays.

## `quantproof.execution`

Cost components (`BpsCommission`, `PercentageCommission`, `PerShareCommission`,
`FixedPerOrderCommission`, `FixedSpread`, `SeriesSpread`, `FixedSlippage`,
`VolatilitySlippage`, `SizeSlippage`, `SquareRootImpact`, `TransactionTax`, `ScaledCost`,
`CostComponent`, `CostContext`), `TransactionCostModel` (`from_bps`, `compose`,
`without`, `scaled`), fill models (`MarketOnClose`, `NextOpen`, `LimitOrder`,
`FillModel`), `simulate`, `SimulationResult`, `turnover_series`, `turnover_stats`,
`analyze_execution_timing`, `roll_spread`, and cost providers (`CostProvider`,
`FeeSchedule`, `ProviderCost`, `india_nse_provider`).

## `quantproof.data`

`load_frame`, `prepare_prices`, `load_prices`, `load_returns`, `validate_data`,
`DATA_RULES`, `generate_prices`, `generate_universe`; panel helpers in
`quantproof.data.panel` (`is_panel`, `symbols`, `unique_times`, `wide`, `to_wide_signals`).

## `quantproof.analyzers.static`

`analyze_source`, `analyze_file`, `analyze_path`, `RULES`, `StaticRule`, `register`,
`list_rules`.

## `quantproof.analyzers.causal`

`run_causality_test`, `causality_findings`, `perturb_after`, `perturb_future`,
`outputs_frame`, `CausalityReport`, `PerturbationTrial`, `SCHEMES`.

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
