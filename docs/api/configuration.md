# Configuration

Configuration is a validated Pydantic model. Build it in Python or load YAML, TOML or JSON
with `AuditConfig.from_file(path)`; `quantproof init-config` writes every default.
Unknown keys are rejected (a typo such as `comission_bps` fails with the offending path).

```python
from quantproof import AuditConfig

cfg = AuditConfig()
cfg.execution.signal_lag = 1
cfg.statistics.trials = 500
cfg = AuditConfig.from_file("quantproof.yaml")
```

## Profiles

`profile` selects a documented preset that is applied on top of the explicit values by
`AuditConfig.effective()` (the audit always uses the effective configuration, and the
manifest records it):

| Setting | `quick` (cap) | `standard` | `strict` (floor) |
|---|---|---|---|
| `statistics.n_bootstrap` | ≤ 200 | as configured (1000) | ≥ 2000 |
| `validation.pbo_max_combinations` | ≤ 500 | as configured (5000) | ≥ 12870 |
| `validation.pbo_partitions` | as configured | as configured | ≥ 16 |
| `validation.cpcv_groups` | as configured | as configured | ≥ 8 |
| `validation.max_trials` (grid points evaluated) | ≤ 50 | as configured (200) | as configured |
| `causality.n_timestamps` | ≤ 4 | as configured (8) | ≥ 16 |
| `static.max_trials_threshold` | as configured | as configured | ≤ 50 |
| `statistics.require_declared_trials` | as configured | as configured | `true` |

- **quick** is for iterating on a strategy; its resampling-based numbers are noisier.
- **standard** is the default.
- **strict** is for a final review: larger resampling sizes, and undeclared trial counts
  become a WARN (QP-STAT-005) instead of INFO, because DSR without a declared trial count
  equals PSR and cannot account for selection.

`AuditConfig.from_profile("strict")` builds a strict configuration; the CLI accepts
`--profile quick|standard|strict` (`--quick` is an alias for `--profile quick`).

## Defaults

```yaml
seed: 42
profile: standard
data:
  enabled: true
  require_ohlc: false
  allow_duplicate_timestamps: false
  max_gap_multiple: 5.0
  max_stale_run: 5
  extreme_return_threshold: 0.5
  disabled_rules: []
static:
  enabled: true
  disabled_rules: []
  max_trials_threshold: 100
execution:
  signal_lag: 1
  fill: close
  commission_bps: 1.0
  spread_bps: 2.0
  slippage_bps: 2.0
  impact_coefficient: 0.0
  tax_bps: 0.0
  capital: 1000000.0
  cost_multipliers:
  - 0.0
  - 0.5
  - 1.0
  - 1.5
  - 2.0
  - 3.0
statistics:
  periods_per_year: 252.0
  risk_free_rate: 0.0
  trials: null
  benchmark_sharpe: 0.0
  confidence: 0.95
  n_bootstrap: 1000
  bootstrap_method: stationary
  block_length: null
  max_plausible_sharpe: 4.0
  min_observations: 60
  require_declared_trials: false
validation:
  walk_forward: true
  train_fraction: 0.5
  n_test_windows: 5
  expanding: true
  gap: 0
  pbo: true
  pbo_partitions: 10
  pbo_max_combinations: 5000
  cpcv: true
  cpcv_groups: 6
  cpcv_test_groups: 2
  embargo: 0.01
  reality_check: true
  max_trials: 200
causality:
  enabled: true
  n_timestamps: 8
  schemes:
  - additive
  - multiplicative
  - permutation
  - replacement
  - extreme
  magnitude: 0.5
  min_history_fraction: 0.2
  rtol: 1.0e-09
  atol: 1.0e-12
  seed: null
regimes:
  enabled: true
  vol_window: 63
  vol_quantiles:
  - 0.3333333333333333
  - 0.6666666666666666
  vol_labels:
  - low
  - normal
  - high
  drawdown_thresholds:
  - -0.1
  - -0.2
  trend_window: 126
  min_observations: 40
sensitivity:
  enabled: true
  robust_ratio: 0.7
  fragile_ratio: 0.4

```

## Sections

Top level: `seed`, `profile`.

| Section | Key fields |
|---|---|
| `data` | `require_ohlc`, `allow_duplicate_timestamps`, `max_gap_multiple`, `max_stale_run`, `extreme_return_threshold`, `disabled_rules` |
| `static` | `disabled_rules`, `max_trials_threshold` |
| `execution` | `signal_lag`, `fill` (`close` or `next_open`), `commission_bps`, `spread_bps`, `slippage_bps`, `impact_coefficient`, `tax_bps`, `capital`, `cost_multipliers` (multiples of the whole audit cost model) |
| `statistics` | `periods_per_year`, `risk_free_rate`, `trials`, `require_declared_trials`, `benchmark_sharpe`, `confidence`, `n_bootstrap`, `bootstrap_method`, `block_length`, `max_plausible_sharpe`, `min_observations` |
| `validation` | `walk_forward`, `train_fraction`, `n_test_windows`, `expanding`, `gap`, `pbo`, `pbo_partitions` (even), `pbo_max_combinations`, `cpcv`, `cpcv_groups`, `cpcv_test_groups`, `embargo`, `reality_check`, `max_trials` |
| `causality` | `enabled`, `n_timestamps` (decision timestamps sampled), `schemes` (`additive`, `multiplicative`, `permutation`, `replacement`, `extreme`), `magnitude`, `seed` (defaults to the audit seed), `min_history_fraction`, `rtol`, `atol` (tolerances for "unchanged") |
| `regimes` | `vol_window`, `vol_quantiles`, `vol_labels`, `drawdown_thresholds`, `trend_window`, `min_observations` |
| `sensitivity` | `robust_ratio`, `fragile_ratio` |
