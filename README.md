# QuantProof

**Trust your backtest before you trust your strategy.**

[![Tests](https://github.com/z33shxn/QuantProof/actions/workflows/tests.yml/badge.svg)](https://github.com/z33shxn/QuantProof/actions/workflows/tests.yml)
[![Lint](https://github.com/z33shxn/QuantProof/actions/workflows/lint.yml/badge.svg)](https://github.com/z33shxn/QuantProof/actions/workflows/lint.yml)
[![Build](https://github.com/z33shxn/QuantProof/actions/workflows/build.yml/badge.svg)](https://github.com/z33shxn/QuantProof/actions/workflows/build.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

QuantProof is a validation and research-auditing layer for quantitative trading
research. It is **not** a backtesting engine. It sits on top of your existing
workflow — pandas research code, CSV/Parquet outputs, scikit-learn pipelines, or any
backtester's returns — and produces evidence about how far the result can be trusted:

```text
your research code / backtester  →  QuantProof  →  audit  →  evidence  →  PASS / WARN / FAIL
```

It looks for look-ahead bias, temporal leakage, random cross-validation on time
series, full-sample normalization, target leakage, same-bar execution, missing
costs, overfitting and multiple testing, weak out-of-sample evidence, fragile
parameter optima, regime dependence, and irreproducible experiments.

A PASS means the configured checks found no problem. It does **not** mean a
strategy is profitable or will work live.

---

## The problem in one example

`examples/lookahead_strategy/strategy.py` contains two common mistakes: it computes
"tomorrow's return" with `shift(-1)` and filters with a centered rolling mean.
A standard vectorized backtest of it looks extraordinary:

```text
WITHOUT QUANTPROOF          (naive backtest: fill at the signal bar's close, no costs)

Sharpe        9.08
CAGR       +225.0%
Max DD        0.0%

Looks excellent.
```

QuantProof audits the same file and data:

```bash
quantproof audit \
  --strategy examples/lookahead_strategy/strategy.py \
  --data examples/data/prices.parquet \
  --output audit-report.html
```

```text
╔══════════════════════════════════════════════════════════╗
║                     QUANTPROOF AUDIT                     ║
╚══════════════════════════════════════════════════════════╝

Strategy: lookahead_momentum   Observations: 1500
Window:   2019-01-01T00:00:00 → 2024-09-30T00:00:00

  Naive (same-bar, no costs)   Sharpe   9.08   CAGR +225.0%   MaxDD    0.0%
  Audit (lag + costs)          Sharpe  -2.19   CAGR  -30.5%   MaxDD  -90.2%

STATIC ANALYSIS
✗ QP001         Look-ahead: negative temporal shift (examples/lookahead_strategy/strategy.py:20:27)
    shift() with a negative period reaches strategy output (assigned to signal/position 'signal').
✗ QP002         Look-ahead: centered rolling window (examples/lookahead_strategy/strategy.py:21:13)
    rolling() is called with center=True; each value uses future observations.
✓ QP003         No future-oriented construction detected
  … (12 more passed static checks)

CAUSALITY
✗ QP-CAUSAL-001 Future perturbation changed historical decisions
    FAIL: historical decision changed after future-data perturbation in 10 of 31 trials …

LEAKAGE
⚠ QP-LEAK-003   Implausible directional accuracy
    Signal direction matches the next bar's return 100.0% of the time over 652 bars (p = 5.4e-197) …

STATISTICS
⚠ QP-STAT-004   Suspiciously strong backtest statistics
    naive backtest: annualized Sharpe 9.08 — above the plausibility threshold 4.0.
  …

Critical failures: 3   Warnings: 5   Passed checks: 29
OVERALL: FAIL
```

Every number above is produced by the analysis (output trimmed with `…`). The
headline was misleading because each position was chosen *after* seeing the return it
was credited with; once decisions are restricted to information available at the
time and executed one bar later with costs, the edge is negative. QuantProof found
this three independent ways — in the code (AST), in behaviour (perturbing the future
changed past decisions), and in the numbers (100 % directional accuracy).

The HTML report is self-contained (inline SVG, works offline) and organised for an
internal research review:

![QuantProof HTML report](docs/images/report-lookahead.png)

---

## Installation

QuantProof requires Python 3.10–3.13. It is not yet published on PyPI; install from
GitHub:

```bash
pip install "quantproof[all] @ git+https://github.com/z33shxn/QuantProof.git"
```

or from a clone (needed to run the bundled examples):

```bash
git clone https://github.com/z33shxn/QuantProof.git
cd QuantProof
pip install -e ".[all]"        # [parquet] → pyarrow, [ml] → scikit-learn, [dev] → tooling
```

## 30-second quickstart

Works offline, from a fresh install, with seeded synthetic data:

```python
import numpy as np
import pandas as pd

from quantproof import AuditConfig, audit
from quantproof.data import generate_prices

prices = generate_prices(1500, seed=2026)  # synthetic OHLCV, no market data needed


def generate_signals(data: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Target weight per bar, using only data up to and including that bar."""
    close = data["close"]
    fast_ma, slow_ma = close.rolling(fast).mean(), close.rolling(slow).mean()
    return pd.Series(np.where(fast_ma > slow_ma, 1.0, 0.0), index=data.index).where(slow_ma.notna())


result = audit(strategy=generate_signals, data=prices, config=AuditConfig(quick=True))

print(result.status)  # Severity.PASS / WARN / FAIL
for finding in result.issues:  # WARN and FAIL findings only
    print(finding.severity.value, finding.id, finding.title)
result.to_json()  # everything, as strict JSON
```

From the command line:

```bash
quantproof generate-data prices.csv                  # synthetic data to try things on
quantproof audit -s my_strategy.py -d prices.csv -o report.html
quantproof audit -s my_strategy.py -d prices.csv --format markdown > AUDIT.md
quantproof scan research/                            # static analysis of a whole tree
quantproof validate prices.csv                       # data-quality checks only
quantproof statistics returns.csv --trials 200       # Sharpe / PSR / DSR / bootstrap
quantproof audit --returns returns.csv --trials variants.csv   # any backtester's output
```

`quantproof audit` exits with status 1 when the verdict is FAIL (configurable with
`--fail-on warn|never`), so it can gate a CI job or pull request.

## The strategy contract

A strategy file exposes one function; everything else is optional, plain-literal
metadata:

```python
NAME = "dual_ma_trend"
PARAMETERS = {"fast": 20, "slow": 100}  # defaults
PARAM_GRID = {"fast": [10, 20, 30, 40], "slow": [60, 100, 140, 180]}  # variants tried
EXECUTION = {"signal_lag": 1, "commission_bps": 1.0, "spread_bps": 2.0, "slippage_bps": 2.0}


def generate_signals(
    data: pd.DataFrame, fast: int = 20, slow: int = 100
) -> pd.Series: ...  # target weight at each bar, decided with data up to and including that bar
```

Declaring `PARAM_GRID` lets QuantProof measure selection bias (Deflated Sharpe Ratio,
PBO, walk-forward and CPCV selection, Reality Check, parameter surface). Declaring
`EXECUTION` lets it compare your assumptions with realistic ones. Code that is not
written against this contract can still be audited: use `quantproof scan` for static
analysis and results mode (`returns=`, `trial_returns=`, or
`quantproof.ResearchArtifacts`) for everything that works on outputs.

## What QuantProof checks

| Area | Checks | Implementation |
|---|---|---|
| Data quality | `QP-DATA-001…013`: invalid, unsorted, duplicated timestamps; missing/impossible OHLC; non-positive prices; nulls; unexplained gaps; timezone mixtures; forward-filled (stale) prices; infinities; extreme returns | `quantproof.data.validate_data` |
| Static code analysis | `QP001…QP015`: negative shifts, centered windows, look-ahead indexing, shuffled splits and CV, fit-before-split, full-sample normalization, target leakage, same-bar execution, missing execution lag, missing costs, future-derived features, large searches, no OOS evaluation, non-causal transforms | AST + taint analysis, `quantproof.analyzers.static` |
| Runtime causality | `QP-CAUSAL-001…004`: perturb data strictly after *t* (additive, multiplicative, permutation, extreme shocks) and verify decisions at or before *t* do not change | `quantproof.analyzers.causal` |
| Leakage (values) | `QP-LEAK-001…003`: features replicating the target or future returns; implausible directional accuracy | `quantproof.analyzers.leakage` |
| Execution realism | `QP-EXEC-001…005`: naive vs declared vs realistic scenarios, lag sensitivity, cost sensitivity and break-even cost, turnover, signal/execution timestamps | `quantproof.execution` |
| Statistics | `QP-STAT-001…007`: sample size, PSR, DSR, implausible Sharpe, undeclared trials, non-normality, serial correlation (Lo-adjusted Sharpe) | `quantproof.statistics` |
| Validation / selection | `QP-VAL-001…004`: walk-forward selection, PBO (CSCV), CPCV paths, White Reality Check / Hansen SPA | `quantproof.validation`, `quantproof.statistics` |
| Regimes | `QP-REGIME-001`: performance by volatility, drawdown and (with a benchmark) bull/bear regime | `quantproof.regimes` |
| Parameter sensitivity | `QP-SENS-001…002`: plateau vs fragile optimum, in-sample/out-of-sample rank persistence | `quantproof.sensitivity` |
| Reproducibility | manifest with data/strategy/config hashes, Git commit, environment, seed, lineage | `quantproof.experiments` |

The full catalogue with severity policy is in [docs/api/rules.md](docs/api/rules.md).

## Statistical methods

All are implemented directly with NumPy/SciPy, documented with their assumptions,
and tested against independent reference calculations or Monte Carlo calibration:

- **Sharpe ratio** with geometric risk-free conversion, non-normal standard error (Mertens 2002), Lo (2002) autocorrelation adjustment
- **Probabilistic Sharpe Ratio** and **Minimum Track Record Length** (Bailey & López de Prado 2012)
- **Deflated Sharpe Ratio** with the expected-maximum-Sharpe hurdle (Bailey & López de Prado 2014)
- **Probability of Backtest Overfitting** via Combinatorially Symmetric Cross-Validation (Bailey, Borwein, López de Prado & Zhu 2017)
- **White's Reality Check** (2000) and **Hansen's SPA** (2005) with the stationary bootstrap; Bonferroni/Holm/Benjamini-Hochberg adjustments
- **Bootstrap**: i.i.d., moving-block, circular-block, stationary; Politis-White automatic block length
- **Temporal validation**: walk-forward (expanding/rolling, gap), purged K-fold with embargo, Combinatorial Purged Cross-Validation (López de Prado 2018)

Methodology pages explain each one — problem, implementation, assumptions, example,
limitations, references: [docs/methodology/](docs/methodology/README.md).

## Architecture

```text
src/quantproof/
├── audit/            engine, result model, verdict rules
│   ├── static/       AST analyzer + rules QP001–QP015 (pluggable via @register)
│   ├── causal/       future-perturbation schemes and runner
│   ├── leakage/      value-based leakage diagnostics
│   ├── execution/    execution-realism analyzer
│   └── statistical/  statistics + selection-bias analyzers
├── statistics/       Sharpe, PSR, DSR, PBO, Reality Check/SPA, bootstrap
├── validation/       walk-forward, purging, embargo, purged K-fold, CPCV
├── execution/        cost components, fills, simulator, turnover, timing, providers/
├── regimes/          volatility, drawdown, trend regimes
├── sensitivity/      parameter-surface analysis
├── data/             loaders, schemas, validation, synthetic data
├── experiments/      hashing, manifest, lineage
├── reports/          HTML, Markdown, JSON, text
├── adapters/         ResearchArtifacts + pandas reference adapter
├── strategy.py       strategy contract and loader
├── config.py         AuditConfig (Python / YAML / TOML / JSON)
└── cli.py            `quantproof` command
```

Analyzers return findings with a common structure (id, category, severity, title,
message, evidence, location, why it matters, recommendation, confidence). The
verdict is not a score; it follows three explicit rules:

1. **FAIL** if any finding is FAIL (a critical research-validity violation);
2. **WARN** if no FAIL but at least one WARN;
3. **PASS** otherwise.

Each analyzer documents the thresholds that decide its findings' severities, and the
report lists the findings that drove the verdict.

## Building blocks

The pieces are usable on their own:

```python
from quantproof.statistics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from quantproof.validation import CPCV, PurgedKFold, WalkForward
from quantproof.execution import TransactionCostModel, simulate, india_nse_provider

dsr = deflated_sharpe_ratio(returns, n_trials=200, trial_sharpes=all_trial_sharpes)
for train, test in PurgedKFold(n_splits=5, label_horizon=5, embargo=0.01).split(X):
    ...
cv = CPCV(n_groups=8, n_test_groups=2, embargo=0.01)  # 28 splits, 7 backtest paths
costs = TransactionCostModel.from_bps(commission_bps=1, spread_bps=2, slippage_bps=2)
```

Jurisdiction-specific statutory charges live in effective-dated providers rather than
in the core. A reference provider for Indian equities (NSE: STT, exchange
transaction charges, SEBI fee, stamp duty, GST — including the F&O STT change
effective 2026-04-01) shows the pattern; see
[docs/methodology/transaction-costs.md](docs/methodology/transaction-costs.md) and
verify all rates against primary sources before use.

## Integrations

QuantProof's core never imports a backtesting engine. Any engine's output can be
converted into `ResearchArtifacts` (returns, trial returns and parameters, benchmark,
prices, signals, positions, trades with signal/execution timestamps, features,
target). The bundled `PandasAdapter` handles pandas objects and CSV/Parquet files;
adapters for VectorBT, Backtrader, LEAN, NautilusTrader or Qlib only need to
implement `Adapter.load`. See [docs/api/adapters.md](docs/api/adapters.md).

## Examples

| Example | What is wrong | What QuantProof reports |
|---|---|---|
| [`clean_strategy`](examples/clean_strategy/strategy.py) | nothing structural | no static, causality, leakage or execution issues; WARN only because the statistical evidence for an edge is weak (it loses money on this synthetic data) |
| [`lookahead_strategy`](examples/lookahead_strategy/strategy.py) | `shift(-1)`, centered window | **FAIL**: QP001, QP002, QP-CAUSAL-001; WARN QP-LEAK-003, QP-STAT-004 |
| [`leakage_strategy`](examples/leakage_strategy/strategy.py) | global scaler, shuffled split, forward return in features | **FAIL**: QP001, QP006, QP012, QP-CAUSAL-001; WARN QP004 |
| [`overfit_strategy`](examples/overfit_strategy/strategy.py) | best of 240 rules on a random walk | WARN: QP013, DSR 0.17, PBO 0.60, walk-forward OOS Sharpe −0.60, CPCV, SPA, fragile optimum |
| [`unrealistic_execution`](examples/unrealistic_execution/strategy.py) | same-bar fills, zero costs | **FAIL**: QP-EXEC-001 (Sharpe 1.64 → 0.10 under audit assumptions); WARN QP009, QP011, cost fragility |

Run them all with `python examples/run_examples.py`. Details and full numbers:
[docs/examples/README.md](docs/examples/README.md).

## Limitations

- Static analysis is heuristic. It reasons about code patterns and data flow within a
  file; leakage hidden behind indirection, other modules, or data-dependent control
  flow can be missed, and unusual but valid code can be flagged (findings carry a
  confidence level and can be suppressed inline with `# quantproof: ignore[QP001]`).
- The future-perturbation test is evidence *against* a class of causal violations for
  the tested timestamps and schemes. It cannot detect leakage already embedded in the
  data (survivorship bias, restated fundamentals, wrong timestamps).
- The reference simulator fills at bar prices with simple cost models. It does not
  reproduce an exchange matching engine, queue position or intrabar latency.
- PSR/DSR rely on the asymptotic distribution of the Sharpe estimator for i.i.d.
  returns; bootstrap intervals are approximate.
- Single-asset signal strategies are the primary runtime target; multi-asset
  portfolios can be audited through results mode.
- **Strategy code is executed in your Python process without a sandbox.** Only audit
  code you trust, or run QuantProof in an isolated environment. See
  [SECURITY.md](SECURITY.md).

## Development

```bash
pip install -e ".[dev]"
pytest                          # unit, integration and property-based tests
ruff check . && ruff format --check .
mypy                            # strict typing of the library
python -m build                 # sdist + wheel
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for adding rules, adapters and cost providers.

## License

MIT — see [LICENSE](LICENSE). All bundled data is synthetic.
