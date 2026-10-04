# Key objects

Short, runnable examples for the objects most users need. Every code block on this page
is executed by `tests/integration/test_docs.py`, so the examples cannot silently go stale.

## `audit` and `AuditResult`

```python
import numpy as np
from quantproof import AuditConfig, audit
from quantproof.data import generate_prices

prices = generate_prices(600, seed=1)


def generate_signals(data, lookback=20):
    return np.sign(data["close"].pct_change(lookback)).fillna(0.0)


result = audit(generate_signals, prices, config=AuditConfig(profile="quick"))
result.status  # Severity.PASS / WARN / FAIL
result.narrative.primary_reason  # the single most important finding, in words
[f.id for f in result.issues]  # WARN and FAIL findings
result.metrics["realistic"]["sharpe"]
result.statistics["dsr"]["dsr"]
result.execution["cost_attribution"]["rows"]
result.reproducibility["fingerprints"]["data"]
html = result.to_html()
md = result.to_markdown()
js = result.to_json()
```

`audit(strategy=None, data=None, *, config=None, benchmark=None, artifacts=None,
returns=None, trial_returns=None, timestamp_column=None, symbol_column=None)`.
`strategy` is a file path, a callable or a `StrategySpec`; results mode uses `returns=`,
`trial_returns=` or `artifacts=` instead.

## `AuditConfig`

```python
from quantproof import AuditConfig

cfg = AuditConfig.from_profile("strict")
cfg.execution.commission_bps = 2.0
cfg.statistics.trials = 120  # variants tried, including discarded ones
effective = cfg.effective()  # the profile applied; what the audit actually uses
assert effective.statistics.n_bootstrap >= 2000
```

## `Finding`

```python
from quantproof import Category, Finding, Severity

f = Finding(
    id="QP-EXEC-002",
    category=Category.EXECUTION,
    severity=Severity.WARN,
    title="Transaction cost model missing",
    message="No costs declared.",
)
f.why_it_matters, f.impact, f.investigate, f.recommendation  # filled from the registry
f.usage  # None, or Usage.LIVE_DECISION / Usage.LABEL_OR_ANALYSIS for look-ahead
```

Rule documentation: `quantproof.rules.get_rule("QP-EXEC-002")` or
`quantproof rules show QP-EXEC-002`.

## `PurgedKFold` and `CPCV`

```python
import numpy as np
import pandas as pd
from quantproof.validation import CPCV, PurgedKFold

idx = pd.date_range("2024-01-01", periods=100, freq="D")
X = pd.DataFrame({"x": np.arange(100.0)}, index=idx)
label_end = pd.Series(idx + pd.Timedelta(days=3), index=idx)  # event-time labels

for train, test in PurgedKFold(5, event_end=label_end, embargo="2D").split(X):
    assert not set(train) & set(test)

cv = CPCV(n_groups=6, n_test_groups=2, label_horizon=3, embargo=2)  # sample-time labels
assert (cv.n_splits, cv.n_paths) == (15, 5)
```

## Sharpe, PSR, DSR, PBO

```python
import numpy as np
from quantproof.statistics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

rng = np.random.default_rng(0)
r = rng.normal(0.0005, 0.01, 1000)
trials = rng.normal(0.0, 0.01, (1000, 20))

sharpe_ratio(r, periods_per_year=252)  # annualized
probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0).psr  # P(true SR > 0)
dsr = deflated_sharpe_ratio(r, n_trials=20, expected_max_method="exact")
dsr.dsr, dsr.to_dict()["expected_max_sharpe_annualized"]
pbo = probability_of_backtest_overfitting(trials, n_partitions=10)
pbo.pbo, pbo.notes
```

## `TransactionCostModel`

```python
from quantproof.execution import (
    BpsCommission,
    FixedSpread,
    SquareRootImpact,
    TransactionCostModel,
    TransactionTax,
    simulate,
)
from quantproof.data import generate_prices

prices = generate_prices(300, seed=2)
model = TransactionCostModel.compose(
    commission=BpsCommission(1),
    spread=FixedSpread(4),
    impact=SquareRootImpact(0.1),
    taxes=TransactionTax(5, side="buy"),
)
weights = (prices["close"].pct_change(10) > 0).astype(float)
sim = simulate(prices, weights, fill="close", lag=1, cost_model=model)
sim.cost_breakdown.sum()  # per component
stressed = simulate(prices, weights, lag=1, cost_model=model.scaled(2.0))
```

## Data validation (`validate_data`)

There is no `DataValidator` class; validation is a function returning findings.

```python
from quantproof.data import generate_universe, validate_data

universe = generate_universe(4, 300, seed=3)  # long format with a symbol column
findings = validate_data(universe)
[(f.id, f.severity.value) for f in findings if f.is_issue]
```

## Runtime causality (`run_causality_test`)

There is no `CausalityTester` class; the test is a function returning a report.

```python
import numpy as np
from quantproof.analyzers.causal import causality_findings, run_causality_test
from quantproof.config import CausalityConfig
from quantproof.data import generate_prices

prices = generate_prices(300, seed=4)
cfg = CausalityConfig(n_timestamps=6, schemes=["additive", "replacement", "extreme"], seed=1)


def peeking(d):
    return np.sign(d["close"].pct_change().shift(-1))


report = run_causality_test(peeking, prices, cfg)
assert report.n_fail > 0
trial = next(t for t in report.trials if t.status == "fail")
trial.decision_time, trial.first_modified, trial.original, trial.perturbed
causality_findings(report)[0].message
```
