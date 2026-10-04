# Adapters and results mode

QuantProof never imports a backtesting engine. Engines are integrated by converting their
outputs into `ResearchArtifacts`:

| Field | Type | Enables |
|---|---|---|
| `returns` | Series | statistics, walk-forward windows, regimes |
| `trial_returns` | DataFrame (T × N) | DSR with trial variance, PBO, CPCV, Reality Check |
| `trial_params` | DataFrame (N rows) | parameter surface |
| `benchmark` | Series | benchmark regimes (incl. bull/bear) |
| `prices` | DataFrame | data validation, asset-relative leakage checks |
| `signals` | Series | directional-foresight check |
| `positions` | Series/DataFrame | turnover |
| `trades` | DataFrame with `signal_time`, `execution_time` | execution-timing check (QP-EXEC-005) |
| `features`, `target` | DataFrame, Series | feature/target leakage checks |
| `metadata` | dict | recorded in the report |

```python
from quantproof import PandasAdapter, audit

artifacts = PandasAdapter().load(
    returns="results/returns.parquet",
    trial_returns="results/all_variants.parquet",
    trial_params="results/variant_params.csv",
    benchmark=spy_returns,
    trades=trade_ledger,
    metadata={"name": "my_backtest", "engine": "in-house"},
)
result = audit(artifacts=artifacts)
```

## Writing an adapter

Implement the `quantproof.adapters.Adapter` protocol:

```python
from quantproof.adapters import ResearchArtifacts


class MyEngineAdapter:
    name = "my_engine"

    def load(self, source, **kwargs) -> ResearchArtifacts:
        portfolio = source  # engine-native object
        return ResearchArtifacts(
            returns=portfolio.returns(),  # per-period, net of the engine's costs
            positions=portfolio.weights(),
            trades=portfolio.trades()[["signal_time", "execution_time"]],
            metadata={"engine": "my_engine", "costs": "net"},
        )
```

Keep engine imports inside the adapter module so the core stays engine-independent. State
in `metadata` whether returns are gross or net of costs.
