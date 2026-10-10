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

## `GenericResultsAdapter`: any engine that can export a table

Most engines can export a returns or equity series and a trade ledger. The generic adapter
maps those exports with explicit column names; it never imports an engine.

```python
from quantproof import GenericResultsAdapter, audit

adapter = GenericResultsAdapter(
    equity_column="equity",  # or returns_column="returns"
    returns_are="net",  # required: "net" (after costs) or "gross"
    trade_columns={"signal_time": "decision_ts", "execution_time": "fill_ts"},
)
artifacts = adapter.load(
    "exports/equity.csv",  # DataFrame, Series or CSV/Parquet path
    trades="exports/trades.csv",
    trial_returns="exports/all_variants.parquet",  # optional, enables PBO/DSR/CPCV/SPA
    metadata={"engine": "my-engine 2.3"},
)
result = audit(artifacts=artifacts)
```

Conversions are recorded in `artifacts.metadata["conversions"]` (e.g. "returns derived
from equity column 'equity' (first bar dropped)"). Column names are matched after the
same normalization the loaders apply (case and spacing insensitive). Duplicate
timestamps, missing columns and non-positive equity are errors, not silent fixes.

## Building adapters for specific engines

QuantProof ships **no** VectorBT, Backtrader or LEAN adapter and has not been tested
against those engines. The notes below describe which outputs to export; attribute and
column names differ between engine versions, so check them against your installed
version and keep the adapter in your own code base.

The key requirement for execution-timing checks (QP-EXEC-005) is a ledger with the time
the **decision** was made (`signal_time`) and the time the order was **filled**
(`execution_time`). Many engines record only entry/exit fill times; in that case omit the
ledger rather than passing fill times as signal times, which would make every trade look
instantaneous.

### VectorBT

- Returns: the portfolio's per-period returns (e.g. `Portfolio.returns()`) or its value
  series (`Portfolio.value()`), exported as a Series indexed by timestamp.
- Variants: when you run a parameter sweep, export the returns of *every* column of the
  portfolio as `trial_returns`. This is what makes PBO/DSR meaningful.
- Ledger: VectorBT's trade records contain entry/exit *fill* timestamps; signal times must
  come from your own signal arrays (the bar at which the entry signal was `True`).
- State `returns_are="net"` only if fees/slippage were configured in the portfolio.

### Backtrader

- Returns: attach a time-return analyzer (`bt.analyzers.TimeReturn`) and convert its
  `get_analysis()` mapping (datetime → return) into a Series.
- Ledger: record `self.datas[0].datetime.datetime(0)` when an order is created in `next()`
  (signal time) and the execution datetime in `notify_order` when the order is completed
  (fill time).
- Backtrader fills market orders at the next bar's open by default ("cheat-on-close"
  changes this); record which mode was used in `metadata`.

### QuantConnect LEAN

- Returns: the backtest result's equity series (strategy equity chart) exported to CSV,
  used with `equity_column=...`.
- Ledger: order events carry submission and fill times; use the time the order was
  submitted by the algorithm as `signal_time` and the fill time as `execution_time`.
- LEAN results are net of the brokerage model's fees; set `returns_are="net"`.

### Checklist for any adapter

1. Returns are per period and aligned to the period they were earned in.
2. `returns_are` states whether costs are included.
3. `trial_returns` contains **all** variants tried, aligned on a common index.
4. The ledger's `signal_time` is the decision time, not a fill time.
5. Write a small test with a known equity curve (see
   `tests/unit/test_strategy_adapters.py::test_generic_adapter_equity_and_ledger`).
