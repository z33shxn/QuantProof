# A bad research workflow (and what QuantProof says about it)

This walkthrough follows a first draft of a cross-sectional momentum strategy as it
often looks after an afternoon in a notebook. Every number below comes from

```bash
python examples/proof_of_value/run.py
```

on the synthetic universe `examples/data/universe.parquet` (10 symbols, 1,250 business
days). The draft is `examples/proof_of_value/flawed_strategy.py`.

## The draft

```python
EXECUTION = {"signal_lag": 0, "fill": "close", "commission_bps": 0.0}


def generate_signals(data, lookback=120, n_side=2):
    close = data["close"].unstack("symbol")
    momentum = close.pct_change(lookback)
    smooth = momentum.rolling(5, center=True).mean()  # "denoise" the score
    ranks = smooth.rank(axis=1, method="first")
    ...
```

The researcher tried 36 combinations of lookback, skip, number of names and rebalance
frequency, kept the best one, and reports a Sharpe ratio of **2.26**.

## Mistake 1: a centered window

`rolling(5, center=True)` averages the score at *t* with the scores at *t+1* and *t+2*.
The strategy reads the future.

- **QP002 FAIL** (static): centered rolling window reaching a live decision, labelled
  *FORBIDDEN IN LIVE DECISION*.
- **QP-CAUSAL-001 FAIL** (runtime): perturbing data after a decision time changed
  historical decisions in 11 of 19 trials. The finding names the decision time, the
  first perturbed observation, the symbol and the before/after weight.

Because the centered window peeks two days ahead, even the audit's one-bar-lag
simulation still benefits from the leak: the "audited" Sharpe of 1.40 is also
contaminated. The report's next steps say so explicitly: *fix look-ahead first; every
other statistic is computed on contaminated returns*.

## Mistake 2: same-bar fills and free trading

`signal_lag: 0` assumes the trade fills at the close that produced the signal, and no
costs are declared.

- **QP009 WARN** / **QP-EXEC-001 WARN**: same-bar execution.
- **QP011 WARN** / **QP-EXEC-002 WARN**: no transaction-cost model.

## Mistake 3: hiding the search

The draft declares no `PARAM_GRID`, so DSR, PBO, CPCV and the Reality Check cannot
account for the 36 variants tried. QuantProof records the trial count as
*not declared (assumed 1)* (QP-STAT-005, INFO; WARN in the strict profile). It cannot
know about variants that are not declared. This is the one mistake an auditor can only
point out, not detect.

## Verdict

**FAIL**, primary reason QP-CAUSAL-001. Nothing else in the report should be read until
the look-ahead is fixed.

Continue with the [good research workflow](good-research-workflow.md).
