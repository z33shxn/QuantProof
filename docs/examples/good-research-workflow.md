# A good research workflow

The corrected strategy is `examples/cross_sectional_momentum/strategy.py`. The changes
relative to the [bad workflow](bad-research-workflow.md):

1. **Causal inputs only.** The score is the trailing return from *t − lookback* to
   *t − skip*; no centered windows, no full-sample statistics.
2. **Explicit execution assumptions.** One bar of lag and declared commission, spread and
   slippage:

   ```python
   EXECUTION = {
       "signal_lag": 1,
       "fill": "close",
       "commission_bps": 1.0,
       "spread_bps": 4.0,
       "slippage_bps": 2.0,
   }
   ```

3. **The whole search is declared.** All 36 variants are in `PARAM_GRID`, so QuantProof
   evaluates every one and uses them for DSR, PBO, CPCV and the data-snooping test.
4. **The audit runs before the results are believed:**

   ```bash
   quantproof audit -s examples/cross_sectional_momentum/strategy.py \
       -d examples/data/universe.parquet -o report.html
   ```

## What the audit says

- Static analysis: no WARN or FAIL. Runtime causality: no decision changed.
- Execution: the declared assumptions are at least as conservative as the audit's.
- **WARN**, primary reason **QP-VAL-001 (weak out-of-sample evidence)**: picking the
  best configuration on each training window and testing it on the next gives a negative
  out-of-sample Sharpe, with only 1 of 5 folds positive.
- Supporting evidence: PBO ≈ 0.8 (the in-sample winner usually lands in the bottom half
  out of sample), DSR ≈ 0.5 with 36 trials (the observed Sharpe is about what the best of
  36 skill-less variants would show), the Hansen SPA test does not reject, PSR just below
  0.95.

The audited full-sample Sharpe of the default parameters (0.82) looks respectable. The
selection-aware evidence says it is not distinguishable from the best of 36 lucky draws.
That is the honest answer, and it is what this workflow is for: QuantProof did not make
the strategy better, it made the claim about it accurate.

## Checklist

- [ ] Every input at *t* is known at *t* (static + runtime checks pass).
- [ ] Execution lag and costs are declared and at least as conservative as reality.
- [ ] Every variant tried is declared (`PARAM_GRID`, `statistics.trials` or
      `trial_returns`); run the **strict** profile before a final decision.
- [ ] The decision is based on out-of-sample evidence (walk-forward, CPCV paths), DSR and
      PBO, not on the in-sample headline.
- [ ] Data-quality findings are resolved, not suppressed.
- [ ] The report (with its manifest and fingerprints) is stored with the research.
