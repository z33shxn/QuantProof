# Temporal cross-validation and walk-forward selection

## Problem

Financial observations are ordered and serially dependent. Random splits put the future
in the training set; even contiguous K-fold trains on data that comes after the test
fold. Estimates of out-of-sample performance from such splits are biased upward.

## Implementation

`quantproof.validation.WalkForward(train_size, test_size=None, step=None, gap=0,
expanding=True, n_splits=None)` yields position arrays:

```text
expanding                         rolling
TRAIN TRAIN | TEST                TRAIN TRAIN | TEST
TRAIN TRAIN TRAIN | TEST                TRAIN TRAIN | TEST
TRAIN TRAIN TRAIN TRAIN | TEST                TRAIN TRAIN | TEST
```

- test windows start `gap` bars after their training window ends and never overlap;
- sizes may be integers (bars) or fractions of the sample;
- `diagram(X)` renders the folds as text (used in the HTML report).

`temporal_train_test_split(n, test_size, gap)` is the single-split equivalent.

**Walk-forward selection (audit).** With a parameter grid, the audit evaluates the
*research procedure*, not a single configuration: in each fold it selects the
configuration with the best training Sharpe and records its returns on the following
test window. The concatenated test returns form an out-of-sample track record of
"pick the best in-sample". QP-VAL-001 warns if that OOS Sharpe is ≤ 0, or below half of a
positive mean in-sample Sharpe.

Without a grid, the audit reports the Sharpe ratio of each sequential window and warns
if more than half are negative.

## Assumptions

- Configuration returns are computed once on the full sample and then sliced. This is
  valid only for causal signals (verified by the [causality test](runtime-causality.md));
  otherwise the diagnostics inherit the contamination.
- Selection uses the Sharpe ratio of realistic net returns.

## Example

For `examples/overfit_strategy` (best of 240 rules on a random walk) the mean in-sample
Sharpe of the selected configurations is 0.62 while the walk-forward OOS Sharpe is −0.60.

## Limitations

A single walk-forward path uses each observation once as test data, so its estimate has
high variance. [CPCV](cpcv.md) provides a distribution of paths.

## References

- Bergmeir, C. & Benítez, J. M. (2012). On the use of cross-validation for time series
  predictor evaluation. *Information Sciences*, 191.
- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*, 2nd ed. Wiley.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7 and 12.
