# Combinatorial Purged Cross-Validation (CPCV)

## Problem

Walk-forward produces one out-of-sample path, so its performance estimate is a single,
noisy draw. CPCV produces many complete out-of-sample paths from the same data.

## Implementation

`quantproof.validation.CPCV(n_groups=N, n_test_groups=k, embargo=0, event_end=None,
event_start=None, label_horizon=0)` (`t1=` is accepted as an alias of `event_end`):

1. Split the sample into *N* contiguous groups (`numpy.array_split`).
2. Each of the `C(N, k)` combinations of *k* groups is a test set (deterministic
   lexicographic order).
3. Training data is every other group after [purging and embargo](purging-and-embargo.md)
   around **each contiguous test block** (adjacent test groups form one block).
4. Each group is a test group in `C(N−1, k−1)` splits, which yields
   `φ = C(N, k) · k / N` backtest paths. Path *p* uses, for group *g*, the *p*-th split (in
   split order) in which *g* is tested; every path covers the sample exactly once.

`CPCV.paths()` returns the `(group, split)` assignments; `assemble_paths(X, preds)` stitches
per-split test outputs into a `(φ, n)` array.

```python
cv = CPCV(n_groups=8, n_test_groups=2, embargo=0.01)
cv.n_splits, cv.n_paths  # (28, 7)
```

**In the audit**, for each split the configuration with the best training Sharpe is
selected and evaluated on the test groups; the paths' Sharpe ratios are reported and
QP-VAL-003 warns if the median path Sharpe is ≤ 0.

## Assumptions

Same as walk-forward selection: configuration returns are precomputed, which is valid only
for causal signals.

## Example

Default audit settings `N=6, k=2` give 15 splits and 5 paths. For
`examples/overfit_strategy` the path Sharpe ratios are mostly negative.

## Limitations

Cost grows as `C(N, k)`: `N=10, k=2` → 45 splits; `N=16, k=4` → 1,820. Training sets in
CPCV contain data *after* the test groups (purged and embargoed), which is appropriate for
evaluating models of stationary relationships but not a simulation of real-time research.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 12.
