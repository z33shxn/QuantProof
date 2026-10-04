# Probability of Backtest Overfitting (PBO)

## Problem

DSR asks whether the selected result beats luck. PBO asks a different question about the
*selection procedure*: how often does the configuration that looks best in-sample end up in
the bottom half out of sample?

## Implementation

`quantproof.statistics.probability_of_backtest_overfitting(M, n_partitions=S,
max_combinations=5000)` implements Combinatorially Symmetric Cross-Validation (CSCV):

1. `M` is a `T × N` matrix of per-period returns (one column per configuration), without
   missing values.
2. Rows are split into *S* (even) contiguous blocks.
3. For every combination *c* of *S/2* blocks: in-sample = those blocks, out-of-sample = the
   complement.
4. `n*` = the configuration with the highest in-sample Sharpe ratio.
5. `ω = rank_OOS(n*) / (N + 1)` with rank 1 = worst and ties averaged;
   `λ = ln(ω / (1 − ω))`.
6. `PBO = share of combinations with λ ≤ 0`.

Also reported: the logit distribution (histogram in the HTML report), the in-sample and
out-of-sample Sharpe of the selected configuration per combination, the regression slope of
OOS on IS Sharpe ("performance degradation"), and the probability that the selected
configuration loses money out of sample.

Sharpe ratios for each combination are computed from per-block sums of `r` and `r²`, so
each combination costs `O(N)` after `O(T·N)` preprocessing. `C(16, 8) = 12,870`; above
`max_combinations` a deterministic seeded subset is used and both counts are reported.

## Assumptions

- Blocks are exchangeable enough that any half of them is a fair in-sample set (CSCV
  ignores order within the IS and OOS sets; the Sharpe metric is order-invariant).
- Counting `λ = 0` as overfit is slightly conservative (relevant only for odd N or ties).

## Example

Tested properties: on pure noise PBO averages ≈ 0.5 across seeds; a configuration that
dominates in every block gives PBO = 0; two configurations that each win one half and lose
the other give PBO = 1. For `examples/overfit_strategy`, PBO = 0.60 (S = 10, 252
combinations, 200 evaluated configurations).

## Limitations

PBO measures the selection procedure on this sample. A low PBO with uniformly poor
configurations is still a poor strategy (check the OOS Sharpe and probability of loss).
Very short blocks make Sharpe estimates noisy; keep `T / S` reasonably large.

## References

- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017). The probability of
  backtest overfitting. *Journal of Computational Finance*, 20(4).
