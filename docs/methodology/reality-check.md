# Data snooping: White's Reality Check, Hansen's SPA, multiple-testing adjustment

## Problem

When many strategies are tested on the same data, the best one's p-value from a single
test is meaningless. A data-snooping test asks: given *all* strategies tried, is there
evidence that *any* of them beats the benchmark?

## Implementation

`quantproof.statistics.reality_check(strategy_returns, benchmark_returns=None,
n_bootstrap=1000, block_length=None, seed=42)` on a `T × K` matrix:

- Differentials `d_k,t = r_k,t − b_t` (benchmark defaults to zero).
- **White's Reality Check**: `V = max_k √T · mean(d_k)`. Null distribution from the
  stationary bootstrap (same resampled indices for all strategies, preserving
  cross-sectional dependence): `V*_b = max_k √T · (mean*(d_k) − mean(d_k))`.
- **Hansen's SPA (consistent version)**: studentized by `ω_k` (bootstrap std of
  `√T · mean*(d_k)`); strategies with `√T · mean(d_k)/ω_k < −√(2 log log T)` keep their
  (negative) mean in the bootstrap, so clearly poor strategies cannot drive the maximum;
  the others are recentred at zero.
- p-values use `(1 + #{T* ≥ T}) / (1 + B)`.
- `ω_k` is estimated from the bootstrap draws (Hansen allows this or a HAC estimator).
  If every differential is constant, the studentized statistic is undefined: the SPA
  p-value is `NaN` and QP-VAL-004 is INFO, never PASS. (Before this was fixed, degenerate
  input produced `p = 1/(1+B)`, a spurious "significant" result.)

**Null hypotheses.** RC and SPA: H0 `max_k E[d_k] ≤ 0` — no supplied strategy beats the
benchmark in expectation. Rejection says the *best supplied* strategy's mean differential
is unlikely to be zero given the whole supplied set. It does **not** account for
strategies tried but not supplied, does not estimate future performance, and says nothing
about economic significance after costs.
- Block length defaults to the Politis-White automatic choice for the best strategy.

`adjust_pvalues(p, method)` provides Bonferroni, Holm (step-down FWER) and
Benjamini-Hochberg (FDR) adjustments for individual tests.

QP-VAL-004 warns when the SPA p-value exceeds `1 − statistics.confidence`.

## Assumptions

- Differentials are strictly stationary and weakly dependent (stationary bootstrap validity).
- The tested set is the full set searched. Omitting tried strategies invalidates the test.

## Example

Tested: 30 noise strategies give RC and SPA p-values > 0.05; adding 30 bps/day to one
gives p < 0.05 with the correct best index; with many clearly poor strategies SPA is no
more conservative than RC.

## Limitations

The Reality Check is conservative when many poor strategies are included (the reason SPA
exists). Both tests have limited power in short samples.

## References

- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5).
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business &
  Economic Statistics*, 23(4).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate.
  *JRSS B*, 57(1).
