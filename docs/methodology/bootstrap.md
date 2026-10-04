# Bootstrap methods for financial returns

## Problem

Confidence intervals for strategy statistics need a sampling distribution. Financial
returns are serially dependent (volatility clustering, autocorrelation from overlapping
positions), so resampling individual observations independently understates uncertainty.

## Implementation

`quantproof.statistics.bootstrap`:

| Method | Function | Notes |
|---|---|---|
| i.i.d. | `iid_indices` | Efron's bootstrap. Destroys dependence; use only for independent data. |
| moving block | `moving_block_indices` | Künsch (1989); fixed block length, no wrap-around. |
| circular block | `circular_block_indices` | Politis & Romano (1992); blocks wrap around the end. |
| stationary | `stationary_indices` | Politis & Romano (1994); geometric block lengths with mean *L*; resamples are stationary. **Default.** |

`optimal_block_length(r)` implements Politis & White (2004) with the Patton, Politis &
White (2009) correction: flat-top lag window, `m̂` chosen as the first lag after which
`K_N = max(5, ⌈√log10 n⌉)` autocorrelations are all below `2√(log10 n / n)`, then

```text
b_SB = (2 Ĝ² / D_SB)^{1/3} n^{1/3},  D_SB = 2 ĝ(0)²
b_CB = (2 Ĝ² / D_CB)^{1/3} n^{1/3},  D_CB = (4/3) ĝ(0)²
```

clipped to `[1, min(3√n, n/3)]`.

`bootstrap_statistic(r, statistic, method, n_boot, block_length, confidence, seed)` takes
a statistic vectorized over rows, resamples in chunks (bounded memory), and returns the
point estimate, bootstrap mean and standard error, percentile interval, method, block
length actually used, seed, and the share of resamples ≤ 0. `bootstrap_sharpe` wraps it for
the annualized Sharpe ratio.

## Assumptions

Block methods assume stationarity and weak dependence. Percentile intervals are
first-order accurate.

## Example

Tested: on an AR(1) series with φ = 0.7, the lag-1 autocorrelation of i.i.d. resamples is
≈ 0, while stationary-bootstrap resamples (L = 30) keep it above 0.5. The automatic block
length for i.i.d. noise is < 3 and grows by more than 5× for φ = 0.8.

## Limitations

Percentile intervals can under-cover in small samples and for heavy-tailed statistics.
Structural breaks violate stationarity. Treat intervals as uncertainty evidence, not exact
coverage guarantees.

## References

- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics*, 7(1).
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary
  observations. *Annals of Statistics*, 17(3).
- Politis, D. N. & Romano, J. P. (1992). A circular block-resampling procedure for
  stationary data. In *Exploring the Limits of Bootstrap*. Wiley.
- Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428).
- Politis, D. N. & White, H. (2004). Automatic block-length selection for the dependent
  bootstrap. *Econometric Reviews*, 23(1); correction: Patton, A., Politis, D. N. &
  White, H. (2009), *Econometric Reviews*, 28(4).
