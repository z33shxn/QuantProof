# Sharpe ratio, Probabilistic Sharpe Ratio, Minimum Track Record Length

## Problem

A Sharpe ratio is an estimate. Its sampling error depends on the sample length and on
skewness and fat tails, which are common in strategy returns. A positive Sharpe ratio on a
short or fat-tailed sample is weak evidence.

## Implementation

`quantproof.statistics`:

- `sharpe_ratio(r, periods_per_year=252, risk_free_rate=0, annualize=True)`:
  `SR = mean(r − rf_p) / std(r − rf_p, ddof=1)`, with the annual risk-free rate converted
  geometrically, `rf_p = (1 + rf)^(1/periods_per_year) − 1`, then multiplied by
  `√periods_per_year`. Returns `nan` for fewer than two observations or zero variance
  (never `inf`). NaNs are dropped by default (`nan_policy="raise"` to refuse them);
  infinite values raise with a hint (they usually come from a zero price).
- `return_moments`: skewness `m3/m2^1.5` and **non-excess** kurtosis `m4/m2²` (biased
  sample moments, normal = 3), matching Bailey & López de Prado (2012).
- `sharpe_standard_error(SR, n, γ3, γ4) = √((1 − γ3·SR + (γ4 − 1)/4·SR²)/(n − 1))` for
  per-period SR (Mertens 2002; Opdyke 2007). With γ3 = 0, γ4 = 3 this is Lo's (2002)
  i.i.d.-normal result.
- `probabilistic_sharpe_ratio(r, benchmark_sharpe=SR*)`:

  ```text
  PSR(SR*) = Φ( (SR − SR*) · √(T − 1) / √(1 − γ3·SR + (γ4 − 1)/4 · SR²) )
  ```

  with per-period SR; the benchmark is given annualized and divided by
  `√periods_per_year`.
- `minimum_track_record_length(SR, γ3, γ4, benchmark_sharpe, confidence)`:

  ```text
  MinTRL = 1 + (1 − γ3·SR + (γ4 − 1)/4 · SR²) · (z_conf / (SR − SR*))²
  ```

  `inf` if `SR ≤ SR*`.
- **Lo (2002) adjustment** (reported in the audit): annualizing by `√q` assumes no serial
  correlation. With autocorrelations ρ_k, `SR_q = SR · q / √(q + 2 Σ_{k<q} (q − k) ρ_k)`;
  QuantProof uses the first 10 sample autocorrelations and treats the rest as zero.

## Assumptions

- Returns are i.i.d. (possibly non-normal) for the PSR/standard-error formulas; the PSR is
  based on the asymptotic normal distribution of the Sharpe estimator.
- Annualization by `√periods_per_year` is exact only for i.i.d. returns.

## Example

Monte Carlo check in the test suite: with zero true Sharpe, 2,000 samples of 250 normal
returns give `P(PSR > 0.95) ≈ 5 %` (asserted between 3.5 % and 6.5 %); the analytic standard
error matches the simulated dispersion of estimated Sharpe ratios within 8 %.

Audit rule QP-STAT-002 warns when PSR < `statistics.confidence` (0.95 by default) and
reports the MinTRL.

## Limitations

Serial correlation, regime changes and heteroskedasticity violate the i.i.d. assumption;
PSR is then over- or under-confident. The bootstrap interval (stationary bootstrap) is
reported alongside as a dependence-aware complement.

## References

- Sharpe, W. F. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1).
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4).
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working paper.
- Opdyke, J. D. (2007). Comparing Sharpe ratios: so where are the p-values?
  *Journal of Asset Management*, 8(5).
- Bailey, D. H. & López de Prado, M. (2012). The Sharpe ratio efficient frontier.
  *Journal of Risk*, 15(2).
