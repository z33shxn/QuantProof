# Deflated Sharpe Ratio (DSR)

## Problem

If many strategy variants are tried and the best is reported, its Sharpe ratio is biased
upward even when no variant has skill. The question is not "is SR > 0?" but "is SR larger
than the best of N lucky draws?".

## Implementation

`quantproof.statistics.deflated_sharpe_ratio(returns, n_trials, trial_sharpes=None,
sharpe_variance=None)` implements Bailey & López de Prado (2014):

```text
SR0 = √V · [ (1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e)) ]      (γ = 0.5772…)
DSR = PSR(SR0)
```

`SR0` is the expected maximum Sharpe ratio of *N* independent zero-skill trials with
cross-trial Sharpe variance *V* (`expected_max_sharpe`). All quantities are per-period.

Choice of *V*, recorded in the result as `variance_source`:

1. `explicit` — passed by the user;
2. `trial_sharpes` — sample variance of the per-period Sharpe ratios of all trials (as in
   the paper). The audit uses this whenever a `PARAM_GRID` or `trial_returns` is supplied;
3. `null_sampling_variance_1/(T-1)` — otherwise, the approximate sampling variance of a
   Sharpe estimate under zero true Sharpe and normal returns.

`N` comes from `statistics.trials`, else the size of `PARAM_GRID` (the full grid, even
when the audit evaluates a sub-sample), else 1 — in which case QP-STAT-005 reminds you
that DSR equals PSR and understates selection bias.

## Assumptions

- Trials are independent. Correlated variants (neighbouring parameters) make the raw count
  conservative — the hurdle is too high. The paper suggests clustering to estimate the
  effective number of trials; QuantProof uses the count you declare.
- The PSR assumptions apply.

## Example

`expected_max_sharpe(N, 1)` matches the Monte Carlo mean of the maximum of N standard
normals within 3 % for N = 10, 100, 1000 (tested). Selecting the best of 200 noise
strategies yields DSR > 0.95 in at most ~5 % of simulations (tested).
For `examples/overfit_strategy` (240 trials) the observed net Sharpe of 0.43 is below the
expected maximum of 0.82, giving DSR ≈ 0.17.

## Limitations

DSR corrects only for the trials you report. Undisclosed exploration (other datasets,
features, ideas discarded early) is invisible to it.

## References

- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for
  selection bias, backtest overfitting and non-normality. *Journal of Portfolio
  Management*, 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014). Pseudo-mathematics
  and financial charlatanism. *Notices of the AMS*, 61(5).
- Harvey, C. R., Liu, Y. & Zhu, H. (2016). … and the cross-section of expected returns.
  *Review of Financial Studies*, 29(1).
