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

**Null hypothesis.** H0: the selected strategy's true Sharpe ratio is no larger than the
expected maximum of *N* zero-skill trials. DSR is the probability (under the PSR
approximation) that the observed Sharpe exceeds that hurdle.

**Expected maximum — approximation vs exact.** `expected_max_sharpe(N, V,
method="approximation")` is the paper's closed form; `method="exact"` integrates
`E[max Z_i] = ∫ x·N·φ(x)·Φ(x)^(N−1) dx` numerically. The audit and the CLI use the exact
value and report `expected_max_method`. Relative error of the closed form (tested against
`1/√π` for N = 2, `3/(2√π)` for N = 3, and Monte Carlo):

| N | 2 | 10 | 100 | 1000 |
|---|---|---|---|---|
| approximation / exact − 1 | −7.9 % | +2.3 % | +0.9 % | +0.4 % |

## What "trials" means

*N* is the number of strategy variants whose results could have been selected: every
parameter set, feature set, universe, rule change or model that was backtested,
**including ones that were discarded**. The audit reports, in `statistics.trials`:

- `declared` / `used_for_dsr` — the count used;
- `source` — `statistics.trials (declared in config)`, `PARAM_GRID size`,
  `columns of trial_returns`, or `not declared (assumed 1)`;
- `effective_estimate` — when trial returns are available, the Li & Ji (2005)
  eigenvalue estimate of the number of effectively independent trials. It is reported
  for context; the DSR still uses the declared count (correlated trials make the declared
  count conservative). The estimate credits fractional eigenvalue parts, so it is a rough
  upper bound on the search's dimension, not an exact correction.

Entering `trials=5000` does **not** produce a perfect multiple-testing correction: it makes
the hurdle consistent with 5000 independent tries, and is only as honest as the number
entered. QuantProof cannot see trials that were never declared.

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
  effective number of trials; QuantProof uses the count you declare and reports the
  Li & Ji estimate alongside it.
- The PSR assumptions apply.

## Example

The exact expected maximum matches the Monte Carlo mean of the maximum of 50 standard
normals within four standard errors, and the closed form's documented error is asserted
(tested). Selecting the best of 200 noise
strategies yields DSR > 0.95 in at most ~5 % of simulations (tested).
For `examples/overfit_strategy` (240 trials) the observed net Sharpe of 0.43 is below the
expected maximum of 0.82, giving DSR ≈ 0.17.

## Limitations

DSR does **not** show that the strategy will be profitable, that its edge is economically
meaningful after costs, or that the declared trial count is complete.
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
