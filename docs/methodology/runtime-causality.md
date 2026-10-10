# Runtime causality: the future-perturbation test

## Problem

Static analysis looks at code. Look-ahead can also hide in library calls (`rank(pct=True)`
over a whole series), in models trained once on the full sample, in backward fills, or in
helper modules. The defining property of a causal strategy is behavioural: **the decision
at time t must not depend on data after t.** That property can be tested directly.

## Implementation

`quantproof.analyzers.causal.run_causality_test(func, data, config)`:

1. Integer columns are cast to float, and the **baseline** is computed on that same frame
   (dtype changes can never masquerade as violations).
2. `func` runs twice on identical data. If outputs differ, the strategy is
   non-deterministic and the test is inconclusive (QP-CAUSAL-002, WARN).
3. `n_timestamps` decision timestamps are spaced evenly (deterministically) over the
   **unique** timestamps between `min_history_fraction · n` and the second-to-last one.
4. For each decision timestamp *τ* and each scheme, `perturb_after(data, τ, scheme)`
   returns a copy in which every row with timestamp **> τ** is perturbed (rows ≤ τ are
   asserted bit-for-bit unchanged), and `func` is re-run. Cut-offs are timestamps, not
   positions, so panels and irregular or timezone-aware indexes are handled correctly.
5. A trial **fails** if any output at a timestamp ≤ τ differs from the baseline beyond
   `rtol`/`atol` (NaN equals NaN). Each trial records the evidence: the decision time
   τ, the first future observation modified, the scheme, the number of changed
   decisions, the first changed decision time and column, the original and perturbed
   values there, and max |Δ|. The finding message quotes one example, e.g. "data from
   2020-06-03 onward was perturbed (extreme); the decision at 2020-05-29 [AAA] changed
   from 0.25 to -0.25 (FORBIDDEN IN LIVE DECISION)".
6. A **control** run perturbs everything after the first row; if outputs never change,
   the strategy ignores its input and the test carries no information (QP-CAUSAL-004).

Perturbation schemes (`quantproof.analyzers.causal.perturbation`):

| Scheme | Rows after τ become |
|---|---|
| additive | value + N(0, (3·magnitude·std(Δx))²), one offset per row shared by price columns, floored to keep prices positive |
| multiplicative | value × exp(3·magnitude·std(Δlog x)·Z + magnitude·std(Δlog x)), shared across price columns |
| permutation | a random permutation of the future rows (values move, timestamps stay); needs ≥ 2 future timestamps |
| replacement | rows drawn with replacement from the data at or before τ (plausible values, wrong time) |
| extreme | value × factor, factor ∈ {0.2, 5} per row (e.g. 103 → 515, 104 → 20.8) |

Configuration (`CausalityConfig`): `n_timestamps`, `schemes`, `magnitude`, `seed`
(defaults to the audit seed), `min_history_fraction`, `rtol`, `atol`.

**Panels.** For `(timestamp, symbol)` data each symbol is perturbed with its own scale,
and permutation/replacement never move values across symbols. Cross-sectional operations
at a single timestamp (ranks, demeaning across symbols) are causal and pass; a
cross-asset look-ahead (e.g. ranking on next-day returns) fails.

Applying the same per-row factor/offset to open, high, low and close keeps OHLC
relationships valid, so strategies that validate their inputs still run. All
randomness is seeded.

## Assumptions

- The strategy is a function of the data frame it receives (no hidden I/O or global state).
- Outputs are aligned to timestamps (Series/DataFrame with the data's index, or arrays of
  the same length).
- Numerical tolerance: by default `rtol=1e-9`, `atol=1e-12`. Rolling and EWM computations
  in pandas are sequential, so causal strategies reproduce exactly.

## Example

| Strategy | Trials changed |
|---|---|
| trailing moving-average crossover (`examples/clean_strategy`) | 0 of 39 evaluated trials (1 permutation trial skipped: one future timestamp): PASS |
| `close.shift(-1)` signal | fails |
| `rolling(11, center=True)` | fails |
| `(x - x.mean()) / x.std()` over the full sample | fails |
| `bfill()` | fails |
| `rank(pct=True)` over the full series | fails |
| scikit-learn model fitted once on all rows | fails |

(See `tests/unit/test_causality.py`.)

## Limitations

- **Passing is evidence, not proof.** Only the tested timestamps and schemes are covered,
  and a strategy could be insensitive to these particular perturbations while still
  reading the future (e.g. a threshold that the perturbation never crosses).
- The test cannot see leakage that is already embedded in the input data: survivorship
  bias, restated fundamentals, timestamps that record availability incorrectly.
- Strategies that crash on perturbed data are reported (QP-CAUSAL-003) but those trials
  are not evaluated.
- Cost: `2 + n_timestamps × n_schemes + 1` strategy evaluations (43 by default, 23 in the
  quick profile, 83 in strict), minus skipped trials (asserted in
  `tests/unit/test_complexity.py`).
- **False positives.** A strategy that legitimately uses future data for *labels* or
  plots will fail if those values are part of the returned decisions. Return only live
  decisions from the strategy function; compute labels elsewhere. Clean transformations
  verified not to trigger the test include rolling/expanding/EWM statistics, cumulative
  max, quantiles, loops, weekly resampling with a shift, expanding model fits,
  cross-sectional ranks and group-wise cumulative sums (`tests/unit/test_causality.py`,
  `tests/unit/test_panel.py`).

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 11
  ("The Dangers of Backtesting").
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014). Pseudo-mathematics
  and financial charlatanism. *Notices of the AMS*, 61(5).
