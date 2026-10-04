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
3. Decision positions `k` are spaced evenly (deterministically) between
   `min_history_fraction · n` and `n − 2`.
4. For each `k` and each scheme, a copy of the data is perturbed **only on rows > k**
   (this is asserted bit-for-bit for rows ≤ k), and `func` is re-run.
5. A trial **fails** if any output at a timestamp ≤ `index[k]` differs from the baseline
   beyond `rtol`/`atol` (NaN equals NaN). The finding records, per trial, the decision
   time, scheme, number of changed decisions, first changed timestamp and max |Δ|.
6. A **control** run perturbs everything after the first row; if outputs never change,
   the strategy ignores its input and the test carries no information (QP-CAUSAL-004).

Perturbation schemes (`quantproof.analyzers.causal.perturbation`):

| Scheme | Rows > k become |
|---|---|
| additive | value + N(0, (3·magnitude·std(Δx))²), one offset per row shared by price columns, floored to keep prices positive |
| multiplicative | value × exp(3·magnitude·std(Δlog x)·Z + magnitude·std(Δlog x)), shared across price columns |
| permutation | a random permutation of the future rows (values move, timestamps stay) |
| shock | value × factor, factor ∈ {0.2, 5} per row (e.g. 103 → 515, 104 → 20.8) |

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
| trailing moving-average crossover | 0 / 31 (PASS) |
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
- Cost: `1 + 2 + n_timestamps × n_schemes` strategy evaluations (35 by default; quick mode 19).

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 11
  ("The Dangers of Backtesting").
- Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014). Pseudo-mathematics
  and financial charlatanism. *Notices of the AMS*, 61(5).
