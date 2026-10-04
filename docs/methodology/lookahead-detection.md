# Look-ahead detection by static analysis

## Problem

Most look-ahead bugs are visible in the code: a negative `shift`, a centered rolling
window, a scaler fitted on the full sample, a signal multiplied by the return of the
same bar. They are easy to write, easy to miss in review, and they make backtests look
far better than anything achievable in real time.

## Implementation

`quantproof.analyzers.static` parses Python source with the standard-library `ast` module
(never regular expressions on code) and builds a `ModuleContext` once per file:

- every call site with its name resolved through import aliases
  (`from sklearn.model_selection import train_test_split as tts` resolves to
  `train_test_split`);
- parent links and scopes for every node;
- module-level literal constants (e.g. `FEATURES = [...]`, `PARAM_GRID = {...}`);
- train/test split events (split functions, slicing assigned to `*train*`/`*test*` names);
- comments, for inline suppression (`# quantproof: ignore[QP002]`);
- a **taint analysis** for future information.

**Taint analysis.** Statements are processed in order within each scope. Sources are
`shift`/`diff`/`pct_change` with a negative period (QP001), `np.roll` with a negative
shift, and `x[i + k]` indexing with a loop variable (QP003). Taint propagates through
assignments to names, to string column keys (`df["fwd"] = ...`), to attributes, and
through `DataFrame.assign`; re-assigning an untainted value clears it. Sinks are:

- **signal** — a value returned from `generate_signals` (or another strategy-like
  function) or assigned to a name/column matching signal/position/weight;
- **feature** — the first argument of `fit`, `fit_transform`, `predict`, `transform`, …;
- **label** — the target argument of `fit`, or a name like `y`, `target`, `label`.

A negative shift that reaches a signal or feature is a **FAIL**; one used only to build
a label is **INFO** (that is how prediction targets are legitimately built); one that
cannot be classified is **WARN**.

**Rules.**

| Rule | Detects | Default severity |
|---|---|---|
| QP001 | negative `shift`/`diff`/`pct_change` | FAIL if it reaches signals/features, INFO for labels, else WARN |
| QP002 | `rolling(..., center=True)` | FAIL |
| QP003 | `x[i + k]` in loops, `np.roll`, future-named model inputs | FAIL if traced to signals/features, else WARN |
| QP004 | `train_test_split` without `shuffle=False` | WARN (context-dependent) |
| QP005 | `KFold(shuffle=True)`, `ShuffleSplit`, `StratifiedKFold`, …; default K-fold in searches | WARN / INFO |
| QP006 | scaler/imputer/PCA/selector fitted before the first split, on non-training data | FAIL |
| QP007 | `(x - x.mean()) / x.std()`, `zscore`, `scale`, transformers fitted with no split | WARN |
| QP008 | target column in the feature matrix, or not dropped, or `fit(X, X)` | FAIL |
| QP009 | declared `signal_lag=0`, or `signal.shift(1) * returns` without a documented assumption | WARN |
| QP010 | un-lagged signal × same-period returns | FAIL |
| QP011 | strategy returns computed with no cost term anywhere / zero declared costs | WARN |
| QP012 | future-derived data used as model input (one finding per origin) | FAIL |
| QP013 | grid/random searches, nested loops, `itertools.product`, `PARAM_GRID` | WARN above the threshold (100), else INFO with the trial count |
| QP014 | models fitted or parameters searched with no OOS construct | WARN |
| QP015 | `filtfilt`, Savitzky-Golay, Gaussian smoothing, HP filter, seasonal decomposition, FFT, linear `interpolate`, `bfill` | WARN |

Rules are classes registered with `@register`; adding one does not change any other
module (see CONTRIBUTING.md).

## Assumptions

- Code is analysed one file at a time. Data flow across modules, through function calls
  with non-literal arguments, or through containers is not tracked.
- Names carry meaning: a variable called `signal` is assumed to be a decision, `returns`
  a realised return. Unusual naming reduces recall.
- Context-dependent patterns (shuffled splits, K-fold) are WARN with an explanation, not
  FAIL: they are valid for genuinely i.i.d. samples.

## Example

```bash
quantproof scan tests/fixtures/broken
```

reports one finding per deliberately broken fixture (`qp001_negative_shift.py` →
QP001 FAIL, …, `qp015_backfill.py` → QP015 WARN); the clean fixtures produce no WARN or
FAIL. These files are part of the test suite.

## Limitations

Static analysis can neither prove the presence nor the absence of look-ahead. It is one
of three independent lines of evidence; the [runtime causality test](runtime-causality.md)
checks behaviour, and [leakage diagnostics](../api/rules.md#leakage) check values.

## References

- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining:
  formulation, detection, and avoidance. *ACM TKDD*, 6(4).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, ch. 7 and 11.
- Python documentation, `ast` — Abstract Syntax Trees.
