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

## Analysis levels

Every static rule works at one of three documented levels. Higher levels find more and
are more fragile; findings always say which rule fired and with what *analysis
confidence* (how sure the analyzer is that the matched pattern means what the rule says
— not a statistical probability).

| Level | Name | What it can resolve | Example |
|---|---|---|---|
| L1 | AST-local | a single call or expression | `rolling(5, center=True)`, `shift(-1)` literal |
| L2 | symbol resolution | import aliases (`import pandas as pd`, `from scipy.signal import filtfilt as ff`), module-qualified calls, callable aliases (`fwd = pd.Series.shift; fwd(x, -1)`), methods, nested functions, lambdas, comprehensions, chained operations (`df.close.shift(-1).rolling(3).mean()`) | `from sklearn.model_selection import train_test_split as tts` |
| L3 | limited data flow | taint from a future-information source through assignments, string column keys, attributes, `DataFrame.assign`, dict/list literals, and **same-file helper functions** (each function gets a summary of what its return value and the columns it writes depend on, iterated to a fixed point over 3 rounds) to a sink | `def fwd(s): return s.shift(-1)` … `df["sig"] = fwd(df.close)` |

Not resolved (documented boundaries, each covered by a test in
`tests/unit/test_static_adversarial.py`): data flow across files/modules, mutation
through container methods (`lst.append(x.shift(-1))`), dynamic attribute access
(`getattr(df, "shift")(-1)`), `exec`/`eval`, and values whose period is computed at
runtime (a non-literal `center=` or shift period is a WARN, not a FAIL).

## Sources, sinks and usage

**Sources** of future information: `shift`/`diff`/`pct_change` with a negative period
(QP001), centered rolling windows (QP002), `np.roll` with a negative shift and `x[i + k]`
in loops (QP003), full-sample normalisation (QP007) and non-causal operations such as
`bfill`, `filtfilt`, Savitzky–Golay, HP filters, seasonal decomposition, FFT and linear
`interpolate` (QP015).

**Sinks** decide how a source is used:

- **live decision** — a value returned from `generate_signals` (or any strategy entry
  point, including the name of a callable passed to `audit`), or assigned to a
  signal/position/weight name or column, or a model *feature*;
- **label / analysis** — the target argument of `fit`, a name like `y`/`target`/`label`,
  or plotting/reporting code.

The same pattern is therefore reported differently by usage:

| Usage | Severity | Label in reports |
|---|---|---|
| reaches a live decision | FAIL | **FORBIDDEN IN LIVE DECISION** |
| only builds a label or analysis output | INFO | **LEGITIMATE FOR LABEL / ANALYSIS** |
| cannot be classified | WARN | — |

Full-sample normalisation (QP007) is always a WARN (it is legitimate for offline analysis
and cross-sectional normalisation across symbols at one timestamp is causal). A scaler
fitted before a split is a FAIL only when the fitted object is traceable to the later
train/test data (QP006); otherwise WARN at low confidence. QP010 (un-lagged signal ×
same-period return) is a FAIL only when both sides are traced to close prices; when the
price sources cannot be traced the finding is a WARN saying "Execution semantics could
not be determined automatically" (an open-based signal traded on open-to-close returns
is not flagged).

## Rules

The exact severity policy, rationale, limitations and an example for every rule are in
the generated [rule catalogue](../api/rules.md) (`quantproof rules show QP010`). Rules are
classes registered with `@register` and read their documentation from the registry; adding
one does not change any other module (see CONTRIBUTING.md).

## Assumptions

- Code is analysed one file at a time (L3 is intra-file).
- Names carry meaning: a variable called `signal` is assumed to be a decision, `returns`
  a realised return. Unusual naming reduces recall.
- Context-dependent patterns (shuffled splits, K-fold) are WARN with an explanation, not
  FAIL: they are valid for genuinely i.i.d. samples.

## Adversarial and false-positive suites

`tests/unit/test_static_adversarial.py` contains cases written to evade the analyzer
(helpers, aliases, hidden centered windows, merged future data, a global scaler, a renamed
target, same-bar fills behind an adapter, excessive search, …) and cases written to
trigger false positives (plots, labels, i.i.d. random splits, offline normalisation after a
split, same-bar execution with open-price data, cross-sectional transforms). Each case
asserts the exact expected rule and severity.

## Example

```bash
quantproof scan tests/fixtures/broken
```

reports one finding per deliberately broken fixture (`qp001_negative_shift.py` →
QP001 FAIL, …, `qp015_backfill.py` → QP015 FAIL because the back-filled series is
returned as the signal); the clean fixtures produce no WARN or FAIL. These files are part of the test suite.

## Limitations

Static analysis can neither prove the presence nor the absence of look-ahead. It is one
of three independent lines of evidence; the [runtime causality test](runtime-causality.md)
checks behaviour, and [leakage diagnostics](../api/rules.md#leakage--methodology) check values.

## References

- Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining:
  formulation, detection, and avoidance. *ACM TKDD*, 6(4).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, ch. 7 and 11.
- Python documentation, `ast` — Abstract Syntax Trees.
